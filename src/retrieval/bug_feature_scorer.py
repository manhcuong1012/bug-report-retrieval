from __future__ import annotations

import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing.clean_text import normalize_text

# ---------------------------------------------------------------------------
# Technical keywords & patterns
# ---------------------------------------------------------------------------

TECHNICAL_KEYWORDS = frozenset({
    "crash", "abort", "segfault", "timeout", "hang",
    "memory", "leak", "exception", "null", "stack",
    "assert", "assertion", "overflow", "underflow",
    "deadlock", "race", "corrupt", "corruption",
    "oom", "sigabrt", "sigsegv", "error", "fail",
    "failed", "failure", "broken", "regression",
})

FILE_EXT_RE = re.compile(
    r"^[\w\-./\\]+\.(?:js|jsx|ts|tsx|cpp|cc|c|h|hpp|py|java|rs|html|css|xml|xul|jsm|json|toml|ini|cfg|idl|webidl)$"
)
CAMEL_CASE_RE = re.compile(r"^[a-z]+(?:[A-Z][a-z0-9]*)+$|^[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+$")
STACK_FRAME_RE = re.compile(r"^#\d+$")

MOZILLA_TOKENS = frozenset({
    "treeherder", "taskcluster", "bugzilla", "gecko", "necko",
    "mochitest", "xpcshell", "reftest", "crashtest", "wpt",
    "spidermonkey", "webrender", "stylo",
})

TECH_NOISE = frozenset({
    "mozilla/5.0", "gecko/20100101", "applewebkit", "firefox",
    "chrome", "safari", "linux", "windows", "x86_64", "x11",
})

# ---------------------------------------------------------------------------
# Tokenize
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    return [t for t in normalized.split() if t]


def tokenize_to_set(text: str) -> set[str]:
    return set(tokenize(text))


# ---------------------------------------------------------------------------
# Technical token extraction
# ---------------------------------------------------------------------------

def extract_technical_tokens(text: str) -> set[str]:
    tokens = tokenize(text)
    tech: set[str] = set()
    for t in tokens:
        if t in TECH_NOISE:
            continue
        if t in TECHNICAL_KEYWORDS:
            tech.add(t)
        elif t in MOZILLA_TOKENS:
            tech.add(t)
        elif FILE_EXT_RE.match(t):
            tech.add(t)
        elif CAMEL_CASE_RE.match(t):
            tech.add(t)
        elif STACK_FRAME_RE.match(t):
            tech.add(t)
        elif "::" in t:
            tech.add(t)
        elif t.startswith("nsi") or t.startswith("moz"):
            tech.add(t)
    return tech


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def jaccard(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def overlap_ratio(query_set: set[str], candidate_set: set[str]) -> float:
    if not query_set:
        return 0.0
    return len(query_set & candidate_set) / len(query_set)


# ---------------------------------------------------------------------------
# Query profile (for heuristics)
# ---------------------------------------------------------------------------

STACKTRACE_RE = re.compile(r"#\d+\s+0x[0-9a-f]|at\s+\S+:\d+", re.IGNORECASE)


def build_query_profile(summary: str, description: str) -> dict:
    sum_tokens = tokenize(summary)
    desc_tokens = tokenize(description)
    raw_text = f"{summary} {description}"

    path_count = sum(1 for t in desc_tokens if FILE_EXT_RE.match(t))
    stacktrace_matches = len(STACKTRACE_RE.findall(raw_text))

    return {
        "is_short": len(sum_tokens) < 8 and len(desc_tokens) < 20,
        "has_stacktrace": stacktrace_matches >= 3,
        "has_many_paths": path_count >= 3,
    }


# ---------------------------------------------------------------------------
# Feature computation for a single (query, candidate) pair
# ---------------------------------------------------------------------------

def compute_features(
    q_summary_tokens: set[str],
    q_desc_tokens: set[str],
    q_component: str,
    q_priority: str,
    q_severity: str,
    q_tech_tokens: set[str],
    c_summary_tokens: set[str],
    c_desc_tokens: set[str],
    c_component: str,
    c_priority: str,
    c_severity: str,
    c_tech_tokens: set[str],
) -> dict[str, float]:
    q_tech_kw = q_tech_tokens & TECHNICAL_KEYWORDS
    c_tech_kw = c_tech_tokens & TECHNICAL_KEYWORDS

    both_priority_unknown = (q_priority == "UNKNOWN" and c_priority == "UNKNOWN")

    return {
        "summary_jaccard": jaccard(q_summary_tokens, c_summary_tokens),
        "summary_overlap": overlap_ratio(q_summary_tokens, c_summary_tokens),
        "desc_jaccard": jaccard(q_desc_tokens, c_desc_tokens),
        "desc_overlap": overlap_ratio(q_desc_tokens, c_desc_tokens),
        "component_match": 1.0 if q_component == c_component else 0.0,
        "severity_match": 1.0 if q_severity == c_severity else 0.0,
        "priority_match": 0.0 if both_priority_unknown else (1.0 if q_priority == c_priority else 0.0),
        "tech_keyword_overlap": overlap_ratio(q_tech_kw, c_tech_kw),
        "tech_token_jaccard": jaccard(q_tech_tokens, c_tech_tokens),
    }


# ---------------------------------------------------------------------------
# Final score
# ---------------------------------------------------------------------------

BASE_WEIGHTS = {
    "summary_jaccard":      0.30,
    "summary_overlap":      0.05,
    "desc_jaccard":         0.15,
    "desc_overlap":         0.05,
    "component_match":      0.22,
    "severity_match":       0.02,
    "priority_match":       0.01,
    "tech_keyword_overlap": 0.08,
    "tech_token_jaccard":   0.12,
}


def compute_final_score(features: dict[str, float], query_profile: dict) -> float:
    W = dict(BASE_WEIGHTS)

    if query_profile["is_short"]:
        W["component_match"] *= 1.5
        W["summary_jaccard"] *= 1.3
    if query_profile["has_stacktrace"]:
        W["tech_token_jaccard"] *= 1.5
    if query_profile["has_many_paths"]:
        W["tech_token_jaccard"] *= 1.3
        W["desc_overlap"] *= 1.2

    total = sum(W.values())
    W = {k: v / total for k, v in W.items()}

    return sum(W[k] * features[k] for k in W)


# ---------------------------------------------------------------------------
# Convenience: precompute per-bug data for reuse
# ---------------------------------------------------------------------------

def precompute_bug(record: dict) -> dict:
    return {
        "bug_id": record["bug_id"],
        "summary_tokens": tokenize_to_set(record.get("summary", "")),
        "desc_tokens": tokenize_to_set(record.get("description", "")),
        "component": record.get("component", "UNKNOWN"),
        "priority": record.get("priority", "UNKNOWN"),
        "severity": record.get("severity", "UNKNOWN"),
        "tech_tokens": extract_technical_tokens(
            f"{record.get('summary', '')} {record.get('description', '')}"
        ),
    }


def score_pair(query_pre: dict, candidate_pre: dict, query_profile: dict) -> float:
    features = compute_features(
        q_summary_tokens=query_pre["summary_tokens"],
        q_desc_tokens=query_pre["desc_tokens"],
        q_component=query_pre["component"],
        q_priority=query_pre["priority"],
        q_severity=query_pre["severity"],
        q_tech_tokens=query_pre["tech_tokens"],
        c_summary_tokens=candidate_pre["summary_tokens"],
        c_desc_tokens=candidate_pre["desc_tokens"],
        c_component=candidate_pre["component"],
        c_priority=candidate_pre["priority"],
        c_severity=candidate_pre["severity"],
        c_tech_tokens=candidate_pre["tech_tokens"],
    )
    return compute_final_score(features, query_profile)
