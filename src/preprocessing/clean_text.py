from __future__ import annotations

import re

MULTISPACE_RE = re.compile(r"\s+")
NON_TEXT_RE = re.compile(r"[^a-z0-9._/#\-\+\s]")


def normalize_whitespace(text: str) -> str:
    return MULTISPACE_RE.sub(" ", text).strip()


def normalize_text(text: str) -> str:
    lowered = text.lower()
    cleaned = NON_TEXT_RE.sub(" ", lowered)
    return normalize_whitespace(cleaned)


def build_text_raw(summary: str, description: str) -> str:
    return normalize_whitespace(f"{summary} {description}".strip())


def build_text_clean(summary: str, description: str) -> str:
    return normalize_text(build_text_raw(summary, description))


def build_weighted_text(summary: str, description: str, summary_repeat: int = 3) -> str:
    repeated_summary = " ".join([summary] * max(summary_repeat, 1))
    return normalize_text(f"{repeated_summary} {description}".strip())


def build_metadata_text(prefix: str, value: str) -> str:
    normalized = normalize_text(value)
    if not normalized:
        return ""
    return " ".join(f"{prefix}.{token}" for token in normalized.split(" ") if token)


def build_retrieval_text(
    summary: str,
    description: str,
    component: str = "",
    priority: str = "",
    severity: str = "",
    summary_repeat: int = 3,
    component_repeat: int = 1,
    priority_repeat: int = 1,
    severity_repeat: int = 1,
) -> str:
    parts = [
        " ".join([summary] * max(summary_repeat, 1)),
        description,
        " ".join([build_metadata_text("component", component)] * max(component_repeat, 0)),
        " ".join([build_metadata_text("priority", priority)] * max(priority_repeat, 0)),
        " ".join([build_metadata_text("severity", severity)] * max(severity_repeat, 0)),
    ]
    return normalize_text(" ".join(part for part in parts if part).strip())
