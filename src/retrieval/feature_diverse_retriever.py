from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from heapq import nlargest
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.metrics import evaluate as evaluate_predictions
from preprocessing.clean_text import normalize_text
from retrieval.bug_feature_scorer import (
    build_query_profile,
    compute_features,
    extract_technical_tokens,
    precompute_bug,
    score_pair,
    tokenize,
)

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S %z"
NUMBER_RE = re.compile(r"\b\d+(?:[./:-]\d+)*\b")
FILE_TOKEN_RE = re.compile(
    r"^[\w\-./\\]+\.(?:js|jsx|ts|tsx|cpp|cc|c|h|hpp|py|java|rs|html|css|xml|xul|jsm|json|toml|ini|cfg|idl|webidl)$"
)


def parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, TIMESTAMP_FORMAT)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def build_bigrams(tokens: list[str]) -> set[str]:
    return {f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)}


def overlap_ratio(query_set: set[str], candidate_set: set[str]) -> float:
    if not query_set:
        return 0.0
    return len(query_set & candidate_set) / len(query_set)


def extract_numbers(text: str) -> set[str]:
    return set(NUMBER_RE.findall(text))


def extract_file_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if FILE_TOKEN_RE.match(token)}


def exact_summary_match(query_summary: str, candidate_summary: str) -> float:
    query_norm = normalize_text(query_summary)
    candidate_norm = normalize_text(candidate_summary)
    if not query_norm or not candidate_norm:
        return 0.0
    if query_norm == candidate_norm:
        return 1.0
    if min(len(query_norm.split()), len(candidate_norm.split())) < 4:
        return 0.0
    return 1.0 if query_norm in candidate_norm or candidate_norm in query_norm else 0.0


def precompute_diverse_bug(record: dict) -> dict:
    summary = record.get("summary", "")
    description = record.get("description", "")
    full_text = f"{summary} {description}"
    summary_tokens = tokenize(summary)
    base = precompute_bug(record)
    base.update({
        "summary": summary,
        "description": description,
        "summary_bigrams": build_bigrams(summary_tokens),
        "numbers": extract_numbers(full_text),
        "file_tokens": extract_file_tokens(full_text),
        "all_tech_tokens": extract_technical_tokens(full_text),
    })
    return base


def compute_detailed_scores(query_pre: dict, candidate_pre: dict, query_profile: dict) -> dict[str, float]:
    base_features = compute_features(
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

    phrase_overlap = overlap_ratio(query_pre["summary_bigrams"], candidate_pre["summary_bigrams"])
    number_overlap = overlap_ratio(query_pre["numbers"], candidate_pre["numbers"])
    file_path_overlap = overlap_ratio(query_pre["file_tokens"], candidate_pre["file_tokens"])
    exact_summary = exact_summary_match(query_pre["summary"], candidate_pre["summary"])
    technical_overlap = base_features["tech_token_jaccard"]
    component_match = base_features["component_match"]
    base_score = score_pair(query_pre, candidate_pre, query_profile)

    # Small additive boosts are intentional: this output is meant to help hybrid
    # fusion recover complementary technical/phrase candidates without replacing
    # the stable feature scorer.
    phrase_boost = 0.025 * phrase_overlap + 0.015 * exact_summary
    numeric_boost = 0.020 * number_overlap
    file_boost = 0.020 * file_path_overlap
    technical_boost = 0.025 * technical_overlap
    component_boost = 0.010 * component_match

    if query_profile["has_stacktrace"] or query_profile["has_many_paths"]:
        technical_boost *= 1.4
        file_boost *= 1.4

    final_score = (
        base_score
        + phrase_boost
        + numeric_boost
        + file_boost
        + technical_boost
        + component_boost
    )

    return {
        "score": final_score,
        "feature_score": base_score,
        "component_match": component_match,
        "summary_jaccard": base_features["summary_jaccard"],
        "summary_overlap": base_features["summary_overlap"],
        "desc_jaccard": base_features["desc_jaccard"],
        "desc_overlap": base_features["desc_overlap"],
        "tech_keyword_overlap": base_features["tech_keyword_overlap"],
        "tech_token_jaccard": technical_overlap,
        "summary_bigram_overlap": phrase_overlap,
        "exact_summary_match": exact_summary,
        "number_overlap": number_overlap,
        "file_path_overlap": file_path_overlap,
        "phrase_boost": phrase_boost,
        "numeric_boost": numeric_boost,
        "file_boost": file_boost,
        "technical_boost": technical_boost,
        "component_boost": component_boost,
    }


def build_index(train_records_path: Path, df_threshold_ratio: float = 0.10):
    inverted: dict[str, list[str]] = {}
    component_index: dict[str, list[str]] = {}
    timestamps: dict[str, float] = {}
    precomputed: dict[str, dict] = {}
    doc_freq: Counter[str] = Counter()
    num_docs = 0

    for record in iter_jsonl(train_records_path):
        num_docs += 1
        bug_id = record["bug_id"]
        timestamps[bug_id] = parse_timestamp(record["created_at"]).timestamp()
        precomputed[bug_id] = precompute_diverse_bug(record)

        all_tokens = tokenize(record.get("summary", "")) + tokenize(record.get("description", ""))
        seen: set[str] = set()
        for token in all_tokens:
            if len(token) <= 2:
                continue
            if token not in seen:
                doc_freq[token] += 1
                seen.add(token)
            inverted.setdefault(token, []).append(bug_id)

        component = record.get("component", "UNKNOWN")
        component_index.setdefault(component, []).append(bug_id)

    high_df = frozenset(
        token for token, freq in doc_freq.items()
        if freq > num_docs * df_threshold_ratio
    )
    clean_inverted = {
        token: bug_ids for token, bug_ids in inverted.items()
        if token not in high_df
    }

    print(
        f"[index-diverse] {num_docs} docs, {len(clean_inverted)} terms "
        f"(dropped {len(inverted) - len(clean_inverted)} high-df), "
        f"{len(component_index)} components"
    )
    return clean_inverted, component_index, timestamps, precomputed


def get_candidates(
    query_record: dict,
    inverted_index: dict[str, list[str]],
    component_index: dict[str, list[str]],
    timestamps: dict[str, float],
    max_candidates: int,
) -> dict[str, int]:
    query_tokens = tokenize(query_record.get("summary", "")) + tokenize(query_record.get("description", ""))
    query_ts = parse_timestamp(query_record["created_at"]).timestamp()

    candidate_hits: Counter[str] = Counter()
    for token in set(query_tokens):
        if len(token) <= 2:
            continue
        for bug_id in inverted_index.get(token, []):
            candidate_hits[bug_id] += 1

    for bug_id in component_index.get(query_record.get("component", ""), []):
        if bug_id not in candidate_hits:
            candidate_hits[bug_id] = 0

    valid = {
        bug_id: hits
        for bug_id, hits in candidate_hits.items()
        if timestamps.get(bug_id, float("inf")) <= query_ts
    }

    if len(valid) > max_candidates:
        valid = dict(Counter(valid).most_common(max_candidates))

    return valid


def retrieve(
    test_records_path: Path,
    train_records_path: Path,
    predictions_output: Path,
    metrics_output: Path,
    top_k: int = 50,
    max_candidates: int = 5000,
) -> dict[str, Any]:
    inverted_index, component_index, timestamps, precomputed = build_index(train_records_path)

    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    total_queries = 0
    total_candidates_scored = 0

    with predictions_output.open("w", encoding="utf-8") as handle:
        for record in iter_jsonl(test_records_path):
            total_queries += 1
            query_pre = precompute_diverse_bug(record)
            query_profile = build_query_profile(
                record.get("summary", ""),
                record.get("description", ""),
            )
            candidates = get_candidates(
                record,
                inverted_index,
                component_index,
                timestamps,
                max_candidates,
            )
            total_candidates_scored += len(candidates)

            scored: list[tuple[str, dict[str, float]]] = []
            for bug_id in candidates:
                candidate_pre = precomputed.get(bug_id)
                if candidate_pre is None:
                    continue
                scores = compute_detailed_scores(query_pre, candidate_pre, query_profile)
                scored.append((bug_id, scores))

            ranked = nlargest(top_k, scored, key=lambda item: item[1]["score"])
            prediction = {
                "query_bug_id": record["bug_id"],
                "project": "mozilla",
                "query_bucket_id": record["bucket_id"],
                "results": [
                    {
                        "bug_id": bug_id,
                        "project": "mozilla",
                        "score": scores["score"],
                        "rank": rank,
                        "feature_score": scores["feature_score"],
                        "component_match": scores["component_match"],
                        "tech_token_jaccard": scores["tech_token_jaccard"],
                        "summary_bigram_overlap": scores["summary_bigram_overlap"],
                        "exact_summary_match": scores["exact_summary_match"],
                        "number_overlap": scores["number_overlap"],
                        "file_path_overlap": scores["file_path_overlap"],
                    }
                    for rank, (bug_id, scores) in enumerate(ranked, start=1)
                ],
            }
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")

            if total_queries % 100 == 0:
                avg_cand = total_candidates_scored / total_queries
                print(f"[retrieve-diverse] {total_queries} queries done, avg candidates/query: {avg_cand:.0f}")

    avg_cand = total_candidates_scored / max(total_queries, 1)
    print(f"[retrieve-diverse] finished {total_queries} queries, avg candidates/query: {avg_cand:.0f}")

    metrics = evaluate_predictions(predictions_output, train_records_path)
    metrics.update({
        "method": "feature_diverse",
        "top_k": top_k,
        "max_candidates": max_candidates,
        "num_queries": total_queries,
        "notes": "Feature output for hybrid fusion: base feature score plus small phrase/technical/numeric boosts and per-result sub-scores.",
    })
    write_json(metrics_output, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run diversified feature retrieval for hybrid fusion.")
    parser.add_argument("--train-records", default="data/train.jsonl")
    parser.add_argument("--test-records", default="data/test.jsonl")
    parser.add_argument("--predictions-output", default="reports/feature_diverse_predictions.jsonl")
    parser.add_argument("--metrics-output", default="reports/feature_diverse_metrics.json")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-candidates", type=int, default=5000)
    args = parser.parse_args()

    metrics = retrieve(
        test_records_path=Path(args.test_records),
        train_records_path=Path(args.train_records),
        predictions_output=Path(args.predictions_output),
        metrics_output=Path(args.metrics_output),
        top_k=args.top_k,
        max_candidates=args.max_candidates,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
