from __future__ import annotations

import argparse
import json
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
from retrieval.bug_feature_scorer import (
    build_query_profile,
    precompute_bug,
    score_pair,
    tokenize,
)

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S %z"


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


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_index(train_records_path: Path, df_threshold_ratio: float = 0.10):
    records: list[dict] = []
    inverted: dict[str, list[str]] = {}
    component_index: dict[str, list[str]] = {}
    timestamps: dict[str, float] = {}
    precomputed: dict[str, dict] = {}
    doc_freq: Counter[str] = Counter()
    num_docs = 0

    for record in iter_jsonl(train_records_path):
        num_docs += 1
        bug_id = record["bug_id"]
        records.append(record)
        timestamps[bug_id] = parse_timestamp(record["created_at"]).timestamp()

        pre = precompute_bug(record)
        precomputed[bug_id] = pre

        all_tokens = tokenize(record.get("summary", "")) + tokenize(record.get("description", ""))
        seen: set[str] = set()
        for token in all_tokens:
            if len(token) <= 2:
                continue
            if token not in seen:
                doc_freq[token] += 1
                seen.add(token)
            inverted.setdefault(token, []).append(bug_id)

        comp = record.get("component", "UNKNOWN")
        component_index.setdefault(comp, []).append(bug_id)

    high_df = frozenset(
        token for token, freq in doc_freq.items()
        if freq > num_docs * df_threshold_ratio
    )

    clean_inverted: dict[str, list[str]] = {
        token: bug_ids for token, bug_ids in inverted.items()
        if token not in high_df
    }

    print(f"[index] {num_docs} docs, {len(clean_inverted)} terms (dropped {len(inverted) - len(clean_inverted)} high-df), {len(component_index)} components")
    return clean_inverted, component_index, timestamps, precomputed, num_docs


# ---------------------------------------------------------------------------
# Candidate pre-filtering
# ---------------------------------------------------------------------------

def get_candidates(
    query_record: dict,
    inverted_index: dict[str, list[str]],
    component_index: dict[str, list[str]],
    timestamps: dict[str, float],
    max_candidates: int = 5000,
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
        bid: hits for bid, hits in candidate_hits.items()
        if timestamps.get(bid, float("inf")) <= query_ts
    }

    if len(valid) > max_candidates:
        valid = dict(Counter(valid).most_common(max_candidates))

    return valid


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    test_records_path: Path,
    train_records_path: Path,
    predictions_output: Path,
    metrics_output: Path,
    top_k: int = 10,
    max_candidates: int = 5000,
) -> dict[str, Any]:
    inverted_index, component_index, timestamps, precomputed, num_docs = build_index(
        train_records_path
    )

    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    total_queries = 0
    total_candidates_scored = 0

    with predictions_output.open("w", encoding="utf-8") as handle:
        for record in iter_jsonl(test_records_path):
            total_queries += 1
            query_pre = precompute_bug(record)
            query_profile = build_query_profile(
                record.get("summary", ""),
                record.get("description", ""),
            )

            candidates = get_candidates(
                record, inverted_index, component_index, timestamps, max_candidates
            )
            total_candidates_scored += len(candidates)

            scored: list[tuple[str, float]] = []
            for bug_id in candidates:
                cand_pre = precomputed.get(bug_id)
                if cand_pre is None:
                    continue
                s = score_pair(query_pre, cand_pre, query_profile)
                scored.append((bug_id, s))

            ranked = nlargest(top_k, scored, key=lambda x: x[1])

            prediction = {
                "query_bug_id": record["bug_id"],
                "project": "mozilla",
                "query_bucket_id": record["bucket_id"],
                "results": [
                    {
                        "bug_id": doc_id,
                        "project": "mozilla",
                        "score": score,
                        "rank": rank,
                    }
                    for rank, (doc_id, score) in enumerate(ranked, start=1)
                ],
            }
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")

            if total_queries % 100 == 0:
                avg_cand = total_candidates_scored / total_queries
                print(f"[retrieve] {total_queries} queries done, avg candidates/query: {avg_cand:.0f}")

    avg_cand = total_candidates_scored / max(total_queries, 1)
    print(f"[retrieve] finished {total_queries} queries, avg candidates/query: {avg_cand:.0f}")

    metrics = evaluate_predictions(predictions_output, train_records_path)
    metrics.update({
        "method": "feature_based",
        "top_k": top_k,
        "max_candidates": max_candidates,
        "num_queries": total_queries,
    })
    write_json(metrics_output, metrics)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run feature-based retrieval on Mozilla.")
    parser.add_argument("--train-records", default="data/train.jsonl")
    parser.add_argument("--test-records", default="data/test.jsonl")
    parser.add_argument("--predictions-output", default="reports/feature_predictions.jsonl")
    parser.add_argument("--metrics-output", default="reports/feature_metrics.json")
    parser.add_argument("--top-k", type=int, default=10)
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
