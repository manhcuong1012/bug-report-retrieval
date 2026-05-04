from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def train_bucket_lookup(train_records_path: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for record in iter_jsonl(train_records_path):
        lookup[record["bug_id"]] = record["bucket_id"]
    return lookup


def hit_at_k(query_bucket: str, ranked_bug_ids: list[str], bugid_to_bucket: dict[str, str], k: int) -> float:
    for bug_id in ranked_bug_ids[:k]:
        if bugid_to_bucket.get(bug_id) == query_bucket:
            return 1.0
    return 0.0


def reciprocal_rank(
    query_bucket: str, ranked_bug_ids: list[str], bugid_to_bucket: dict[str, str]
) -> float:
    for rank, bug_id in enumerate(ranked_bug_ids, start=1):
        if bugid_to_bucket.get(bug_id) == query_bucket:
            return 1.0 / rank
    return 0.0


def evaluate(predictions_path: Path, train_records_path: Path) -> dict[str, Any]:
    bugid_to_bucket = train_bucket_lookup(train_records_path)

    num_queries = 0
    recall_at_1 = 0.0
    recall_at_5 = 0.0
    recall_at_10 = 0.0
    mrr = 0.0

    for prediction in iter_jsonl(predictions_path):
        ranked_bug_ids = [result["bug_id"] for result in prediction.get("results", [])]
        query_bucket = prediction["query_bucket_id"]
        num_queries += 1
        recall_at_1 += hit_at_k(query_bucket, ranked_bug_ids, bugid_to_bucket, 1)
        recall_at_5 += hit_at_k(query_bucket, ranked_bug_ids, bugid_to_bucket, 5)
        recall_at_10 += hit_at_k(query_bucket, ranked_bug_ids, bugid_to_bucket, 10)
        mrr += reciprocal_rank(query_bucket, ranked_bug_ids, bugid_to_bucket)

    divisor = max(num_queries, 1)
    return {
        "num_queries": num_queries,
        "Recall@1": recall_at_1 / divisor,
        "Recall@5": recall_at_5 / divisor,
        "Recall@10": recall_at_10 / divisor,
        "MRR": mrr / divisor,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval predictions.")
    parser.add_argument("--predictions", required=True, help="Predictions JSONL path")
    parser.add_argument(
        "--train-records",
        default="data/train.jsonl",
        help="Train records JSONL path",
    )
    parser.add_argument("--output", required=True, help="Metrics JSON output path")
    args = parser.parse_args()

    metrics = evaluate(Path(args.predictions), Path(args.train_records))
    write_json(Path(args.output), metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
