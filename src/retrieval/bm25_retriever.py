from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from heapq import nlargest
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.metrics import evaluate as evaluate_predictions  # type: ignore
from preprocessing.clean_text import build_weighted_text  # type: ignore

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


def record_tokens(record: dict[str, str], summary_repeat: int = 3) -> list[str]:
    weighted_text = build_weighted_text(
        record.get("summary", ""),
        record.get("description", ""),
        summary_repeat=summary_repeat,
    )
    return [token for token in weighted_text.split(" ") if token]


def bm25_idf(document_frequency: int, num_docs: int) -> float:
    return math.log(1.0 + (num_docs - document_frequency + 0.5) / (document_frequency + 0.5))


def build_index(train_records_path: Path, summary_repeat: int = 3):
    postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
    df: Counter[str] = Counter()
    doc_lengths: dict[str, int] = {}
    doc_timestamps: dict[str, float] = {}
    num_docs = 0

    for record in iter_jsonl(train_records_path):
        num_docs += 1
        doc_id = record["bug_id"]
        tokens = record_tokens(record, summary_repeat=summary_repeat)
        tf = Counter(tokens)
        doc_lengths[doc_id] = len(tokens)
        doc_timestamps[doc_id] = parse_timestamp(record["created_at"]).timestamp()
        for term, raw_tf in tf.items():
            postings[term].append((doc_id, raw_tf))
            df[term] += 1

    avgdl = (sum(doc_lengths.values()) / num_docs) if num_docs else 0.0
    return postings, df, doc_lengths, doc_timestamps, num_docs, avgdl


def retrieve(
    test_records_path: Path,
    train_records_path: Path,
    predictions_output: Path,
    metrics_output: Path,
    top_k: int = 10,
    summary_repeat: int = 3,
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, Any]:
    postings, df, doc_lengths, doc_timestamps, num_docs, avgdl = build_index(
        train_records_path,
        summary_repeat=summary_repeat,
    )

    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    with predictions_output.open("w", encoding="utf-8") as handle:
        for record in iter_jsonl(test_records_path):
            query_tokens = record_tokens(record, summary_repeat=summary_repeat)
            query_terms = Counter(query_tokens)
            query_timestamp = parse_timestamp(record["created_at"]).timestamp()
            scores: dict[str, float] = defaultdict(float)

            for term in query_terms:
                if term not in postings:
                    continue
                idf = bm25_idf(df[term], num_docs)
                for doc_id, raw_tf in postings[term]:
                    if doc_timestamps[doc_id] > query_timestamp:
                        continue
                    doc_length = doc_lengths[doc_id]
                    numerator = raw_tf * (k1 + 1.0)
                    denominator = raw_tf + k1 * (1.0 - b + b * doc_length / max(avgdl, 1.0))
                    scores[doc_id] += idf * (numerator / denominator)

            ranked = nlargest(top_k, scores.items(), key=lambda item: item[1])
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

    metrics = evaluate_predictions(predictions_output, train_records_path)
    metrics.update(
        {
            "method": "bm25",
            "top_k": top_k,
            "summary_repeat": summary_repeat,
            "k1": k1,
            "b": b,
        }
    )
    write_json(metrics_output, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BM25 retrieval on Mozilla.")
    parser.add_argument("--train-records", default="data/train.jsonl")
    parser.add_argument("--test-records", default="data/test.jsonl")
    parser.add_argument("--predictions-output", default="reports/bm25_predictions.jsonl")
    parser.add_argument("--metrics-output", default="reports/bm25_metrics.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--summary-repeat", type=int, default=3)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()

    metrics = retrieve(
        test_records_path=Path(args.test_records),
        train_records_path=Path(args.train_records),
        predictions_output=Path(args.predictions_output),
        metrics_output=Path(args.metrics_output),
        top_k=args.top_k,
        summary_repeat=args.summary_repeat,
        k1=args.k1,
        b=args.b,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
