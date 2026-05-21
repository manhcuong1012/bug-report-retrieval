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
from preprocessing.clean_text import build_retrieval_text, normalize_text  # type: ignore

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S %z"
METADATA_PREFIXES = ("component.", "priority.", "severity.")


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


def is_metadata_term(term: str) -> bool:
    return term.startswith(METADATA_PREFIXES)


def should_prune_term(
    term: str,
    document_frequency: int,
    num_docs: int,
    max_term_df_ratio: float | None,
    max_metadata_df_ratio: float | None,
) -> bool:
    if max_term_df_ratio is not None and document_frequency > num_docs * max_term_df_ratio:
        return True
    return (
        max_metadata_df_ratio is not None
        and is_metadata_term(term)
        and document_frequency > num_docs * max_metadata_df_ratio
    )


def tokenize(text: str) -> list[str]:
    return [token for token in normalize_text(text).split(" ") if token]


def record_tokens(
    record: dict[str, str],
    summary_repeat: int = 3,
    component_repeat: int = 1,
    priority_repeat: int = 1,
    severity_repeat: int = 1,
) -> list[str]:
    retrieval_text = build_retrieval_text(
        record.get("summary", ""),
        record.get("description", ""),
        component=record.get("component", ""),
        priority=record.get("priority", ""),
        severity=record.get("severity", ""),
        summary_repeat=summary_repeat,
        component_repeat=component_repeat,
        priority_repeat=priority_repeat,
        severity_repeat=severity_repeat,
    )
    return [token for token in retrieval_text.split(" ") if token]


def compute_idf(document_frequency: int, num_docs: int) -> float:
    return math.log((num_docs + 1.0) / (document_frequency + 1.0)) + 1.0


def build_index(
    train_records_path: Path,
    summary_repeat: int = 3,
    component_repeat: int = 1,
    priority_repeat: int = 1,
    severity_repeat: int = 1,
    max_term_df_ratio: float | None = None,
    max_metadata_df_ratio: float | None = 0.1,
):
    postings: dict[str, list[tuple[str, float]]] = defaultdict(list)
    document_frequency: Counter[str] = Counter()
    doc_norms: dict[str, float] = {}
    doc_timestamps: dict[str, float] = {}
    doc_bucket_ids: dict[str, str] = {}
    doc_norm_squares: Counter[str] = Counter()
    idf: dict[str, float] = {}

    doc_term_counts: dict[str, Counter[str]] = {}
    num_docs = 0
    for record in iter_jsonl(train_records_path):
        num_docs += 1
        doc_id = record["bug_id"]
        tokens = record_tokens(
            record,
            summary_repeat=summary_repeat,
            component_repeat=component_repeat,
            priority_repeat=priority_repeat,
            severity_repeat=severity_repeat,
        )
        tf = Counter(tokens)
        doc_term_counts[doc_id] = tf
        doc_timestamps[doc_id] = parse_timestamp(record["created_at"]).timestamp()
        doc_bucket_ids[doc_id] = record["bucket_id"]
        for term in tf:
            document_frequency[term] += 1

    for term in list(document_frequency):
        if should_prune_term(
            term,
            document_frequency[term],
            num_docs,
            max_term_df_ratio=max_term_df_ratio,
            max_metadata_df_ratio=max_metadata_df_ratio,
        ):
            del document_frequency[term]

    for term, df in document_frequency.items():
        idf[term] = compute_idf(df, num_docs)

    for doc_id, tf in doc_term_counts.items():
        for term, raw_tf in tf.items():
            weight = raw_tf * idf[term]
            postings[term].append((doc_id, weight))
            doc_norm_squares[doc_id] += weight * weight

    for doc_id, square_sum in doc_norm_squares.items():
        doc_norms[doc_id] = math.sqrt(square_sum) if square_sum > 0 else 1.0

    return postings, idf, doc_norms, doc_timestamps, doc_bucket_ids


def vectorize_query(tokens: list[str], idf: dict[str, float]) -> tuple[dict[str, float], float]:
    tf = Counter(tokens)
    vector: dict[str, float] = {}
    norm_square = 0.0
    for term, raw_tf in tf.items():
        if term not in idf:
            continue
        weight = raw_tf * idf[term]
        vector[term] = weight
        norm_square += weight * weight
    return vector, math.sqrt(norm_square) if norm_square > 0 else 1.0


def retrieve(
    test_records_path: Path,
    train_records_path: Path,
    predictions_output: Path,
    metrics_output: Path,
    top_k: int = 10,
    summary_repeat: int = 3,
    component_repeat: int = 1,
    priority_repeat: int = 1,
    severity_repeat: int = 1,
    max_term_df_ratio: float | None = None,
    max_metadata_df_ratio: float | None = 0.1,
) -> dict[str, Any]:
    postings, idf, doc_norms, doc_timestamps, _doc_bucket_ids = build_index(
        train_records_path,
        summary_repeat=summary_repeat,
        component_repeat=component_repeat,
        priority_repeat=priority_repeat,
        severity_repeat=severity_repeat,
        max_term_df_ratio=max_term_df_ratio,
        max_metadata_df_ratio=max_metadata_df_ratio,
    )

    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    with predictions_output.open("w", encoding="utf-8") as handle:
        for record in iter_jsonl(test_records_path):
            query_tokens = record_tokens(
                record,
                summary_repeat=summary_repeat,
                component_repeat=component_repeat,
                priority_repeat=priority_repeat,
                severity_repeat=severity_repeat,
            )
            query_vector, query_norm = vectorize_query(query_tokens, idf)
            query_timestamp = parse_timestamp(record["created_at"]).timestamp()

            scores: dict[str, float] = defaultdict(float)
            for term, query_weight in query_vector.items():
                for doc_id, doc_weight in postings.get(term, []):
                    if doc_timestamps[doc_id] > query_timestamp:
                        continue
                    scores[doc_id] += query_weight * doc_weight

            ranked = nlargest(
                top_k,
                (
                    (
                        doc_id,
                        score / (query_norm * doc_norms.get(doc_id, 1.0)),
                    )
                    for doc_id, score in scores.items()
                    if score > 0
                ),
                key=lambda item: item[1],
            )

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
            "method": "tfidf",
            "top_k": top_k,
            "summary_repeat": summary_repeat,
            "component_repeat": component_repeat,
            "priority_repeat": priority_repeat,
            "severity_repeat": severity_repeat,
            "max_term_df_ratio": max_term_df_ratio,
            "max_metadata_df_ratio": max_metadata_df_ratio,
        }
    )
    write_json(metrics_output, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TF-IDF retrieval on Mozilla.")
    parser.add_argument("--train-records", default="data/train.jsonl")
    parser.add_argument("--test-records", default="data/test.jsonl")
    parser.add_argument("--predictions-output", default="reports/tfidf_predictions.jsonl")
    parser.add_argument("--metrics-output", default="reports/tfidf_metrics.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--summary-repeat", type=int, default=3)
    parser.add_argument("--component-repeat", type=int, default=1)
    parser.add_argument("--priority-repeat", type=int, default=1)
    parser.add_argument("--severity-repeat", type=int, default=1)
    parser.add_argument("--max-term-df-ratio", type=float, default=None)
    parser.add_argument("--max-metadata-df-ratio", type=float, default=0.1)
    args = parser.parse_args()

    metrics = retrieve(
        test_records_path=Path(args.test_records),
        train_records_path=Path(args.train_records),
        predictions_output=Path(args.predictions_output),
        metrics_output=Path(args.metrics_output),
        top_k=args.top_k,
        summary_repeat=args.summary_repeat,
        component_repeat=args.component_repeat,
        priority_repeat=args.priority_repeat,
        severity_repeat=args.severity_repeat,
        max_term_df_ratio=args.max_term_df_ratio,
        max_metadata_df_ratio=args.max_metadata_df_ratio,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
