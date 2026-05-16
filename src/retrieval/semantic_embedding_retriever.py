from __future__ import annotations

import argparse
import json
import pickle
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
from retrieval.bm25_retriever import bm25_idf, build_index as build_bm25_index, record_tokens  # type: ignore

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S %z"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


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


def choose_device(torch: Any, requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_embedding_backend(model_name: str, requested_device: str):
    try:
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Semantic embedding retrieval requires optional dependencies. "
            "Install them with: .venv/bin/python -m pip install -r requirements-semantic.txt"
        ) from exc

    device = choose_device(torch, requested_device)
    return np, SentenceTransformer(model_name, device=device), device


def build_semantic_text(record: dict[str, Any], max_description_chars: int = 1000) -> str:
    summary = record.get("summary", "")
    description = record.get("description", "")[:max_description_chars]
    component = record.get("component", "UNKNOWN")
    severity = record.get("severity", "UNKNOWN")
    priority = record.get("priority", "UNKNOWN")
    return (
        f"Summary: {summary}\n"
        f"Component: {component}\n"
        f"Severity: {severity}\n"
        f"Priority: {priority}\n"
        f"Description: {description}"
    )


def train_file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def cache_matches(cache_payload: dict[str, Any], metadata: dict[str, Any]) -> bool:
    return cache_payload.get("metadata") == metadata


def load_train_records(train_records_path: Path, max_train_records: int = 0) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in iter_jsonl(train_records_path):
        records.append(record)
        if max_train_records > 0 and len(records) >= max_train_records:
            break
    return records


def build_or_load_train_embeddings(
    train_records_path: Path,
    cache_path: Path,
    model: Any,
    np: Any,
    model_name: str,
    batch_size: int,
    max_description_chars: int,
    max_train_records: int = 0,
) -> tuple[list[dict[str, Any]], Any, dict[str, int]]:
    train_records = load_train_records(train_records_path, max_train_records=max_train_records)
    metadata = {
        "model_name": model_name,
        "train_file": train_file_fingerprint(train_records_path),
        "record_count": len(train_records),
        "max_description_chars": max_description_chars,
        "max_train_records": max_train_records,
    }

    if cache_path.exists():
        with cache_path.open("rb") as handle:
            cache_payload = pickle.load(handle)
        if cache_matches(cache_payload, metadata):
            bug_id_to_index = {
                bug_id: index
                for index, bug_id in enumerate(cache_payload["bug_ids"])
            }
            return train_records, cache_payload["embeddings"], bug_id_to_index

    texts = [build_semantic_text(record, max_description_chars) for record in train_records]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    bug_ids = [record["bug_id"] for record in train_records]
    with cache_path.open("wb") as handle:
        pickle.dump(
            {
                "metadata": metadata,
                "bug_ids": bug_ids,
                "embeddings": embeddings,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    bug_id_to_index = {bug_id: index for index, bug_id in enumerate(bug_ids)}
    return train_records, embeddings, bug_id_to_index


def compute_bm25_scores(
    query_record: dict[str, Any],
    postings: dict[str, list[tuple[str, int]]],
    df: Counter[str],
    doc_lengths: dict[str, int],
    doc_timestamps: dict[str, float],
    num_docs: int,
    avgdl: float,
    summary_repeat: int,
    k1: float,
    b: float,
) -> dict[str, float]:
    query_tokens = record_tokens(query_record, summary_repeat=summary_repeat)
    query_terms = Counter(query_tokens)
    query_timestamp = parse_timestamp(query_record["created_at"]).timestamp()

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

    return dict(scores)


def semantic_rank_candidates(
    query_record: dict[str, Any],
    candidate_ids: list[str],
    train_embeddings: Any,
    bug_id_to_index: dict[str, int],
    model: Any,
    np: Any,
    max_description_chars: int,
    top_k: int,
) -> list[tuple[str, float]]:
    valid_ids = [bug_id for bug_id in candidate_ids if bug_id in bug_id_to_index]
    if not valid_ids:
        return []

    query_embedding = model.encode(
        [build_semantic_text(query_record, max_description_chars)],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)[0]
    candidate_indices = np.array([bug_id_to_index[bug_id] for bug_id in valid_ids])
    candidate_embeddings = train_embeddings[candidate_indices]
    scores = candidate_embeddings @ query_embedding
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    return [(valid_ids[index], float(scores[index])) for index in ranked_indices]


def retrieve(
    test_records_path: Path,
    train_records_path: Path,
    predictions_output: Path,
    metrics_output: Path,
    cache_path: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    top_k: int = 10,
    candidate_depth: int = 1000,
    batch_size: int = 64,
    max_description_chars: int = 1000,
    summary_repeat: int = 3,
    k1: float = 1.5,
    b: float = 0.75,
    max_test_queries: int = 0,
    max_train_records: int = 0,
    device: str = "auto",
) -> dict[str, Any]:
    np, model, resolved_device = load_embedding_backend(model_name, device)
    train_records, train_embeddings, bug_id_to_index = build_or_load_train_embeddings(
        train_records_path=train_records_path,
        cache_path=cache_path,
        model=model,
        np=np,
        model_name=model_name,
        batch_size=batch_size,
        max_description_chars=max_description_chars,
        max_train_records=max_train_records,
    )

    postings, df, doc_lengths, doc_timestamps, num_docs, avgdl = build_bm25_index(
        train_records_path,
        summary_repeat=summary_repeat,
    )
    if max_train_records > 0:
        allowed_bug_ids = {record["bug_id"] for record in train_records}
        postings = {
            term: [(bug_id, raw_tf) for bug_id, raw_tf in entries if bug_id in allowed_bug_ids]
            for term, entries in postings.items()
        }
        doc_lengths = {bug_id: length for bug_id, length in doc_lengths.items() if bug_id in allowed_bug_ids}
        doc_timestamps = {bug_id: ts for bug_id, ts in doc_timestamps.items() if bug_id in allowed_bug_ids}

    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    total_queries = 0
    total_candidates = 0

    with predictions_output.open("w", encoding="utf-8") as handle:
        for record in iter_jsonl(test_records_path):
            total_queries += 1
            bm25_scores = compute_bm25_scores(
                record,
                postings,
                df,
                doc_lengths,
                doc_timestamps,
                num_docs,
                avgdl,
                summary_repeat,
                k1,
                b,
            )
            bm25_ranked = nlargest(candidate_depth, bm25_scores.items(), key=lambda item: item[1])
            candidate_ids = [bug_id for bug_id, _ in bm25_ranked]
            total_candidates += len(candidate_ids)

            ranked = semantic_rank_candidates(
                query_record=record,
                candidate_ids=candidate_ids,
                train_embeddings=train_embeddings,
                bug_id_to_index=bug_id_to_index,
                model=model,
                np=np,
                max_description_chars=max_description_chars,
                top_k=top_k,
            )

            prediction = {
                "query_bug_id": record["bug_id"],
                "project": "mozilla",
                "query_bucket_id": record["bucket_id"],
                "results": [
                    {
                        "bug_id": bug_id,
                        "project": "mozilla",
                        "score": score,
                        "rank": rank,
                    }
                    for rank, (bug_id, score) in enumerate(ranked, start=1)
                ],
            }
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")

            if max_test_queries > 0 and total_queries >= max_test_queries:
                break

    metrics = evaluate_predictions(predictions_output, train_records_path)
    metrics.update(
        {
            "method": "semantic_embedding_bm25_prefilter",
            "model_name": model_name,
            "top_k": top_k,
            "candidate_depth": candidate_depth,
            "batch_size": batch_size,
            "max_description_chars": max_description_chars,
            "summary_repeat": summary_repeat,
            "k1": k1,
            "b": b,
            "num_queries": total_queries,
            "avg_candidates": total_candidates / max(total_queries, 1),
            "cache_path": str(cache_path),
            "device": resolved_device,
        }
    )
    write_json(metrics_output, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic embedding retrieval on Mozilla.")
    parser.add_argument("--train-records", default="data/train.jsonl")
    parser.add_argument("--test-records", default="data/test.jsonl")
    parser.add_argument("--predictions-output", default="reports/semantic_embedding_predictions.jsonl")
    parser.add_argument("--metrics-output", default="reports/semantic_embedding_metrics.json")
    parser.add_argument("--cache-path", default="reports/cache/semantic_train_embeddings.pkl")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-depth", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-description-chars", type=int, default=1000)
    parser.add_argument("--summary-repeat", type=int, default=3)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--max-test-queries", type=int, default=0)
    parser.add_argument("--max-train-records", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()

    metrics = retrieve(
        test_records_path=Path(args.test_records),
        train_records_path=Path(args.train_records),
        predictions_output=Path(args.predictions_output),
        metrics_output=Path(args.metrics_output),
        cache_path=Path(args.cache_path),
        model_name=args.model_name,
        top_k=args.top_k,
        candidate_depth=args.candidate_depth,
        batch_size=args.batch_size,
        max_description_chars=args.max_description_chars,
        summary_repeat=args.summary_repeat,
        k1=args.k1,
        b=args.b,
        max_test_queries=args.max_test_queries,
        max_train_records=args.max_train_records,
        device=args.device,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
