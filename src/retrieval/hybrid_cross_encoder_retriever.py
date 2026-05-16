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
DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"


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


def load_embedding_backends(
    dense_model_name: str,
    cross_encoder_model_name: str,
    requested_device: str,
):
    try:
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer, CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "Hybrid Cross-Encoder retrieval requires optional dependencies. "
            "Install them with: .venv/bin/python -m pip install -r requirements-semantic.txt"
        ) from exc

    device = choose_device(torch, requested_device)
    dense_model = SentenceTransformer(dense_model_name, device=device)
    cross_encoder_model = CrossEncoder(cross_encoder_model_name, device=device)
    return np, dense_model, cross_encoder_model, device


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


def build_or_load_dense_embeddings(
    train_records_path: Path,
    cache_path: Path,
    dense_model: Any,
    np: Any,
    model_name: str,
    batch_size: int,
    max_description_chars: int,
    max_train_records: int = 0,
) -> tuple[list[dict[str, Any]], Any, dict[str, int]]:
    """Build or load dense embeddings for all training records."""
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
    embeddings = dense_model.encode(
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
    """Compute BM25 scores for all documents."""
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


def compute_dense_scores(
    query_record: dict[str, Any],
    candidate_ids: list[str],
    train_embeddings: Any,
    bug_id_to_index: dict[str, int],
    dense_model: Any,
    np: Any,
    max_description_chars: int,
) -> dict[str, float]:
    """Compute dense embedding similarity scores."""
    valid_ids = [bug_id for bug_id in candidate_ids if bug_id in bug_id_to_index]
    if not valid_ids:
        return {}

    query_embedding = dense_model.encode(
        [build_semantic_text(query_record, max_description_chars)],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)[0]
    
    candidate_indices = np.array([bug_id_to_index[bug_id] for bug_id in valid_ids])
    candidate_embeddings = train_embeddings[candidate_indices]
    scores = candidate_embeddings @ query_embedding
    
    return {bug_id: float(score) for bug_id, score in zip(valid_ids, scores)}


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Normalize scores to [0, 1] range."""
    if not scores:
        return scores
    min_score = min(scores.values())
    max_score = max(scores.values())
    if max_score - min_score == 0:
        return {bug_id: 0.5 for bug_id in scores}
    return {
        bug_id: (score - min_score) / (max_score - min_score)
        for bug_id, score in scores.items()
    }


def hybrid_score_fusion(
    bm25_scores: dict[str, float],
    dense_scores: dict[str, float],
    bm25_weight: float = 0.3,
    dense_weight: float = 0.7,
) -> dict[str, float]:
    """Combine BM25 and dense scores using weighted sum."""
    # Normalize both scores
    norm_bm25 = normalize_scores(bm25_scores)
    norm_dense = normalize_scores(dense_scores)
    
    # Get all bug IDs
    all_bug_ids = set(norm_bm25.keys()) | set(norm_dense.keys())
    
    # Weighted fusion
    fused_scores = {}
    for bug_id in all_bug_ids:
        bm25_score = norm_bm25.get(bug_id, 0.0)
        dense_score = norm_dense.get(bug_id, 0.0)
        fused_scores[bug_id] = bm25_weight * bm25_score + dense_weight * dense_score
    
    return fused_scores


def rerank_with_cross_encoder(
    query_record: dict[str, Any],
    candidate_ids: list[str],
    train_records: list[dict[str, Any]],
    cross_encoder_model: Any,
    max_description_chars: int,
    top_k: int,
) -> list[tuple[str, float]]:
    """Re-rank candidates using cross-encoder."""
    if not candidate_ids:
        return []
    
    bug_id_to_record = {record["bug_id"]: record for record in train_records}
    query_text = build_semantic_text(query_record, max_description_chars)
    
    # Prepare pairs for cross-encoder
    pairs = []
    valid_ids = []
    for bug_id in candidate_ids:
        if bug_id in bug_id_to_record:
            candidate_text = build_semantic_text(bug_id_to_record[bug_id], max_description_chars)
            pairs.append([query_text, candidate_text])
            valid_ids.append(bug_id)
    
    if not pairs:
        return []
    
    # Get cross-encoder scores
    scores = cross_encoder_model.predict(pairs)
    
    # Rank by score and return top-k
    ranked = sorted(
        zip(valid_ids, scores),
        key=lambda x: x[1],
        reverse=True,
    )
    
    return ranked[:top_k]


def retrieve(
    test_records_path: Path,
    train_records_path: Path,
    predictions_output: Path,
    metrics_output: Path,
    cache_path: Path,
    dense_model_name: str = DEFAULT_DENSE_MODEL,
    cross_encoder_model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
    top_k: int = 10,
    retrieval_depth: int = 100,  # Top 100 from hybrid search
    batch_size: int = 64,
    max_description_chars: int = 1000,
    summary_repeat: int = 3,
    bm25_weight: float = 0.3,
    dense_weight: float = 0.7,
    k1: float = 1.5,
    b: float = 0.75,
    max_test_queries: int = 0,
    max_train_records: int = 0,
    device: str = "auto",
) -> dict[str, Any]:
    """
    Hybrid Cross-Encoder Retrieval:
    Stage 1: Hybrid Search (BM25 + Dense) -> Top 100
    Stage 2: Cross-Encoder Re-ranking -> Top K
    """
    np, dense_model, cross_encoder_model, resolved_device = load_embedding_backends(
        dense_model_name,
        cross_encoder_model_name,
        device,
    )
    
    # Load training data and embeddings
    train_records, train_embeddings, bug_id_to_index = build_or_load_dense_embeddings(
        train_records_path=train_records_path,
        cache_path=cache_path,
        dense_model=dense_model,
        np=np,
        model_name=dense_model_name,
        batch_size=batch_size,
        max_description_chars=max_description_chars,
        max_train_records=max_train_records,
    )

    # Build BM25 index
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
    total_stage1_candidates = 0

    with predictions_output.open("w", encoding="utf-8") as handle:
        for record in iter_jsonl(test_records_path):
            total_queries += 1
            
            # ========== STAGE 1: HYBRID SEARCH ==========
            # Compute BM25 scores
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
            
            # Compute dense embedding scores
            all_bug_ids = list(bug_id_to_index.keys())
            dense_scores = compute_dense_scores(
                query_record=record,
                candidate_ids=all_bug_ids,
                train_embeddings=train_embeddings,
                bug_id_to_index=bug_id_to_index,
                dense_model=dense_model,
                np=np,
                max_description_chars=max_description_chars,
            )
            
            # Hybrid fusion
            fused_scores = hybrid_score_fusion(
                bm25_scores,
                dense_scores,
                bm25_weight=bm25_weight,
                dense_weight=dense_weight,
            )
            
            # Get top-100 candidates
            hybrid_ranked = nlargest(retrieval_depth, fused_scores.items(), key=lambda item: item[1])
            candidate_ids = [bug_id for bug_id, _ in hybrid_ranked]
            total_stage1_candidates += len(candidate_ids)
            
            # ========== STAGE 2: CROSS-ENCODER RE-RANKING ==========
            ranked = rerank_with_cross_encoder(
                query_record=record,
                candidate_ids=candidate_ids,
                train_records=train_records,
                cross_encoder_model=cross_encoder_model,
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
            handle.write(json.dumps(prediction, ensure_ascii=False, default=float) + "\n")

            if max_test_queries > 0 and total_queries >= max_test_queries:
                break

    metrics = evaluate_predictions(predictions_output, train_records_path)
    metrics.update(
        {
            "method": "hybrid_cross_encoder",
            "dense_model_name": dense_model_name,
            "cross_encoder_model_name": cross_encoder_model_name,
            "top_k": top_k,
            "retrieval_depth": retrieval_depth,
            "batch_size": batch_size,
            "max_description_chars": max_description_chars,
            "summary_repeat": summary_repeat,
            "bm25_weight": bm25_weight,
            "dense_weight": dense_weight,
            "k1": k1,
            "b": b,
            "num_queries": total_queries,
            "avg_stage1_candidates": total_stage1_candidates / max(total_queries, 1),
            "cache_path": str(cache_path),
            "device": resolved_device,
        }
    )
    write_json(metrics_output, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hybrid cross-encoder retrieval on Mozilla.")
    parser.add_argument("--train-records", default="data/train.jsonl")
    parser.add_argument("--test-records", default="data/test.jsonl")
    parser.add_argument("--predictions-output", default="reports/hybrid_cross_encoder_predictions.jsonl")
    parser.add_argument("--metrics-output", default="reports/hybrid_cross_encoder_metrics.json")
    parser.add_argument("--cache-path", default="reports/cache/hybrid_dense_embeddings.pkl")
    parser.add_argument("--dense-model-name", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--cross-encoder-model-name", default=DEFAULT_CROSS_ENCODER_MODEL)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--retrieval-depth", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-description-chars", type=int, default=1000)
    parser.add_argument("--summary-repeat", type=int, default=3)
    parser.add_argument("--bm25-weight", type=float, default=0.3)
    parser.add_argument("--dense-weight", type=float, default=0.7)
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
        dense_model_name=args.dense_model_name,
        cross_encoder_model_name=args.cross_encoder_model_name,
        top_k=args.top_k,
        retrieval_depth=args.retrieval_depth,
        batch_size=args.batch_size,
        max_description_chars=args.max_description_chars,
        summary_repeat=args.summary_repeat,
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
        k1=args.k1,
        b=args.b,
        max_test_queries=args.max_test_queries,
        max_train_records=args.max_train_records,
        device=args.device,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
