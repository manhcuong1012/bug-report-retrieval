"""
Hybrid Cross-Encoder Retriever
================================
Pipeline 3 stage:
  Stage 1a — BM25 prefilter   : load offline BM25 index -> top bm25_prefilter_depth candidates
  Stage 1b — FAISS dense ANN  : load offline FAISS index -> top faiss_prefilter_depth candidates
  Stage 1c — Hybrid fusion     : weighted min-max normalize + fuse -> top retrieval_depth candidates
  Stage 2  — Cross-Encoder    : rerank top retrieval_depth with cross-encoder -> top_k results

Sử dụng BM25 index từ build_bm25_index.py và FAISS index từ build_faiss_index.py.
Không build lại index trong giờ chạy — load từ disk.
"""
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
from retrieval.bm25_retriever import bm25_idf, record_tokens  # type: ignore

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S %z"
DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# Mặc định path index đã build sẵn
DEFAULT_BM25_INDEX = "reports/cache/bm25_index.pkl"
DEFAULT_FAISS_INDEX = "reports/cache/faiss_index"          # faiss_index + faiss_index_ids.json
DEFAULT_DENSE_CACHE = "reports/cache/hybrid_dense_embeddings.pkl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def cache_matches(cache_payload: dict[str, Any], metadata: dict[str, Any]) -> bool:
    return cache_payload.get("metadata") == metadata


def load_train_records(train_records_path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(train_records_path))


# ---------------------------------------------------------------------------
# Load backends
# ---------------------------------------------------------------------------

def load_ml_backends(
    dense_model_name: str,
    cross_encoder_model_name: str,
    requested_device: str,
):
    """Load SentenceTransformer + CrossEncoder."""
    try:
        import numpy as np
        import torch
        from sentence_transformers import CrossEncoder, SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Hybrid Cross-Encoder retrieval requires optional dependencies.\n"
            "Install: .venv/bin/python -m pip install -r requirements-semantic.txt"
        ) from exc

    device = choose_device(torch, requested_device)
    print(f"[init] Device: {device}")
    dense_model = SentenceTransformer(dense_model_name, device=device)
    cross_encoder_model = CrossEncoder(cross_encoder_model_name, device=device)
    return np, dense_model, cross_encoder_model, device


# ---------------------------------------------------------------------------
# Load offline BM25 index
# ---------------------------------------------------------------------------

def load_bm25_index(index_path: Path) -> dict[str, Any]:
    """Load BM25 index đã build sẵn từ build_bm25_index.py."""
    if not index_path.exists():
        raise FileNotFoundError(
            f"BM25 index không tìm thấy tại {index_path}.\n"
            "Chạy trước: python scripts/build_bm25_index.py"
        )
    print(f"[bm25] Loading index from {index_path} ...")
    with index_path.open("rb") as handle:
        data = pickle.load(handle)
    print(
        f"[bm25] Loaded: {data['num_docs']:,} docs, "
        f"vocab={len(data['df']):,}, avgdl={data['avgdl']:.1f}"
    )
    return data


# ---------------------------------------------------------------------------
# Load offline FAISS index
# ---------------------------------------------------------------------------

def load_faiss_index(faiss_index_path: Path):
    """Load FAISS index + bug_id mapping đã build sẵn từ build_faiss_index.py."""
    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "FAISS is required. Install: pip install faiss-cpu"
        ) from exc

    if not faiss_index_path.exists():
        raise FileNotFoundError(
            f"FAISS index không tìm thấy tại {faiss_index_path}.\n"
            "Chạy trước: python scripts/build_faiss_index.py"
        )

    ids_path = faiss_index_path.parent / f"{faiss_index_path.stem}_ids.json"
    if not ids_path.exists():
        raise FileNotFoundError(
            f"FAISS ID mapping không tìm thấy tại {ids_path}."
        )

    print(f"[faiss] Loading index from {faiss_index_path} ...")
    index = faiss.read_index(str(faiss_index_path))
    with ids_path.open("r") as f:
        bug_ids: list[str] = json.load(f)

    # Tăng efSearch để cải thiện recall (HNSW)
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = 128

    print(f"[faiss] Loaded: {index.ntotal:,} vectors, dim={index.d}")
    return index, bug_ids, np


# ---------------------------------------------------------------------------
# Build or load dense embedding cache (dùng để lookup embedding cho cross-encoder pool)
# ---------------------------------------------------------------------------

def build_or_load_dense_embeddings(
    train_records: list[dict[str, Any]],
    train_records_path: Path,
    cache_path: Path,
    dense_model: Any,
    np: Any,
    model_name: str,
    batch_size: int,
    max_description_chars: int,
) -> tuple[Any, dict[str, int]]:
    """
    Trả về (train_embeddings [N x D], bug_id_to_index).
    Cache để tái dùng. Nếu FAISS index đã load thì embedding này chỉ dùng
    để lấy candidate_embeddings khi tính dense_scores cho BM25 expansion pool.
    """
    metadata = {
        "model_name": model_name,
        "train_file": train_file_fingerprint(train_records_path),
        "record_count": len(train_records),
        "max_description_chars": max_description_chars,
    }

    if cache_path.exists():
        print(f"[embed] Loading embedding cache from {cache_path} ...")
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        if cache_matches(payload, metadata):
            bug_id_to_index = {
                bug_id: idx for idx, bug_id in enumerate(payload["bug_ids"])
            }
            print(f"[embed] Cache hit: {len(bug_id_to_index):,} embeddings")
            return payload["embeddings"], bug_id_to_index
        print("[embed] Cache mismatch — rebuilding ...")

    print(f"[embed] Encoding {len(train_records):,} training records ...")
    texts = [build_semantic_text(r, max_description_chars) for r in train_records]
    embeddings = dense_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    bug_ids = [r["bug_id"] for r in train_records]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(
            {"metadata": metadata, "bug_ids": bug_ids, "embeddings": embeddings},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"[embed] Saved cache to {cache_path}")

    bug_id_to_index = {bug_id: idx for idx, bug_id in enumerate(bug_ids)}
    return embeddings, bug_id_to_index


# ---------------------------------------------------------------------------
# Stage 1a: BM25
# ---------------------------------------------------------------------------

def bm25_top_candidates(
    query_record: dict[str, Any],
    postings: dict[str, list[tuple[str, int]]],
    df: Counter,
    doc_lengths: dict[str, int],
    doc_timestamps: dict[str, float],
    num_docs: int,
    avgdl: float,
    summary_repeat: int,
    k1: float,
    b: float,
    depth: int,
) -> list[tuple[str, float]]:
    """BM25 retrieve top-depth candidates với time filter."""
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
            dl = doc_lengths[doc_id]
            tf_norm = raw_tf * (k1 + 1.0) / (raw_tf + k1 * (1.0 - b + b * dl / max(avgdl, 1.0)))
            scores[doc_id] += idf * tf_norm

    return nlargest(depth, scores.items(), key=lambda x: x[1])


# ---------------------------------------------------------------------------
# Stage 1b: FAISS dense ANN
# ---------------------------------------------------------------------------

def faiss_top_candidates(
    query_record: dict[str, Any],
    faiss_index: Any,
    faiss_bug_ids: list[str],
    doc_timestamps: dict[str, float],
    dense_model: Any,
    np: Any,
    max_description_chars: int,
    depth: int,
    query_timestamp: float,
) -> list[tuple[str, float]]:
    """
    FAISS ANN search rồi filter theo thời gian.
    Tìm thêm để bù cho các doc bị loại do time filter.
    """
    query_text = build_semantic_text(query_record, max_description_chars)
    query_emb = dense_model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    # Lấy depth * 3 để đủ sau khi lọc thời gian
    search_k = min(depth * 3, faiss_index.ntotal)
    scores_arr, indices_arr = faiss_index.search(query_emb, search_k)
    raw_scores = scores_arr[0]
    raw_indices = indices_arr[0]

    results = []
    for score, idx in zip(raw_scores, raw_indices):
        if idx < 0 or idx >= len(faiss_bug_ids):
            continue
        bug_id = faiss_bug_ids[idx]
        ts = doc_timestamps.get(bug_id, 0.0)
        if ts > query_timestamp:
            continue
        results.append((bug_id, float(score)))
        if len(results) >= depth:
            break

    return results


# ---------------------------------------------------------------------------
# Stage 1c: Hybrid fusion
# ---------------------------------------------------------------------------

def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalization về [0, 1]."""
    if not scores:
        return {}
    min_s = min(scores.values())
    max_s = max(scores.values())
    rng = max_s - min_s
    if rng == 0.0:
        return {k: 0.5 for k in scores}
    return {k: (v - min_s) / rng for k, v in scores.items()}


def hybrid_fusion(
    bm25_candidates: list[tuple[str, float]],
    faiss_candidates: list[tuple[str, float]],
    bm25_weight: float,
    dense_weight: float,
    depth: int,
) -> list[tuple[str, float]]:
    """
    Fuse BM25 + FAISS scores:
    - Normalize riêng từng phương pháp
    - Weighted sum; doc chỉ có ở 1 phương pháp thì score phương pháp kia = 0
    - Trả về top `depth`
    """
    norm_bm25 = normalize_scores(dict(bm25_candidates))
    norm_dense = normalize_scores(dict(faiss_candidates))

    all_ids = set(norm_bm25) | set(norm_dense)
    fused: dict[str, float] = {}
    for bug_id in all_ids:
        fused[bug_id] = (
            bm25_weight * norm_bm25.get(bug_id, 0.0)
            + dense_weight * norm_dense.get(bug_id, 0.0)
        )

    return nlargest(depth, fused.items(), key=lambda x: x[1])


# ---------------------------------------------------------------------------
# Stage 2: Cross-Encoder reranking
# ---------------------------------------------------------------------------

def rerank_with_cross_encoder(
    query_record: dict[str, Any],
    candidate_ids: list[str],
    bug_id_to_record: dict[str, dict[str, Any]],
    cross_encoder_model: Any,
    max_description_chars: int,
    top_k: int,
) -> list[tuple[str, float]]:
    """Cross-encoder rerank trên pool candidates."""
    if not candidate_ids:
        return []

    query_text = build_semantic_text(query_record, max_description_chars)
    pairs = []
    valid_ids = []
    for bug_id in candidate_ids:
        rec = bug_id_to_record.get(bug_id)
        if rec is None:
            continue
        pairs.append([query_text, build_semantic_text(rec, max_description_chars)])
        valid_ids.append(bug_id)

    if not pairs:
        return []

    scores = cross_encoder_model.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(valid_ids, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Main retrieve function
# ---------------------------------------------------------------------------

def retrieve(
    test_records_path: Path,
    train_records_path: Path,
    predictions_output: Path,
    metrics_output: Path,
    # Index paths (offline, đã build sẵn)
    bm25_index_path: Path,
    faiss_index_path: Path,
    dense_cache_path: Path,
    # Models
    dense_model_name: str = DEFAULT_DENSE_MODEL,
    cross_encoder_model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
    # Pipeline config
    top_k: int = 10,
    bm25_prefilter_depth: int = 500,   # BM25 lấy bao nhiêu candidates Stage 1a
    faiss_prefilter_depth: int = 500,  # FAISS lấy bao nhiêu candidates Stage 1b
    retrieval_depth: int = 100,        # Sau fusion, lấy top bao nhiêu vào cross-encoder
    batch_size: int = 64,
    max_description_chars: int = 1000,
    summary_repeat: int = 3,
    bm25_weight: float = 0.35,
    dense_weight: float = 0.65,
    k1: float = 1.5,
    b: float = 0.75,
    max_test_queries: int = 0,
    device: str = "auto",
) -> dict[str, Any]:
    """
    3-stage Hybrid Cross-Encoder Retrieval.

    Stage 1a: BM25 offline index  -> top bm25_prefilter_depth
    Stage 1b: FAISS offline index -> top faiss_prefilter_depth
    Stage 1c: Hybrid fusion       -> top retrieval_depth
    Stage 2 : Cross-Encoder       -> top_k
    """
    # ── Load ML backends ──────────────────────────────────────────────────
    np, dense_model, cross_encoder_model, resolved_device = load_ml_backends(
        dense_model_name, cross_encoder_model_name, device
    )

    # ── Load offline BM25 index ───────────────────────────────────────────
    bm25 = load_bm25_index(bm25_index_path)
    postings: dict[str, list[tuple[str, int]]] = bm25["postings"]
    df: Counter = bm25["df"]
    doc_lengths: dict[str, int] = bm25["doc_lengths"]
    doc_timestamps: dict[str, float] = bm25["doc_timestamps"]
    num_docs: int = bm25["num_docs"]
    avgdl: float = bm25["avgdl"]

    # ── Load offline FAISS index ──────────────────────────────────────────
    faiss_index, faiss_bug_ids, np = load_faiss_index(faiss_index_path)

    # ── Load train records + embedding cache ──────────────────────────────
    print(f"[data] Loading train records from {train_records_path} ...")
    train_records = load_train_records(train_records_path)
    print(f"[data] {len(train_records):,} train records")

    # Build lookup dict một lần duy nhất — dùng cho cross-encoder
    bug_id_to_record: dict[str, dict[str, Any]] = {
        r["bug_id"]: r for r in train_records
    }

    # Dense embedding cache (dùng như fallback hoặc debug; FAISS là primary ANN)
    # Vẫn load để có thể dùng compute_dense_scores nếu cần mở rộng sau này
    train_embeddings, bug_id_to_index = build_or_load_dense_embeddings(
        train_records=train_records,
        train_records_path=train_records_path,
        cache_path=dense_cache_path,
        dense_model=dense_model,
        np=np,
        model_name=dense_model_name,
        batch_size=batch_size,
        max_description_chars=max_description_chars,
    )

    # ── Retrieve ──────────────────────────────────────────────────────────
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    total_queries = 0
    total_stage1_size = 0

    print(f"\n[retrieve] Starting retrieval on {test_records_path} ...")
    with predictions_output.open("w", encoding="utf-8") as out_handle:
        for query_record in iter_jsonl(test_records_path):
            total_queries += 1
            query_timestamp = parse_timestamp(query_record["created_at"]).timestamp()

            # ── Stage 1a: BM25 prefilter ──────────────────────────────────
            bm25_candidates = bm25_top_candidates(
                query_record=query_record,
                postings=postings,
                df=df,
                doc_lengths=doc_lengths,
                doc_timestamps=doc_timestamps,
                num_docs=num_docs,
                avgdl=avgdl,
                summary_repeat=summary_repeat,
                k1=k1,
                b=b,
                depth=bm25_prefilter_depth,
            )

            # ── Stage 1b: FAISS dense ANN ─────────────────────────────────
            faiss_candidates = faiss_top_candidates(
                query_record=query_record,
                faiss_index=faiss_index,
                faiss_bug_ids=faiss_bug_ids,
                doc_timestamps=doc_timestamps,
                dense_model=dense_model,
                np=np,
                max_description_chars=max_description_chars,
                depth=faiss_prefilter_depth,
                query_timestamp=query_timestamp,
            )

            # ── Stage 1c: Hybrid fusion -> top retrieval_depth ────────────
            fused_candidates = hybrid_fusion(
                bm25_candidates=bm25_candidates,
                faiss_candidates=faiss_candidates,
                bm25_weight=bm25_weight,
                dense_weight=dense_weight,
                depth=retrieval_depth,
            )
            candidate_ids = [bug_id for bug_id, _ in fused_candidates]
            total_stage1_size += len(candidate_ids)

            # ── Stage 2: Cross-Encoder reranking ──────────────────────────
            ranked = rerank_with_cross_encoder(
                query_record=query_record,
                candidate_ids=candidate_ids,
                bug_id_to_record=bug_id_to_record,
                cross_encoder_model=cross_encoder_model,
                max_description_chars=max_description_chars,
                top_k=top_k,
            )

            prediction = {
                "query_bug_id": query_record["bug_id"],
                "project": "mozilla",
                "query_bucket_id": query_record["bucket_id"],
                "results": [
                    {
                        "bug_id": bug_id,
                        "project": "mozilla",
                        "score": float(score),
                        "rank": rank,
                    }
                    for rank, (bug_id, score) in enumerate(ranked, start=1)
                ],
            }
            out_handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")

            if total_queries % 100 == 0:
                print(f"  [retrieve] {total_queries} queries done ...")

            if max_test_queries > 0 and total_queries >= max_test_queries:
                break

    print(f"[retrieve] Done: {total_queries} queries, avg stage1={total_stage1_size / max(total_queries, 1):.1f} candidates")

    # ── Evaluate ──────────────────────────────────────────────────────────
    metrics = evaluate_predictions(predictions_output, test_records_path)
    metrics.update(
        {
            "method": "hybrid_cross_encoder",
            "dense_model_name": dense_model_name,
            "cross_encoder_model_name": cross_encoder_model_name,
            "top_k": top_k,
            "bm25_prefilter_depth": bm25_prefilter_depth,
            "faiss_prefilter_depth": faiss_prefilter_depth,
            "retrieval_depth": retrieval_depth,
            "bm25_weight": bm25_weight,
            "dense_weight": dense_weight,
            "k1": k1,
            "b": b,
            "batch_size": batch_size,
            "max_description_chars": max_description_chars,
            "summary_repeat": summary_repeat,
            "num_queries": total_queries,
            "avg_stage1_candidates": total_stage1_size / max(total_queries, 1),
            "device": resolved_device,
        }
    )
    write_json(metrics_output, metrics)
    print(f"[eval] Metrics saved to {metrics_output}")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid Cross-Encoder retrieval (BM25 index + FAISS index + Cross-Encoder).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    parser.add_argument("--train-records", default="data/train.jsonl")
    parser.add_argument("--test-records", default="data/test.jsonl")
    # Offline indexes
    parser.add_argument("--bm25-index", default=DEFAULT_BM25_INDEX, help="Path tới BM25 index pkl")
    parser.add_argument("--faiss-index", default=DEFAULT_FAISS_INDEX, help="Path tới FAISS index (không cần .faiss extension)")
    parser.add_argument("--dense-cache", default=DEFAULT_DENSE_CACHE, help="Path cache embedding pkl")
    # Output
    parser.add_argument("--predictions-output", default="reports/hybrid_cross_encoder_predictions.jsonl")
    parser.add_argument("--metrics-output", default="reports/hybrid_cross_encoder_metrics.json")
    # Models
    parser.add_argument("--dense-model-name", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--cross-encoder-model-name", default=DEFAULT_CROSS_ENCODER_MODEL)
    # Pipeline
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bm25-prefilter-depth", type=int, default=500, help="BM25 stage lấy top bao nhiêu")
    parser.add_argument("--faiss-prefilter-depth", type=int, default=500, help="FAISS stage lấy top bao nhiêu")
    parser.add_argument("--retrieval-depth", type=int, default=100, help="Sau fusion, feed vào cross-encoder bao nhiêu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-description-chars", type=int, default=1000)
    parser.add_argument("--summary-repeat", type=int, default=3)
    parser.add_argument("--bm25-weight", type=float, default=0.35)
    parser.add_argument("--dense-weight", type=float, default=0.65)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--max-test-queries", type=int, default=0, help="0 = chạy hết")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()

    metrics = retrieve(
        test_records_path=Path(args.test_records),
        train_records_path=Path(args.train_records),
        predictions_output=Path(args.predictions_output),
        metrics_output=Path(args.metrics_output),
        bm25_index_path=Path(args.bm25_index),
        faiss_index_path=Path(args.faiss_index),
        dense_cache_path=Path(args.dense_cache),
        dense_model_name=args.dense_model_name,
        cross_encoder_model_name=args.cross_encoder_model_name,
        top_k=args.top_k,
        bm25_prefilter_depth=args.bm25_prefilter_depth,
        faiss_prefilter_depth=args.faiss_prefilter_depth,
        retrieval_depth=args.retrieval_depth,
        batch_size=args.batch_size,
        max_description_chars=args.max_description_chars,
        summary_repeat=args.summary_repeat,
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
        k1=args.k1,
        b=args.b,
        max_test_queries=args.max_test_queries,
        device=args.device,
    )

    print("\n" + "=" * 60)
    print(json.dumps(
        {k: v for k, v in metrics.items() if k in (
            "recall_at_1", "recall_at_5", "recall_at_10", "mrr",
            "num_queries", "avg_stage1_candidates", "method",
        )},
        indent=2,
    ))


if __name__ == "__main__":
    main()
