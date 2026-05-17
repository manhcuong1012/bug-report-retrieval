"""
Hybrid Cross-Encoder Retriever
================================
Pipeline 3 stage:
  Stage 1a — BM25 prefilter   : load offline BM25 index  -> top bm25_prefilter_depth
  Stage 1b — FAISS dense ANN  : load offline FAISS index -> top faiss_prefilter_depth
  Stage 1c — Hybrid fusion    : weighted min-max fuse    -> top retrieval_depth
  Stage 2  — Cross-Encoder    : rerank                   -> top_k kết quả cuối

Yêu cầu build sẵn 2 index trước khi chạy:
  python src/offline/build_bm25_index.py
  python src/offline/build_faiss_index.py

Bugs đã fix so với phiên bản trước:
  [BUG-1] evaluate_predictions phải nhận train_records_path cho result bug_ids
          → nếu truyền test_records_path, result bug_ids không có trong lookup → 0.0
  [BUG-2] doc_timestamps và bug_id_to_record có thể lưu key dạng int
          trong khi postings/faiss_bug_ids dùng string → time filter sai,
          cross-encoder lookup miss toàn bộ → candidate list rỗng
  [BUG-3] build_or_load_dense_embeddings encode lại 154k records không cần thiết
          (18 phút) vì FAISS đã cover stage 1b và cross-encoder chỉ cần text gốc
          → đã bỏ hoàn toàn, retriever giờ chạy không cần cache pkl
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
DEFAULT_BM25_INDEX = "reports/cache/bm25_index.pkl"
DEFAULT_FAISS_INDEX = "reports/cache/faiss_index"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_timestamp(value: str) -> float:
    return datetime.strptime(value, TIMESTAMP_FORMAT).timestamp()


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


def choose_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_semantic_text(record: dict[str, Any], max_description_chars: int = 1000) -> str:
    """
    Tạo text đưa vào dense model / cross-encoder.
    PHẢI khớp với build_semantic_text trong build_faiss_index.py.
    """
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


# ---------------------------------------------------------------------------
# Load ML backends
# ---------------------------------------------------------------------------

def load_ml_backends(
    dense_model_name: str,
    cross_encoder_model_name: str,
    requested_device: str,
):
    try:
        import numpy as np
        import torch
        from sentence_transformers import CrossEncoder, SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu dependency.\n"
            "Install: .venv/bin/python -m pip install -r requirements-semantic.txt"
        ) from exc

    device = choose_device(torch, requested_device)
    print(f"[init] Device: {device}")
    dense_model = SentenceTransformer(dense_model_name, device=device)
    cross_encoder = CrossEncoder(cross_encoder_model_name, device=device)
    return np, dense_model, cross_encoder, device


# ---------------------------------------------------------------------------
# Load offline BM25 index
# ---------------------------------------------------------------------------

def load_bm25_index(index_path: Path) -> dict[str, Any]:
    if not index_path.exists():
        raise FileNotFoundError(
            f"BM25 index không tìm thấy: {index_path}\n"
            "Chạy: python src/offline/build_bm25_index.py"
        )
    print(f"[bm25] Loading: {index_path}")
    with index_path.open("rb") as f:
        data = pickle.load(f)

    # [BUG-2 FIX] Normalize tất cả key sang string.
    # bug_id trong Mozilla dataset là số nguyên, có thể được lưu dạng int.
    # Postings, results, và faiss_bug_ids đều cần string để lookup nhất quán.
    data["doc_timestamps"] = {str(k): v for k, v in data["doc_timestamps"].items()}
    data["doc_lengths"] = {str(k): v for k, v in data["doc_lengths"].items()}
    # Normalize posting keys sang string nếu cần
    new_postings: dict[str, list[tuple[str, int]]] = {}
    for term, plist in data["postings"].items():
        new_postings[term] = [(str(doc_id), tf) for doc_id, tf in plist]
    data["postings"] = new_postings

    print(
        f"[bm25] Loaded: {data['num_docs']:,} docs | "
        f"vocab={len(data['df']):,} | avgdl={data['avgdl']:.1f}"
    )
    return data


# ---------------------------------------------------------------------------
# Load offline FAISS index
# ---------------------------------------------------------------------------

def load_faiss_index(faiss_index_path: Path):
    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("FAISS chưa install: pip install faiss-cpu") from exc

    if not faiss_index_path.exists():
        raise FileNotFoundError(
            f"FAISS index không tìm thấy: {faiss_index_path}\n"
            "Chạy: python src/offline/build_faiss_index.py"
        )

    ids_path = faiss_index_path.parent / f"{faiss_index_path.stem}_ids.json"
    if not ids_path.exists():
        raise FileNotFoundError(f"FAISS ID mapping không tìm thấy: {ids_path}")

    print(f"[faiss] Loading: {faiss_index_path}")
    index = faiss.read_index(str(faiss_index_path))

    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = 128  # tăng recall cho HNSW

    with ids_path.open("r") as f:
        # [BUG-2 FIX] Đảm bảo list[str]
        faiss_bug_ids: list[str] = [str(x) for x in json.load(f)]

    print(f"[faiss] Loaded: {index.ntotal:,} vectors | dim={index.d}")
    return index, faiss_bug_ids, np


# ---------------------------------------------------------------------------
# Stage 1a: BM25 retrieval
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
    query_tokens = record_tokens(query_record, summary_repeat=summary_repeat)
    query_terms = Counter(query_tokens)
    query_ts = parse_timestamp(query_record["created_at"])

    scores: dict[str, float] = defaultdict(float)
    for term in query_terms:
        if term not in postings:
            continue
        idf = bm25_idf(df[term], num_docs)
        for doc_id, raw_tf in postings[term]:
            # doc_id đã là string sau normalize trong load_bm25_index
            ts = doc_timestamps.get(doc_id, 0.0)
            if ts > query_ts:
                continue
            dl = doc_lengths.get(doc_id, 1)
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
    query_ts: float,
) -> list[tuple[str, float]]:
    query_text = build_semantic_text(query_record, max_description_chars)
    query_emb = dense_model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    # Lấy depth * 4 để bù cho docs bị loại bởi time filter
    search_k = min(depth * 4, faiss_index.ntotal)
    scores_arr, indices_arr = faiss_index.search(query_emb, search_k)

    results: list[tuple[str, float]] = []
    for score, idx in zip(scores_arr[0], indices_arr[0]):
        if idx < 0 or idx >= len(faiss_bug_ids):
            continue
        bug_id = faiss_bug_ids[idx]  # đã là string
        ts = doc_timestamps.get(bug_id, 0.0)
        if ts > query_ts:
            continue
        results.append((bug_id, float(score)))
        if len(results) >= depth:
            break

    return results


# ---------------------------------------------------------------------------
# Stage 1c: Hybrid fusion
# ---------------------------------------------------------------------------

def rrf_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    weights: list[float],
    depth: int,
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion — dùng rank thay vì score.

    Công thức: RRF(doc) = sum( weight_i / (k + rank_i) )
      k=60 là hằng số chuẩn.

    Tại sao tốt hơn min-max cho bài toán này:
      Min-max kéo 500 scores về [0,1]: bug ở hạng 380/500 bị gán ~0.05
      dù cosine score 0.61 thực ra là khá tốt.
      RRF chỉ dùng rank tương đối, không bị lệch bởi score distribution
      khác nhau giữa BM25 (sparse, range rộng) và FAISS (cosine, hẹp).
      Kết quả: pool 86.5% recall được giữ nhiều hơn sau khi cắt top-100.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, (bug_id, _) in enumerate(ranked_list, start=1):
            rrf_scores[bug_id] += weight / (k + rank)
    return nlargest(depth, rrf_scores.items(), key=lambda x: x[1])


def hybrid_fusion(
    bm25_candidates: list[tuple[str, float]],
    faiss_candidates: list[tuple[str, float]],
    bm25_weight: float,
    dense_weight: float,
    depth: int,
) -> list[tuple[str, float]]:
    """Wrapper dùng RRF thay cho min-max normalize."""
    return rrf_fusion(
        ranked_lists=[bm25_candidates, faiss_candidates],
        weights=[bm25_weight, dense_weight],
        depth=depth,
        k=60,
    )


# ---------------------------------------------------------------------------
# Stage 2: Cross-Encoder reranking
# ---------------------------------------------------------------------------

def rerank_with_cross_encoder(
    query_record: dict[str, Any],
    candidate_ids: list[str],
    bug_id_to_record: dict[str, dict[str, Any]],
    cross_encoder: Any,
    max_description_chars: int,
    top_k: int,
) -> list[tuple[str, float]]:
    query_text = build_semantic_text(query_record, max_description_chars)

    pairs: list[list[str]] = []
    valid_ids: list[str] = []
    for bug_id in candidate_ids:
        rec = bug_id_to_record.get(bug_id)
        if rec is None:
            continue
        pairs.append([query_text, build_semantic_text(rec, max_description_chars)])
        valid_ids.append(bug_id)

    if not pairs:
        return []

    raw_scores = cross_encoder.predict(pairs, show_progress_bar=False)
    # predict() trả về numpy array hoặc list — chuẩn hóa về list[float]
    score_list = raw_scores.tolist() if hasattr(raw_scores, "tolist") else list(raw_scores)
    ranked = sorted(zip(valid_ids, score_list), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Main retrieve function
# ---------------------------------------------------------------------------

def retrieve(
    test_records_path: Path,
    train_records_path: Path,
    predictions_output: Path,
    metrics_output: Path,
    bm25_index_path: Path,
    faiss_index_path: Path,
    dense_model_name: str = DEFAULT_DENSE_MODEL,
    cross_encoder_model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
    top_k: int = 10,
    bm25_prefilter_depth: int = 500,
    faiss_prefilter_depth: int = 500,
    retrieval_depth: int = 100,
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

    # ── Load ML backends ──────────────────────────────────────────────────
    np, dense_model, cross_encoder, resolved_device = load_ml_backends(
        dense_model_name, cross_encoder_model_name, device
    )

    # ── Load offline indexes ──────────────────────────────────────────────
    bm25 = load_bm25_index(bm25_index_path)
    postings: dict[str, list[tuple[str, int]]] = bm25["postings"]
    df: Counter = bm25["df"]
    doc_lengths: dict[str, int] = bm25["doc_lengths"]
    doc_timestamps: dict[str, float] = bm25["doc_timestamps"]
    num_docs: int = bm25["num_docs"]
    avgdl: float = bm25["avgdl"]

    faiss_index, faiss_bug_ids, np = load_faiss_index(faiss_index_path)

    # ── Load train records ────────────────────────────────────────────────
    print(f"[data] Loading train records: {train_records_path}")
    train_records = list(iter_jsonl(train_records_path))
    print(f"[data] {len(train_records):,} records")

    # [BUG-2 FIX] key phải là string để cross-encoder lookup khớp với
    # bug_id từ postings (string) và faiss_bug_ids (string)
    bug_id_to_record: dict[str, dict[str, Any]] = {
        str(r["bug_id"]): r for r in train_records
    }

    # ── Retrieve ──────────────────────────────────────────────────────────
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    total_queries = 0
    total_stage1 = 0

    print(f"\n[retrieve] Running on {test_records_path} ...")
    with predictions_output.open("w", encoding="utf-8") as out:
        for query_record in iter_jsonl(test_records_path):
            total_queries += 1
            query_ts = parse_timestamp(query_record["created_at"])

            # Stage 1a: BM25
            bm25_cands = bm25_top_candidates(
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

            # Stage 1b: FAISS ANN
            faiss_cands = faiss_top_candidates(
                query_record=query_record,
                faiss_index=faiss_index,
                faiss_bug_ids=faiss_bug_ids,
                doc_timestamps=doc_timestamps,
                dense_model=dense_model,
                np=np,
                max_description_chars=max_description_chars,
                depth=faiss_prefilter_depth,
                query_ts=query_ts,
            )

            # Stage 1c: Hybrid fusion → top retrieval_depth
            fused = hybrid_fusion(
                bm25_candidates=bm25_cands,
                faiss_candidates=faiss_cands,
                bm25_weight=bm25_weight,
                dense_weight=dense_weight,
                depth=retrieval_depth,
            )
            candidate_ids = [bug_id for bug_id, _ in fused]
            total_stage1 += len(candidate_ids)

            # Stage 2: Cross-Encoder rerank → top_k
            ranked = rerank_with_cross_encoder(
                query_record=query_record,
                candidate_ids=candidate_ids,
                bug_id_to_record=bug_id_to_record,
                cross_encoder=cross_encoder,
                max_description_chars=max_description_chars,
                top_k=top_k,
            )

            prediction = {
                "query_bug_id": str(query_record["bug_id"]),
                "project": "mozilla",
                "query_bucket_id": query_record["bucket_id"],
                "results": [
                    {
                        "bug_id": str(bug_id),
                        "project": "mozilla",
                        "score": float(score),
                        "rank": rank,
                    }
                    for rank, (bug_id, score) in enumerate(ranked, start=1)
                ],
            }
            out.write(json.dumps(prediction, ensure_ascii=False) + "\n")

            if total_queries % 100 == 0:
                print(f"  [retrieve] {total_queries} queries | avg_stage1={total_stage1/total_queries:.0f}")

            if max_test_queries > 0 and total_queries >= max_test_queries:
                break

    avg_stage1 = total_stage1 / max(total_queries, 1)
    print(f"[retrieve] Done: {total_queries} queries | avg_stage1={avg_stage1:.1f}")

    # ── Evaluate ──────────────────────────────────────────────────────────
    # Predictions contain train-set result bug_ids. The query bucket is already
    # stored in each prediction, so the evaluator must map result bug_ids using
    # train records, not test records.
    metrics = evaluate_predictions(predictions_output, train_records_path)
    metrics.update({
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
        "avg_stage1_candidates": avg_stage1,
        "device": resolved_device,
    })
    write_json(metrics_output, metrics)
    print(f"[eval] Saved → {metrics_output}")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid Cross-Encoder: BM25 index + FAISS index + Cross-Encoder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-records", default="data/train.jsonl")
    parser.add_argument("--test-records", default="data/test.jsonl")
    parser.add_argument("--bm25-index", default=DEFAULT_BM25_INDEX)
    parser.add_argument("--faiss-index", default=DEFAULT_FAISS_INDEX)
    parser.add_argument("--predictions-output", default="reports/hybrid_cross_encoder_predictions.jsonl")
    parser.add_argument("--metrics-output", default="reports/hybrid_cross_encoder_metrics.json")
    parser.add_argument("--dense-model-name", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--cross-encoder-model-name", default=DEFAULT_CROSS_ENCODER_MODEL)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bm25-prefilter-depth", type=int, default=500)
    parser.add_argument("--faiss-prefilter-depth", type=int, default=500)
    parser.add_argument("--retrieval-depth", type=int, default=100)
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
        {k: metrics[k] for k in (
            "Recall@1", "Recall@5", "Recall@10", "MRR",
            "num_queries", "avg_stage1_candidates",
        ) if k in metrics},
        indent=2,
    ))


if __name__ == "__main__":
    main()
