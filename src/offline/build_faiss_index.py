"""
Build FAISS vector index for dense embeddings offline.
Chạy một lần để tạo persistent FAISS index dùng cho hybrid_cross_encoder_retriever.py.

Output:
  <output-index>          — FAISS HNSW binary index
  <output-index>_ids.json — danh sách bug_id theo thứ tự index
  <output-index>_meta.json— metadata để kiểm tra tính hợp lệ của index

Fix macOS Apple Silicon (M1/M2/M3):
  FAISS HNSW dùng OpenMP multi-thread, conflict với PyTorch MPS gây segfault.
  Giải pháp: set OMP_NUM_THREADS=1 TRƯỚC KHI import bất kỳ thứ gì,
  và encode embedding trên CPU thay vì MPS.
"""
from __future__ import annotations

# ── macOS / Apple Silicon segfault fix ────────────────────────────────────────
# PHẢI set trước khi import torch, faiss, hay bất kỳ thư viện nào dùng OpenMP.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S %z"
DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OUTPUT_INDEX = "reports/cache/faiss_index"

# HNSW tuning
HNSW_M = 32              # số connections mỗi node — tăng lên cải thiện recall, tốn RAM hơn
HNSW_EF_CONSTRUCTION = 200  # beam width khi build — tăng lên index chính xác hơn, build chậm hơn


# ---------------------------------------------------------------------------
# Helpers (standalone, không import từ các retriever khác)
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_train_records(path: Path) -> list[dict]:
    print(f"[data] Loading train records from {path} ...")
    records = list(iter_jsonl(path))
    print(f"[data] {len(records):,} records loaded")
    return records


def build_semantic_text(record: dict, max_description_chars: int) -> str:
    """Phải khớp với build_semantic_text trong hybrid_cross_encoder_retriever.py."""
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


def choose_device(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    # Trên macOS Apple Silicon: KHÔNG dùng MPS khi build FAISS.
    # MPS + FAISS OpenMP multi-thread = segfault.
    # Encode trên CPU vẫn đủ nhanh (~18 phút cho 154k records với MiniLM).
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_file_fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


# ---------------------------------------------------------------------------
# Load embedding model
# ---------------------------------------------------------------------------

def load_embedding_model(model_name: str, device: str):
    try:
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Cần torch và sentence-transformers.\n"
            "Install: .venv/bin/python -m pip install -r requirements-semantic.txt"
        ) from exc

    resolved_device = choose_device(torch, device)
    print(f"[model] Loading '{model_name}' on {resolved_device} ...")
    model = SentenceTransformer(model_name, device=resolved_device)
    return np, model, resolved_device


# ---------------------------------------------------------------------------
# Build & save FAISS index
# ---------------------------------------------------------------------------

def build_and_save_faiss_index(
    embeddings,          # np.ndarray float32 [N x D]
    bug_ids: list[str],
    index_path: Path,
    model_name: str,
    train_file_meta: dict,
    max_description_chars: int,
):
    """Build FAISS HNSW index và lưu 3 file: index, ids, metadata."""
    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "FAISS is required.\n"
            "Install: pip install faiss-cpu   # hoặc faiss-gpu nếu có GPU"
        ) from exc

    n, dim = embeddings.shape
    assert embeddings.dtype == np.float32, "embeddings phải là float32"
    assert len(bug_ids) == n, "số bug_id phải bằng số vector"

    print(f"[faiss] Building HNSW index: {n:,} vectors × dim={dim}, M={HNSW_M}, efConstruction={HNSW_EF_CONSTRUCTION}")

    index = faiss.IndexHNSWFlat(dim, HNSW_M)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.add(embeddings)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    print(f"✅ FAISS index saved  → {index_path}")
    print(f"   vectors : {index.ntotal:,}")
    print(f"   dim     : {dim}")
    print(f"   M       : {HNSW_M}  |  efConstruction: {HNSW_EF_CONSTRUCTION}")

    # Bug ID mapping
    ids_path = index_path.parent / f"{index_path.stem}_ids.json"
    with ids_path.open("w", encoding="utf-8") as f:
        json.dump(bug_ids, f)
    print(f"✅ Bug ID mapping saved → {ids_path}")

    # Metadata (để kiểm tra tính hợp lệ khi load)
    meta = {
        "model_name": model_name,
        "train_file": train_file_meta,
        "num_vectors": n,
        "dimension": dim,
        "hnsw_m": HNSW_M,
        "hnsw_ef_construction": HNSW_EF_CONSTRUCTION,
        "max_description_chars": max_description_chars,
        "built_at": datetime.now().isoformat(),
    }
    meta_path = index_path.parent / f"{index_path.stem}_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ Metadata saved       → {meta_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS HNSW index offline từ train records.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-records", default="data/train.jsonl",
                        help="Path tới train.jsonl")
    parser.add_argument("--output-index", default=DEFAULT_OUTPUT_INDEX,
                        help="Output path cho FAISS index (không cần extension)")
    parser.add_argument("--model-name", default=DEFAULT_DENSE_MODEL,
                        help="SentenceTransformer model name")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size khi encode")
    parser.add_argument("--max-description-chars", type=int, default=1000,
                        help="Cắt description tối đa bao nhiêu ký tự (phải khớp với retriever)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                        help="Device để encode embedding")
    args = parser.parse_args()

    train_records_path = Path(args.train_records)
    output_index = Path(args.output_index)

    # ── Load model ────────────────────────────────────────────────────────
    np, model, resolved_device = load_embedding_model(args.model_name, args.device)
    print(f"[model] Device resolved: {resolved_device}")

    # ── Load records ──────────────────────────────────────────────────────
    train_records = load_train_records(train_records_path)
    bug_ids = [r["bug_id"] for r in train_records]

    # ── Encode ────────────────────────────────────────────────────────────
    print(f"\n[encode] Building semantic texts (max_desc={args.max_description_chars}) ...")
    texts = [build_semantic_text(r, args.max_description_chars) for r in train_records]

    print(f"[encode] Encoding {len(texts):,} texts with batch_size={args.batch_size} ...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # normalize để dùng inner product ≡ cosine
    ).astype(np.float32)

    print(f"[encode] Done — shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    # Giải phóng bộ nhớ model trước khi build FAISS index.
    # Tránh OOM và conflict thread trên Apple Silicon.
    del model
    del texts
    import gc; gc.collect()

    # ── Build & save FAISS index ──────────────────────────────────────────
    print()
    build_and_save_faiss_index(
        embeddings=embeddings,
        bug_ids=bug_ids,
        index_path=output_index,
        model_name=args.model_name,
        train_file_meta=train_file_fingerprint(train_records_path),
        max_description_chars=args.max_description_chars,
    )

    print("\n[done] FAISS index build complete.")
    print(f"       Chạy retriever với: --faiss-index {output_index}")


if __name__ == "__main__":
    main()
