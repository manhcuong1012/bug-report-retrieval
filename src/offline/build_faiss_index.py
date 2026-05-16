"""
Build FAISS vector index for dense embeddings offline.
Run once to create persistent FAISS index.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrieval.semantic_embedding_retriever import (
    build_semantic_text,
    choose_device,
    iter_jsonl,
    load_train_records,
    train_file_fingerprint,
)  # type: ignore

DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model(model_name: str, device: str):
    """Load dense embedding model."""
    try:
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Vector DB building requires torch and sentence-transformers. "
            "Install with: pip install -r requirements-semantic.txt"
        ) from exc

    resolved_device = choose_device(torch, device)
    model = SentenceTransformer(model_name, device=resolved_device)
    return np, model, resolved_device


def build_faiss_index(
    embeddings: Any,
    bug_ids: list[str],
    index_path: Path,
):
    """Build and save FAISS index."""
    try:
        import faiss
    except ImportError:
        raise RuntimeError(
            "FAISS is required for vector indexing. "
            "Install with: pip install faiss-cpu  # or faiss-gpu"
        )

    np = embeddings.__class__.__module__.split('.')[0]
    dimension = embeddings.shape[1]
    
    # Create FAISS index (HNSW for fast search)
    index = faiss.IndexHNSWFlat(dimension, 32)  # 32 = M param (connections per node)
    index.add(embeddings)
    
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    
    print(f"✅ FAISS index saved to {index_path}")
    print(f"   - Dimension: {dimension}")
    print(f"   - Num vectors: {len(bug_ids)}")
    print(f"   - Index type: HNSW (Hierarchical Navigable Small World)")
    
    # Also save bug_ids mapping
    ids_path = index_path.parent / f"{index_path.stem}_ids.json"
    with ids_path.open("w") as f:
        json.dump(bug_ids, f)
    print(f"✅ Bug ID mappings saved to {ids_path}")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS vector index offline.")
    parser.add_argument("--train-records", default="data/train.jsonl")
    parser.add_argument("--output-index", default="reports/cache/faiss_index")
    parser.add_argument("--model-name", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-description-chars", type=int, default=1000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()

    train_records_path = Path(args.train_records)
    output_index = Path(args.output_index)

    print(f"Loading embeddings model: {args.model_name}")
    np, model, resolved_device = load_embedding_model(args.model_name, args.device)
    print(f"Device: {resolved_device}")

    print(f"\nLoading training records from {train_records_path}...")
    train_records = load_train_records(train_records_path)
    print(f"Loaded {len(train_records)} records")

    print(f"\nEncoding texts to dense embeddings...")
    texts = [build_semantic_text(record, args.max_description_chars) for record in train_records]
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    
    # Ensure float32 for FAISS
    embeddings = embeddings.astype(np.float32)
    bug_ids = [record["bug_id"] for record in train_records]

    print(f"\nBuilding FAISS index...")
    build_faiss_index(embeddings, bug_ids, output_index)


if __name__ == "__main__":
    main()
