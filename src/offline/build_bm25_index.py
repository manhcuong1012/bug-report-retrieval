"""
Build Inverted Index for BM25 offline.
Run once to create persistent index.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrieval.bm25_retriever import build_index as build_bm25_index, record_tokens  # type: ignore

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S %z"


def save_bm25_index(
    index_path: Path,
    postings: dict[str, list[tuple[str, int]]],
    df: Counter[str],
    doc_lengths: dict[str, int],
    doc_timestamps: dict[str, float],
    num_docs: int,
    avgdl: float,
    config: dict[str, Any],
):
    """Save BM25 index to disk."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("wb") as handle:
        pickle.dump(
            {
                "postings": postings,
                "df": df,
                "doc_lengths": doc_lengths,
                "doc_timestamps": doc_timestamps,
                "num_docs": num_docs,
                "avgdl": avgdl,
                "config": config,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"✅ BM25 index saved to {index_path}")
    print(f"   - Num docs: {num_docs}")
    print(f"   - Vocab size: {len(df)}")
    print(f"   - Avg doc length: {avgdl:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Build and save BM25 inverted index offline.")
    parser.add_argument("--train-records", default="data/train.jsonl")
    parser.add_argument("--output-index", default="reports/cache/bm25_index.pkl")
    parser.add_argument("--summary-repeat", type=int, default=5)
    parser.add_argument("--component-repeat", type=int, default=1)
    parser.add_argument("--priority-repeat", type=int, default=1)
    parser.add_argument("--severity-repeat", type=int, default=1)
    parser.add_argument("--max-term-df-ratio", type=float, default=0.1)
    parser.add_argument("--max-metadata-df-ratio", type=float, default=0.1)
    args = parser.parse_args()

    train_records_path = Path(args.train_records)
    output_index = Path(args.output_index)

    print(f"Building BM25 inverted index from {train_records_path}...")
    postings, df, doc_lengths, doc_timestamps, num_docs, avgdl = build_bm25_index(
        train_records_path,
        summary_repeat=args.summary_repeat,
        component_repeat=args.component_repeat,
        priority_repeat=args.priority_repeat,
        severity_repeat=args.severity_repeat,
        max_term_df_ratio=args.max_term_df_ratio,
        max_metadata_df_ratio=args.max_metadata_df_ratio,
    )
    config = {
        "summary_repeat": args.summary_repeat,
        "component_repeat": args.component_repeat,
        "priority_repeat": args.priority_repeat,
        "severity_repeat": args.severity_repeat,
        "max_term_df_ratio": args.max_term_df_ratio,
        "max_metadata_df_ratio": args.max_metadata_df_ratio,
    }

    save_bm25_index(
        output_index,
        postings,
        df,
        doc_lengths,
        doc_timestamps,
        num_docs,
        avgdl,
        config,
    )


if __name__ == "__main__":
    main()
