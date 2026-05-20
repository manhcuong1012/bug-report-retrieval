"""Pre-build all caches for the demo. Run once, demo loads in seconds."""
from __future__ import annotations

import json
import math
import pickle
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrieval.bug_feature_scorer import precompute_bug

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S %z"
MULTISPACE_RE = re.compile(r"\s+")
NON_TEXT_RE = re.compile(r"[^a-z0-9._/#\-\+\s]")

TRAIN_PATH = ROOT_DIR / "data" / "train.jsonl"
CACHE_DIR = ROOT_DIR / "reports" / "cache"


def normalize_text(text: str) -> str:
    return MULTISPACE_RE.sub(" ", NON_TEXT_RE.sub(" ", text.lower())).strip()


def build_weighted_text(summary: str, description: str, summary_repeat: int = 3) -> str:
    repeated = " ".join([summary] * max(summary_repeat, 1))
    return normalize_text(f"{repeated} {description}".strip())


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    demo_cache = CACHE_DIR / "demo_cache.pkl"

    print("Loading training records...")
    train_records = list(iter_jsonl(TRAIN_PATH))
    print(f"  {len(train_records):,} records")

    print("Building BM25 index...")
    postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
    df: Counter = Counter()
    doc_lengths: dict[str, int] = {}
    doc_timestamps: dict[str, float] = {}
    num_docs = 0

    for record in train_records:
        num_docs += 1
        doc_id = record["bug_id"]
        tokens = build_weighted_text(record.get("summary", ""), record.get("description", "")).split()
        tf = Counter(tokens)
        doc_lengths[doc_id] = len(tokens)
        doc_timestamps[doc_id] = datetime.strptime(record["created_at"], TIMESTAMP_FORMAT).timestamp()
        for term, raw_tf in tf.items():
            postings[term].append((doc_id, raw_tf))
            df[term] += 1

    avgdl = sum(doc_lengths.values()) / num_docs if num_docs else 0.0
    print(f"  {num_docs:,} docs, {len(postings):,} terms")

    print("Precomputing features for all bugs...")
    precomputed = {}
    for i, r in enumerate(train_records):
        precomputed[r["bug_id"]] = precompute_bug(r)
        if (i + 1) % 50000 == 0:
            print(f"  {i + 1:,}/{len(train_records):,}")
    print(f"  Done: {len(precomputed):,} bugs")

    bug_id_to_record = {r["bug_id"]: r for r in train_records}

    print(f"Saving cache to {demo_cache}...")
    with demo_cache.open("wb") as f:
        pickle.dump({
            "postings": dict(postings),
            "df": df,
            "doc_lengths": doc_lengths,
            "doc_timestamps": doc_timestamps,
            "num_docs": num_docs,
            "avgdl": avgdl,
            "bug_id_to_record": bug_id_to_record,
            "precomputed": precomputed,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = demo_cache.stat().st_size / 1e6
    print(f"Cache saved: {size_mb:.1f} MB")
    print("Done! Now run: streamlit run src/app/demo.py")


if __name__ == "__main__":
    main()
