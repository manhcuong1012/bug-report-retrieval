from __future__ import annotations

import json
import math
import pickle
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from heapq import nlargest
from pathlib import Path

import streamlit as st

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrieval.bug_feature_scorer import precompute_bug, score_pair, build_query_profile
from retrieval.reranker import fuse_scores, rank_candidates

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S %z"
MULTISPACE_RE = re.compile(r"\s+")
NON_TEXT_RE = re.compile(r"[^a-z0-9._/#\-\+\s]")

TRAIN_PATH = ROOT_DIR / "data" / "train.jsonl"
TEST_PATH = ROOT_DIR / "data" / "test.jsonl"
DEMO_CACHE = ROOT_DIR / "reports" / "cache" / "demo_cache.pkl"


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


def bm25_idf(df_val: int, n: int) -> float:
    return math.log(1.0 + (n - df_val + 0.5) / (df_val + 0.5))


@st.cache_resource(show_spinner="Loading data cache...")
def load_data():
    if not DEMO_CACHE.exists():
        st.error("Cache not found! Run: python src/app/build_cache.py")
        st.stop()
    with DEMO_CACHE.open("rb") as f:
        cache = pickle.load(f)
    return (
        cache["postings"],
        cache["df"],
        cache["doc_lengths"],
        cache["doc_timestamps"],
        cache["num_docs"],
        cache["avgdl"],
        cache["bug_id_to_record"],
        cache["precomputed"],
    )


@st.cache_data(show_spinner="Loading test queries...")
def load_test_records():
    return list(iter_jsonl(TEST_PATH))


def compute_bm25(query_summary, query_description, postings, df, doc_lengths, num_docs, avgdl, candidate_depth, k1=1.5, b=0.75):
    tokens = build_weighted_text(query_summary, query_description).split()
    query_terms = Counter(tokens)
    scores: dict[str, float] = defaultdict(float)
    for term in query_terms:
        if term not in postings:
            continue
        idf = bm25_idf(df[term], num_docs)
        for doc_id, raw_tf in postings[term]:
            dl = doc_lengths[doc_id]
            num = raw_tf * (k1 + 1.0)
            den = raw_tf + k1 * (1.0 - b + b * dl / max(avgdl, 1.0))
            scores[doc_id] += idf * (num / den)
    return dict(scores), nlargest(candidate_depth, scores.items(), key=lambda x: x[1])


def rerank_hybrid(query_record, bm25_all_scores, candidate_ids, precomputed, top_k):
    query_pre = precompute_bug(query_record)
    query_profile = build_query_profile(query_record.get("summary", ""), query_record.get("description", ""))

    feature_scores = {}
    candidate_pre_map = {}
    for bug_id in candidate_ids:
        cpre = precomputed.get(bug_id)
        if cpre is None:
            continue
        candidate_pre_map[bug_id] = cpre
        feature_scores[bug_id] = score_pair(query_pre, cpre, query_profile)

    fused = fuse_scores(
        bm25_scores={bid: bm25_all_scores.get(bid, 0.0) for bid in candidate_ids},
        feature_scores=feature_scores,
        query_record={**query_record, **query_pre},
        candidate_records=candidate_pre_map,
        query_profile=query_profile,
    )
    return rank_candidates(fused, top_k)


def render_results(ranked, bug_id_to_record, query_bucket):
    if not ranked:
        st.warning("No results found.")
        return

    hits = 0
    for rank, (bug_id, score) in enumerate(ranked, start=1):
        candidate = bug_id_to_record.get(bug_id, {})
        cand_bucket = candidate.get("bucket_id", "")
        is_match = query_bucket and cand_bucket == query_bucket
        if is_match:
            hits += 1

        badge = " ✅ DUPLICATE" if is_match else ""
        with st.expander(f"#{rank} — Bug `{bug_id}` (score: {score:.4f}){badge}", expanded=(rank <= 3)):
            c1, c2, c3 = st.columns([3, 1, 1])
            with c1:
                st.markdown(f"**{candidate.get('summary', 'N/A')}**")
            with c2:
                st.markdown(f"Component: `{candidate.get('component', 'UNKNOWN')}`")
            with c3:
                st.markdown(f"Severity: `{candidate.get('severity', 'UNKNOWN')}`")
            desc = candidate.get("description", "")
            if desc:
                st.text(desc[:500])
            if is_match:
                st.success(f"Same bucket: `{cand_bucket}`")

    if query_bucket:
        st.info(f"Duplicates found in top-{len(ranked)}: **{hits}**")


def main():
    st.set_page_config(page_title="Bug Duplicate Finder", page_icon="🔍", layout="wide")
    st.title("🔍 Bug Duplicate Finder")
    st.caption("Pipeline: BM25 Inverted Index Pre-filter → Hybrid Rerank (BM25 + Feature Scoring)")

    postings, df, doc_lengths, doc_timestamps, num_docs, avgdl, bug_id_to_record, precomputed = load_data()
    test_records = load_test_records()

    st.sidebar.header("⚙️ Settings")
    candidate_depth = st.sidebar.slider("BM25 candidates", 50, 500, 200, step=50)
    top_k = st.sidebar.slider("Top-K results", 5, 20, 10)

    tab_input, tab_test = st.tabs(["✏️ Input Bug Report", "📋 Select from Test Set"])

    query_record = None

    with tab_input:
        col1, col2 = st.columns([1, 1])
        with col1:
            summary = st.text_input("Summary", placeholder="e.g. Crash when opening new tab in private mode")
            component = st.text_input("Component", placeholder="e.g. Networking: HTTP")
        with col2:
            severity = st.selectbox("Severity", ["normal", "critical", "major", "minor", "trivial", "blocker", "enhancement"])
            priority = st.selectbox("Priority", ["UNKNOWN", "P1", "P2", "P3", "P4", "P5"])
        description = st.text_area("Description", height=150, placeholder="Detailed bug description...")

        if st.button("Find Duplicates", type="primary", key="btn_input"):
            if summary.strip():
                query_record = {
                    "bug_id": "USER_INPUT",
                    "summary": summary,
                    "description": description,
                    "component": component,
                    "severity": severity,
                    "priority": priority,
                    "bucket_id": "",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S +0000"),
                }
            else:
                st.warning("Please enter a summary.")

    with tab_test:
        options = {f"{r['bug_id']} — {r['summary'][:80]}": r for r in test_records[:500]}
        selected = st.selectbox("Select a test bug", list(options.keys()))
        if st.button("Find Duplicates", type="primary", key="btn_test"):
            query_record = options[selected]
            st.markdown(f"**Bug `{query_record['bug_id']}`** | Component: `{query_record.get('component', '?')}` | Bucket: `{query_record.get('bucket_id', '?')}`")
            with st.expander("Query details"):
                st.markdown(f"**Summary**: {query_record['summary']}")
                st.text(query_record.get("description", "")[:500])

    if query_record is None:
        return

    query_bucket = query_record.get("bucket_id", "")

    with st.spinner("BM25 pre-filtering..."):
        bm25_all_scores, bm25_ranked = compute_bm25(
            query_record.get("summary", ""),
            query_record.get("description", ""),
            postings, df, doc_lengths, num_docs, avgdl,
            candidate_depth,
        )
        candidate_ids = [bid for bid, _ in bm25_ranked]

    st.success(f"BM25 returned **{len(candidate_ids)}** candidates → Hybrid reranking")

    with st.spinner("Hybrid reranking (BM25 + Feature)..."):
        results = rerank_hybrid(query_record, bm25_all_scores, candidate_ids, precomputed, top_k)

    render_results(results, bug_id_to_record, query_bucket)


if __name__ == "__main__":
    main()
