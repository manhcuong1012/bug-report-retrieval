from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.metrics import evaluate as evaluate_predictions
from retrieval.bm25_retriever import bm25_idf, build_index as build_bm25_index, record_tokens
from retrieval.bug_feature_scorer import build_query_profile, precompute_bug, score_pair
from retrieval.feature_retriever import (
    build_index as build_feature_index,
    get_candidates as get_feature_candidates,
)
from retrieval.reranker import fuse_scores, rank_candidates

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S %z"


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


def compute_bm25_scores(
    query_record: dict,
    postings: dict[str, list[tuple[str, int]]],
    df: Counter[str],
    doc_lengths: dict[str, int],
    doc_timestamps: dict[str, float],
    num_docs: int,
    avgdl: float,
    summary_repeat: int,
    component_repeat: int,
    priority_repeat: int,
    severity_repeat: int,
    k1: float,
    b: float,
) -> dict[str, float]:
    query_tokens = record_tokens(
        query_record,
        summary_repeat=summary_repeat,
        component_repeat=component_repeat,
        priority_repeat=priority_repeat,
        severity_repeat=severity_repeat,
    )
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


def compute_feature_scores(
    query_record: dict,
    candidate_records_raw: dict[str, dict],
    candidate_ids: list[str],
) -> tuple[dict[str, float], dict[str, dict], dict]:
    query_pre = precompute_bug(query_record)
    query_profile = build_query_profile(query_record.get("summary", ""), query_record.get("description", ""))

    feature_scores: dict[str, float] = {}
    candidate_precomputed: dict[str, dict] = {}
    for bug_id in candidate_ids:
        candidate_raw = candidate_records_raw.get(bug_id)
        if candidate_raw is None:
            continue
        candidate_pre = precompute_bug(candidate_raw)
        candidate_precomputed[bug_id] = candidate_pre
        feature_scores[bug_id] = score_pair(query_pre, candidate_pre, query_profile)

    return feature_scores, candidate_precomputed, query_profile


def retrieve(
    test_records_path: Path,
    train_records_path: Path,
    predictions_output: Path,
    metrics_output: Path,
    top_k: int = 10,
    rerank_depth: int = 300,
    summary_repeat: int = 5,
    component_repeat: int = 1,
    priority_repeat: int = 1,
    severity_repeat: int = 1,
    max_term_df_ratio: float | None = 0.1,
    max_metadata_df_ratio: float | None = 0.1,
    k1: float = 2.0,
    b: float = 0.5,
    alpha: float = 0.6,
    beta: float = 0.4,
    use_rank_normalization: bool = False,
    feature_candidate_depth: int = 0,
    max_feature_candidates: int = 5000,
) -> dict[str, Any]:
    postings, df, doc_lengths, doc_timestamps, num_docs, avgdl = build_bm25_index(
        train_records_path,
        summary_repeat=summary_repeat,
        component_repeat=component_repeat,
        priority_repeat=priority_repeat,
        severity_repeat=severity_repeat,
        max_term_df_ratio=max_term_df_ratio,
        max_metadata_df_ratio=max_metadata_df_ratio,
    )
    candidate_records_raw = {
        record["bug_id"]: record
        for record in iter_jsonl(train_records_path)
    }
    feature_index = None
    if feature_candidate_depth > 0:
        feature_index = build_feature_index(train_records_path)

    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    total_queries = 0
    total_reranked = 0
    total_bm25_candidates = 0
    total_feature_candidates = 0

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
                component_repeat,
                priority_repeat,
                severity_repeat,
                k1,
                b,
            )

            bm25_ranked = rank_candidates(bm25_scores, rerank_depth)
            bm25_candidate_ids = [bug_id for bug_id, _ in bm25_ranked]
            candidate_ids = list(bm25_candidate_ids)
            total_bm25_candidates += len(bm25_candidate_ids)

            if feature_index is not None:
                inverted_index, component_index, feature_timestamps, feature_precomputed, _ = feature_index
                feature_candidate_hits = get_feature_candidates(
                    record,
                    inverted_index,
                    component_index,
                    feature_timestamps,
                    max_feature_candidates=max_feature_candidates,
                )
                query_pre_for_feature = precompute_bug(record)
                query_profile_for_feature = build_query_profile(
                    record.get("summary", ""),
                    record.get("description", ""),
                )
                feature_prefilter_scores: dict[str, float] = {}
                for bug_id in feature_candidate_hits:
                    candidate_pre = feature_precomputed.get(bug_id)
                    if candidate_pre is None:
                        continue
                    feature_prefilter_scores[bug_id] = score_pair(
                        query_pre_for_feature,
                        candidate_pre,
                        query_profile_for_feature,
                    )

                feature_ranked = rank_candidates(feature_prefilter_scores, feature_candidate_depth)
                feature_candidate_ids = [bug_id for bug_id, _ in feature_ranked]
                total_feature_candidates += len(feature_candidate_ids)
                seen = set(candidate_ids)
                for bug_id in feature_candidate_ids:
                    if bug_id not in seen:
                        candidate_ids.append(bug_id)
                        seen.add(bug_id)

            total_reranked += len(candidate_ids)

            feature_scores, candidate_records, query_profile = compute_feature_scores(
                record,
                candidate_records_raw,
                candidate_ids,
            )

            query_pre = precompute_bug(record)
            fused_scores = fuse_scores(
                bm25_scores={bug_id: bm25_scores.get(bug_id, 0.0) for bug_id in candidate_ids},
                feature_scores=feature_scores,
                query_record={**record, **query_pre},
                candidate_records=candidate_records,
                query_profile=query_profile,
                alpha=alpha,
                beta=beta,
                use_rank_normalization=use_rank_normalization,
            )

            ranked = rank_candidates(fused_scores, top_k)
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

    metrics = evaluate_predictions(predictions_output, train_records_path)
    metrics.update(
        {
            "method": "hybrid",
            "top_k": top_k,
            "rerank_depth": rerank_depth,
            "summary_repeat": summary_repeat,
            "component_repeat": component_repeat,
            "priority_repeat": priority_repeat,
            "severity_repeat": severity_repeat,
            "max_term_df_ratio": max_term_df_ratio,
            "max_metadata_df_ratio": max_metadata_df_ratio,
            "k1": k1,
            "b": b,
            "alpha": alpha,
            "beta": beta,
            "normalization": "rank" if use_rank_normalization else "min-max",
            "feature_candidate_depth": feature_candidate_depth,
            "max_feature_candidates": max_feature_candidates,
            "num_queries": total_queries,
            "avg_reranked_candidates": total_reranked / max(total_queries, 1),
            "avg_bm25_candidates": total_bm25_candidates / max(total_queries, 1),
            "avg_feature_candidates": total_feature_candidates / max(total_queries, 1),
        }
    )
    write_json(metrics_output, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hybrid retrieval on Mozilla.")
    parser.add_argument("--train-records", default="data/train.jsonl")
    parser.add_argument("--test-records", default="data/test.jsonl")
    parser.add_argument("--predictions-output", default="reports/hybrid_predictions.jsonl")
    parser.add_argument("--metrics-output", default="reports/hybrid_metrics.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rerank-depth", type=int, default=300)
    parser.add_argument("--summary-repeat", type=int, default=5)
    parser.add_argument("--component-repeat", type=int, default=1)
    parser.add_argument("--priority-repeat", type=int, default=1)
    parser.add_argument("--severity-repeat", type=int, default=1)
    parser.add_argument("--max-term-df-ratio", type=float, default=0.1)
    parser.add_argument("--max-metadata-df-ratio", type=float, default=0.1)
    parser.add_argument("--k1", type=float, default=2.0)
    parser.add_argument("--b", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--rank-normalization", action="store_true")
    parser.add_argument("--feature-candidate-depth", type=int, default=0)
    parser.add_argument("--max-feature-candidates", type=int, default=5000)
    args = parser.parse_args()

    metrics = retrieve(
        test_records_path=Path(args.test_records),
        train_records_path=Path(args.train_records),
        predictions_output=Path(args.predictions_output),
        metrics_output=Path(args.metrics_output),
        top_k=args.top_k,
        rerank_depth=args.rerank_depth,
        summary_repeat=args.summary_repeat,
        component_repeat=args.component_repeat,
        priority_repeat=args.priority_repeat,
        severity_repeat=args.severity_repeat,
        max_term_df_ratio=args.max_term_df_ratio,
        max_metadata_df_ratio=args.max_metadata_df_ratio,
        k1=args.k1,
        b=args.b,
        alpha=args.alpha,
        beta=args.beta,
        use_rank_normalization=args.rank_normalization,
        feature_candidate_depth=args.feature_candidate_depth,
        max_feature_candidates=args.max_feature_candidates,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
