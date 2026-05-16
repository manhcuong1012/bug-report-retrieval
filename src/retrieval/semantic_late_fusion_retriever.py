from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.metrics import evaluate as evaluate_predictions


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


def load_predictions(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def add_unique_results(
    output: list[dict[str, Any]],
    seen_bug_ids: set[str],
    source_results: list[dict[str, Any]],
    limit: int,
) -> None:
    for result in source_results:
        if len(output) >= limit:
            return
        bug_id = result["bug_id"]
        if bug_id in seen_bug_ids:
            continue
        output.append(dict(result))
        seen_bug_ids.add(bug_id)


def rerank_results(
    base_prediction: dict[str, Any],
    semantic_prediction: dict[str, Any],
    hybrid_prediction: dict[str, Any],
    feature_prediction: dict[str, Any],
    bm25_prediction: dict[str, Any],
    keep_depth: int,
    top_k: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen_bug_ids: set[str] = set()

    add_unique_results(output, seen_bug_ids, base_prediction.get("results", [])[:keep_depth], top_k)
    for source in [semantic_prediction, hybrid_prediction, feature_prediction, bm25_prediction, base_prediction]:
        add_unique_results(output, seen_bug_ids, source.get("results", []), top_k)

    for rank, result in enumerate(output, start=1):
        result["rank"] = rank
    return output


def retrieve(
    base_predictions_path: Path,
    semantic_predictions_path: Path,
    hybrid_predictions_path: Path,
    feature_predictions_path: Path,
    bm25_predictions_path: Path,
    train_records_path: Path,
    predictions_output: Path,
    metrics_output: Path,
    top_k: int = 10,
    keep_depth: int = 3,
) -> dict[str, Any]:
    base_predictions = load_predictions(base_predictions_path)
    semantic_predictions = load_predictions(semantic_predictions_path)
    hybrid_predictions = load_predictions(hybrid_predictions_path)
    feature_predictions = load_predictions(feature_predictions_path)
    bm25_predictions = load_predictions(bm25_predictions_path)

    lengths = {
        len(base_predictions),
        len(semantic_predictions),
        len(hybrid_predictions),
        len(feature_predictions),
        len(bm25_predictions),
    }
    if len(lengths) != 1:
        raise ValueError("Prediction files must contain the same number of queries.")

    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    with predictions_output.open("w", encoding="utf-8") as handle:
        for base_prediction, semantic_prediction, hybrid_prediction, feature_prediction, bm25_prediction in zip(
            base_predictions,
            semantic_predictions,
            hybrid_predictions,
            feature_predictions,
            bm25_predictions,
        ):
            prediction = {
                "query_bug_id": base_prediction["query_bug_id"],
                "project": base_prediction.get("project", "mozilla"),
                "query_bucket_id": base_prediction["query_bucket_id"],
                "results": rerank_results(
                    base_prediction=base_prediction,
                    semantic_prediction=semantic_prediction,
                    hybrid_prediction=hybrid_prediction,
                    feature_prediction=feature_prediction,
                    bm25_prediction=bm25_prediction,
                    keep_depth=keep_depth,
                    top_k=top_k,
                ),
            }
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    metrics = evaluate_predictions(predictions_output, train_records_path)
    metrics.update(
        {
            "method": "semantic_late_fusion",
            "top_k": top_k,
            "keep_depth": keep_depth,
            "base_predictions": str(base_predictions_path),
            "semantic_predictions": str(semantic_predictions_path),
            "num_queries": len(base_predictions),
        }
    )
    write_json(metrics_output, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic late fusion over existing prediction files.")
    parser.add_argument("--base-predictions", default="reports/ensemble_predictions.jsonl")
    parser.add_argument("--semantic-predictions", default="reports/semantic_embedding_predictions.jsonl")
    parser.add_argument("--hybrid-predictions", default="reports/hybrid_predictions.jsonl")
    parser.add_argument("--feature-predictions", default="reports/feature_predictions.jsonl")
    parser.add_argument("--bm25-predictions", default="reports/bm25_predictions.jsonl")
    parser.add_argument("--train-records", default="data/train.jsonl")
    parser.add_argument("--predictions-output", default="reports/semantic_late_fusion_predictions.jsonl")
    parser.add_argument("--metrics-output", default="reports/semantic_late_fusion_metrics.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--keep-depth", type=int, default=3)
    args = parser.parse_args()

    metrics = retrieve(
        base_predictions_path=Path(args.base_predictions),
        semantic_predictions_path=Path(args.semantic_predictions),
        hybrid_predictions_path=Path(args.hybrid_predictions),
        feature_predictions_path=Path(args.feature_predictions),
        bm25_predictions_path=Path(args.bm25_predictions),
        train_records_path=Path(args.train_records),
        predictions_output=Path(args.predictions_output),
        metrics_output=Path(args.metrics_output),
        top_k=args.top_k,
        keep_depth=args.keep_depth,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
