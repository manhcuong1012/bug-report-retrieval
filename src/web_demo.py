from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import threading
from collections import Counter, defaultdict
from heapq import nlargest
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrieval.bm25_retriever import bm25_idf, record_tokens  # type: ignore
from retrieval.bug_feature_scorer import (  # type: ignore
    build_query_profile,
    precompute_bug,
    score_pair,
)
from retrieval.reranker import FusionConfig, rerank_candidates  # type: ignore


TIMESTAMP_FUTURE = "2099-01-01 00:00:00 +0000"
INDEX_CACHE_VERSION = 3
BM25_SUMMARY_REPEAT = 5
BM25_MAX_DF_RATIO = 0.10
BM25_K1 = 2.0
BM25_B = 0.5


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {
        prediction["query_bug_id"]: prediction
        for prediction in iter_jsonl(path)
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def text_excerpt(value: str, limit: int = 280) -> str:
    text = " ".join(str(value or "").replace("\n", " ").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def as_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(parsed) or math.isinf(parsed):
        return 0.0
    return parsed


class DemoStore:
    def __init__(
        self,
        root_dir: Path,
        rebuild_index: bool = False,
        max_train_records: int | None = None,
        rerank_depth: int = 100,
    ) -> None:
        self.root_dir = root_dir
        self.train_path = root_dir / "data" / "train.jsonl"
        self.test_path = root_dir / "data" / "test.jsonl"
        self.predictions_path = root_dir / "reports" / "hybrid_predictions.jsonl"
        self.metrics_path = root_dir / "reports" / "hybrid_metrics.json"
        cache_name = "web_full_index.pkl"
        if max_train_records is not None:
            cache_name = f"web_full_index.limit{max_train_records}.pkl"
        self.cache_path = root_dir / "data" / "processed" / cache_name
        self.rebuild_index = rebuild_index
        self.max_train_records = max_train_records
        self.rerank_depth = rerank_depth
        self.lock = threading.Lock()

        self.metrics: dict[str, Any] = {}
        self.test_records: dict[str, dict[str, Any]] = {}
        self.hybrid_predictions: dict[str, dict[str, Any]] = {}
        self.train_records: dict[str, dict[str, Any]] = {}
        self.records: dict[str, dict[str, Any]] = {}
        self.precomputed: dict[str, dict[str, Any]] = {}
        self.postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self.df: Counter[str] = Counter()
        self.doc_lengths: dict[str, int] = {}
        self.num_docs = 0
        self.avgdl = 1.0
        self.index_source = "not_loaded"
        self.examples: list[dict[str, str]] = []

        self.load()

    def load(self) -> None:
        print("[web] loading test records")
        self.metrics = load_json(self.metrics_path)
        self.test_records = {
            record["bug_id"]: record
            for record in iter_jsonl(self.test_path)
        }
        self.hybrid_predictions = load_predictions(self.predictions_path)
        self._load_or_build_full_index()
        self.records = {**self.train_records, **self.test_records}
        self.examples = self._build_examples()
        print(
            "[web] ready: "
            f"{len(self.test_records)} test bugs, "
            f"{len(self.train_records)} train bugs indexed "
            f"({self.index_source})"
        )

    def _load_or_build_full_index(self) -> None:
        if not self.rebuild_index and self._load_index_cache():
            self.index_source = "full_cache"
            return

        print("[web] building full train index; first run can take a few minutes")
        self._build_full_index()
        self._save_index_cache()
        self.index_source = "full_rebuilt"

    def _load_index_cache(self) -> bool:
        if not self.cache_path.exists():
            return False
        try:
            with self.cache_path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:  # noqa: BLE001 - cache can be safely rebuilt.
            print(f"[web] cannot load cache, rebuilding: {exc}")
            return False

        metadata = payload.get("metadata", {})
        expected_train_mtime = self.train_path.stat().st_mtime
        if metadata.get("version") != INDEX_CACHE_VERSION:
            return False
        if metadata.get("train_mtime") != expected_train_mtime:
            return False
        if metadata.get("max_train_records") != self.max_train_records:
            return False

        self.train_records = payload["train_records"]
        self.precomputed = payload["precomputed"]
        self.postings = payload["postings"]
        self.df = payload["df"]
        self.doc_lengths = payload["doc_lengths"]
        self.num_docs = payload["num_docs"]
        self.avgdl = payload["avgdl"]
        print(f"[web] loaded full index cache: {self.cache_path}")
        return True

    def _save_index_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {
                "version": INDEX_CACHE_VERSION,
                "train_mtime": self.train_path.stat().st_mtime,
                "max_train_records": self.max_train_records,
            },
            "train_records": self.train_records,
            "precomputed": self.precomputed,
            "postings": self.postings,
            "df": self.df,
            "doc_lengths": self.doc_lengths,
            "num_docs": self.num_docs,
            "avgdl": self.avgdl,
        }
        with self.cache_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[web] saved full index cache: {self.cache_path}")

    def _build_full_index(self) -> None:
        doc_lengths: dict[str, int] = {}
        df: Counter[str] = Counter()
        rows = 0

        print("[web] pass 1/2: document frequencies")
        for record in iter_jsonl(self.train_path):
            bug_id = record["bug_id"]
            tokens = [
                token
                for token in record_tokens(record, summary_repeat=BM25_SUMMARY_REPEAT)
                if len(token) > 2
            ]
            unique_tokens = set(tokens)
            doc_lengths[bug_id] = len(tokens)
            for term in unique_tokens:
                df[term] += 1
            rows += 1
            if rows % 25000 == 0:
                print(f"[web] pass 1 indexed {rows} train records")
            if self.max_train_records is not None and rows >= self.max_train_records:
                break

        num_docs = len(doc_lengths)
        max_df = max(1, int(num_docs * BM25_MAX_DF_RATIO))
        allowed_terms = {
            term
            for term, term_df in df.items()
            if term_df <= max_df
        }

        print("[web] pass 2/2: postings and feature precompute")
        postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
        train_records: dict[str, dict[str, Any]] = {}
        precomputed: dict[str, dict[str, Any]] = {}
        rows = 0
        for record in iter_jsonl(self.train_path):
            bug_id = record["bug_id"]
            tokens = [
                token
                for token in record_tokens(record, summary_repeat=BM25_SUMMARY_REPEAT)
                if len(token) > 2 and token in allowed_terms
            ]
            tf = Counter(tokens)
            for term, raw_tf in tf.items():
                postings[term].append((bug_id, raw_tf))
            train_records[bug_id] = {
                "bug_id": bug_id,
                "bucket_id": record.get("bucket_id", ""),
                "summary": record.get("summary", ""),
                "description": text_excerpt(record.get("description", ""), 600),
                "component": record.get("component", "UNKNOWN"),
                "severity": record.get("severity", "UNKNOWN"),
                "priority": record.get("priority", "UNKNOWN"),
                "created_at": record.get("created_at", ""),
            }
            precomputed[bug_id] = precompute_bug(record)
            rows += 1
            if rows % 25000 == 0:
                print(f"[web] pass 2 indexed {rows} train records")
            if self.max_train_records is not None and rows >= self.max_train_records:
                break

        self.train_records = train_records
        self.precomputed = precomputed
        self.postings = postings
        self.df = df
        self.doc_lengths = doc_lengths
        self.num_docs = num_docs
        self.avgdl = sum(doc_lengths.values()) / max(num_docs, 1)

    def _build_examples(self) -> list[dict[str, str]]:
        examples: list[dict[str, str]] = []
        for bug_id in sorted(self.hybrid_predictions)[:12]:
            record = self.test_records.get(bug_id)
            if not record:
                continue
            examples.append(
                {
                    "bug_id": bug_id,
                    "summary": text_excerpt(record.get("summary", ""), 96),
                    "component": record.get("component", "UNKNOWN"),
                    "severity": record.get("severity", "UNKNOWN"),
                }
            )
        return examples

    def status(self) -> dict[str, Any]:
        return {
            "ready": True,
            "test_records": len(self.test_records),
            "candidate_records": len(self.train_records),
            "index_records": len(self.train_records),
            "index_source": self.index_source,
            "rerank_depth": self.rerank_depth,
            "prediction_queries": len(self.hybrid_predictions),
            "metrics": {
                "recall_at_1": self.metrics.get("Recall@1"),
                "recall_at_5": self.metrics.get("Recall@5"),
                "recall_at_10": self.metrics.get("Recall@10"),
                "mrr": self.metrics.get("MRR"),
            },
        }

    def search(self, payload: dict[str, Any]) -> dict[str, Any]:
        top_k = max(1, min(int(payload.get("top_k") or 10), 20))
        bug_id = str(payload.get("bug_id") or "").strip()
        if bug_id:
            return self.search_bug_id(bug_id, top_k)
        return self.search_text(payload, top_k)

    def search_bug_id(self, bug_id: str, top_k: int) -> dict[str, Any]:
        with self.lock:
            prediction = self.hybrid_predictions.get(bug_id)
            if prediction is not None:
                query_record = self.test_records.get(bug_id, {"bug_id": bug_id})
                results = [
                    self._format_prediction_result(result, rank, query_record)
                    for rank, result in enumerate(
                        prediction.get("results", [])[:top_k],
                        start=1,
                    )
                ]
                return {
                    "mode": "bug_id",
                    "source": "precomputed_hybrid",
                    "query": self._format_query(query_record),
                    "results": results,
                }

            query_record = self.records.get(bug_id)
            if query_record is None:
                query_record = self._scan_record_by_id(bug_id)
            if query_record is None:
                return {
                    "mode": "bug_id",
                    "source": "not_found",
                    "query": {"bug_id": bug_id},
                    "results": [],
                    "error": (
                        "Bug ID is not in the loaded train/test data. Try one of "
                        "the example test bug IDs or paste the bug content."
                    ),
                }
            return self._search_live(query_record, top_k, exclude_bug_id=bug_id)

    def search_text(self, payload: dict[str, Any], top_k: int) -> dict[str, Any]:
        summary = str(payload.get("summary") or "").strip()
        description = str(payload.get("description") or "").strip()
        if not summary and not description:
            return {
                "mode": "text",
                "source": "empty_query",
                "query": {},
                "results": [],
                "error": "Enter a bug ID or bug content.",
            }

        query_record = {
            "bug_id": "input",
            "project": "mozilla",
            "created_at": TIMESTAMP_FUTURE,
            "summary": summary,
            "description": description,
            "component": str(payload.get("component") or "UNKNOWN").strip() or "UNKNOWN",
            "priority": str(payload.get("priority") or "UNKNOWN").strip() or "UNKNOWN",
            "severity": str(payload.get("severity") or "normal").strip() or "normal",
            "bucket_id": "input",
        }
        with self.lock:
            return self._search_live(query_record, top_k)

    def _search_live(
        self,
        query_record: dict[str, Any],
        top_k: int,
        exclude_bug_id: str | None = None,
    ) -> dict[str, Any]:
        query_profile = build_query_profile(
            query_record.get("summary", ""),
            query_record.get("description", ""),
        )
        query_precomputed = precompute_bug(query_record)
        bm25_candidates = self._full_bm25_top(
            query_record,
            rerank_depth=max(self.rerank_depth, top_k * 5),
            exclude_bug_id=exclude_bug_id,
        )

        candidates: list[dict[str, Any]] = []
        for candidate_id, bm25_score in bm25_candidates:
            candidate_precomputed = self.precomputed.get(candidate_id)
            if candidate_precomputed is None:
                continue
            feature_score = score_pair(
                query_precomputed,
                candidate_precomputed,
                query_profile,
            )
            candidates.append(
                {
                    "bug_id": candidate_id,
                    "bm25_score": bm25_score,
                    "feature_score": feature_score,
                    "candidate_precomputed": candidate_precomputed,
                }
            )

        ranked = rerank_candidates(
            query_precomputed=query_precomputed,
            query_profile=query_profile,
            candidates=candidates,
            config=FusionConfig(alpha=0.5, beta=0.5),
        )[:top_k]

        return {
            "mode": "live",
            "source": "full_train_hybrid",
            "query": self._format_query(query_record),
            "results": [
                self._format_ranked_result(result, rank, query_record)
                for rank, result in enumerate(ranked, start=1)
            ],
        }

    def _full_bm25_top(
        self,
        query_record: dict[str, Any],
        rerank_depth: int,
        exclude_bug_id: str | None,
        summary_repeat: int = BM25_SUMMARY_REPEAT,
        max_query_terms: int = 220,
        max_df_ratio: float = BM25_MAX_DF_RATIO,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> list[tuple[str, float]]:
        query_terms = Counter(record_tokens(query_record, summary_repeat=summary_repeat))
        summary_terms = set(
            record_tokens(
                {
                    "summary": query_record.get("summary", ""),
                    "description": "",
                },
                summary_repeat=summary_repeat,
            )
        )

        selected_terms: list[tuple[int, float, str]] = []
        max_df = self.num_docs * max_df_ratio
        for term in query_terms:
            if term not in self.postings:
                continue
            term_df = self.df[term]
            in_summary = term in summary_terms
            if not in_summary and term_df > max_df:
                continue
            selected_terms.append(
                (1 if in_summary else 0, bm25_idf(term_df, self.num_docs), term)
            )

        selected_terms.sort(reverse=True)
        selected_terms = selected_terms[:max_query_terms]

        scores: dict[str, float] = defaultdict(float)
        for _summary_priority, idf, term in selected_terms:
            for doc_id, raw_tf in self.postings.get(term, []):
                if doc_id == exclude_bug_id:
                    continue
                doc_length = self.doc_lengths.get(doc_id, 0)
                numerator = raw_tf * (k1 + 1.0)
                denominator = raw_tf + k1 * (
                    1.0 - b + b * doc_length / max(self.avgdl, 1.0)
                )
                scores[doc_id] += idf * (numerator / denominator)

        return nlargest(rerank_depth, scores.items(), key=lambda item: item[1])

    def _scan_record_by_id(self, bug_id: str) -> dict[str, Any] | None:
        for path in (self.test_path, self.train_path):
            for record in iter_jsonl(path):
                if record.get("bug_id") == bug_id:
                    self.records[bug_id] = record
                    return record
        return None

    def _format_prediction_result(
        self,
        result: dict[str, Any],
        rank: int,
        query_record: dict[str, Any],
    ) -> dict[str, Any]:
        bug_id = result.get("bug_id")
        record = self.records.get(bug_id, {"bug_id": bug_id})
        return {
            **self._format_record(record),
            "rank": rank,
            "score": as_float(result.get("score")),
            "bucket_match": self._bucket_match(query_record, record),
        }

    def _format_ranked_result(
        self,
        result: dict[str, Any],
        rank: int,
        query_record: dict[str, Any],
    ) -> dict[str, Any]:
        record = self.records.get(result["bug_id"], {"bug_id": result["bug_id"]})
        return {
            **self._format_record(record),
            "rank": rank,
            "score": as_float(result.get("score")),
            "bm25_score": as_float(result.get("bm25_score")),
            "feature_score": as_float(result.get("feature_score")),
            "metadata_boost": as_float(result.get("metadata_boost")),
            "bucket_match": self._bucket_match(query_record, record),
        }

    def _format_query(self, record: dict[str, Any]) -> dict[str, Any]:
        return self._format_record(record, description_limit=420)

    def _format_record(
        self,
        record: dict[str, Any],
        description_limit: int = 220,
    ) -> dict[str, Any]:
        return {
            "bug_id": record.get("bug_id", ""),
            "bucket_id": record.get("bucket_id", ""),
            "summary": text_excerpt(record.get("summary", ""), 180),
            "description": text_excerpt(record.get("description", ""), description_limit),
            "component": record.get("component", "UNKNOWN"),
            "severity": record.get("severity", "UNKNOWN"),
            "priority": record.get("priority", "UNKNOWN"),
            "created_at": record.get("created_at", ""),
        }

    def _bucket_match(
        self,
        query_record: dict[str, Any],
        candidate_record: dict[str, Any],
    ) -> bool:
        query_bucket = query_record.get("bucket_id")
        candidate_bucket = candidate_record.get("bucket_id")
        return bool(query_bucket and query_bucket == candidate_bucket)


class DemoHandler(BaseHTTPRequestHandler):
    store: DemoStore
    static_dir = ROOT_DIR / "src" / "web" / "static"

    def log_message(self, format: str, *args: Any) -> None:
        print("[web]", format % args)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/status":
            self._send_json(self.store.status())
            return
        if path == "/api/examples":
            self._send_json({"examples": self.store.examples})
            return
        if path in {"/", "/index.html"}:
            self._send_static("index.html")
            return
        if path.startswith("/static/"):
            self._send_static(path.removeprefix("/static/"))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/search":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8") or "{}")
            response = self.store.search(payload)
        except Exception as exc:  # noqa: BLE001 - returned to the local demo UI.
            self._send_json({"error": str(exc), "results": []}, status=500)
            return

        status = 400 if response.get("error") else 200
        self._send_json(response, status=status)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_static(self, relative_path: str) -> None:
        safe_name = unquote(relative_path).replace("\\", "/")
        if safe_name.startswith("../") or "/../" in safe_name:
            self.send_error(HTTPStatus.BAD_REQUEST)
            return

        path = (self.static_dir / safe_name).resolve()
        if not str(path).startswith(str(self.static_dir.resolve())) or not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
        }.get(path.suffix, "application/octet-stream")
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bug retrieval web demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Ignore the saved full-index cache and rebuild it from data/train.jsonl.",
    )
    parser.add_argument(
        "--max-train-records",
        type=int,
        default=None,
        help="Debug only: limit indexed train records for a quick smoke run.",
    )
    parser.add_argument(
        "--rerank-depth",
        type=int,
        default=100,
        help="BM25 candidates to rerank with feature scoring for live searches.",
    )
    args = parser.parse_args()

    DemoHandler.store = DemoStore(
        ROOT_DIR,
        rebuild_index=args.rebuild_index,
        max_train_records=args.max_train_records,
        rerank_depth=args.rerank_depth,
    )
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"[web] serving http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[web] stopped")


if __name__ == "__main__":
    main()
