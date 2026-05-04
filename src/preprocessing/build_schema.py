from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from clean_text import build_text_clean, build_text_raw


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, item: str) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item: str) -> str:
        self.add(item)
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return

        left_rank = self.rank[left_root]
        right_rank = self.rank[right_root]
        if left_rank < right_rank:
            self.parent[left_root] = right_root
        elif left_rank > right_rank:
            self.parent[right_root] = left_root
        else:
            self.parent[right_root] = left_root
            self.rank[left_root] += 1


def normalize_scalar(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
        return fallback
    return str(value).strip() or fallback


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


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


def extract_duplicate_targets(raw_dup_id: Any) -> list[str]:
    if raw_dup_id is None:
        return []
    if isinstance(raw_dup_id, list):
        return [normalize_scalar(item) for item in raw_dup_id if normalize_scalar(item)]
    value = normalize_scalar(raw_dup_id)
    return [value] if value else []


def normalize_bug(raw_bug: dict[str, Any]) -> dict[str, str]:
    bug_id = normalize_scalar(raw_bug.get("bug_id"))
    summary = normalize_scalar(raw_bug.get("short_desc"))
    description = normalize_scalar(raw_bug.get("description"))
    priority = normalize_scalar(raw_bug.get("priority"), "UNKNOWN")
    if priority == "--":
        priority = "UNKNOWN"

    return {
        "bug_id": bug_id,
        "project": "mozilla",
        "created_at": normalize_scalar(raw_bug.get("creation_ts")),
        "summary": summary,
        "description": description,
        "component": normalize_scalar(raw_bug.get("component"), "UNKNOWN"),
        "priority": priority,
        "severity": normalize_scalar(raw_bug.get("bug_severity"), "UNKNOWN"),
        "text_raw": build_text_raw(summary, description),
        "text_clean": build_text_clean(summary, description),
    }


def record_key(bug_id: str) -> str:
    return f"mozilla:{bug_id}"


def canonical_sort_key(raw_bug_id: str) -> tuple[int, str]:
    try:
        return (0, f"{int(raw_bug_id):020d}")
    except ValueError:
        return (1, raw_bug_id)


def build_bucket_lookup(
    union_find: UnionFind, known_keys: set[str], raw_bug_ids_by_key: dict[str, str]
) -> tuple[dict[str, str], dict[str, list[str]]]:
    component_members: dict[str, list[str]] = defaultdict(list)
    for key in known_keys:
        component_members[union_find.find(key)].append(key)

    key_to_bucket: dict[str, str] = {}
    bucket_map: dict[str, list[str]] = {}
    for members in component_members.values():
        canonical_key = min(
            members,
            key=lambda item: canonical_sort_key(raw_bug_ids_by_key[item]),
        )
        bucket_id = canonical_key
        sorted_members = sorted(
            members,
            key=lambda item: canonical_sort_key(raw_bug_ids_by_key[item]),
        )
        bucket_map[bucket_id] = sorted_members
        for member in sorted_members:
            key_to_bucket[member] = bucket_id

    return key_to_bucket, bucket_map


def build_schema(
    input_path: Path,
    processed_output: Path,
    bucket_output: Path,
    summary_output: Path | None = None,
) -> dict[str, Any]:
    processed_output.parent.mkdir(parents=True, exist_ok=True)
    bucket_output.parent.mkdir(parents=True, exist_ok=True)

    union_find = UnionFind()
    known_keys: set[str] = set()
    raw_bug_ids_by_key: dict[str, str] = {}
    processed_count = 0

    with TemporaryDirectory(prefix="mozilla_schema_") as temp_dir:
        temp_processed = Path(temp_dir) / "processed_without_bucket.jsonl"

        with temp_processed.open("w", encoding="utf-8") as temp_handle:
            for raw_bug in iter_jsonl(input_path):
                normalized = normalize_bug(raw_bug)
                bug_id = normalized["bug_id"]
                if not bug_id:
                    continue

                current_key = record_key(bug_id)
                union_find.add(current_key)
                known_keys.add(current_key)
                raw_bug_ids_by_key[current_key] = bug_id

                for dup_target in extract_duplicate_targets(raw_bug.get("dup_id")):
                    dup_key = record_key(dup_target)
                    union_find.add(dup_key)
                    known_keys.add(dup_key)
                    raw_bug_ids_by_key.setdefault(dup_key, dup_target)
                    union_find.union(current_key, dup_key)

                temp_handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                processed_count += 1

        key_to_bucket, bucket_map = build_bucket_lookup(
            union_find=union_find,
            known_keys=known_keys,
            raw_bug_ids_by_key=raw_bug_ids_by_key,
        )

        with processed_output.open("w", encoding="utf-8") as handle:
            for record in iter_jsonl(temp_processed):
                key = record_key(record["bug_id"])
                record["bucket_id"] = key_to_bucket[key]
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    duplicate_bucket_count = sum(1 for members in bucket_map.values() if len(members) > 1)
    duplicate_record_count = sum(len(members) for members in bucket_map.values() if len(members) > 1)

    summary = {
        "input_path": str(input_path),
        "processed_output": str(processed_output),
        "bucket_output": str(bucket_output),
        "num_records": processed_count,
        "num_buckets": len(bucket_map),
        "num_duplicate_buckets": duplicate_bucket_count,
        "num_duplicate_records": duplicate_record_count,
        "schema_fields": [
            "bug_id",
            "project",
            "created_at",
            "summary",
            "description",
            "component",
            "priority",
            "severity",
            "bucket_id",
            "text_raw",
            "text_clean",
        ],
        "field_mapping": {
            "created_at": "creation_ts",
            "summary": "short_desc",
            "description": "description",
            "component": "component",
            "priority": "priority",
            "severity": "bug_severity",
            "bucket_id": "union-find over dup_id with canonical minimum bug_id",
        },
    }
    write_json(bucket_output, bucket_map)
    if summary_output is not None:
        write_json(summary_output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalized Mozilla bug schema.")
    parser.add_argument(
        "--input",
        default="data/raw/mozilla.json",
        help="Path to mozilla.json",
    )
    parser.add_argument(
        "--processed-output",
        default="data/processed/processed_bugs.jsonl",
        help="Output path for processed records",
    )
    parser.add_argument(
        "--bucket-output",
        default="data/bucket_map.json",
        help="Output path for bucket map",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/schema_summary.json",
        help="Output path for schema summary JSON",
    )
    args = parser.parse_args()

    summary = build_schema(
        input_path=Path(args.input),
        processed_output=Path(args.processed_output),
        bucket_output=Path(args.bucket_output),
        summary_output=Path(args.summary_output),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
