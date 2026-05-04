from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


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


def load_sorted_records(records_path: Path) -> tuple[list[dict[str, Any]], bool]:
    records: list[dict[str, Any]] = []
    previous_timestamp: datetime | None = None
    was_sorted = True

    for record in iter_jsonl(records_path):
        current_timestamp = parse_timestamp(record["created_at"])
        if previous_timestamp and current_timestamp < previous_timestamp:
            was_sorted = False
        previous_timestamp = current_timestamp
        records.append(record)

    if not was_sorted:
        records.sort(key=lambda record: parse_timestamp(record["created_at"]))
    return records, was_sorted


def split_by_time(
    records_path: Path,
    train_output: Path,
    test_output: Path,
    summary_output: Path | None = None,
    train_ratio: float = 0.8,
) -> dict[str, Any]:
    records, was_sorted = load_sorted_records(records_path)
    bucket_sizes: Counter[str] = Counter()
    for record in records:
        bucket_sizes[record["bucket_id"]] += 1
    total_count = len(records)

    split_index = max(1, int(total_count * train_ratio))
    train_output.parent.mkdir(parents=True, exist_ok=True)
    test_output.parent.mkdir(parents=True, exist_ok=True)

    seen_train_buckets: Counter[str] = Counter()
    train_count = 0
    test_count = 0
    dropped_test_count = 0

    with train_output.open("w", encoding="utf-8") as train_handle, test_output.open(
        "w", encoding="utf-8"
    ) as test_handle:
        for position, record in enumerate(records, start=1):
            bucket_id = record["bucket_id"]
            if position <= split_index:
                train_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                seen_train_buckets[bucket_id] += 1
                train_count += 1
                continue

            if bucket_sizes[bucket_id] <= 1 or seen_train_buckets[bucket_id] == 0:
                dropped_test_count += 1
                continue

            test_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            test_count += 1

    summary = {
        "records_path": str(records_path),
        "train_output": str(train_output),
        "test_output": str(test_output),
        "train_ratio": train_ratio,
        "total_records": total_count,
        "split_index": split_index,
        "train_count": train_count,
        "test_count": test_count,
        "dropped_test_count": dropped_test_count,
        "input_was_sorted": was_sorted,
    }
    if summary_output is not None:
        write_json(summary_output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Split processed Mozilla bugs by time.")
    parser.add_argument(
        "--records",
        default="data/processed/processed_bugs.jsonl",
        help="Path to processed records JSONL",
    )
    parser.add_argument(
        "--train-output",
        default="data/train.jsonl",
        help="Output path for train JSONL",
    )
    parser.add_argument(
        "--test-output",
        default="data/test.jsonl",
        help="Output path for test JSONL",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/split_summary.json",
        help="Output path for split summary JSON",
    )
    args = parser.parse_args()

    summary = split_by_time(
        records_path=Path(args.records),
        train_output=Path(args.train_output),
        test_output=Path(args.test_output),
        summary_output=Path(args.summary_output),
        train_ratio=args.train_ratio,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
