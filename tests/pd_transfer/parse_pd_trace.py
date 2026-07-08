#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


EVENT_COLUMNS = [
    "proxy_request_received",
    "proxy_prefill_request_start",
    "proxy_prefill_request_end",
    "proxy_decode_request_start",
    "proxy_decode_request_end",
    "pull_prefill_finished",
    "layerwise_prefill_finished",
    "pull_kv_load_start",
    "layerwise_kv_load_start",
    "pull_transfer_start",
    "pull_transfer_end",
    "layerwise_transfer_start",
    "layerwise_transfer_end",
    "pull_kv_recv_done",
    "layerwise_kv_recv_done",
    "decode_remote_kv_ready",
    "decode_first_token_out",
    "proxy_first_response_chunk",
]

LAST_EVENT_COLUMNS = {
    "proxy_prefill_request_end",
    "proxy_decode_request_end",
    "pull_transfer_end",
    "layerwise_transfer_end",
    "pull_kv_recv_done",
    "layerwise_kv_recv_done",
    "decode_remote_kv_ready",
    "decode_first_token_out",
    "proxy_first_response_chunk",
}

def iter_trace_records(root: Path):
    for path in root.rglob("*.trace.jsonl"):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record["_trace_file"] = str(path)
                yield record


def record_req_ids(record: dict[str, Any]) -> list[str]:
    if "request_id" in record:
        return [str(record["request_id"])]
    if "request_ids" in record:
        return [str(req_id) for req_id in record["request_ids"]]
    return []


def ms_delta(row: dict[str, Any], start_event: str, end_event: str) -> float | str:
    start = row.get(start_event)
    end = row.get(end_event)
    if start is None or end is None:
        return ""
    return (end - start) / 1_000_000


def summarize(root: Path) -> list[dict[str, Any]]:
    by_req: dict[str, dict[str, Any]] = defaultdict(dict)

    for record in iter_trace_records(root):
        event = record.get("event")
        ts_ns = record.get("ts_ns")
        if event is None or ts_ns is None:
            continue
        for req_id in record_req_ids(record):
            row = by_req[req_id]
            row["request_id"] = req_id
            key = str(event)
            if key not in row:
                row[key] = int(ts_ns)
            elif key in LAST_EVENT_COLUMNS:
                row[key] = max(row[key], int(ts_ns))
            else:
                row[key] = min(row[key], int(ts_ns))
            row[f"{key}_count"] = row.get(f"{key}_count", 0) + 1

    rows = []
    for row in by_req.values():
        row["proxy_to_first_chunk_ms"] = ms_delta(
            row, "proxy_request_received", "proxy_first_response_chunk")
        row["pull_visible_transfer_ms"] = ms_delta(
            row, "pull_prefill_finished", "pull_transfer_end")
        row["layerwise_visible_transfer_tail_ms"] = ms_delta(
            row, "layerwise_prefill_finished", "layerwise_transfer_end")
        row["kv_ready_to_first_token_ms"] = ms_delta(
            row, "decode_remote_kv_ready", "decode_first_token_out")
        rows.append(row)

    rows.sort(key=lambda item: item["request_id"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = summarize(args.run_root)
    output = args.output or args.run_root / "pd_trace_summary.csv"

    fieldnames = ["request_id"]
    fieldnames.extend(EVENT_COLUMNS)
    fieldnames.extend(f"{event}_count" for event in EVENT_COLUMNS)
    fieldnames.extend([
        "proxy_to_first_chunk_ms",
        "pull_visible_transfer_ms",
        "layerwise_visible_transfer_tail_ms",
        "kv_ready_to_first_token_ms",
    ])

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} request rows to {output}")


if __name__ == "__main__":
    main()
