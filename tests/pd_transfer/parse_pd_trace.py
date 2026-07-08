#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


REQUEST_EVENTS = [
    "proxy_request_received",
    "proxy_prefill_request_start",
    "proxy_prefill_request_end",
    "proxy_decode_request_start",
    "proxy_decode_request_end",
    "pull_prefill_finished",
    "pull_kv_load_start",
    "pull_transfer_start",
    "pull_transfer_end",
    "pull_kv_recv_done",
    "decode_request_added",
    "decode_remote_kv_ready",
    "decode_first_token_out",
    "proxy_first_response_chunk",
    "layerwise_prefill_finished",
    "layerwise_transfer_start",
    "layerwise_transfer_end",
    "layerwise_kv_recv_done",
]

LAST_EVENTS = {
    "proxy_prefill_request_end",
    "proxy_decode_request_end",
    "pull_transfer_end",
    "pull_kv_recv_done",
    "decode_remote_kv_ready",
    "decode_first_token_out",
    "proxy_first_response_chunk",
    "layerwise_transfer_end",
    "layerwise_kv_recv_done",
}

REQUEST_ID_PREFIX_RE = re.compile(r"^cmpl-")
REQUEST_ID_SUFFIX_RE = re.compile(r"-\d+$")


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


def normalize_request_id(request_id: Any) -> str:
    normalized = str(request_id)
    normalized = REQUEST_ID_PREFIX_RE.sub("", normalized)
    normalized = REQUEST_ID_SUFFIX_RE.sub("", normalized)
    return normalized


def req_ids(record: dict[str, Any]) -> list[str]:
    if "request_id" in record:
        return [normalize_request_id(record["request_id"])]
    if "request_ids" in record:
        return [normalize_request_id(req_id) for req_id in record["request_ids"]]
    return []


def to_ms(delta_ns: int | float | None) -> float | str:
    if delta_ns is None:
        return ""
    return round(delta_ns / 1_000_000, 3)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return ordered[idx]


def find_cases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    starts: dict[str, dict[str, Any]] = {}
    cases: list[dict[str, Any]] = []

    for record in sorted(records, key=lambda r: int(r.get("ts_ns", 0))):
        event = record.get("event")
        case_id = record.get("case_id")
        if not case_id:
            continue
        if event == "bench_case_start":
            starts[str(case_id)] = record
        elif event == "bench_case_end" and str(case_id) in starts:
            start = starts.pop(str(case_id))
            cases.append({
                "case_id": str(case_id),
                "mode": record.get("mode", start.get("mode", "")),
                "input_len": int(record.get("input_len", start.get("input_len", 0))),
                "output_len": int(record.get("output_len", start.get("output_len", 0))),
                "concurrency": int(record.get("concurrency", start.get("concurrency", 0))),
                "num_prompts": int(record.get("num_prompts", start.get("num_prompts", 0))),
                "start_ns": int(start["ts_ns"]),
                "end_ns": int(record["ts_ns"]),
            })

    cases.sort(key=lambda case: case["start_ns"])
    return cases


def case_for_ts(cases: list[dict[str, Any]], ts_ns: int) -> dict[str, Any] | None:
    for case in cases:
        if case["start_ns"] <= ts_ns <= case["end_ns"]:
            return case
    return None


def build_request_rows(records: list[dict[str, Any]],
                       cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_req: dict[str, dict[str, Any]] = defaultdict(dict)

    for record in records:
        event = record.get("event")
        ts_ns = record.get("ts_ns")
        if event is None or ts_ns is None:
            continue
        event = str(event)
        ts_ns = int(ts_ns)
        case = case_for_ts(cases, ts_ns)

        for req_id in req_ids(record):
            row = by_req[req_id]
            row["request_id"] = req_id
            if case is not None:
                row.setdefault("case_id", case["case_id"])
                row.setdefault("mode", case["mode"])
                row.setdefault("input_len", case["input_len"])
                row.setdefault("output_len", case["output_len"])
                row.setdefault("concurrency", case["concurrency"])

            if event not in row:
                row[event] = ts_ns
            elif event in LAST_EVENTS:
                row[event] = max(row[event], ts_ns)
            else:
                row[event] = min(row[event], ts_ns)
            row[f"{event}_count"] = row.get(f"{event}_count", 0) + 1

            if event == "pull_transfer_end" and "elapsed_ms" in record:
                row["pull_transfer_elapsed_ms"] = float(record["elapsed_ms"])
                row["pull_transfer_bytes"] = int(record.get("total_bytes", 0))
            if event == "layerwise_transfer_end" and "elapsed_ms" in record:
                row["layerwise_transfer_elapsed_ms_sum"] = (
                    row.get("layerwise_transfer_elapsed_ms_sum", 0.0)
                    + float(record["elapsed_ms"]))
                row["layerwise_transfer_bytes_sum"] = (
                    row.get("layerwise_transfer_bytes_sum", 0)
                    + int(record.get("total_bytes", 0)))

    rows = []
    for row in by_req.values():
        row["ttft_trace_ms"] = to_ms(delta(row, "proxy_request_received",
                                           "proxy_first_response_chunk"))
        row["pull_transfer_wall_ms"] = to_ms(delta(row, "pull_transfer_start",
                                                  "pull_transfer_end"))
        row["pull_transfer_end_minus_start_ms"] = row["pull_transfer_wall_ms"]
        row["pull_prefill_to_transfer_start_ms"] = to_ms(
            delta(row, "pull_prefill_finished", "pull_transfer_start"))
        row["pull_prefill_to_proxy_decode_start_ms"] = to_ms(
            delta(row, "pull_prefill_finished", "proxy_decode_request_start"))
        row["pull_prefill_to_decode_request_added_ms"] = to_ms(
            delta(row, "pull_prefill_finished", "decode_request_added"))
        row["pull_visible_tail_ms"] = to_ms(delta(row, "pull_prefill_finished",
                                                 "pull_transfer_end"))
        row["decode_kv_ready_to_first_token_ms"] = to_ms(
            delta(row, "decode_remote_kv_ready", "decode_first_token_out"))
        row["layerwise_transfer_wall_ms"] = to_ms(
            delta(row, "layerwise_transfer_start", "layerwise_transfer_end"))
        row["layerwise_visible_tail_ms"] = to_ms(
            delta(row, "layerwise_prefill_finished", "layerwise_transfer_end"))
        rows.append(row)

    rows.sort(key=lambda row: (
        str(row.get("case_id", "")),
        str(row.get("request_id", "")),
    ))
    return rows


def delta(row: dict[str, Any], start: str, end: str) -> int | None:
    if start not in row or end not in row:
        return None
    return int(row[end]) - int(row[start])


def aggregate_pull(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "pull_transfer_elapsed_ms" not in row:
            continue
        key = (
            row.get("case_id", "unknown"),
            row.get("mode", "pull"),
            row.get("input_len", ""),
            row.get("output_len", ""),
            row.get("concurrency", ""),
        )
        grouped[key].append(row)

    result = []
    for key, items in sorted(grouped.items()):
        values = [float(item["pull_transfer_elapsed_ms"]) for item in items]
        wall_values = [
            float(item["pull_transfer_wall_ms"])
            for item in items
            if item.get("pull_transfer_wall_ms") != ""
        ]
        visible_values = [
            float(item["pull_visible_tail_ms"])
            for item in items
            if item.get("pull_visible_tail_ms") != ""
        ]
        prefill_to_transfer_values = [
            float(item["pull_prefill_to_transfer_start_ms"])
            for item in items
            if item.get("pull_prefill_to_transfer_start_ms") != ""
        ]
        prefill_to_proxy_decode_values = [
            float(item["pull_prefill_to_proxy_decode_start_ms"])
            for item in items
            if item.get("pull_prefill_to_proxy_decode_start_ms") != ""
        ]
        prefill_to_decode_added_values = [
            float(item["pull_prefill_to_decode_request_added_ms"])
            for item in items
            if item.get("pull_prefill_to_decode_request_added_ms") != ""
        ]
        result.append({
            "case_id": key[0],
            "mode": key[1],
            "input_len": key[2],
            "output_len": key[3],
            "concurrency": key[4],
            "request_count": len(values),
            "transfer_mean_ms": round(statistics.mean(values), 3),
            "transfer_median_ms": round(statistics.median(values), 3),
            "transfer_p90_ms": round(percentile(values, 90), 3),
            "transfer_p99_ms": round(percentile(values, 99), 3),
            "transfer_min_ms": round(min(values), 3),
            "transfer_max_ms": round(max(values), 3),
            "transfer_wall_mean_ms": round(statistics.mean(wall_values), 3)
            if wall_values else "",
            "prefill_to_transfer_start_mean_ms":
            round(statistics.mean(prefill_to_transfer_values), 3)
            if prefill_to_transfer_values else "",
            "prefill_to_proxy_decode_start_mean_ms":
            round(statistics.mean(prefill_to_proxy_decode_values), 3)
            if prefill_to_proxy_decode_values else "",
            "prefill_to_decode_request_added_mean_ms":
            round(statistics.mean(prefill_to_decode_added_values), 3)
            if prefill_to_decode_added_values else "",
            "visible_tail_mean_ms": round(statistics.mean(visible_values), 3)
            if visible_values else "",
        })
    return result


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, pull_rows: list[dict[str, Any]]) -> None:
    lines = ["# PD Transfer Trace Report", ""]
    if not pull_rows:
        lines.extend([
            "No pull transfer events were found.",
            "",
            "For existing runs without `bench_case_start/end` markers, rerun the",
            "benchmark with the updated script to get per-case grouping.",
        ])
    else:
        lines.extend([
            "## Pull Transfer Latency",
            "",
            "| case | input | output | concurrency | requests | pure transfer mean ms | prefill->decode request mean ms | prefill->transfer start mean ms | prefill->transfer end mean ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in pull_rows:
            lines.append(
                f"| {row['case_id']} | {row['input_len']} | {row['output_len']} | "
                f"{row['concurrency']} | {row['request_count']} | "
                f"{row['transfer_mean_ms']} | "
                f"{row['prefill_to_decode_request_added_mean_ms']} | "
                f"{row['prefill_to_transfer_start_mean_ms']} | "
                f"{row['visible_tail_mean_ms']} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()

    records = list(iter_trace_records(args.run_root))
    cases = find_cases(records)
    request_rows = build_request_rows(records, cases)
    pull_rows = aggregate_pull(request_rows)

    request_fields = [
        "case_id", "mode", "input_len", "output_len", "concurrency",
        "request_id", "ttft_trace_ms", "pull_transfer_elapsed_ms",
        "pull_transfer_wall_ms", "pull_transfer_end_minus_start_ms",
        "pull_prefill_to_proxy_decode_start_ms",
        "pull_prefill_to_decode_request_added_ms",
        "pull_prefill_to_transfer_start_ms", "pull_visible_tail_ms",
        "decode_kv_ready_to_first_token_ms", "pull_transfer_bytes",
        "layerwise_transfer_wall_ms", "layerwise_transfer_elapsed_ms_sum",
        "layerwise_visible_tail_ms", "layerwise_transfer_bytes_sum",
    ]
    pull_fields = [
        "case_id", "mode", "input_len", "output_len", "concurrency",
        "request_count", "transfer_mean_ms", "transfer_median_ms",
        "transfer_p90_ms", "transfer_p99_ms", "transfer_min_ms",
        "transfer_max_ms", "transfer_wall_mean_ms",
        "prefill_to_proxy_decode_start_mean_ms",
        "prefill_to_decode_request_added_mean_ms",
        "prefill_to_transfer_start_mean_ms", "visible_tail_mean_ms",
    ]

    request_csv = args.run_root / "pd_request_timeline_ms.csv"
    pull_csv = args.run_root / "pull_transfer_summary.csv"
    report_md = args.run_root / "pd_trace_report.md"

    write_csv(request_csv, request_rows, request_fields)
    write_csv(pull_csv, pull_rows, pull_fields)
    write_markdown(report_md, pull_rows)

    print(f"Wrote request timeline: {request_csv}")
    print(f"Wrote pull transfer summary: {pull_csv}")
    print(f"Wrote markdown report: {report_md}")
    if not cases:
        print("No bench_case_start/end markers found; per-case grouping is only "
              "available for runs produced by the updated run script.")


if __name__ == "__main__":
    main()
