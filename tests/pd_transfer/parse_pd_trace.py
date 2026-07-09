#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import json
import os
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
BENCH_RESULT_RE = re.compile(
    r"(?P<mode>pull|layerwise)-input-(?P<input_len>\d+)-output-"
    r"(?P<output_len>\d+)-concurrency-(?P<concurrency>\d+)\.json$")


def iter_trace_records(root: Path):
    for path in root.rglob("*.trace.jsonl"):
        with path.open("r", encoding="utf-8-sig") as f:
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


def as_float(value: Any) -> float | str:
    if value is None or value == "":
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def nested_percentile(data: dict[str, Any], metric: str,
                      pct: int) -> float | str:
    candidates = [
        f"percentiles_{metric}_ms",
        f"{metric}_percentiles_ms",
        f"{metric}_percentiles",
        f"percentiles_{metric}",
    ]
    for key in candidates:
        value = data.get(key)
        if isinstance(value, dict):
            for pct_key in (str(pct), f"p{pct}", f"P{pct}", float(pct), pct):
                if pct_key in value:
                    return as_float(value[pct_key])
    return ""


def metric_value(data: dict[str, Any], *keys: str) -> float | str:
    for key in keys:
        value = as_float(data.get(key))
        if value != "":
            return value
    return ""


def percentile_metric(data: dict[str, Any], metric: str,
                      pct: int) -> float | str:
    direct = metric_value(data, f"p{pct}_{metric}_ms",
                          f"P{pct}_{metric}_ms", f"{metric}_p{pct}_ms",
                          f"{metric}_P{pct}_ms")
    if direct != "":
        return direct
    return nested_percentile(data, metric, pct)


def parse_benchmark_results(root: Path, ttft_p99_slo_ms: float,
                            tpot_p99_slo_ms: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in root.rglob("*.json"):
        match = BENCH_RESULT_RE.search(path.name)
        if match is None:
            continue
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)

        row: dict[str, Any] = {
            "case_id":
            path.stem,
            "mode":
            match.group("mode"),
            "input_len":
            int(match.group("input_len")),
            "output_len":
            int(match.group("output_len")),
            "concurrency":
            int(match.group("concurrency")),
            "result_file":
            str(path),
            "completed":
            data.get("completed", data.get("num_completed_requests", "")),
            "request_throughput":
            metric_value(data, "request_throughput", "requests_per_second"),
            "input_throughput":
            metric_value(data, "input_throughput"),
            "output_throughput":
            metric_value(data, "output_throughput"),
            "total_token_throughput":
            metric_value(data, "total_token_throughput",
                         "total_tokens_per_second", "tokens_per_second"),
            "mean_ttft_ms":
            metric_value(data, "mean_ttft_ms"),
            "median_ttft_ms":
            metric_value(data, "median_ttft_ms"),
            "p99_ttft_ms":
            percentile_metric(data, "ttft", 99),
            "mean_tpot_ms":
            metric_value(data, "mean_tpot_ms"),
            "median_tpot_ms":
            metric_value(data, "median_tpot_ms"),
            "p99_tpot_ms":
            percentile_metric(data, "tpot", 99),
            "mean_itl_ms":
            metric_value(data, "mean_itl_ms"),
            "median_itl_ms":
            metric_value(data, "median_itl_ms"),
            "p99_itl_ms":
            percentile_metric(data, "itl", 99),
            "mean_e2el_ms":
            metric_value(data, "mean_e2el_ms"),
            "median_e2el_ms":
            metric_value(data, "median_e2el_ms"),
            "p99_e2el_ms":
            percentile_metric(data, "e2el", 99),
        }

        ttft = row["p99_ttft_ms"]
        tpot = row["p99_tpot_ms"]
        ttft_ok = ttft == "" or ttft_p99_slo_ms <= 0 or ttft <= ttft_p99_slo_ms
        tpot_ok = tpot == "" or tpot_p99_slo_ms <= 0 or tpot <= tpot_p99_slo_ms
        row["ttft_p99_slo_ms"] = ttft_p99_slo_ms
        row["tpot_p99_slo_ms"] = tpot_p99_slo_ms
        row["slo_met"] = bool(ttft_ok and tpot_ok)
        row["goodput_request_throughput"] = (
            row["request_throughput"] if row["slo_met"] else 0.0)
        row["goodput_output_throughput"] = (
            row["output_throughput"] if row["slo_met"] else 0.0)
        rows.append(row)

    rows.sort(key=lambda row: (str(row["mode"]), int(row["input_len"]),
                               int(row["output_len"]),
                               int(row["concurrency"])))
    return rows


def summarize_goodput(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["mode"], row["input_len"], row["output_len"])].append(row)

    result = []
    for key, items in sorted(grouped.items()):
        best = max(items,
                   key=lambda item: float(item["goodput_output_throughput"]
                                          or 0.0))
        result.append({
            "mode": key[0],
            "input_len": key[1],
            "output_len": key[2],
            "best_concurrency": best["concurrency"],
            "slo_met": best["slo_met"],
            "goodput_output_throughput": best["goodput_output_throughput"],
            "goodput_request_throughput": best["goodput_request_throughput"],
            "p99_ttft_ms": best["p99_ttft_ms"],
            "p99_tpot_ms": best["p99_tpot_ms"],
        })
    return result


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


def aggregate_npu_util(records: list[dict[str, Any]],
                       cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for record in records:
        if record.get("event") != "npu_util_sample":
            continue
        util = as_float(record.get("util_percent"))
        if util == "":
            continue
        case = case_for_ts(cases, int(record.get("ts_ns", 0)))
        if case is None:
            continue
        role = str(record.get("role", ""))
        device = str(record.get("device", ""))
        grouped[(case["case_id"], case["mode"], case["input_len"],
                 case["output_len"], case["concurrency"], role,
                 device)].append(float(util))

    by_case_role: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for key, values in grouped.items():
        case_key = key[:5]
        role = key[5]
        by_case_role[(*case_key, role)].extend(values)

    case_keys = sorted({key[:5] for key in by_case_role})
    result = []
    for case_key in case_keys:
        prefill_values = by_case_role.get((*case_key, "prefill_util"), [])
        decode_values = by_case_role.get((*case_key, "decode_util"), [])
        prefill_mean = (round(statistics.mean(prefill_values), 3)
                        if prefill_values else "")
        decode_mean = (round(statistics.mean(decode_values), 3)
                       if decode_values else "")
        balance_gap = ""
        decode_to_prefill_ratio = ""
        if prefill_mean != "" and decode_mean != "":
            balance_gap = round(abs(float(prefill_mean) - float(decode_mean)),
                                3)
            decode_to_prefill_ratio = (
                round(float(decode_mean) / float(prefill_mean), 3)
                if float(prefill_mean) > 0 else "")
        result.append({
            "case_id": case_key[0],
            "mode": case_key[1],
            "input_len": case_key[2],
            "output_len": case_key[3],
            "concurrency": case_key[4],
            "prefill_util_mean": prefill_mean,
            "decode_util_mean": decode_mean,
            "util_balance_gap": balance_gap,
            "decode_to_prefill_util_ratio": decode_to_prefill_ratio,
            "prefill_sample_count": len(prefill_values),
            "decode_sample_count": len(decode_values),
        })
    return result


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, pull_rows: list[dict[str, Any]],
                   serve_rows: list[dict[str, Any]],
                   goodput_rows: list[dict[str, Any]],
                   util_rows: list[dict[str, Any]]) -> None:
    lines = ["# PD Transfer Trace Report", ""]
    if serve_rows:
        lines.extend([
            "## Serve-Level Metrics",
            "",
            "| case | req/s | output tok/s | TTFT P99 ms | TPOT P99 ms | SLO met | goodput output tok/s |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in serve_rows:
            lines.append(
                f"| {row['case_id']} | {row['request_throughput']} | "
                f"{row['output_throughput']} | {row['p99_ttft_ms']} | "
                f"{row['p99_tpot_ms']} | {row['slo_met']} | "
                f"{row['goodput_output_throughput']} |")
        lines.append("")

    if goodput_rows:
        lines.extend([
            "## Best Goodput Under SLO",
            "",
            "| mode | input | output | best concurrency | goodput output tok/s | TTFT P99 ms | TPOT P99 ms |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in goodput_rows:
            lines.append(
                f"| {row['mode']} | {row['input_len']} | "
                f"{row['output_len']} | {row['best_concurrency']} | "
                f"{row['goodput_output_throughput']} | "
                f"{row['p99_ttft_ms']} | {row['p99_tpot_ms']} |")
        lines.append("")

    if util_rows:
        lines.extend([
            "## P/D Utilization Balance",
            "",
            "| case | P util mean | D util mean | abs gap | D/P ratio |",
            "|---|---:|---:|---:|---:|",
        ])
        for row in util_rows:
            lines.append(
                f"| {row['case_id']} | {row['prefill_util_mean']} | "
                f"{row['decode_util_mean']} | {row['util_balance_gap']} | "
                f"{row['decode_to_prefill_util_ratio']} |")
        lines.append("")

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
    parser.add_argument("--ttft-p99-slo-ms",
                        type=float,
                        default=float(os.getenv("TTFT_P99_SLO_MS", "0")))
    parser.add_argument("--tpot-p99-slo-ms",
                        type=float,
                        default=float(os.getenv("TPOT_P99_SLO_MS", "0")))
    args = parser.parse_args()

    records = list(iter_trace_records(args.run_root))
    cases = find_cases(records)
    request_rows = build_request_rows(records, cases)
    pull_rows = aggregate_pull(request_rows)
    serve_rows = parse_benchmark_results(args.run_root, args.ttft_p99_slo_ms,
                                         args.tpot_p99_slo_ms)
    goodput_rows = summarize_goodput(serve_rows)
    util_rows = aggregate_npu_util(records, cases)

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
    serve_fields = [
        "case_id", "mode", "input_len", "output_len", "concurrency",
        "completed", "request_throughput", "input_throughput",
        "output_throughput", "total_token_throughput", "mean_ttft_ms",
        "median_ttft_ms", "p99_ttft_ms", "mean_tpot_ms",
        "median_tpot_ms", "p99_tpot_ms", "mean_itl_ms", "median_itl_ms",
        "p99_itl_ms", "mean_e2el_ms", "median_e2el_ms", "p99_e2el_ms",
        "ttft_p99_slo_ms", "tpot_p99_slo_ms", "slo_met",
        "goodput_request_throughput", "goodput_output_throughput",
        "result_file",
    ]
    goodput_fields = [
        "mode", "input_len", "output_len", "best_concurrency", "slo_met",
        "goodput_output_throughput", "goodput_request_throughput",
        "p99_ttft_ms", "p99_tpot_ms",
    ]
    util_fields = [
        "case_id", "mode", "input_len", "output_len", "concurrency",
        "prefill_util_mean", "decode_util_mean", "util_balance_gap",
        "decode_to_prefill_util_ratio", "prefill_sample_count",
        "decode_sample_count",
    ]

    request_csv = args.run_root / "pd_request_timeline_ms.csv"
    pull_csv = args.run_root / "pull_transfer_summary.csv"
    serve_csv = args.run_root / "serve_summary.csv"
    goodput_csv = args.run_root / "goodput_summary.csv"
    util_csv = args.run_root / "npu_util_summary.csv"
    report_md = args.run_root / "pd_trace_report.md"

    write_csv(request_csv, request_rows, request_fields)
    write_csv(pull_csv, pull_rows, pull_fields)
    write_csv(serve_csv, serve_rows, serve_fields)
    write_csv(goodput_csv, goodput_rows, goodput_fields)
    write_csv(util_csv, util_rows, util_fields)
    write_markdown(report_md, pull_rows, serve_rows, goodput_rows, util_rows)

    print(f"Wrote request timeline: {request_csv}")
    print(f"Wrote pull transfer summary: {pull_csv}")
    print(f"Wrote serve summary: {serve_csv}")
    print(f"Wrote goodput summary: {goodput_csv}")
    print(f"Wrote NPU utilization summary: {util_csv}")
    print(f"Wrote markdown report: {report_md}")
    if not cases:
        print("No bench_case_start/end markers found; per-case grouping is only "
              "available for runs produced by the updated run script.")


if __name__ == "__main__":
    main()
