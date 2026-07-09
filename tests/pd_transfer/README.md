# PD Transfer Benchmark

This directory contains lightweight scripts for measuring PD KV-transfer
timelines in a Linux NPU environment.

The default experiment targets a 1P1D, single-concurrency setup:

1. Start one prefiller server.
2. Start one decoder server.
3. Start the matching PD proxy.
4. Run random-input serving benchmarks for selected input lengths and injected
   transfer-delay levels.
5. Save vLLM benchmark JSON, process logs, PD trace JSONL files, and optional
   NPU utilization samples.

Tracing is disabled by default in the codebase. These scripts enable it by
setting:

```bash
VLLM_ASCEND_PD_TRACE_PATH=<case-dir>/<role>.trace.jsonl
VLLM_ASCEND_PD_TRACE_ROLE=<prefill|decode|proxy>
```

## Usage

```bash
cd /path/to/vllm-ascend-study
cp tests/pd_transfer/config.example.env tests/pd_transfer/config.env
# Edit MODEL and device/network settings.
bash tests/pd_transfer/run_pd_transfer_bench.sh tests/pd_transfer/config.env
```

After a run, parse trace files:

```bash
python tests/pd_transfer/parse_pd_trace.py results/pd_transfer/<run-dir> \
  --ttft-p99-slo-ms 1000 \
  --tpot-p99-slo-ms 50
```

The parser produces:

- `serve_summary.csv`: benchmark throughput, TTFT P99, TPOT P99, SLO status,
  and goodput for every case.
- `goodput_summary.csv`: best goodput under SLO for each mode/input/output
  workload.
- `sleep_sensitivity_summary.csv`: metric deltas relative to `sleep_ms=0`,
  useful for comparing whether transfer-delay impact is similar across input
  lengths.
- `npu_util_summary.csv`: optional P/D utilization balance from `npu-smi`
  samples.
- `pd_request_timeline_ms.csv`: per-request event timestamps and derived PD
  durations.
- `pull_transfer_summary.csv`: pull connector transfer latency summary.
- `pd_trace_report.md`: a compact markdown report.

By default the sample config uses `CONCURRENCIES="1"` and
`TRANSFER_SLEEP_MS_LIST="0 5 10 20 50 100"`. The pull connector reads
`VLLM_ASCEND_PD_TRANSFER_SLEEP_MS` and injects that delay into the measured
transfer path. The script restarts the P/D/proxy processes for each sleep value
so every case has a stable setting.

The default `OUTPUT_LENS="32"` keeps TPOT meaningful while avoiding a large
decode-heavy sweep. Use `OUTPUT_LENS="1"` only as a diagnostic case when you
want to isolate prefill plus PD handoff latency.
