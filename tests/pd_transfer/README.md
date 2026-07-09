# PD Transfer Benchmark

This directory contains lightweight scripts for measuring PD KV-transfer
timelines in a Linux NPU environment.

The first version targets a 1P1D setup:

1. Start one prefiller server.
2. Start one decoder server.
3. Start the matching PD proxy.
4. Run random-input serving benchmarks for selected input/output lengths and
   concurrency levels.
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
- `npu_util_summary.csv`: optional P/D utilization balance from `npu-smi`
  samples.
- `pd_request_timeline_ms.csv`: per-request event timestamps and derived PD
  durations.
- `pull_transfer_summary.csv`: pull connector transfer latency summary.
- `pd_trace_report.md`: a compact markdown report.

By default the sample config uses `OUTPUT_LENS="32 128"` because the main
evaluation is serve-level quality. Use `OUTPUT_LENS="1"` only as a diagnostic
case when you want to isolate prefill plus PD handoff latency.
