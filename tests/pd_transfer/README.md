# PD Transfer Benchmark

This directory contains lightweight scripts for measuring PD KV-transfer
timelines in a Linux NPU environment.

The first version targets a 1P1D setup:

1. Start one prefiller server.
2. Start one decoder server.
3. Start the matching PD proxy.
4. Run random-input serving benchmarks for selected input/output lengths and
   concurrency levels.
5. Save vLLM benchmark JSON, process logs, and PD trace JSONL files.

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
python tests/pd_transfer/parse_pd_trace.py results/pd_transfer/<run-dir>
```

The parser produces `pd_trace_summary.csv` with per-request event timestamps and
basic derived durations.
