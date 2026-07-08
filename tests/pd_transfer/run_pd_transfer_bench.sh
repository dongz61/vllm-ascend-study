#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-"tests/pd_transfer/config.env"}
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${CONFIG_PATH}"

RUN_ID=$(date "+%Y%m%d-%H%M%S")
RUN_ROOT="${RESULT_ROOT}/${RUN_ID}"
mkdir -p "${RUN_ROOT}"

PIDS=()

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
  sleep 2
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill -9 "${pid}" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup EXIT

wait_for_server() {
  local port=$1
  local name=$2
  local waited=0
  local timeout_sec=1200

  until curl -fsS "http://${HOST}:${port}/v1/models" >/dev/null 2>&1; do
    sleep 1
    waited=$((waited + 1))
    if (( waited >= timeout_sec )); then
      echo "Timeout waiting for ${name} on port ${port}" >&2
      return 1
    fi
  done
}

wait_for_proxy() {
  local waited=0
  local timeout_sec=300

  until curl -fsS "http://${HOST}:${PROXY_PORT}/healthcheck" >/dev/null 2>&1; do
    sleep 1
    waited=$((waited + 1))
    if (( waited >= timeout_sec )); then
      echo "Timeout waiting for proxy on port ${PROXY_PORT}" >&2
      return 1
    fi
  done
}

kv_config() {
  local connector=$1
  local module_path=$2
  local role=$3
  local kv_port=$4
  local engine_id=$5

  cat <<EOF
{"kv_connector":"${connector}","kv_role":"${role}","kv_port":"${kv_port}","engine_id":"${engine_id}","kv_connector_module_path":"${module_path}","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":${TP_SIZE}},"decode":{"dp_size":1,"tp_size":${TP_SIZE}}}}
EOF
}

start_vllm_server() {
  local mode=$1
  local role=$2
  local port=$3
  local kv_port=$4
  local engine_id=$5
  local devices=$6
  local trace_file=$7
  local log_file=$8
  local connector module_path extra_args

  if [[ "${mode}" == "pull" ]]; then
    connector="MooncakeConnectorV1"
    module_path="vllm_ascend.distributed.mooncake_connector"
  elif [[ "${mode}" == "layerwise" ]]; then
    connector="MooncakeLayerwiseConnector"
    module_path="vllm_ascend.distributed.mooncake_layerwise_connector"
  else
    echo "Unknown mode: ${mode}" >&2
    exit 1
  fi

  if [[ "${role}" == "prefill" ]]; then
    kv_role="kv_producer"
    extra_args="${PREFILL_EXTRA_ARGS}"
  else
    kv_role="kv_consumer"
    extra_args="${DECODE_EXTRA_ARGS}"
  fi

  local kv_json
  kv_json=$(kv_config "${connector}" "${module_path}" "${kv_role}" "${kv_port}" "${engine_id}")

  echo "Starting ${mode}/${role} on port ${port}, log=${log_file}"
  (
    export ASCEND_RT_VISIBLE_DEVICES="${devices}"
    export VLLM_ASCEND_PD_TRACE_PATH="${trace_file}"
    export VLLM_ASCEND_PD_TRACE_ROLE="${role}"
    export HCCL_EXEC_TIMEOUT HCCL_CONNECT_TIMEOUT TASK_QUEUE_ENABLE VLLM_USE_V1
    export TRANSFORMERS_OFFLINE HF_HUB_OFFLINE
    vllm serve "${MODEL}" \
      --host "0.0.0.0" \
      --port "${port}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --tensor-parallel-size "${TP_SIZE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      ${COMMON_VLLM_ARGS} \
      ${extra_args} \
      --kv-transfer-config "${kv_json}"
  ) >"${log_file}" 2>&1 &
  PIDS+=("$!")
}

start_proxy() {
  local mode=$1
  local trace_file=$2
  local log_file=$3
  local proxy_script

  if [[ "${mode}" == "pull" ]]; then
    proxy_script="examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py"
  else
    proxy_script="examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py"
  fi

  echo "Starting ${mode}/proxy on port ${PROXY_PORT}, log=${log_file}"
  (
    export VLLM_ASCEND_PD_TRACE_PATH="${trace_file}"
    export VLLM_ASCEND_PD_TRACE_ROLE="proxy"
    python "${proxy_script}" \
      --host "${HOST}" \
      --port "${PROXY_PORT}" \
      --prefiller-hosts "${HOST}" \
      --prefiller-ports "${PREFILL_PORT}" \
      --decoder-hosts "${HOST}" \
      --decoder-ports "${DECODE_PORT}"
  ) >"${log_file}" 2>&1 &
  PIDS+=("$!")
}

run_benchmark_case() {
  local mode=$1
  local case_dir=$2
  local input_len=$3
  local output_len=$4
  local concurrency=$5
  local num_prompts=$((concurrency * NUM_FOLDS))
  local result_name="${mode}-input-${input_len}-output-${output_len}-concurrency-${concurrency}.json"
  local bench_log="${case_dir}/bench-input-${input_len}-output-${output_len}-concurrency-${concurrency}.log"

  echo "Benchmark ${mode}: input=${input_len}, output=${output_len}, concurrency=${concurrency}, prompts=${num_prompts}"
  vllm bench serve \
    --backend vllm \
    --model "${SERVED_MODEL_NAME}" \
    --tokenizer "${BENCH_TOKENIZER:-${MODEL}}" \
    --host "${HOST}" \
    --port "${PROXY_PORT}" \
    --dataset-name random \
    --random-input-len "${input_len}" \
    --random-output-len "${output_len}" \
    --random-prefix-len "${RANDOM_PREFIX_LEN}" \
    --num-prompts "${num_prompts}" \
    --max-concurrency "${concurrency}" \
    --save-result \
    --result-dir "${case_dir}" \
    --result-filename "${result_name}" \
    ${BENCH_EXTRA_ARGS} 2>&1 | tee "${bench_log}"
}

run_mode() {
  local mode=$1
  local mode_dir="${RUN_ROOT}/${mode}"
  mkdir -p "${mode_dir}"

  start_vllm_server "${mode}" "prefill" "${PREFILL_PORT}" "${PREFILL_KV_PORT}" "0" \
    "${PREFILL_DEVICES}" "${mode_dir}/prefill.trace.jsonl" "${mode_dir}/prefill.log"
  start_vllm_server "${mode}" "decode" "${DECODE_PORT}" "${DECODE_KV_PORT}" "1" \
    "${DECODE_DEVICES}" "${mode_dir}/decode.trace.jsonl" "${mode_dir}/decode.log"
  wait_for_server "${PREFILL_PORT}" "${mode}/prefill"
  wait_for_server "${DECODE_PORT}" "${mode}/decode"

  start_proxy "${mode}" "${mode_dir}/proxy.trace.jsonl" "${mode_dir}/proxy.log"
  wait_for_proxy

  for input_len in ${INPUT_LENS}; do
    for output_len in ${OUTPUT_LENS}; do
      for concurrency in ${CONCURRENCIES}; do
        run_benchmark_case "${mode}" "${mode_dir}" "${input_len}" "${output_len}" "${concurrency}"
      done
    done
  done

  cleanup
  PIDS=()
}

echo "Results will be saved to ${RUN_ROOT}"
for mode in ${MODES}; do
  run_mode "${mode}"
done

echo "Done. Parse traces with:"
echo "  python tests/pd_transfer/parse_pd_trace.py ${RUN_ROOT}"
