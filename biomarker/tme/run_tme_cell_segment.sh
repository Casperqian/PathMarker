#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/tme_common.sh"

usage() {
  cat >&2 <<USAGE
Usage:
  COHORT=BLCA LIMIT_SLIDES=10 RANDOM_SAMPLE=1 CUDA_ID=2,4,5 ${0}

Runs only the TME 40x cell stage. Existing tissue masks are required.
Multi-GPU runs shard slides across GPUs and write final cells.csv per slide.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${DEVICE}" == cuda* && "${#CUDA_ID_LIST[@]}" -gt 1 ]]; then
  mkdir -p "${SHARD_LOG_DIR}"
  shard_count="${#CUDA_ID_LIST[@]}"
  echo "[tme:cell] multi-GPU slide sharding: CUDA_IDS=${CUDA_IDS}, shards=${shard_count}"
  pids=()
  logs=()
  for shard_index in "${!CUDA_ID_LIST[@]}"; do
    gpu_id="${CUDA_ID_LIST[$shard_index]}"
    log_path="${SHARD_LOG_DIR}/${TME_LOG_PREFIX}_cell_slide_shard${shard_index}_of${shard_count}_gpu${gpu_id}.log"
    logs+=("${log_path}")
    echo "[tme:cell] shard $((shard_index + 1))/${shard_count} -> GPU ${gpu_id}, log=${log_path}"
    tme_run_python "${gpu_id}" \
      --stage cell \
      --slide-shard-count "${shard_count}" \
      --slide-shard-index "${shard_index}" \
      >"${log_path}" 2>&1 &
    pids+=("$!")
  done

  status=0
  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      echo "[tme:cell] shard $((idx + 1))/${shard_count} failed; see ${logs[$idx]}" >&2
      status=1
    fi
  done
  exit "${status}"
fi

gpu_id="${CUDA_ID_LIST[0]:-0}"
tme_run_python "${gpu_id}" --stage cell
