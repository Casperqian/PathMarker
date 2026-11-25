#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/tme_common.sh"

usage() {
  cat >&2 <<USAGE
Usage:
  COHORT=BLCA LIMIT_SLIDES=10 RANDOM_SAMPLE=1 CUDA_ID=2,4,5 ${0}

Runs only the readout stage. It merges cell shards when multiple CUDA IDs are supplied,
then writes TME_Readout.csv.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

shard_count="${#CUDA_ID_LIST[@]}"
"${PYTHON}" "${tme_base_args[@]}" --stage readout --cell-shard-count "${shard_count}"
