#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -z "${PYTHON:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON="python"
  else
    PYTHON="python3"
  fi
fi

COHORT="${COHORT:-BLCA}"
SLIDE_ID="${SLIDE_ID:-}"
if [[ -z "${OUTPUT_ROOT:-}" ]]; then
  if [[ -d "/project/vcompath/zq/PathSeg/biomarker" ]]; then
    OUTPUT_ROOT="/project/vcompath/zq/PathSeg/biomarker"
  else
    OUTPUT_ROOT="/jhcnas3/MRI_jhcnas3/biomarker/tme_readout"
  fi
fi
FORMAT_CSV="${FORMAT_CSV:-${PROJECT_ROOT}/biomarker/tme/TME_Readout_Format.csv}"
RADII_UM="${RADII_UM:-40,80}"
MIN_CELL_AREA_PX="${MIN_CELL_AREA_PX:-50}"
CELL_TARGET_MPP="${CELL_TARGET_MPP:-0.25}"
READOUT_WORKERS="${READOUT_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
SPATIAL_WORKING_MB="${SPATIAL_WORKING_MB:-256}"
LIMIT_SLIDES="${LIMIT_SLIDES:-0}"
RESUME="${RESUME:-1}"
MATRIX_CSV="${MATRIX_CSV:-}"
NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
if [[ -n "${SHARD_INDEX:-}" ]]; then
  SHARD_INDEX="${SHARD_INDEX}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  SHARD_INDEX="$((SLURM_ARRAY_TASK_ID - ${SLURM_ARRAY_TASK_MIN:-0}))"
else
  SHARD_INDEX=0
fi
WRITE_MATRIX="${WRITE_MATRIX:-}"
if [[ -z "${WRITE_MATRIX}" ]]; then
  if (( NUM_SHARDS > 1 )); then
    WRITE_MATRIX=0
  else
    WRITE_MATRIX=1
  fi
fi

usage() {
  cat >&2 <<USAGE
Usage:
  COHORT=BRCA OUTPUT_ROOT=/project/vcompath/zq/PathSeg/biomarker READOUT_WORKERS=16 SPATIAL_WORKING_MB=256 MIN_CELL_AREA_PX=50 ${0}

Two disjoint jobs (or use the same variables with a Slurm array 0-1):
  NUM_SHARDS=2 SHARD_INDEX=0 COHORT=ARGO_CRC sbatch ${0}
  NUM_SHARDS=2 SHARD_INDEX=1 COHORT=ARGO_CRC sbatch ${0}

Sharded jobs skip the shared matrix by default. After all shards finish, run once
with NUM_SHARDS=1 RESUME=1 WRITE_MATRIX=1 to rebuild TME_Readout_Matrix.csv.

CPU-only readout stage. It scans:
  OUTPUT_ROOT/<COHORT>/<SLIDE_ID>/tissue_mask.png
  OUTPUT_ROOT/<COHORT>/<SLIDE_ID>/tissue_mask.json
  OUTPUT_ROOT/<COHORT>/<SLIDE_ID>/cells.csv

and writes:
  OUTPUT_ROOT/<COHORT>/<SLIDE_ID>/TME_Readout.csv
  OUTPUT_ROOT/<COHORT>/TME_Readout_Matrix.csv
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

args=(
  "${PROJECT_ROOT}/biomarker/tme/compute_tme_readouts_cpu.py"
  --cohort "${COHORT}"
  --output-root "${OUTPUT_ROOT}"
  --format-csv "${FORMAT_CSV}"
  --radii-um "${RADII_UM}"
  --min-cell-area-px "${MIN_CELL_AREA_PX}"
  --cell-target-mpp "${CELL_TARGET_MPP}"
  --spatial-working-mb "${SPATIAL_WORKING_MB}"
  --workers "${READOUT_WORKERS}"
  --limit-slides "${LIMIT_SLIDES}"
  --num-shards "${NUM_SHARDS}"
  --shard-index "${SHARD_INDEX}"
)

if [[ -n "${SLIDE_ID}" ]]; then
  args+=(--slide-id "${SLIDE_ID}")
fi
if [[ -n "${MATRIX_CSV}" ]]; then
  args+=(--matrix-csv "${MATRIX_CSV}")
fi
if [[ "${RESUME}" == "0" ]]; then
  args+=(--no-resume)
else
  args+=(--resume)
fi
if [[ "${WRITE_MATRIX}" == "0" ]]; then
  args+=(--skip-matrix)
fi

PYTHONUNBUFFERED=1 "${PYTHON}" "${args[@]}"
