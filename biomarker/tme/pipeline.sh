#!/usr/bin/env bash
#SBATCH -t 120:00:00
#SBATCH --mem=28G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH -N 1
#SBATCH --requeue
#SBATCH --job-name=tme
#SBATCH --open-mode=append
#SBATCH --partition=vcompath
#SBATCH --account=vcompath
#SBATCH --output=./watch_folder/%x_%j.log
#SBATCH --error=./watch_folder/%x_%j.err
#SBATCH --mail-user=qzengar@connect.ust.hk

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SAM3_PYTHON="${SAM3_PYTHON:-/mnt/hdd2/zq/anaconda3/envs/sam3/bin/python}"
TME_TISSUE_SCRIPT="${TME_TISSUE_SCRIPT:-${PROJECT_ROOT}/biomarker/tme/run_tme_tissue_segment.sh}"
TME_CELL_SCRIPT="${TME_CELL_SCRIPT:-${PROJECT_ROOT}/biomarker/tme/run_tme_cell_segment.sh}"
if [[ ! -x "${SAM3_PYTHON}" ]]; then
  echo "sam3 python not found: ${SAM3_PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${TME_TISSUE_SCRIPT}" || ! -f "${TME_CELL_SCRIPT}" ]]; then
  echo "TME stage script not found: ${TME_TISSUE_SCRIPT} or ${TME_CELL_SCRIPT}" >&2
  exit 1
fi

export PYTHON="${SAM3_PYTHON}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TCGA_ROOT="${TCGA_ROOT:-/jhcnas7/Pathology/original_data/TCGA}"
export SOURCE_MANIFEST="${SOURCE_MANIFEST:-}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/jhcnas3/MRI_jhcnas3/biomarker/tme_readout}"
export HISTOQC_ROOT="${HISTOQC_ROOT:-/jhcnas3/MRI_jhcnas3/biomarker}"
export LIMIT_SLIDES="${LIMIT_SLIDES:-0}"
export RANDOM_SAMPLE="${RANDOM_SAMPLE:-1}"
export RANDOM_SEED="${RANDOM_SEED:-42}"
export CUDA_ID="${CUDA_ID:-${SLURM_JOB_GPUS:-2,3}}"
export RESUME="${RESUME:-1}"
export PRIORITIZE_CASES="${PRIORITIZE_CASES:-1}"
export DYNAMIC_SLIDE_CLAIM="${DYNAMIC_SLIDE_CLAIM:-1}"
export SLIDE_MASK_DOWNSAMPLE="${SLIDE_MASK_DOWNSAMPLE:-16}"
export TISSUE_BATCH_SIZE="${TISSUE_BATCH_SIZE:-2}"
export CELL_TILE_SIZE="${CELL_TILE_SIZE:-576}"
export CELL_OVERLAP="${CELL_OVERLAP:-64}"
export CELL_BATCH_SIZE="${CELL_BATCH_SIZE:-8}"
export CELL_TISSUE_FILTER_THRESHOLD="${CELL_TISSUE_FILTER_THRESHOLD:-0.1}"
export CELL_PREFETCH_WORKERS="${CELL_PREFETCH_WORKERS:-2}"
export CELL_POSTPROCESS_WORKERS="${CELL_POSTPROCESS_WORKERS:-4}"
export CELL_POSTPROCESS_MAX_PENDING="${CELL_POSTPROCESS_MAX_PENDING:-4}"
export TME_RUN_TAG="${TME_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

# Edit this ordered, comma-separated cohort list before submitting the job.
COHORTS="NanFang_CRC"
IFS=',' read -r -a cohort_list <<< "${COHORTS}"
if [[ "${#cohort_list[@]}" -eq 0 ]]; then
  echo "No cohorts configured; set COHORTS in this script." >&2
  exit 2
fi

for raw_cohort in "${cohort_list[@]}"; do
  cohort="${raw_cohort//[[:space:]]/}"
  if [[ -z "${cohort}" ]]; then
    echo "Invalid empty cohort in COHORTS=${COHORTS}" >&2
    exit 2
  fi
  log_prefix="${TME_RUN_TAG}_${cohort}"

  echo "[tme] cohort=${cohort}: tissue"
  COHORT="${cohort}" TME_LOG_PREFIX="${log_prefix}" bash "${TME_TISSUE_SCRIPT}"

  echo "[tme] cohort=${cohort}: cell"
  COHORT="${cohort}" TME_LOG_PREFIX="${log_prefix}" bash "${TME_CELL_SCRIPT}"
done
