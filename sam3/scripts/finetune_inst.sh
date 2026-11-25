#!/usr/bin/env bash
#SBATCH -t 120:00:00
#SBATCH --mem=40G
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH -N 1
#SBATCH -w dgx-34
#SBATCH --requeue
#SBATCH --job-name=finetune_inst
#SBATCH --open-mode=append
#SBATCH --partition=vcompath
#SBATCH --account=vcompath
#SBATCH --output=./watch_folder/%x_%j.log
#SBATCH --error=./watch_folder/%x_%j.err

set -euo pipefail

# Slurm stages the script in a spool directory. Submit from sam3/ or sam3/scripts/.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR%/scripts}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${PROJECT_ROOT}"

HOSTNAME="$(hostname -s 2>/dev/null || hostname)"

CONFIG_PATH="${CONFIG_PATH:-${PWD}/sam3/train/configs/pathseg_train/sam3_pathseg_instance_ft.yaml}"
INSTANCE_ROOT="${INSTANCE_ROOT:-/path/to/your/data/instances}"
NUM_GPUS="${NUM_GPUS:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
TRAIN_RESIZE_MIN_SIZE="${TRAIN_RESIZE_MIN_SIZE:-640}"
EXPERIMENT_LOG_DIR="${EXPERIMENT_LOG_DIR:-${PWD}/exps/pathseg_instance_ft}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
VAL_LIMIT_PER_DATASET="${VAL_LIMIT_PER_DATASET:-null}"
EVAL_PATCHIFY="${EVAL_PATCHIFY:-true}"
EVAL_PATCH_SIZE="${EVAL_PATCH_SIZE:-256}"
EVAL_PATCH_STRIDE="${EVAL_PATCH_STRIDE:-192}"
NUM_QUERIES="${NUM_QUERIES:-300}"
MAX_ANN_PER_IMG="${MAX_ANN_PER_IMG:-1000000}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_LOG_DIR%/}"
EXPERIMENT_NAME="${EXPERIMENT_NAME##*/}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXPERIMENT_NAME}-${RUN_TS}}"

local_env() {
  [[ "${HOSTNAME}" == jhcpu5* ]]
}

setup_env() {
  export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
  export PYTHONPATH="${PWD}"

  if ! local_env; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    conda activate sam3
  fi
}

check_file() {
  local path="$1"
  local name="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] ${name} not found: ${path}" >&2
    exit 1
  fi
}

main() {
  setup_env
  check_file "${CONFIG_PATH}" "config"
  echo "[RUN] wandb_run_name=${WANDB_RUN_NAME}"
  local checkpoint_args=()
  if [[ -n "${INITIALIZATION_CHECKPOINT_PATH:-}" ]]; then
    check_file "${INITIALIZATION_CHECKPOINT_PATH}" "initialization checkpoint"
    checkpoint_args+=("paths.sam3_checkpoint=${INITIALIZATION_CHECKPOINT_PATH}")
  fi

  python sam3/train/train.py \
    -c "${CONFIG_PATH}" \
    --use-cluster 0 \
    --num-gpus "${NUM_GPUS}" \
    scratch.train_batch_size="${TRAIN_BATCH_SIZE}" \
    scratch.val_batch_size="${VAL_BATCH_SIZE}" \
    scratch.train_resize_min_size="${TRAIN_RESIZE_MIN_SIZE}" \
    scratch.num_queries="${NUM_QUERIES}" \
    scratch.max_ann_per_img="${MAX_ANN_PER_IMG}" \
    trainer.data.val.dataset.patchify_eval="${EVAL_PATCHIFY}" \
    scratch.eval_patch_size="${EVAL_PATCH_SIZE}" \
    scratch.eval_patch_stride="${EVAL_PATCH_STRIDE}" \
    scratch.gradient_accumulation_steps="${GRAD_ACCUM_STEPS}" \
    trainer.max_epochs="${MAX_EPOCHS}" \
    paths.pathseg_instance_root="${INSTANCE_ROOT}" \
    paths.experiment_log_dir="${EXPERIMENT_LOG_DIR}" \
    "${checkpoint_args[@]}" \
    trainer.logging.wandb_writer.name="${WANDB_RUN_NAME}" \
    trainer.data.val.dataset.limit_per_dataset="${VAL_LIMIT_PER_DATASET}"
}

main "$@"
