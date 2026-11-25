#!/usr/bin/env bash
#SBATCH -t 120:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH -N 1
#SBATCH -w dgx-34
#SBATCH --requeue
#SBATCH --job-name=finetune
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

CONFIG_PATH="${PWD}/sam3/train/configs/pathseg_train/sam3_pathseg_labeled_ft.yaml"
NUM_GPUS="8"
TRAIN_BATCH_SIZE="64"
GRAD_ACCUM_STEPS="2"
EXPERIMENT_LOG_DIR="${PROJECT_ROOT}/exps/siglip_labeled_ft"
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

main() {
  setup_env

  echo "[RUN] wandb_run_name=${WANDB_RUN_NAME}"
  python sam3/train/train.py \
    -c "${CONFIG_PATH}" \
    --use-cluster 0 \
    --num-gpus "${NUM_GPUS}" \
    scratch.train_batch_size="${TRAIN_BATCH_SIZE}" \
    scratch.gradient_accumulation_steps="${GRAD_ACCUM_STEPS}" \
    paths.experiment_log_dir="${EXPERIMENT_LOG_DIR}" \
    trainer.logging.wandb_writer.name="${WANDB_RUN_NAME}"
}

main "$@"
