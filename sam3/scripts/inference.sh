#!/usr/bin/env bash
#SBATCH -t 120:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=28
#SBATCH -N 1
#SBATCH --requeue
#SBATCH --job-name=infer
#SBATCH --open-mode=append
#SBATCH --partition=vcompath
#SBATCH --account=vcompath

set -euo pipefail

cd ../

HOSTNAME="$(hostname -s 2>/dev/null || hostname)"

K="${K:-3}"
DATASETS_STR="${DATASETS_STR:-DatasetA}"
CONFIGS_STR="${CONFIGS_STR:-siglip}"
PROMPT_MODES_STR="${PROMPT_MODES_STR:-none}"
POINT_COUNTS_STR="${POINT_COUNTS_STR:-1 2 5 10}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
RETURN_GEOMETRY="${RETURN_GEOMETRY:-true}"
TEXT_PROMPT_OVERRIDE="${TEXT_PROMPT_OVERRIDE:-}"
WANDB_PROJECT="${WANDB_PROJECT:-pathseg_evals}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-infer}"
DISABLE_WANDB="${DISABLE_WANDB:-0}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-/home/qzengar/PathSegmentorV2/sam3/exps/pathseg_evals/logs}"
AUTO_SELECT_GPU="${AUTO_SELECT_GPU:-1}"
GPU_LIST="${GPU_LIST:-1,3,4}"

local_env() {
  [[ "${HOSTNAME}" == jhcpu5* ]]
}

setup_env() {
  export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
  export PYTHONPATH="${PWD}"

  if local_env; then
    :
  else
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    conda activate sam3
  fi
}

pick_best_gpu() {
  [[ "${AUTO_SELECT_GPU}" == "1" ]] || return 1
  command -v nvidia-smi >/dev/null 2>&1 || return 1

  local best_gpu=""
  local best_free=-1
  local candidates="${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-}}"
  local query_output

  query_output="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null)" || return 1

  while IFS=',' read -r gpu free_mem; do
    gpu="${gpu//[[:space:]]/}"
    free_mem="${free_mem//[[:space:]]/}"
    [[ "${gpu}" =~ ^[0-9]+$ && "${free_mem}" =~ ^[0-9]+$ ]] || continue

    if [[ -n "${candidates}" && ",${candidates// /}," != *",${gpu},"* ]]; then
      continue
    fi

    if (( free_mem > best_free )); then
      best_free="${free_mem}"
      best_gpu="${gpu}"
    fi
  done <<< "${query_output}"

  [[ -n "${best_gpu}" ]] || return 1
  echo "${best_gpu}"
}

init_maps() {
  declare -gA EXP_BASE_DIRS
  declare -gA CHECKPOINT_PATHS
  declare -gA CFG_PATHS

  EXP_BASE_DIRS["siglip"]="${SIGLIP_EXP_BASE_DIR:-/home/qzengar/PathSegmentorV2/sam3/exps/pathseg_evals/siglip}"
  CHECKPOINT_PATHS["siglip"]="/jhcnas5/zq/code/ckpt/finetune/ours/exps/siglip_labeled_ft/checkpoints/best.pt"
  CFG_PATHS["siglip"]="${PWD}/sam3/train/configs/pathseg_evals/eval_base.yaml"

  EXP_BASE_DIRS["siglip_wo_pretrain"]="/home/qzengar/PathSegmentorV2/sam3/exps/pathseg_evals/siglip_wo_pretrain"
  CHECKPOINT_PATHS["siglip_wo_pretrain"]="/jhcnas5/zq/code/ckpt/finetune/ours/exps/siglip_labeled_ft_wo_pretrain/checkpoints/best.pt"
  CFG_PATHS["siglip_wo_pretrain"]="${PWD}/sam3/train/configs/pathseg_evals/eval_base.yaml"

  EXP_BASE_DIRS["siglip_wo_hierarchy"]="/home/qzengar/PathSegmentorV2/sam3/exps/pathseg_evals/siglip_wo_hierarchy"
  CHECKPOINT_PATHS["siglip_wo_hierarchy"]="/jhcnas5/zq/code/ckpt/finetune/ours/exps/siglip_labeled_ft_wo_hierarchy/checkpoints/best.pt"
  CFG_PATHS["siglip_wo_hierarchy"]="${PWD}/sam3/train/configs/pathseg_evals/eval_base.yaml"

  EXP_BASE_DIRS["siglip_partial"]="/home/qzengar/PathSegmentorV2/sam3/exps/pathseg_evals/siglip_partial"
  CHECKPOINT_PATHS["siglip_partial"]="/jhcnas5/zq/code/ckpt/finetune/ours/exps/siglip_labeled_ft_partial/checkpoints/best.pt"
  CFG_PATHS["siglip_partial"]="${PWD}/sam3/train/configs/pathseg_evals/eval_base.yaml"

  EXP_BASE_DIRS["test"]="/home/qzengar/PathSegmentorV2/sam3/exps/pathseg_evals/test"
  CHECKPOINT_PATHS["test"]="/jhcnas5/zq/code/ckpt/finetune/ours/exps/pathseg_instance_ft/checkpoints/best.pt"
  CFG_PATHS["test"]="${PWD}/sam3/train/configs/pathseg_evals/eval_base.yaml"
}

run_one_dataset() {
  local config_name="$1"
  local prompt_mode="$2"
  local point_count="$3"
  local dataset_key="$4"
  local run_idx="${5:-0}"
  local run_idx_tag
  printf -v run_idx_tag "%03d" "${run_idx}"

  local base_dir="${EXP_BASE_DIRS[$config_name]}"
  local checkpoint_path="${CHECKPOINT_PATHS[$config_name]}"
  local cfg_path="${CFG_PATHS[$config_name]}"

  if [[ -z "${base_dir:-}" || -z "${checkpoint_path:-}" || -z "${cfg_path:-}" ]]; then
    echo "Unknown config: ${config_name}" >&2
    return 1
  fi

  local return_box_prob=0
  local return_point_prob=0
  local prompt_tag="${prompt_mode}"
  local exp_log_dir="${base_dir}/${dataset_key}"

  case "${prompt_mode}" in
    box)
      return_box_prob=1
      return_point_prob=0
      prompt_tag="box"
      exp_log_dir="${exp_log_dir}/box"
      ;;
    point)
      if [[ -z "${point_count}" ]]; then
        echo "point_count must be set when prompt_mode=point" >&2
        return 1
      fi
      return_box_prob=0
      return_point_prob=1
      prompt_tag="point_${point_count}"
      exp_log_dir="${exp_log_dir}/${prompt_tag}"
      ;;
    none)
      return_box_prob=0
      return_point_prob=0
      prompt_tag="no_prompt"
      exp_log_dir="${exp_log_dir}/${prompt_tag}"
      ;;
    *)
      echo "Unknown prompt mode: ${prompt_mode}. Expected one of: point box none" >&2
      return 1
      ;;
  esac

  if [[ -n "${TEXT_PROMPT_OVERRIDE}" ]]; then
    local text_tag="${TEXT_PROMPT_OVERRIDE// /_}"
    text_tag="${text_tag//\//_}"
    prompt_tag="${prompt_tag}_text_${text_tag}"
    exp_log_dir="${base_dir}/${dataset_key}/${prompt_tag}"
  fi

  mkdir -p "${exp_log_dir}"
  local log_subdir="${LOG_DIR}/${config_name}/${dataset_key}"
  mkdir -p "${log_subdir}"
  local log_file="${log_subdir}/${prompt_tag}.log"
  local wandb_run_name="${WANDB_RUN_PREFIX}-${config_name}-${dataset_key}-${prompt_tag}-r${run_idx_tag}-${RUN_TS}"

  local -a cmd=(
    python sam3/train/train.py
    -c "${cfg_path}"
    --use-cluster 0
    --num-gpus 1
    dataset_key="${dataset_key}"
    paths.experiment_log_dir="${exp_log_dir}"
    paths.checkpoint_path="${checkpoint_path}"
    trainer.data.val.dataset.return_geometry="${RETURN_GEOMETRY}"
    trainer.data.val.dataset.return_box_prob="${return_box_prob}"
    trainer.data.val.dataset.return_point_prob="${return_point_prob}"
  )

  if [[ "${prompt_mode}" == "point" ]]; then
    cmd+=(trainer.data.val.dataset.max_input_points="${point_count}")
  fi

  if [[ -n "${VAL_BATCH_SIZE}" ]]; then
    cmd+=(scratch.val_batch_size="${VAL_BATCH_SIZE}")
  fi

  if [[ -n "${TEXT_PROMPT_OVERRIDE}" ]]; then
    cmd+=(trainer.data.val.dataset.text_prompt_override="${TEXT_PROMPT_OVERRIDE}")
  fi

  if [[ "${DISABLE_WANDB}" == "1" ]]; then
    cmd+=(trainer.logging.wandb_writer=null)
  elif [[ -n "${WANDB_PROJECT}" ]]; then
    cmd+=(
      trainer.logging.wandb_writer.project="${WANDB_PROJECT}"
      trainer.logging.wandb_writer.group="${dataset_key}"
      trainer.logging.wandb_writer.name="${wandb_run_name}"
    )
  fi

  echo "------------------------------------------------------------"
  echo "[RUN] CONFIG=${config_name}, DATASET=${dataset_key}, PROMPT_MODE=${prompt_mode}, POINT_COUNT=${point_count:-none}, TEXT_PROMPT=${TEXT_PROMPT_OVERRIDE:-dataset-default}"
  echo "[RUN] experiment_log_dir=${exp_log_dir}"
  echo "[RUN] log_file=${log_file}"
  echo "[RUN] wandb_run_name=${wandb_run_name}"

  local selected_gpu=""
  selected_gpu="$(pick_best_gpu || true)"
  if [[ -n "${selected_gpu}" ]]; then
    echo "[RUN] gpu=${selected_gpu}"
    CUDA_VISIBLE_DEVICES="${selected_gpu}" "${cmd[@]}" > "${log_file}" 2>&1
  else
    echo "[RUN] gpu=${CUDA_VISIBLE_DEVICES:-all}"
    "${cmd[@]}" > "${log_file}" 2>&1
  fi
}

main() {
  setup_env
  init_maps
  mkdir -p "${LOG_DIR}"

  local -a datasets configs prompt_modes point_counts
  read -r -a datasets <<< "${DATASETS_STR}"
  read -r -a configs <<< "${CONFIGS_STR}"
  read -r -a prompt_modes <<< "${PROMPT_MODES_STR}"
  read -r -a point_counts <<< "${POINT_COUNTS_STR}"

  local -i running=0
  local -i failed=0
  local -i max_parallel="${K}"
  local -i run_idx=0
  for cfg in "${configs[@]}"; do
    for prompt_mode in "${prompt_modes[@]}"; do
      for dataset_key in "${datasets[@]}"; do
        if [[ "${prompt_mode}" == "point" ]]; then
          for point_count in "${point_counts[@]}"; do
            run_idx=$((run_idx + 1))
            run_one_dataset "${cfg}" "${prompt_mode}" "${point_count}" "${dataset_key}" "${run_idx}" &
            running=$((running + 1))
            sleep 60
            if (( running >= max_parallel )); then
              if ! wait -n; then
                failed=$((failed + 1))
              fi
              running=$((running - 1))
            fi
          done
        else
          run_idx=$((run_idx + 1))
          run_one_dataset "${cfg}" "${prompt_mode}" "" "${dataset_key}" "${run_idx}" &
          running=$((running + 1))
          sleep 60
          if (( running >= max_parallel )); then
            if ! wait -n; then
              failed=$((failed + 1))
            fi
            running=$((running - 1))
          fi
        fi
      done
    done
  done

  while (( running > 0 )); do
    if ! wait -n; then
      failed=$((failed + 1))
    fi
    running=$((running - 1))
  done

  if (( failed > 0 )); then
    echo "All eval jobs finished with ${failed} failed job(s)." >&2
    return 1
  fi

  echo "All eval jobs finished."
}

main "$@"
