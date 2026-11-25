#!/usr/bin/env bash
#SBATCH -t 120:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH -N 1
#SBATCH --requeue
#SBATCH --job-name=infer_inst
#SBATCH --open-mode=append
#SBATCH --partition=vcompath
#SBATCH --account=vcompath
#SBATCH --output=./watch_folder/%x_%j.log
#SBATCH --error=./watch_folder/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

HOSTNAME="$(hostname -s 2>/dev/null || hostname)"
RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"

CONFIG_PATH="${CONFIG_PATH:-${PWD}/sam3/train/configs/pathseg_train/sam3_pathseg_instance_ft.yaml}"
INSTANCE_ROOT="${INSTANCE_ROOT:-/path/to/your/data/instances}"
EXP_ROOT="${EXP_ROOT:-/home/qzengar/PathSegmentorV2/sam3/exps/pathseg_instance_eval/siglip}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/jhcnas5/zq/code/ckpt/finetune/ours/exps/pathseg_instance_ft/checkpoints/best.pt}"
DATASETS_STR="${DATASETS_STR:-auto}"
DETECTION_THRESHOLDS="${DETECTION_THRESHOLDS:-0.5}"

NUM_GPUS="${NUM_GPUS:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-${K:-1}}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
NUM_QUERIES="${NUM_QUERIES:-300}"
MAX_ANN_PER_IMG="${MAX_ANN_PER_IMG:-1000000}"
MAX_DETS_PER_PATCH="${MAX_DETS_PER_PATCH:-300}"
PATCH_PRE_NMS_TOPK="${PATCH_PRE_NMS_TOPK:-2000}"
PATCH_NMS_IOU_THRESHOLD="${PATCH_NMS_IOU_THRESHOLD:-0.7}"
USE_PRESENCE_EVAL="${USE_PRESENCE_EVAL:-false}"
ENABLE_MORPH_POSTPROCESS="${ENABLE_MORPH_POSTPROCESS:-false}"
POSTPROCESS_MIN_INSTANCE_AREA="${POSTPROCESS_MIN_INSTANCE_AREA:-16}"
POSTPROCESS_HOLE_AREA="${POSTPROCESS_HOLE_AREA:-16}"
POSTPROCESS_SPLIT_COMPONENTS="${POSTPROCESS_SPLIT_COMPONENTS:-true}"

EVAL_PATCHIFY="${EVAL_PATCHIFY:-true}"
EVAL_PATCH_SIZE="${EVAL_PATCH_SIZE:-256}"
EVAL_PATCH_STRIDE="${EVAL_PATCH_STRIDE:-192}"
EVAL_ADAPTIVE_INFERENCE="${EVAL_ADAPTIVE_INFERENCE:-false}"
EVAL_ADAPTIVE_THRESHOLD="${EVAL_ADAPTIVE_THRESHOLD:-250}"
EVAL_ADAPTIVE_MIN_SIZE="${EVAL_ADAPTIVE_MIN_SIZE:-256}"
EVAL_ADAPTIVE_MAX_DEPTH="${EVAL_ADAPTIVE_MAX_DEPTH:-2}"
EVAL_ADAPTIVE_CONTEXT="${EVAL_ADAPTIVE_CONTEXT:-0}"
VAL_LIMIT_PER_DATASET="${VAL_LIMIT_PER_DATASET:-}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-pathseg_instance_eval}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-infer-inst}"
SUMMARY_FILE="${SUMMARY_FILE:-}"

resolve_output_root() {
  if [[ "${ENABLE_MORPH_POSTPROCESS}" == "true" ]]; then
    EXP_ROOT="${EXP_ROOT%/}/postproc"
  fi
}

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

check_file() {
  local path="$1"
  local name="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] ${name} not found: ${path}" >&2
    exit 1
  fi
}

check_dir() {
  local path="$1"
  local name="$2"
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] ${name} not found: ${path}" >&2
    exit 1
  fi
}

resolve_checkpoint_path() {
  if [[ -n "${CHECKPOINT_PATH}" ]]; then
    return
  fi

  local best_ckpt="/jhcnas5/zq/code/ckpt/finetune/ours/exps/pathseg_instance_ft/checkpoints/best.pt"
  local last_ckpt="/home/qzengar/PathSegmentorV2/sam3/exps/pathseg_instance_ft/checkpoints/checkpoint.pt"
  if [[ -f "${best_ckpt}" ]]; then
    CHECKPOINT_PATH="${best_ckpt}"
  else
    CHECKPOINT_PATH="${last_ckpt}"
  fi
}

detection_threshold_tag() {
  local detection_threshold="$1"
  echo "thres_${detection_threshold//./}"
}

discover_datasets() {
  find "${INSTANCE_ROOT}" -mindepth 1 -maxdepth 1 -type d \
    -exec sh -c 'for d; do [ -d "$d/test" ] && [ -d "$d/test_inst_mask" ] && basename "$d"; done' sh {} + \
    | sort
}

append_summary() {
  local dataset_key="$1"
  local result_file="$2"
  local summary_file="$3"

  python - "$dataset_key" "$result_file" "$summary_file" <<'PY'
import json
import pathlib
import sys

dataset, result_path, summary_path = sys.argv[1:4]
result_path = pathlib.Path(result_path)
summary_path = pathlib.Path(summary_path)
payload = json.loads(result_path.read_text())
summary = payload.get("summary", {})
summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("a", encoding="utf-8") as f:
    f.write(
        (
            "{dataset}\t{dice:.6f}\t{aji:.6f}\t{dq:.6f}\t{sq:.6f}\t{pq:.6f}\t"
            "{aji_plus:.6f}\t{num_images:.0f}\t{num_gt:.0f}\t{num_pred:.0f}\t{json_path}\n"
        ).format(
            dataset=dataset,
            dice=float(summary.get("Dice", 0.0)),
            aji=float(summary.get("AJI", 0.0)),
            dq=float(summary.get("DQ", 0.0)),
            sq=float(summary.get("SQ", 0.0)),
            pq=float(summary.get("PQ", 0.0)),
            aji_plus=float(summary.get("AJI+", 0.0)),
            num_images=float(summary.get("num_images", 0.0)),
            num_gt=float(summary.get("num_gt", 0.0)),
            num_pred=float(summary.get("num_pred", 0.0)),
            json_path=str(result_path),
        )
    )
PY
}

merge_summary() {
  local threshold_dir="$1"
  local summary_file="$2"

  printf "dataset\tDice\tAJI\tDQ\tSQ\tPQ\tAJI+\tnum_images\tnum_gt\tnum_pred\tjson\n" > "${summary_file}"
  find "${threshold_dir}" -mindepth 2 -maxdepth 2 -path "*/logs/summary_row.tsv" -print0 \
    | sort -z \
    | xargs -0 -r cat >> "${summary_file}"
}

run_one_dataset() {
  local dataset_key="$1"
  local detection_threshold="$2"
  local threshold_dir="$3"
  local summary_file="$4"
  local exp_dir="${threshold_dir}/${dataset_key}"
  local dump_dir="${exp_dir}/dumps/pathseg_instance/${dataset_key}"
  local log_dir="${exp_dir}/logs"
  local log_file="${log_dir}/infer.log"
  local summary_row_file="${log_dir}/summary_row.tsv"
  local result_file="${dump_dir}/instance_predictions_segm.json"
  local wandb_run_name="${WANDB_RUN_PREFIX}-${dataset_key}-${RUN_TS}"

  mkdir -p "${log_dir}" "${dump_dir}"

  local -a cmd=(
    python sam3/train/train.py
    -c "${CONFIG_PATH}"
    --use-cluster 0
    --num-gpus "${NUM_GPUS}"
    trainer.mode=val
    trainer.skip_saving_ckpts=true
    trainer.checkpoint.resume_from=null
    "trainer.checkpoint.save_dir=${exp_dir}/checkpoints_unused"
    "pathseg_instance_train.supercategory=${dataset_key}"
    "trainer.data.val.dataset.dataset_name=${dataset_key}"
    trainer.data.val.dataset.split=test
    "paths.pathseg_instance_root=${INSTANCE_ROOT}"
    "paths.experiment_log_dir=${exp_dir}"
    "launcher.experiment_log_dir=${exp_dir}"
    "paths.sam3_checkpoint=${CHECKPOINT_PATH}"
    paths.semantic_checkpoint=null
    trainer.model.checkpoint_path_partial=null
    "scratch.val_batch_size=${VAL_BATCH_SIZE}"
    "scratch.num_queries=${NUM_QUERIES}"
    "scratch.max_ann_per_img=${MAX_ANN_PER_IMG}"
    "scratch.eval_max_dets_per_patch=${MAX_DETS_PER_PATCH}"
    "scratch.eval_patch_pre_nms_topk=${PATCH_PRE_NMS_TOPK}"
    "scratch.eval_patch_nms_iou_threshold=${PATCH_NMS_IOU_THRESHOLD}"
    "scratch.instance_postprocessor.detection_threshold=${detection_threshold}"
    "scratch.use_presence_eval=${USE_PRESENCE_EVAL}"
    "scratch.instance_enable_morph_postprocess=${ENABLE_MORPH_POSTPROCESS}"
    "scratch.instance_postprocess_min_instance_area=${POSTPROCESS_MIN_INSTANCE_AREA}"
    "scratch.instance_postprocess_hole_area=${POSTPROCESS_HOLE_AREA}"
    "scratch.instance_postprocess_split_components=${POSTPROCESS_SPLIT_COMPONENTS}"
    "trainer.data.val.dataset.patchify_eval=${EVAL_PATCHIFY}"
    "scratch.eval_patch_size=${EVAL_PATCH_SIZE}"
    "scratch.eval_patch_stride=${EVAL_PATCH_STRIDE}"
    "scratch.eval_adaptive_inference=${EVAL_ADAPTIVE_INFERENCE}"
    "scratch.eval_adaptive_threshold=${EVAL_ADAPTIVE_THRESHOLD}"
    "scratch.eval_adaptive_min_size=${EVAL_ADAPTIVE_MIN_SIZE}"
    "scratch.eval_adaptive_max_depth=${EVAL_ADAPTIVE_MAX_DEPTH}"
    "scratch.eval_adaptive_context=${EVAL_ADAPTIVE_CONTEXT}"
    "trainer.meters.val.all.instance_seg.dump_dir=${dump_dir}"
    "trainer.logging.wandb_writer.should_log=${WANDB_ENABLED}"
    "trainer.logging.wandb_writer.project=${WANDB_PROJECT}"
    "trainer.logging.wandb_writer.group=${dataset_key}"
    "trainer.logging.wandb_writer.name=${wandb_run_name}"
  )

  if [[ -n "${VAL_LIMIT_PER_DATASET}" ]]; then
    cmd+=("trainer.data.val.dataset.limit_per_dataset=${VAL_LIMIT_PER_DATASET}")
  fi

  echo "------------------------------------------------------------"
  echo "[RUN] dataset=${dataset_key}"
  echo "[RUN] checkpoint=${CHECKPOINT_PATH}"
  echo "[RUN] instance_root=${INSTANCE_ROOT}"
  echo "[RUN] detection_threshold=${detection_threshold}"
  echo "[RUN] gpu=${CUDA_VISIBLE_DEVICES:-5}"
  echo "[RUN] exp_dir=${exp_dir}"
  echo "[RUN] log_file=${log_file}"

  "${cmd[@]}" > "${log_file}" 2>&1

  check_file "${result_file}" "result json"
  : > "${summary_row_file}"
  append_summary "${dataset_key}" "${result_file}" "${summary_row_file}"
  echo "[DONE] dataset=${dataset_key}, result=${result_file}"
}

wait_for_slot() {
  local -n running_ref="$1"
  local -n failed_ref="$2"

  if (( running_ref < MAX_PARALLEL )); then
    return
  fi

  if ! wait -n; then
    failed_ref=$((failed_ref + 1))
  fi
  running_ref=$((running_ref - 1))
}

wait_for_all() {
  local -n running_ref="$1"
  local -n failed_ref="$2"

  while (( running_ref > 0 )); do
    if ! wait -n; then
      failed_ref=$((failed_ref + 1))
    fi
    running_ref=$((running_ref - 1))
  done
}

main() {
  setup_env
  resolve_output_root
  resolve_checkpoint_path
  check_file "${CONFIG_PATH}" "config"
  check_file "${CHECKPOINT_PATH}" "instance checkpoint"
  check_dir "${INSTANCE_ROOT}" "instance root"

  if [[ "${EVAL_ADAPTIVE_INFERENCE}" == "true" && "${NUM_GPUS}" != "1" ]]; then
    echo "[ERROR] EVAL_ADAPTIVE_INFERENCE=true requires NUM_GPUS=1" >&2
    exit 1
  fi
  if (( MAX_PARALLEL < 1 )); then
    echo "[ERROR] MAX_PARALLEL must be >= 1, got ${MAX_PARALLEL}" >&2
    exit 1
  fi

  local -a datasets
  if [[ "${DATASETS_STR}" == "auto" ]]; then
    mapfile -t datasets < <(discover_datasets)
  else
    read -r -a datasets <<< "${DATASETS_STR}"
  fi
  if (( ${#datasets[@]} == 0 )); then
    echo "[ERROR] no datasets found under ${INSTANCE_ROOT}" >&2
    exit 1
  fi

  local -a detection_thresholds
  read -r -a detection_thresholds <<< "${DETECTION_THRESHOLDS}"
  if (( ${#detection_thresholds[@]} == 0 )); then
    echo "[ERROR] DETECTION_THRESHOLDS must contain at least one value" >&2
    exit 1
  fi

  mkdir -p "${EXP_ROOT}"
  local -i running=0
  local -i failed=0

  for detection_threshold in "${detection_thresholds[@]}"; do
    local tag
    tag="$(detection_threshold_tag "${detection_threshold}")"
    local threshold_dir="${EXP_ROOT}/${tag}"
    local summary_file="${SUMMARY_FILE:-${threshold_dir}/summary.tsv}"

    mkdir -p "${threshold_dir}"

    echo "============================================================"
    echo "[RUN] detection_threshold=${detection_threshold}, output=${threshold_dir}"
    echo "[RUN] datasets=${datasets[*]}"
    echo "[RUN] max_parallel=${MAX_PARALLEL}, shared_gpu=${CUDA_VISIBLE_DEVICES:-slurm-visible-gpu}"
    for dataset_key in "${datasets[@]}"; do
      wait_for_slot running failed
      run_one_dataset "${dataset_key}" "${detection_threshold}" "${threshold_dir}" "${summary_file}" &
      running=$((running + 1))
    done
  done

  wait_for_all running failed
  if (( failed > 0 )); then
    echo "[ERROR] inference finished with ${failed} failed job(s)" >&2
    exit 1
  fi

  for detection_threshold in "${detection_thresholds[@]}"; do
    local tag
    tag="$(detection_threshold_tag "${detection_threshold}")"
    local threshold_dir="${EXP_ROOT}/${tag}"
    local summary_file="${SUMMARY_FILE:-${threshold_dir}/summary.tsv}"
    merge_summary "${threshold_dir}" "${summary_file}"
    echo "[DONE] detection_threshold=${detection_threshold}, summary=${summary_file}"
  done

  echo "------------------------------------------------------------"
  echo "[DONE] all thresholds finished"
  echo "[DONE] root=${EXP_ROOT}"
}

main "$@"
