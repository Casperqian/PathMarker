#!/usr/bin/env bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIOMARKER_ROOT="${PROJECT_ROOT}/biomarker"

if [[ -z "${PYTHON:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON="python"
  else
    PYTHON="python3"
  fi
fi

COHORT="${COHORT:-BLCA}"
SLIDE_ID="${SLIDE_ID:-}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-}"
TCGA_ROOT="${TCGA_ROOT:-/jhcnas7/Pathology/original_data/TCGA}"
HISTOQC_ROOT="${HISTOQC_ROOT:-/jhcnas5/zq/data/PathSeg2/biomarker}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/jhcnas3/MRI_jhcnas3/biomarker/tme_readout}"
FORMAT_CSV="${FORMAT_CSV:-${PROJECT_ROOT}/biomarker/tme/TME_Readout_Format.csv}"

TISSUE_CHECKPOINT="${TISSUE_CHECKPOINT:-/jhcnas5/zq/code/ckpt/finetune/ours/exps/siglip_labeled_ft/checkpoints/best.pt}"
CELL_CHECKPOINT="${CELL_CHECKPOINT:-/jhcnas5/zq/code/ckpt/finetune/ours/exps/pathseg_instance_ft_mix/checkpoints/best.pt}"
BPE_PATH="${BPE_PATH:-${PROJECT_ROOT}/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz}"
DEVICE="${DEVICE:-cuda}"

if [[ -n "${CUDA_ID:-}" ]]; then
  DEFAULT_CUDA_IDS="${CUDA_ID}"
else
  DEFAULT_CUDA_IDS="2,4,5"
fi
CUDA_IDS="${CUDA_IDS:-${DEFAULT_CUDA_IDS}}"
IFS=',' read -r -a CUDA_ID_LIST <<< "${CUDA_IDS}"
for idx in "${!CUDA_ID_LIST[@]}"; do
  CUDA_ID_LIST[$idx]="$(echo "${CUDA_ID_LIST[$idx]}" | xargs)"
done

LIMIT_SLIDES="${LIMIT_SLIDES:-1}"
RANDOM_SAMPLE="${RANDOM_SAMPLE:-0}"
RANDOM_SEED="${RANDOM_SEED:-0}"
RESUME="${RESUME:-1}"
PRIORITIZE_CASES="${PRIORITIZE_CASES:-0}"
DYNAMIC_SLIDE_CLAIM="${DYNAMIC_SLIDE_CLAIM:-0}"
SHARD_LOG_DIR="${SHARD_LOG_DIR:-${PROJECT_ROOT}/biomarker/logs}"
TME_RUN_TAG="${TME_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
TME_LOG_PREFIX="${TME_LOG_PREFIX:-${TME_RUN_TAG}_${COHORT}}"

TISSUE_TARGET_MPP="${TISSUE_TARGET_MPP:-0.5}"
TISSUE_PATCH_SIZE="${TISSUE_PATCH_SIZE:-1024}"
TISSUE_OVERLAP="${TISSUE_OVERLAP:-128}"
TISSUE_BATCH_SIZE="${TISSUE_BATCH_SIZE:-8}"
TISSUE_MODEL_RESOLUTION="${TISSUE_MODEL_RESOLUTION:-1008}"
TISSUE_PROB_THRESHOLD="${TISSUE_PROB_THRESHOLD:-0.5}"
TISSUE_THRESHOLD="${TISSUE_THRESHOLD:-0.2}"
SLIDE_MASK_DOWNSAMPLE="${SLIDE_MASK_DOWNSAMPLE:-32}"
LIMIT_TISSUE_PATCHES="${LIMIT_TISSUE_PATCHES:-0}"

CELL_TARGET_MPP="${CELL_TARGET_MPP:-0.25}"
CELL_TILE_SIZE="${CELL_TILE_SIZE:-576}"
CELL_OVERLAP="${CELL_OVERLAP:-64}"
CELL_BATCH_SIZE="${CELL_BATCH_SIZE:-64}"
CELL_MODEL_RESOLUTION="${CELL_MODEL_RESOLUTION:-1008}"
CELL_NUM_QUERIES="${CELL_NUM_QUERIES:-300}"
CELL_DETECTION_THRESHOLD="${CELL_DETECTION_THRESHOLD:-0.5}"
CELL_MAX_DETS_PER_TILE="${CELL_MAX_DETS_PER_TILE:-300}"
CELL_NMS_IOU_THRESHOLD="${CELL_NMS_IOU_THRESHOLD:-0.5}"
CELL_DEDUPE_DISTANCE_UM="${CELL_DEDUPE_DISTANCE_UM:-2.0}"
LIMIT_CELL_TILES="${LIMIT_CELL_TILES:-0}"
CELL_PREFETCH_WORKERS="${CELL_PREFETCH_WORKERS:-2}"
CELL_POSTPROCESS_WORKERS="${CELL_POSTPROCESS_WORKERS:-0}"
CELL_POSTPROCESS_MAX_PENDING="${CELL_POSTPROCESS_MAX_PENDING:-1}"
CELL_TISSUE_FILTER_LABELS="${CELL_TISSUE_FILTER_LABELS:-Tumor,Epithelial Tissue,Stroma}"
CELL_TISSUE_FILTER_THRESHOLD="${CELL_TISSUE_FILTER_THRESHOLD:-0.1}"

RADII_UM="${RADII_UM:-40,80}"
FOREGROUND_SOURCE="${FOREGROUND_SOURCE:-online}"
FOREGROUND_DOWNSAMPLE="${FOREGROUND_DOWNSAMPLE:-32}"
FOREGROUND_SATURATION_THRESHOLD="${FOREGROUND_SATURATION_THRESHOLD:-8}"
FOREGROUND_USE_OTSU="${FOREGROUND_USE_OTSU:-0}"
FOREGROUND_MEDIAN_SIZE="${FOREGROUND_MEDIAN_SIZE:-7}"
FOREGROUND_CLOSE="${FOREGROUND_CLOSE:-4}"
FOREGROUND_MIN_OBJECT_SIZE="${FOREGROUND_MIN_OBJECT_SIZE:-64}"
FOREGROUND_HOLE_SIZE="${FOREGROUND_HOLE_SIZE:-64}"
FOREGROUND_VALUE_LOWER="${FOREGROUND_VALUE_LOWER:-20}"
FOREGROUND_VALUE_UPPER="${FOREGROUND_VALUE_UPPER:-245}"
FOREGROUND_SUPPRESS_MARKERS="${FOREGROUND_SUPPRESS_MARKERS:-1}"
FOREGROUND_MARKER_DILATION="${FOREGROUND_MARKER_DILATION:-3}"

tme_base_args=(
  "${PROJECT_ROOT}/biomarker/tme/tme_wsi_readout_pipeline.py"
  --cohort "${COHORT}"
  --tcga-root "${TCGA_ROOT}"
  --histoqc-root "${HISTOQC_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --format-csv "${FORMAT_CSV}"
  --tissue-checkpoint "${TISSUE_CHECKPOINT}"
  --cell-checkpoint "${CELL_CHECKPOINT}"
  --bpe-path "${BPE_PATH}"
  --device "${DEVICE}"
  --limit-slides "${LIMIT_SLIDES}"
  --seed "${RANDOM_SEED}"
  --tissue-target-mpp "${TISSUE_TARGET_MPP}"
  --tissue-patch-size "${TISSUE_PATCH_SIZE}"
  --tissue-overlap "${TISSUE_OVERLAP}"
  --tissue-batch-size "${TISSUE_BATCH_SIZE}"
  --tissue-model-resolution "${TISSUE_MODEL_RESOLUTION}"
  --tissue-prob-threshold "${TISSUE_PROB_THRESHOLD}"
  --tissue-threshold "${TISSUE_THRESHOLD}"
  --slide-mask-downsample "${SLIDE_MASK_DOWNSAMPLE}"
  --limit-tissue-patches "${LIMIT_TISSUE_PATCHES}"
  --cell-target-mpp "${CELL_TARGET_MPP}"
  --cell-tile-size "${CELL_TILE_SIZE}"
  --cell-overlap "${CELL_OVERLAP}"
  --cell-batch-size "${CELL_BATCH_SIZE}"
  --cell-model-resolution "${CELL_MODEL_RESOLUTION}"
  --cell-num-queries "${CELL_NUM_QUERIES}"
  --cell-detection-threshold "${CELL_DETECTION_THRESHOLD}"
  --cell-max-dets-per-tile "${CELL_MAX_DETS_PER_TILE}"
  --cell-nms-iou-threshold "${CELL_NMS_IOU_THRESHOLD}"
  --cell-dedupe-distance-um "${CELL_DEDUPE_DISTANCE_UM}"
  --limit-cell-tiles "${LIMIT_CELL_TILES}"
  --cell-prefetch-workers "${CELL_PREFETCH_WORKERS}"
  --cell-postprocess-workers "${CELL_POSTPROCESS_WORKERS}"
  --cell-postprocess-max-pending "${CELL_POSTPROCESS_MAX_PENDING}"
  --cell-tissue-filter-labels "${CELL_TISSUE_FILTER_LABELS}"
  --cell-tissue-filter-threshold "${CELL_TISSUE_FILTER_THRESHOLD}"
  --radii-um "${RADII_UM}"
  --foreground-source "${FOREGROUND_SOURCE}"
  --foreground-downsample "${FOREGROUND_DOWNSAMPLE}"
  --foreground-saturation-threshold "${FOREGROUND_SATURATION_THRESHOLD}"
  --foreground-median-size "${FOREGROUND_MEDIAN_SIZE}"
  --foreground-close "${FOREGROUND_CLOSE}"
  --foreground-min-object-size "${FOREGROUND_MIN_OBJECT_SIZE}"
  --foreground-hole-size "${FOREGROUND_HOLE_SIZE}"
  --foreground-value-lower "${FOREGROUND_VALUE_LOWER}"
  --foreground-value-upper "${FOREGROUND_VALUE_UPPER}"
  --foreground-marker-dilation "${FOREGROUND_MARKER_DILATION}"
)

if [[ -n "${SLIDE_ID}" ]]; then
  tme_base_args+=(--slide-id "${SLIDE_ID}")
fi
if [[ -n "${SOURCE_MANIFEST}" ]]; then
  tme_base_args+=(--source-manifest "${SOURCE_MANIFEST}")
fi
if [[ "${RANDOM_SAMPLE}" == "1" ]]; then
  tme_base_args+=(--random-sample)
fi
if [[ "${RESUME}" == "1" ]]; then
  tme_base_args+=(--resume)
fi
if [[ "${PRIORITIZE_CASES}" == "1" ]]; then
  tme_base_args+=(--prioritize-cases)
fi
if [[ "${DYNAMIC_SLIDE_CLAIM}" == "1" ]]; then
  tme_base_args+=(--dynamic-slide-claim)
fi
if [[ "${FOREGROUND_USE_OTSU}" == "1" ]]; then
  tme_base_args+=(--foreground-use-otsu)
fi
if [[ "${FOREGROUND_SUPPRESS_MARKERS}" == "1" ]]; then
  tme_base_args+=(--foreground-suppress-markers)
else
  tme_base_args+=(--no-foreground-suppress-markers)
fi

tme_run_python() {
  local gpu_id="$1"
  shift
  if [[ "${DEVICE}" == cuda* ]]; then
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON}" "${tme_base_args[@]}" "$@"
  else
    "${PYTHON}" "${tme_base_args[@]}" "$@"
  fi
}
