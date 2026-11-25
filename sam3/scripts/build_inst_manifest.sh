#!/usr/bin/env bash
#SBATCH -t 120:00:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=10
#SBATCH -N 1
#SBATCH --requeue
#SBATCH --job-name=inst_manifest
#SBATCH --open-mode=append
#SBATCH --partition=vcompath
#SBATCH --account=vcompath
#SBATCH --output=./watch_folder/%x_%j.log
#SBATCH --error=./watch_folder/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

INSTANCE_ROOT="${INSTANCE_ROOT:-/path/to/your/data/instances}"
DATASET_NAME="${DATASET_NAME:-all}"
SPLITS="${SPLITS:-train test}"
CONDA_ENV="${CONDA_ENV:-sam3}"

source "${HOME}/anaconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

echo "[RUN] instance_root=${INSTANCE_ROOT}"
echo "[RUN] dataset_name=${DATASET_NAME}"
echo "[RUN] splits=${SPLITS}"

INSTANCE_ROOT="${INSTANCE_ROOT}" \
DATASET_NAME="${DATASET_NAME}" \
SPLITS="${SPLITS}" \
python - <<'PY'
import os
from pathlib import Path

from sam3.train.data.custom_instance_dataset import (
    _load_or_build_manifest,
    _manifest_path,
    _resolve_dataset_names,
)

root = Path(os.environ["INSTANCE_ROOT"])
dataset_name = os.environ["DATASET_NAME"]
splits = os.environ["SPLITS"].split()

total = 0
for split in splits:
    split_total = 0
    datasets = _resolve_dataset_names(root, dataset_name, split)
    if not datasets:
        print(f"[MANIFEST] split={split} no datasets found under {root}")
        continue

    for ds in datasets:
        image_paths, instance_paths = _load_or_build_manifest(root, ds, split)
        manifest_path = _manifest_path(root, ds, split)
        count = len(instance_paths)
        split_total += count
        print(
            f"[MANIFEST] dataset={ds} split={split} "
            f"pairs={count} cache={manifest_path}"
        )

    total += split_total
    print(f"[MANIFEST] split={split} total_pairs={split_total}")

print(f"[MANIFEST] all_done total_pairs={total}")
PY
