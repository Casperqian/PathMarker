#!/usr/bin/env bash
#SBATCH -t 120:00:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=10
#SBATCH -N 1
#SBATCH --requeue
#SBATCH --job-name=manifest
#SBATCH --open-mode=append
#SBATCH --partition=vcompath
#SBATCH --account=vcompath
#SBATCH --output=./watch_folder/%x_%j.log
#SBATCH --error=./watch_folder/%x_%j.err

source "${HOME}/anaconda3/etc/profile.d/conda.sh"
conda activate sam3

cd ../

build_main_dataset() {
    python -m sam3.train.data.custom_image_dataset \
        --dataset-type path \
        --build-manifests
}

build_external_dataset() {
    local dataset_name="$1"
    python -m sam3.train.data.custom_image_dataset \
        --dataset-type path \
        --dataset-name "${dataset_name}" \
        --build-manifests \
        --split test
}

main() {
    # build_main_dataset

    for dataset_name in "${EXTERNAL_DATASETS[@]}"; do
        build_external_dataset "${dataset_name}"
    done
}

EXTERNAL_DATASETS=("DatasetB")

main "$@"
