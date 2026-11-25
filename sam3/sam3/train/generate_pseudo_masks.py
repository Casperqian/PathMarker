#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from sam3.model.utils.misc import copy_data_to_device
from sam3.model_builder import build_pathmarker_semantic_model
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.hed_augmentation import HedColorAugmenter1
from sam3.train.data.pretrain_dataset import NORM_MEAN, NORM_STD, PretrainDataset
from sam3.train.data.sam3_image_dataset import Datapoint, Image


DEFAULT_BPE_PATH = "/home/qzengar/PathSegmentorV2/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
DEFAULT_CHECKPOINT_PATH = (
    "/home/qzengar/PathSegmentorV2/sam3/exps/siglip_labeled_ft-l8-wo-pathgen/checkpoints/checkpoint.pt"
)
DEFAULT_OUTPUT_DIR = (
    "/home/qzengar/PathSegmentorV2/sam3/exps/pretrain_pseudo_masks/siglip-l8-wo-pathgen"
)

class DeterministicHEDAugmentTransform:
    """
    Deterministic HED augmentation.

    `HedColorAugmenter1.randomize()` uses the global numpy RNG. We temporarily seed
    it before every call so the augmentation strength stays identical for every image.
    """

    def __init__(self, thresh: float, seed: int) -> None:
        self.augmenter = HedColorAugmenter1(thresh)
        self.seed = seed

    def __call__(self, image: PILImage.Image) -> PILImage.Image:
        image_np = np.asarray(image).copy()
        rng_state = np.random.get_state()
        np.random.seed(self.seed)
        try:
            self.augmenter.randomize()
        finally:
            np.random.set_state(rng_state)
        aug_np = self.augmenter.transform(image_np)
        return PILImage.fromarray(aug_np)


class ModelInputTransform:
    def __init__(
        self,
        resolution: int,
        mean: Sequence[float],
        std: Sequence[float],
        augment: Optional[DeterministicHEDAugmentTransform] = None,
    ) -> None:
        self.resolution = resolution
        self.mean = list(mean)
        self.std = list(std)
        self.augment = augment

    def __call__(self, image: PILImage.Image) -> torch.Tensor:
        if self.augment is not None:
            image = self.augment(image)
        image = TF.resize(
            image,
            [self.resolution, self.resolution],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, self.mean, self.std)
        return tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate pseudo masks for pretraining data using a finetuned SAM3 checkpoint. "
            "The script estimates a confidence threshold from a subset of images, then "
            "applies that threshold to the full pretrain set."
        )
    )
    parser.add_argument("--dataset-name", default="all")
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--bpe-path", default=DEFAULT_BPE_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--subset-image-count", type=int, default=1000)
    parser.add_argument("--keep-top-ratio", type=float, default=0.10)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional pre-defined confidence threshold. If provided, threshold estimation is skipped.",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--limit-samples", type=int, default=-1)
    parser.add_argument(
        "--generate-dataset-name",
        "--generate-dataset-key",
        dest="generate_dataset_name",
        default="",
        help=(
            "Optional dataset name used only when generating pseudo labels. "
            "If set, generation uses a separate PretrainDataset built from this dataset only."
        ),
    )
    parser.add_argument(
        "--save-rejected-masks",
        action="store_true",
        help="Deprecated no-op. Rejected masks are no longer saved.",
    )
    parser.add_argument("--hed-thresh", type=float, default=0.05)
    parser.add_argument("--hed-seed", type=int, default=123)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.keep_top_ratio <= 1.0:
        raise ValueError(f"--keep-top-ratio must be in (0, 1], got {args.keep_top_ratio}")
    if args.threshold is None and args.subset_image_count <= 0:
        raise ValueError(f"--subset-image-count must be > 0, got {args.subset_image_count}")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be > 0, got {args.batch_size}")
    if not 0.0 <= args.mask_threshold <= 1.0:
        raise ValueError(f"--mask-threshold must be in [0, 1], got {args.mask_threshold}")
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        raise ValueError(f"--threshold must be in [0, 1], got {args.threshold}")


def build_pretrain_dataset(dataset_name: str) -> PretrainDataset:
    return PretrainDataset(
        dataset_name=dataset_name,
        transform=lambda image: image.copy(),
        num_copy=1,
    )


def build_indices(dataset: PretrainDataset, limit_samples: int) -> List[int]:
    if limit_samples > 0:
        return list(range(min(limit_samples, len(dataset))))
    return list(range(len(dataset)))


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    model = build_pathmarker_semantic_model(
        bpe_path=args.bpe_path,
        device=args.device,
        eval_mode=True,
        checkpoint_path=args.checkpoint_path,
        load_from_HF=False,
        enable_segmentation=True,
    )
    model.eval()
    return model


def make_transforms(args: argparse.Namespace) -> tuple[ModelInputTransform, ModelInputTransform]:
    clean_transform = ModelInputTransform(
        resolution=args.resolution,
        mean=NORM_MEAN,
        std=NORM_STD,
        augment=None,
    )
    aug_transform = ModelInputTransform(
        resolution=args.resolution,
        mean=NORM_MEAN,
        std=NORM_STD,
        augment=DeterministicHEDAugmentTransform(
            thresh=args.hed_thresh,
            seed=args.hed_seed,
        ),
    )
    return clean_transform, aug_transform


def clone_datapoint_with_tensor(base_sample: Datapoint, image_tensor: torch.Tensor) -> Datapoint:
    image_meta = base_sample.images[0]
    sample = Datapoint(
        find_queries=list(base_sample.find_queries),
        images=[
            Image(
                data=image_tensor,
                objects=[],
                size=image_meta.size,
                blurring_mask=None,
            )
        ],
        raw_images=None,
    )
    sample.label = copy.deepcopy(getattr(base_sample, "label", None))
    sample.image_path = getattr(base_sample, "image_path", None)
    return sample


def build_model_batches(
    dataset: PretrainDataset,
    indices: Sequence[int],
    clean_transform: ModelInputTransform,
    aug_transform: ModelInputTransform,
) -> tuple[Dict, Dict]:
    clean_samples: List[Datapoint] = []
    aug_samples: List[Datapoint] = []

    for idx in indices:
        base_sample = dataset[idx]
        pil_image = base_sample.images[0].data
        if not isinstance(pil_image, PILImage.Image):
            raise TypeError(f"Expected PIL image from base dataset, got {type(pil_image)}")

        clean_tensor = clean_transform(pil_image.copy())
        aug_tensor = aug_transform(pil_image.copy())

        clean_samples.append(clone_datapoint_with_tensor(base_sample, clean_tensor))
        aug_samples.append(clone_datapoint_with_tensor(base_sample, aug_tensor))

    clean_batch = collate_fn_api(
        clean_samples,
        dict_key="pseudo-label",
        with_seg_masks=False,
    )
    aug_batch = collate_fn_api(
        aug_samples,
        dict_key="pseudo-label",
        with_seg_masks=False,
    )
    return clean_batch, aug_batch


def select_best_masks(stage_out: Dict) -> tuple[torch.Tensor, torch.Tensor]:
    logits = stage_out["pred_logits"].squeeze(-1)
    best_idx = logits.argmax(dim=1)
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    best_logits = logits[batch_idx, best_idx]
    best_mask_logits = stage_out["pred_masks"][batch_idx, best_idx]
    return best_logits, best_mask_logits


def extract_mask_logits(stage_out: Dict) -> torch.Tensor:
    semantic_logits = stage_out.get("semantic_seg")
    if semantic_logits is not None:
        if semantic_logits.ndim == 4 and semantic_logits.shape[1] == 1:
            return semantic_logits[:, 0]
        if semantic_logits.ndim == 3:
            return semantic_logits
        raise ValueError(f"Unexpected semantic_seg shape: {tuple(semantic_logits.shape)}")

    _, best_mask_logits = select_best_masks(stage_out)
    return best_mask_logits


def resize_probs_to_original_sizes(
    mask_logits_batch: torch.Tensor,
    original_sizes: torch.Tensor,
) -> List[torch.Tensor]:
    probs: List[torch.Tensor] = []
    for mask_logits, size in zip(mask_logits_batch, original_sizes):
        height = int(size[0].item())
        width = int(size[1].item())
        prob = torch.sigmoid(
            F.interpolate(
                mask_logits[None, None],
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        )
        probs.append(prob)
    return probs


def compute_binary_entropy(prob: torch.Tensor) -> torch.Tensor:
    prob = prob.clamp(1e-6, 1.0 - 1e-6)
    entropy = -(prob * prob.log() + (1.0 - prob) * (1.0 - prob).log())
    return entropy / math.log(2.0)


def score_prediction(
    clean_prob: torch.Tensor,
    aug_prob: torch.Tensor,
    mask_threshold: float,
) -> tuple[float, float, int, np.ndarray]:
    clean_mask = clean_prob > mask_threshold
    aug_mask = aug_prob > mask_threshold
    foreground_area = int(clean_mask.sum().item())

    if foreground_area == 0:
        empty_mask = np.zeros(tuple(clean_prob.shape), dtype=np.uint8)
        return 1.0, 0.0, 0, empty_mask

    entropy = float(compute_binary_entropy(clean_prob).mean().item())
    # avg_foreground_entropy = float(entropy_map[clean_mask].mean().item())

    intersection = (clean_mask & aug_mask).sum().item()
    denom = clean_mask.sum().item() + aug_mask.sum().item()
    consistency = 1.0 if denom == 0 else float((2.0 * intersection) / denom)

    # confidence = float(np.clip((1.0 - avg_foreground_entropy) * consistency, 0.0, 1.0))
    binary_mask = clean_mask.to(torch.uint8).cpu().numpy() # 0/1
    return entropy, consistency, foreground_area, binary_mask


def iter_inference_batch(
    model: torch.nn.Module,
    clean_batch: Dict,
    aug_batch: Dict,
    device: str,
    mask_threshold: float,
) -> Iterator[Dict[str, object]]:
    torch_device = torch.device(device)
    clean_input = copy_data_to_device(clean_batch["pseudo-label"], torch_device, non_blocking=True)
    aug_input = copy_data_to_device(aug_batch["pseudo-label"], torch_device, non_blocking=True)

    autocast_enabled = torch_device.type == "cuda"
    autocast_dtype = None
    if autocast_enabled:
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    with torch.no_grad():
        with torch.amp.autocast(
            device_type=torch_device.type,
            enabled=autocast_enabled,
            dtype=autocast_dtype,
        ):
            clean_out = model(clean_input)[0]
            aug_out = model(aug_input)[0]

    clean_mask_logits = extract_mask_logits(clean_out)
    aug_mask_logits = extract_mask_logits(aug_out)
    original_sizes = clean_input.find_metadatas[0].original_size
    clean_probs = resize_probs_to_original_sizes(clean_mask_logits, original_sizes)
    aug_probs = resize_probs_to_original_sizes(aug_mask_logits, original_sizes)

    image_paths = clean_batch.get("img_path_batch", [])
    labels = clean_batch.get("label_batch", [])
    for i in range(len(clean_probs)):
        entropy, consistency, foreground_area, binary_mask = score_prediction(
            clean_prob=clean_probs[i].float(),
            aug_prob=aug_probs[i].float(),
            mask_threshold=mask_threshold,
        )
        confidence = float(np.clip((1.0 - entropy) * consistency, 0.0, 1.0))
        yield {
            "image_path": image_paths[i],
            "label": labels[i],
            "confidence": confidence,
            "foreground_area": foreground_area,
            "mask": binary_mask,
        }


def batched(indices: Sequence[int], batch_size: int) -> Iterable[List[int]]:
    for start in range(0, len(indices), batch_size):
        yield list(indices[start : start + batch_size])


def collect_scores(
    dataset: PretrainDataset,
    indices: Sequence[int],
    model: torch.nn.Module,
    clean_transform: ModelInputTransform,
    aug_transform: ModelInputTransform,
    args: argparse.Namespace,
    desc: str,
) -> List[float]:
    scores: List[float] = []
    total_batches = math.ceil(len(indices) / args.batch_size) if indices else 0
    for batch_indices in tqdm(batched(indices, args.batch_size), total=total_batches, desc=desc):
        batch = build_model_batches(dataset, batch_indices, clean_transform, aug_transform)
        scores.extend(
            float(result["confidence"])
            for result in iter_inference_batch(
                model=model,
                clean_batch=batch[0],
                aug_batch=batch[1],
                device=args.device,
                mask_threshold=args.mask_threshold,
            )
        )
    return scores


def select_subset_indices(dataset: PretrainDataset, subset_image_count: int, seed: int) -> tuple[List[int], int]:
    sample_path_indices = dataset.sample_path_indices.numpy()
    unique_path_indices = np.unique(sample_path_indices)
    rng = np.random.default_rng(seed)
    chosen_size = min(subset_image_count, len(unique_path_indices))
    chosen_path_indices = rng.choice(unique_path_indices, size=chosen_size, replace=False)
    subset_indices = np.flatnonzero(np.isin(sample_path_indices, chosen_path_indices)).tolist()
    return subset_indices, int(chosen_size)


def compute_threshold(scores: Sequence[float], keep_top_ratio: float) -> float:
    if not scores:
        raise RuntimeError("No subset scores were computed, cannot estimate threshold.")
    return float(np.quantile(np.asarray(scores, dtype=np.float32), 1.0 - keep_top_ratio))


def summarize_scores(scores: Sequence[float]) -> Dict[str, float]:
    if not scores:
        raise RuntimeError("No scores were provided, cannot summarize distribution.")

    scores_np = np.asarray(scores, dtype=np.float32)
    return {
        "count": float(scores_np.size),
        "mean": float(np.mean(scores_np)),
        "std": float(np.std(scores_np)),
        "min": float(np.min(scores_np)),
        "p10": float(np.quantile(scores_np, 0.10)),
        "p25": float(np.quantile(scores_np, 0.25)),
        "median": float(np.quantile(scores_np, 0.50)),
        "p75": float(np.quantile(scores_np, 0.75)),
        "p90": float(np.quantile(scores_np, 0.90)),
        "max": float(np.max(scores_np)),
    }


def make_mask_path(image_path: str, label: Dict[str, str]) -> Path:
    image_path_obj = Path(image_path)
    split_dir = image_path_obj.parent
    mask_dir = split_dir.parent / f"{split_dir.name}_mask"

    level_token = str(label["level"]).strip().replace(" ", "+")
    object_token = str(label["object"]).strip().replace(" ", "+")
    suffix = level_token if not object_token else f"{level_token}+{object_token}"
    return mask_dir / f"{image_path_obj.stem}_{suffix}.png"


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(mask).save(path)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def generate_pseudo_labels(
    dataset: PretrainDataset,
    indices: Sequence[int],
    model: torch.nn.Module,
    clean_transform: ModelInputTransform,
    aug_transform: ModelInputTransform,
    threshold: float,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, float]:
    # output_dir.mkdir(parents=True, exist_ok=True)

    kept_count = 0
    rejected_count = 0
    mask_saved_count = 0
    confidence_sum = 0.0

    total_batches = math.ceil(len(indices) / args.batch_size) if indices else 0
    for batch_indices in tqdm(
        batched(indices, args.batch_size),
        total=total_batches,
        desc="Label full set",
    ):
        batch = build_model_batches(dataset, batch_indices, clean_transform, aug_transform)
        for result in iter_inference_batch(
            model=model,
            clean_batch=batch[0],
            aug_batch=batch[1],
            device=args.device,
            mask_threshold=args.mask_threshold,
        ):
            confidence = float(result["confidence"])
            foreground_area = int(result["foreground_area"])
            keep = confidence >= threshold and foreground_area > 0

            if keep:
                mask_path_obj = make_mask_path(
                    image_path=str(result["image_path"]),
                    label=result["label"],
                )
                save_mask(result["mask"], mask_path_obj)
                kept_count += 1
                mask_saved_count += 1
            else:
                rejected_count += 1

            confidence_sum += confidence

    total = len(indices)
    return {
        "num_samples": total,
        "num_kept": kept_count,
        "num_rejected": rejected_count,
        "num_masks_saved": mask_saved_count,
        "mean_confidence": confidence_sum / max(total, 1),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    clean_transform, aug_transform = make_transforms(args)
    generate_dataset_name = args.generate_dataset_name or args.dataset_name
    generate_dataset = build_pretrain_dataset(generate_dataset_name)
    generate_indices = build_indices(generate_dataset, args.limit_samples)

    model = load_model(args)

    subset_score_stats = None
    subset_indices: List[int] = []
    subset_image_count = 0
    if args.threshold is None:
        threshold_dataset = build_pretrain_dataset(args.dataset_name)
        threshold_indices = build_indices(threshold_dataset, args.limit_samples)
        subset_indices, subset_image_count = select_subset_indices(
            dataset=threshold_dataset,
            subset_image_count=args.subset_image_count,
            seed=args.seed,
        )
        if args.limit_samples > 0:
            threshold_index_set = set(threshold_indices)
            subset_indices = [idx for idx in subset_indices if idx in threshold_index_set]

        subset_scores = collect_scores(
            dataset=threshold_dataset,
            indices=subset_indices,
            model=model,
            clean_transform=clean_transform,
            aug_transform=aug_transform,
            args=args,
            desc="Score subset",
        )
        subset_score_stats = summarize_scores(subset_scores)
        print(
            "Subset score distribution: "
            "count={count:.0f}, mean={mean:.6f}, std={std:.6f}, min={min:.6f}, "
            "p10={p10:.6f}, p25={p25:.6f}, median={median:.6f}, "
            "p75={p75:.6f}, p90={p90:.6f}, max={max:.6f}".format(
                **subset_score_stats
            )
        )
        threshold = compute_threshold(subset_scores, keep_top_ratio=args.keep_top_ratio)
        threshold_source = "estimated"
    else:
        threshold = float(args.threshold)
        threshold_source = "provided"
        print(f"Using provided threshold: {threshold:.6f}")

    print(
        "Generate dataset: {dataset_name} -> {count} samples".format(
            dataset_name=generate_dataset_name,
            count=len(generate_indices),
        )
    )

    summary = generate_pseudo_labels(
        dataset=generate_dataset,
        indices=generate_indices,
        model=model,
        clean_transform=clean_transform,
        aug_transform=aug_transform,
        threshold=threshold,
        args=args,
        output_dir=output_dir,
    )

    threshold_payload = {
        "threshold_source": threshold_source,
        "dataset_name": args.dataset_name,
        "generate_dataset_name": generate_dataset_name,
        "subset_image_count_requested": args.subset_image_count,
        "subset_image_count_used": subset_image_count if args.threshold is None else 0,
        "subset_sample_count_used": len(subset_indices) if args.threshold is None else 0,
        "generate_sample_count_used": len(generate_indices),
        "keep_top_ratio": args.keep_top_ratio,
        "threshold": threshold,
    }
    if subset_score_stats is not None:
        threshold_payload.update(
            {
                "subset_score_count": subset_score_stats["count"],
                "subset_score_mean": subset_score_stats["mean"],
                "subset_score_std": subset_score_stats["std"],
                "subset_score_min": subset_score_stats["min"],
                "subset_score_p10": subset_score_stats["p10"],
                "subset_score_p25": subset_score_stats["p25"],
                "subset_score_median": subset_score_stats["median"],
                "subset_score_p75": subset_score_stats["p75"],
                "subset_score_p90": subset_score_stats["p90"],
                "subset_score_max": subset_score_stats["max"],
            }
        )

    write_json(output_dir / "threshold_selection.json", threshold_payload)
    write_json(
        output_dir / "summary.json",
        {
            **summary,
            **threshold_payload,
            "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
            "device": args.device,
            "resolution": args.resolution,
            "mask_threshold": args.mask_threshold,
            "seed": args.seed,
            "augmentation": {
                "type": "HedColorAugmenter1",
                "thresh": args.hed_thresh,
                "seed": args.hed_seed,
            },
        },
    )

    print(f"Pseudo-label summaries written under: {output_dir}")
    print(f"Threshold ({threshold_source}): {threshold:.6f}")
    print(
        "Kept {num_kept}/{num_samples} samples, saved {num_masks_saved} masks.".format(
            **summary
        )
    )


if __name__ == "__main__":
    main()
