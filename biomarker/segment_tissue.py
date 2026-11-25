from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

try:
    from . import online_foreground, utils
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from biomarker import online_foreground, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-2 text-prompt tissue segmentation on TCGA WSIs.")
    parser.add_argument("--cohort", default="BLCA", help="TCGA cohort, comma list, or all.")
    parser.add_argument("--tcga-root", type=Path, default=utils.TCGA_ROOT)
    parser.add_argument("--histoqc-root", type=Path, default=utils.HISTOQC_ROOT)
    parser.add_argument("--output-root", type=Path, default=utils.PATCH_MASK_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=utils.CHECKPOINT_PATH)
    parser.add_argument("--bpe-path", type=Path, default=utils.BPE_PATH)
    parser.add_argument("--target-mpp", type=float, default=0.25)
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--context-size", type=int, default=None)
    parser.add_argument("--tissue-threshold", type=float, default=0.2)
    parser.add_argument("--prob-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-resolution", type=int, default=1008)
    parser.add_argument("--foreground-source", choices=("online", "histoqc", "none"), default="online")
    parser.add_argument("--foreground-downsample", type=int, default=32)
    parser.add_argument("--foreground-saturation-threshold", type=int, default=8)
    parser.add_argument("--foreground-use-otsu", action="store_true")
    parser.add_argument("--foreground-median-size", type=int, default=7)
    parser.add_argument("--foreground-close", type=int, default=4)
    parser.add_argument("--foreground-min-object-size", type=int, default=64)
    parser.add_argument("--foreground-hole-size", type=int, default=64)
    parser.add_argument("--foreground-value-lower", type=int, default=20)
    parser.add_argument("--foreground-value-upper", type=int, default=245)
    parser.add_argument("--foreground-suppress-markers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--foreground-marker-dilation", type=int, default=3)
    parser.add_argument("--slide-id", default="", help="Optional exact/substring slide id filter.")
    parser.add_argument("--limit-slides", type=int, default=0)
    parser.add_argument("--limit-patches", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve_cohorts(value: str, tcga_root: Path) -> list[str]:
    if value.lower() == "all":
        return sorted(path.name for path in tcga_root.iterdir() if path.is_dir())
    return [item.strip() for item in value.split(",") if item.strip()]


def shard_items(items: Sequence[Path], shard_index: int, shard_count: int) -> list[Path]:
    if shard_count <= 0:
        raise ValueError("--shard-count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index must be in [0, shard_count)")
    return [item for idx, item in enumerate(items) if idx % shard_count == shard_index]


def load_model(args: argparse.Namespace):
    sys.path.insert(0, str(utils.SAM3_ROOT))
    import torch
    from sam3.model_builder import build_pathmarker_semantic_model

    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device) if ":" in args.device else torch.cuda.current_device())
    model = build_pathmarker_semantic_model(
        bpe_path=str(args.bpe_path),
        device=args.device,
        eval_mode=True,
        checkpoint_path=str(args.checkpoint),
        load_from_HF=False,
        enable_segmentation=True,
    )
    model.to(args.device).eval()
    torch.set_grad_enabled(False)
    return model


def resolve_context_size(patch_size: int, context_size: int | None) -> int:
    if context_size is None:
        return int(patch_size + 256)
    if context_size <= 0:
        return int(patch_size)
    if context_size < patch_size:
        raise ValueError("--context-size must be greater than or equal to --patch-size")
    if (context_size - patch_size) % 2 != 0:
        raise ValueError("--context-size minus --patch-size must be an even number")
    return int(context_size)


def make_find_stage(num_prompts: int, device: str, num_images: int = 1):
    import torch

    if str(utils.SAM3_ROOT) not in sys.path:
        sys.path.insert(0, str(utils.SAM3_ROOT))
    from sam3.model.data_misc import FindStage

    total_prompts = int(num_prompts) * int(num_images)
    img_ids = torch.arange(num_images, device=device, dtype=torch.long).repeat_interleave(num_prompts)
    text_ids = torch.arange(num_prompts, device=device, dtype=torch.long).repeat(num_images)
    return FindStage(
        img_ids=img_ids,
        text_ids=text_ids,
        input_boxes=torch.zeros(0, total_prompts, 4, device=device),
        input_boxes_mask=torch.ones(total_prompts, 0, device=device, dtype=torch.bool),
        input_boxes_label=torch.zeros(0, total_prompts, device=device, dtype=torch.long),
        input_points=torch.empty(total_prompts, 0, 3, device=device),
        input_points_mask=torch.ones(total_prompts, 0, device=device, dtype=torch.bool),
    )


def preprocess_image(image: Image.Image, resolution: int, device: str):
    import torch
    import torch.nn.functional as F

    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    tensor = F.interpolate(tensor, size=(resolution, resolution), mode="bilinear", align_corners=False)
    return (tensor - 0.5) / 0.5


def preprocess_images(images: Sequence[Image.Image], resolution: int, device: str):
    import torch
    import torch.nn.functional as F

    tensors = []
    for image in images:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        tensors.append(tensor)
    batch = torch.stack(tensors, dim=0).to(device)
    batch = F.interpolate(batch, size=(resolution, resolution), mode="bilinear", align_corners=False)
    return (batch - 0.5) / 0.5


class PromptSegmentor:
    def __init__(self, model, prompts: Sequence[str], device: str, resolution: int = 1008):
        self.model = model
        self.prompts = list(prompts)
        self.device = device
        self.resolution = resolution
        self.find_stage = make_find_stage(len(prompts), device)
        self.geometric_prompt = model._get_dummy_prompt(num_prompts=len(prompts))
        self._find_stage_cache = {1: self.find_stage}
        self._geometric_prompt_cache = {len(prompts): self.geometric_prompt}
        import torch

        with torch.inference_mode():
            self.text_features = self.model.backbone.forward_text(self.prompts, device=self.device)

    def _find_stage_for_batch(self, batch_size: int):
        if batch_size not in self._find_stage_cache:
            self._find_stage_cache[batch_size] = make_find_stage(len(self.prompts), self.device, num_images=batch_size)
        return self._find_stage_cache[batch_size]

    def _geometric_prompt_for_batch(self, batch_size: int):
        total_prompts = batch_size * len(self.prompts)
        if total_prompts not in self._geometric_prompt_cache:
            self._geometric_prompt_cache[total_prompts] = self.model._get_dummy_prompt(num_prompts=total_prompts)
        return self._geometric_prompt_cache[total_prompts]

    def predict_probabilities(self, image: Image.Image) -> np.ndarray:
        return self.predict_probabilities_batch([image])[0]

    def predict_probabilities_batch(self, images: Sequence[Image.Image]) -> list[np.ndarray]:
        import torch
        import torch.nn.functional as F

        if not images:
            return []
        with torch.inference_mode():
            image_tensor = preprocess_images(images, self.resolution, self.device)
            backbone_out = self.model.backbone.forward_image(image_tensor)
            backbone_out.update(dict(self.text_features))
            batch_size = len(images)
            out = self.model.forward_grounding(
                backbone_out=backbone_out,
                find_input=self._find_stage_for_batch(batch_size),
                find_target=None,
                geometric_prompt=self._geometric_prompt_for_batch(batch_size),
            )
            logits = out["semantic_seg"]
            if logits.ndim == 4 and logits.shape[1] == 1:
                logits = logits[:, 0]
            logits = F.interpolate(
                logits[:, None],
                size=(images[0].height, images[0].width),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            probs = logits.sigmoid().detach().cpu().numpy().astype(np.float32)
            return [probs[i * len(self.prompts) : (i + 1) * len(self.prompts)] for i in range(batch_size)]


def read_region_with_white_padding(
    slide,
    x0: int,
    y0: int,
    width: int,
    height: int,
    output_size: int,
    source_mpp: float | None = None,
    target_mpp: float | None = None,
) -> Image.Image:
    slide_w, slide_h = map(int, slide.dimensions)
    x1 = x0 + width
    y1 = y0 + height
    ix0 = max(0, x0)
    iy0 = max(0, y0)
    ix1 = min(slide_w, x1)
    iy1 = min(slide_h, y1)

    canvas = Image.new("RGB", (output_size, output_size), "white")
    if ix1 <= ix0 or iy1 <= iy0:
        return canvas

    dst_x0 = int(round((ix0 - x0) * output_size / float(width)))
    dst_y0 = int(round((iy0 - y0) * output_size / float(height)))
    dst_x1 = int(round((ix1 - x0) * output_size / float(width)))
    dst_y1 = int(round((iy1 - y0) * output_size / float(height)))
    dst_w = max(1, dst_x1 - dst_x0)
    dst_h = max(1, dst_y1 - dst_y0)

    level = 0
    level_downsample = 1.0
    if source_mpp is not None and target_mpp is not None and source_mpp > 0 and target_mpp > 0:
        desired_downsample = float(target_mpp) / float(source_mpp)
        get_best_level = getattr(slide, "get_best_level_for_downsample", None)
        level_downsamples = getattr(slide, "level_downsamples", None)
        if callable(get_best_level) and level_downsamples is not None and len(level_downsamples) > 0:
            level = int(get_best_level(desired_downsample))
            level = max(0, min(level, len(level_downsamples) - 1))
            next_level = level + 1
            if (
                next_level < len(level_downsamples)
                and float(source_mpp) * float(level_downsamples[next_level]) <= float(target_mpp) * 1.02
            ):
                level = next_level
            level_downsample = float(level_downsamples[level])

    read_width = max(1, int(math.ceil((ix1 - ix0) / level_downsample)))
    read_height = max(1, int(math.ceil((iy1 - iy0) / level_downsample)))
    region = slide.read_region((ix0, iy0), level, (read_width, read_height)).convert("RGB")
    region = region.resize((dst_w, dst_h), utils.RESAMPLE_BILINEAR)
    canvas.paste(region, (dst_x0, dst_y0))
    return canvas


def read_patch(slide, geom: utils.PatchGeometry, context_size: int | None = None) -> Image.Image:
    context_size = geom.target_size if context_size is None else resolve_context_size(geom.target_size, context_size)
    if context_size == geom.target_size:
        return read_region_with_white_padding(
            slide,
            x0=geom.x0,
            y0=geom.y0,
            width=geom.width,
            height=geom.height,
            output_size=geom.target_size,
        )

    margin_target = (context_size - geom.target_size) // 2
    margin_x = int(round(margin_target * geom.width / float(geom.target_size)))
    margin_y = int(round(margin_target * geom.height / float(geom.target_size)))
    return read_region_with_white_padding(
        slide,
        x0=geom.x0 - margin_x,
        y0=geom.y0 - margin_y,
        width=geom.width + 2 * margin_x,
        height=geom.height + 2 * margin_y,
        output_size=context_size,
    )


def crop_context_probabilities(probs: np.ndarray, patch_size: int) -> np.ndarray:
    if probs.ndim != 3:
        raise ValueError(f"Expected [classes,H,W] probabilities, got {probs.shape}")
    height, width = probs.shape[1:]
    if height == patch_size and width == patch_size:
        return probs
    if height < patch_size or width < patch_size:
        raise ValueError(f"Cannot crop probabilities of shape {probs.shape} to patch_size={patch_size}")
    top = (height - patch_size) // 2
    left = (width - patch_size) // 2
    return probs[:, top : top + patch_size, left : left + patch_size]


def load_foreground_mask(slide, slide_id: str, cohort: str, args: argparse.Namespace) -> tuple[np.ndarray | None, dict]:
    source = args.foreground_source
    if source == "histoqc":
        mask_path = utils.find_histoqc_mask(args.histoqc_root / cohort, slide_id)
        if mask_path is None:
            return None, {"foreground_source": source, "status": "skip_no_histoqc_mask"}
        return utils.load_binary_mask(mask_path), {
            "foreground_source": source,
            "foreground_mask_path": str(mask_path),
            "histoqc_mask_path": str(mask_path),
        }

    if source == "none":
        slide_w, slide_h = map(int, slide.dimensions)
        downsample = int(args.foreground_downsample)
        mask = np.ones(
            (
                max(1, int(np.ceil(slide_h / float(downsample)))),
                max(1, int(np.ceil(slide_w / float(downsample)))),
            ),
            dtype=bool,
        )
        return mask, {"foreground_source": source, "foreground_downsample": downsample}

    mask = online_foreground.build_foreground_mask_from_slide(
        slide,
        downsample=int(args.foreground_downsample),
        saturation_threshold=int(args.foreground_saturation_threshold),
        use_otsu=bool(args.foreground_use_otsu),
        median_size=int(args.foreground_median_size),
        close_size=int(args.foreground_close),
        min_object_size=int(args.foreground_min_object_size),
        hole_size=int(args.foreground_hole_size),
        value_lower=int(args.foreground_value_lower),
        value_upper=int(args.foreground_value_upper),
        suppress_markers=bool(args.foreground_suppress_markers),
        marker_dilation=int(args.foreground_marker_dilation),
    )
    return mask, {
        "foreground_source": source,
        "foreground_downsample": int(args.foreground_downsample),
        "foreground_saturation_threshold": int(args.foreground_saturation_threshold),
        "foreground_use_otsu": bool(args.foreground_use_otsu),
        "foreground_median_size": int(args.foreground_median_size),
        "foreground_close": int(args.foreground_close),
        "foreground_min_object_size": int(args.foreground_min_object_size),
        "foreground_hole_size": int(args.foreground_hole_size),
        "foreground_value_lower": int(args.foreground_value_lower),
        "foreground_value_upper": int(args.foreground_value_upper),
        "foreground_suppress_markers": bool(args.foreground_suppress_markers),
        "foreground_marker_dilation": int(args.foreground_marker_dilation),
    }


def save_patch_outputs(
    output_dir: Path,
    cohort: str,
    slide_path: Path,
    foreground_metadata: dict,
    geom: utils.PatchGeometry,
    slide_size: tuple[int, int],
    source_mpp: float,
    magnification: float | None,
    target_mpp: float,
    tissue_fraction: float,
    prob_threshold: float,
    context_size: int,
    class_mask: np.ndarray,
) -> None:
    name = utils.relative_patch_name(geom.x0, geom.y0)
    out_mask = output_dir / f"{name}.png"
    tmp_mask = output_dir / f".{name}.png.tmp"
    try:
        Image.fromarray(class_mask).save(tmp_mask, format="PNG")
        tmp_mask.replace(out_mask)
    except OSError as exc:
        tmp_mask.unlink(missing_ok=True)
        raise OSError(
            f"Failed to save patch mask to {out_mask}. "
            f"Check output filesystem space/quota/permissions."
        ) from exc
    utils.write_patch_metadata(
        output_dir / f"{name}.json",
        {
            "cohort": cohort,
            "slide_id": slide_path.name,
            "slide_path": str(slide_path),
            "mask_path": str(out_mask),
            "x0": geom.x0,
            "y0": geom.y0,
            "width": geom.width,
            "height": geom.height,
            "patch_size": geom.target_size,
            "slide_width": slide_size[0],
            "slide_height": slide_size[1],
            "source_mpp": source_mpp,
            "magnification": magnification,
            "target_mpp": target_mpp,
            "level0_to_target_scale": geom.scale,
            "tissue_fraction": tissue_fraction,
            "prob_threshold": prob_threshold,
            "context_size": context_size,
            "context_margin": (context_size - geom.target_size) // 2,
            "class_names": list(utils.PROMPTS),
            **foreground_metadata,
        },
    )


def process_slide(
    slide_path: Path,
    cohort: str,
    segmentor: PromptSegmentor,
    args: argparse.Namespace,
) -> dict:
    import openslide

    slide_id = utils.slide_id_from_path(slide_path)

    context_size = resolve_context_size(args.patch_size, args.context_size)

    with openslide.OpenSlide(str(slide_path)) as slide:
        magnification = utils.get_slide_magnification(slide)
        if magnification is None:
            return {"slide_id": slide_id, "status": "skip_no_magnification", "patches": 0}
        source_mpp = utils.get_slide_mpp(slide)
        if source_mpp is None:
            source_mpp = utils.infer_mpp_from_magnification(magnification)
        if source_mpp is None:
            return {"slide_id": slide_id, "status": "skip_no_mpp", "patches": 0}
        slide_size = tuple(map(int, slide.dimensions))
        valid_mask, foreground_metadata = load_foreground_mask(slide, slide_id, cohort, args)
        if valid_mask is None:
            return {"slide_id": slide_id, "patches": 0, **foreground_metadata}
        out_dir = args.output_root / cohort / slide_id
        out_dir.mkdir(parents=True, exist_ok=True)

        kept = 0
        skipped = 0
        for geom in utils.iter_patch_geometries(
            slide_width=slide_size[0],
            slide_height=slide_size[1],
            patch_size=args.patch_size,
            source_mpp=source_mpp,
            target_mpp=args.target_mpp,
        ):
            name = utils.relative_patch_name(geom.x0, geom.y0)
            if args.resume and (out_dir / f"{name}.json").exists():
                kept += 1
                continue
            frac = utils.tissue_fraction_for_patch(
                valid_mask, slide_size, geom.x0, geom.y0, geom.width, geom.height
            )
            if frac < args.tissue_threshold:
                skipped += 1
                continue
            patch = read_patch(slide, geom, context_size=context_size)
            probs = segmentor.predict_probabilities(patch)
            probs = crop_context_probabilities(probs, patch_size=args.patch_size)
            class_mask = utils.combine_prompt_probabilities(probs, threshold=args.prob_threshold)
            save_patch_outputs(
                output_dir=out_dir,
                cohort=cohort,
                slide_path=slide_path,
                foreground_metadata=foreground_metadata,
                geom=geom,
                slide_size=slide_size,
                source_mpp=source_mpp,
                magnification=magnification,
                target_mpp=args.target_mpp,
                tissue_fraction=frac,
                prob_threshold=args.prob_threshold,
                context_size=context_size,
                class_mask=class_mask,
            )
            kept += 1
            if args.limit_patches and kept >= args.limit_patches:
                break

    return {"slide_id": slide_id, "status": "ok", "patches": kept, "skipped_low_tissue": skipped}


def main() -> None:
    args = parse_args()
    cohorts = resolve_cohorts(args.cohort, args.tcga_root)
    segmentor = PromptSegmentor(load_model(args), utils.PROMPTS, args.device, resolution=args.model_resolution)

    for cohort in cohorts:
        slide_paths = utils.select_slide_paths(
            list(utils.iter_slide_paths(cohort, args.tcga_root)),
            slide_id=args.slide_id,
            limit=args.limit_slides,
        )
        slide_paths = shard_items(slide_paths, args.shard_index, args.shard_count)
        for index, slide_path in enumerate(slide_paths, start=1):
            result = process_slide(slide_path, cohort, segmentor, args)
            print(
                f"[{cohort} shard {args.shard_index}/{args.shard_count} "
                f"{index}/{len(slide_paths)}] {result}"
            )


if __name__ == "__main__":
    main()
