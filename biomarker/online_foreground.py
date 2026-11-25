from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

try:
    from . import utils
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from biomarker import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a lightweight online foreground mask for one WSI.")
    parser.add_argument("--slide", type=Path, required=True)
    parser.add_argument("--output-mask", type=Path, required=True)
    parser.add_argument("--output-overlay", type=Path, default=None)
    parser.add_argument("--downsample", type=int, default=32)
    parser.add_argument("--saturation-threshold", type=int, default=8)
    parser.add_argument("--use-otsu", action="store_true")
    parser.add_argument("--median-size", type=int, default=7)
    parser.add_argument("--close-size", type=int, default=4)
    parser.add_argument("--min-object-size", type=int, default=64)
    parser.add_argument("--hole-size", type=int, default=64)
    parser.add_argument("--value-lower", type=int, default=20)
    parser.add_argument("--value-upper", type=int, default=245)
    parser.add_argument("--suppress-markers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--marker-dilation", type=int, default=3)
    return parser.parse_args()


def otsu_threshold_u8(values: np.ndarray) -> int:
    values = values.astype(np.uint8, copy=False)
    hist = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0
    bins = np.arange(256, dtype=np.float64)
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    mean_bg = np.cumsum(hist * bins) / np.maximum(weight_bg, 1.0)
    mean_fg = (np.cumsum((hist * bins)[::-1]) / np.maximum(np.cumsum(hist[::-1]), 1.0))[::-1]
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    return int(np.argmax(between))


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 0:
        return mask.astype(bool, copy=False)
    labels, count = ndi.label(mask)
    if count == 0:
        return mask.astype(bool, copy=False)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(min_size)
    keep[0] = False
    return keep[labels]


def fill_small_holes(mask: np.ndarray, max_hole_size: int) -> np.ndarray:
    if max_hole_size <= 0:
        return mask.astype(bool, copy=False)
    filled = ndi.binary_fill_holes(mask)
    holes = filled & ~mask
    labels, count = ndi.label(holes)
    if count == 0:
        return mask.astype(bool, copy=False)
    sizes = np.bincount(labels.ravel())
    fill = sizes <= int(max_hole_size)
    fill[0] = False
    return mask | fill[labels]


def marker_mask_from_hsv(
    hsv: np.ndarray,
    saturation_min: int = 90,
    value_min: int = 45,
) -> np.ndarray:
    hue = hsv[..., 0].astype(np.float32) * (360.0 / 255.0)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    vivid = (saturation >= saturation_min) & (value >= value_min)

    green = (hue >= 70) & (hue <= 170)
    blue = (hue >= 185) & (hue <= 255) & (saturation >= 120)
    red = ((hue <= 15) | (hue >= 345)) & (saturation >= 120)
    return vivid & (green | blue | red)


def build_foreground_mask(
    image: Image.Image | np.ndarray,
    saturation_threshold: int = 8,
    use_otsu: bool = False,
    median_size: int = 7,
    close_size: int = 4,
    min_object_size: int = 64,
    hole_size: int = 64,
    value_lower: int = 20,
    value_upper: int = 245,
    suppress_markers: bool = True,
    marker_dilation: int = 3,
) -> np.ndarray:
    if isinstance(image, np.ndarray):
        rgb = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    else:
        rgb = image.convert("RGB")
    hsv = np.array(rgb.convert("HSV"), dtype=np.uint8)
    saturation = hsv[..., 1]
    value = hsv[..., 2]

    if median_size > 1:
        saturation_for_threshold = ndi.median_filter(saturation, size=int(median_size))
    else:
        saturation_for_threshold = saturation

    threshold = otsu_threshold_u8(saturation_for_threshold) if use_otsu else int(saturation_threshold)
    threshold = max(int(saturation_threshold), int(threshold))
    mask = (saturation_for_threshold > threshold) & (value > int(value_lower)) & (value < int(value_upper))

    if close_size > 1:
        mask = ndi.binary_closing(mask, structure=np.ones((int(close_size), int(close_size)), dtype=bool))
    mask = fill_small_holes(mask, int(hole_size))
    mask = remove_small_components(mask, int(min_object_size))

    if suppress_markers:
        markers = marker_mask_from_hsv(hsv)
        if marker_dilation > 0:
            markers = ndi.binary_dilation(
                markers,
                structure=np.ones((int(marker_dilation), int(marker_dilation)), dtype=bool),
            )
        mask = mask & ~markers

    return mask.astype(bool, copy=False)


def build_foreground_mask_from_slide(
    slide,
    downsample: int = 32,
    **kwargs,
) -> np.ndarray:
    slide_w, slide_h = map(int, slide.dimensions)
    mask_size = (
        max(1, int(math.ceil(slide_w / float(downsample)))),
        max(1, int(math.ceil(slide_h / float(downsample)))),
    )
    thumbnail = slide.get_thumbnail(mask_size).convert("RGB").resize(mask_size, utils.RESAMPLE_BILINEAR)
    return build_foreground_mask(thumbnail, **kwargs)


def save_mask_overlay(mask: np.ndarray, thumbnail: Image.Image, output_path: Path, alpha: float = 0.45) -> None:
    base = thumbnail.convert("RGB").resize((mask.shape[1], mask.shape[0]), utils.RESAMPLE_BILINEAR)
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    mask_rgb[mask.astype(bool)] = (30, 180, 90)
    overlay = Image.blend(base, Image.fromarray(mask_rgb), alpha)
    arr = np.array(base)
    fg = mask.astype(bool)
    arr[fg] = np.array(overlay)[fg]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(output_path)


def main() -> None:
    args = parse_args()
    import openslide

    with openslide.OpenSlide(str(args.slide)) as slide:
        mask = build_foreground_mask_from_slide(
            slide,
            downsample=args.downsample,
            saturation_threshold=args.saturation_threshold,
            use_otsu=args.use_otsu,
            median_size=args.median_size,
            close_size=args.close_size,
            min_object_size=args.min_object_size,
            hole_size=args.hole_size,
            value_lower=args.value_lower,
            value_upper=args.value_upper,
            suppress_markers=args.suppress_markers,
            marker_dilation=args.marker_dilation,
        )
        args.output_mask.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((mask.astype(np.uint8) * 255)).save(args.output_mask)
        if args.output_overlay is not None:
            thumbnail = slide.get_thumbnail((mask.shape[1], mask.shape[0]))
            save_mask_overlay(mask, thumbnail, args.output_overlay)


if __name__ == "__main__":
    main()
