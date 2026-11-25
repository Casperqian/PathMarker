from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image


TCGA_ROOT = Path("/jhcnas7/Pathology/original_data/TCGA")
HISTOQC_ROOT = Path("/jhcnas5/zq/data/PathSeg2/biomarker")
PATCH_MASK_ROOT = Path("/jhcnas5/zq/data/PathSeg2/biomarker/stage2_patch_mask")
SLIDE_MASK_ROOT = Path("/jhcnas5/zq/data/PathSeg2/biomarker/stage2_slide_mask")

CHECKPOINT_PATH = Path(
    "/jhcnas5/zq/code/ckpt/finetune/ours/exps/siglip_labeled_ft/checkpoints/best.pt"
)
SAM3_ROOT = Path(__file__).resolve().parents[1] / "sam3"
BPE_PATH = SAM3_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"

PROMPTS = ("tumor", "epithelial tissue", "stroma", "blood", "necrosis")
SLIDE_SUFFIXES = frozenset({".svs", ".kfb", ".tif", ".tiff"})

try:
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE_NEAREST = Image.NEAREST
    RESAMPLE_BILINEAR = Image.BILINEAR


@dataclass(frozen=True)
class PatchGeometry:
    x0: int
    y0: int
    width: int
    height: int
    target_size: int
    scale: float


def _resolve_manifest_slide_source(
    stem: str,
    candidates: Sequence[Path],
    selected_source: str,
) -> Path:
    selected_path = Path(selected_source)
    matches = [
        candidate
        for candidate in candidates
        if candidate.name == selected_source
        or candidate == selected_path
        or (selected_path.is_absolute() and candidate.resolve() == selected_path.resolve())
    ]
    if len(matches) == 1:
        return matches[0]
    available = ", ".join(str(candidate) for candidate in candidates)
    raise ValueError(
        f"Source manifest selection for slide stem {stem!r} did not match exactly one candidate: "
        f"{selected_source!r}; available: {available}"
    )


def _select_slide_source(
    stem: str,
    candidates: Sequence[Path],
    source_manifest: Mapping[str, str] | None,
) -> Path:
    if source_manifest and stem in source_manifest:
        return _resolve_manifest_slide_source(stem, candidates, source_manifest[stem])
    if len(candidates) == 1:
        return candidates[0]

    svs_candidates = [candidate for candidate in candidates if candidate.suffix.lower() == ".svs"]
    if len(svs_candidates) == 1:
        return svs_candidates[0]

    available = ", ".join(str(candidate) for candidate in candidates)
    raise ValueError(
        f"Ambiguous source files for slide stem {stem!r}; expected one candidate or one .svs: {available}. "
        "Use a source manifest to select the intended file."
    )


def iter_slide_paths(
    cohort: str,
    tcga_root: Path = TCGA_ROOT,
    source_manifest: Mapping[str, str] | None = None,
) -> Iterable[Path]:
    cohort_root = tcga_root / cohort
    roots = [cohort_root / "slides", cohort_root]
    seen: set[Path] = set()
    candidates_by_stem: dict[str, list[Path]] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in SLIDE_SUFFIXES or path in seen:
                continue
            seen.add(path)
            candidates_by_stem.setdefault(path.stem, []).append(path)

    for stem, candidates in sorted(candidates_by_stem.items()):
        yield _select_slide_source(stem, candidates, source_manifest)


def slide_id_from_path(slide_path: Path) -> str:
    return slide_path.name


def case_id_from_slide_path(slide_path: Path) -> str:
    stem = slide_path.stem
    parts = stem.split("-")
    if stem.startswith("TCGA-") and len(parts) >= 3:
        return "-".join(parts[:3])
    if stem.startswith(("C3N-", "C3L-")) and len(parts) >= 2:
        return "-".join(parts[:2])
    return stem.split("_", 1)[0]


def prioritize_slide_paths_by_case(
    selected: Sequence[tuple[str, Path]],
    completed: Sequence[tuple[str, Path]],
) -> list[tuple[str, Path]]:
    covered = {
        (cohort, case_id_from_slide_path(slide_path))
        for cohort, slide_path in completed
    }
    first_pass: list[tuple[str, Path]] = []
    remaining: list[tuple[str, Path]] = []
    for item in selected:
        cohort, slide_path = item
        case_key = (cohort, case_id_from_slide_path(slide_path))
        if case_key in covered:
            remaining.append(item)
            continue
        covered.add(case_key)
        first_pass.append(item)
    return first_pass + remaining


def slide_id_matches(path_or_name: Path | str, slide_id: str | None) -> bool:
    if not slide_id:
        return True
    path = Path(path_or_name)
    value = str(slide_id)
    candidates = (path.name, path.stem, str(path))
    return any(value == candidate or value in candidate for candidate in candidates)


def select_slide_paths(
    slide_paths: Sequence[Path],
    slide_id: str | None = "",
    limit: int = 0,
    random_sample: bool = False,
    seed: int = 0,
) -> list[Path]:
    selected = [path for path in slide_paths if slide_id_matches(path, slide_id)]
    if limit <= 0:
        return selected
    if not random_sample:
        return selected[:limit]
    rng = random.Random(seed)
    if limit >= len(selected):
        shuffled = list(selected)
        rng.shuffle(shuffled)
        return shuffled
    return rng.sample(selected, limit)


def select_slide_paths_by_cohort(
    cohort_slide_paths: dict[str, Sequence[Path]],
    slide_id: str | None = "",
    total_limit: int = 0,
    random_sample: bool = False,
    seed: int = 0,
) -> list[tuple[str, Path]]:
    cohorts = list(cohort_slide_paths)
    if not cohorts:
        return []

    if total_limit > 0:
        per_cohort = total_limit // len(cohorts)
        extras = total_limit % len(cohorts)
    else:
        per_cohort = 0
        extras = 0

    selected: list[tuple[str, Path]] = []
    for idx, cohort in enumerate(cohorts):
        limit = 0
        if total_limit > 0:
            limit = per_cohort + (1 if idx < extras else 0)
            if limit <= 0:
                continue
        cohort_selected = select_slide_paths(
            cohort_slide_paths[cohort],
            slide_id=slide_id,
            limit=limit,
            random_sample=random_sample and not bool(slide_id),
            seed=seed + idx,
        )
        selected.extend((cohort, path) for path in cohort_selected)
    return selected


def find_histoqc_mask(cohort_mask_root: Path, slide_id: str) -> Path | None:
    slide_dir = cohort_mask_root / slide_id
    candidates: list[Path] = []
    if slide_dir.exists():
        candidates.extend(sorted(slide_dir.glob("*_mask_use.png")))
        candidates.extend(sorted(slide_dir.glob("*mask_use*.png")))
    if not candidates:
        candidates.extend(sorted(cohort_mask_root.glob(f"**/{slide_id}_mask_use.png")))
        candidates.extend(sorted(cohort_mask_root.glob(f"**/{slide_id}*mask_use*.png")))
    return candidates[0] if candidates else None


def load_binary_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 0


def get_slide_mpp(slide) -> float | None:
    props = getattr(slide, "properties", {})
    mpp_x = props.get("openslide.mpp-x")
    mpp_y = props.get("openslide.mpp-y")
    mpp_pair = []
    for value in (mpp_x, mpp_y):
        if value is None:
            continue
        try:
            mpp = float(value)
        except (TypeError, ValueError):
            continue
        if mpp > 0:
            mpp_pair.append(mpp)
    if mpp_pair:
        return float(sum(mpp_pair) / len(mpp_pair))

    keys = (
        "aperio.MPP",
        "tiff.XResolution",
    )
    for key in keys:
        value = props.get(key)
        if value is None:
            continue
        try:
            mpp = float(value)
        except (TypeError, ValueError):
            continue
        if key == "tiff.XResolution":
            unit = str(props.get("tiff.ResolutionUnit", "")).lower()
            if unit in {"centimeter", "cm"} and mpp > 0:
                return 10000.0 / mpp
            continue
        if mpp > 0:
            return mpp
    return None


def get_slide_magnification(slide) -> float | None:
    props = getattr(slide, "properties", {})
    for key in ("openslide.objective-power", "aperio.AppMag"):
        value = props.get(key)
        if value is None:
            continue
        try:
            mag = float(value)
        except (TypeError, ValueError):
            continue
        if mag > 0:
            return mag
    return None


def infer_mpp_from_magnification(magnification: float | None) -> float | None:
    if magnification is None or magnification <= 0:
        return None
    return 10.0 / magnification


def patch_geometry_at_target_mpp(
    x_target: int,
    y_target: int,
    patch_size: int,
    source_mpp: float,
    target_mpp: float,
) -> PatchGeometry:
    scale = target_mpp / source_mpp
    width = int(round(patch_size * scale))
    return PatchGeometry(
        x0=int(round(x_target * scale)),
        y0=int(round(y_target * scale)),
        width=width,
        height=width,
        target_size=patch_size,
        scale=scale,
    )


def iter_patch_geometries(
    slide_width: int,
    slide_height: int,
    patch_size: int,
    source_mpp: float,
    target_mpp: float,
) -> Iterable[PatchGeometry]:
    scale = target_mpp / source_mpp
    target_width = int(math.ceil(slide_width / scale))
    target_height = int(math.ceil(slide_height / scale))
    for y in range(0, target_height, patch_size):
        for x in range(0, target_width, patch_size):
            geom = patch_geometry_at_target_mpp(x, y, patch_size, source_mpp, target_mpp)
            if geom.x0 >= slide_width or geom.y0 >= slide_height:
                continue
            width = min(geom.width, slide_width - geom.x0)
            height = min(geom.height, slide_height - geom.y0)
            if width <= 0 or height <= 0:
                continue
            yield PatchGeometry(geom.x0, geom.y0, width, height, patch_size, geom.scale)


def tissue_fraction_for_patch(
    mask: np.ndarray,
    slide_size: tuple[int, int],
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> float:
    mask = mask > 0
    slide_w, slide_h = slide_size
    mask_h, mask_w = mask.shape[:2]
    x1 = min(slide_w, x0 + width)
    y1 = min(slide_h, y0 + height)
    mx0 = max(0, int(math.floor(x0 * mask_w / slide_w)))
    my0 = max(0, int(math.floor(y0 * mask_h / slide_h)))
    mx1 = min(mask_w, int(math.ceil(x1 * mask_w / slide_w)))
    my1 = min(mask_h, int(math.ceil(y1 * mask_h / slide_h)))
    if mx1 <= mx0 or my1 <= my0:
        return 0.0
    return float(mask[my0:my1, mx0:mx1].mean())


def combine_prompt_probabilities(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    if probs.ndim != 3:
        raise ValueError(f"Expected [classes,H,W] probabilities, got {probs.shape}")
    max_idx = probs.argmax(axis=0).astype(np.uint8) + 1
    max_prob = probs.max(axis=0)
    return np.where(max_prob >= threshold, max_idx, 0).astype(np.uint8)


def write_patch_metadata(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(row, f, indent=2, sort_keys=True)


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def save_json(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(row, f, indent=2, sort_keys=True)


def relative_patch_name(x0: int, y0: int) -> str:
    return f"patch_x{x0}_y{y0}"


def aggregate_patch_masks(
    metadata_paths: Sequence[Path],
    output_path: Path,
    downsample: int,
) -> dict:
    if downsample <= 0:
        raise ValueError("downsample must be positive")
    rows = [read_json(path) for path in metadata_paths]
    if not rows:
        raise ValueError("No patch metadata found")

    slide_w = max(int(row["slide_width"]) for row in rows)
    slide_h = max(int(row["slide_height"]) for row in rows)
    mask_w = int(math.ceil(slide_w / downsample))
    mask_h = int(math.ceil(slide_h / downsample))
    slide_mask = np.zeros((mask_h, mask_w), dtype=np.uint8)

    for row in rows:
        patch = Image.open(row["mask_path"]).convert("L")
        x0 = int(row["x0"])
        y0 = int(row["y0"])
        x1 = min(slide_w, x0 + int(row["width"]))
        y1 = min(slide_h, y0 + int(row["height"]))
        mx0 = int(math.floor(x0 / downsample))
        my0 = int(math.floor(y0 / downsample))
        mx1 = min(mask_w, int(math.ceil(x1 / downsample)))
        my1 = min(mask_h, int(math.ceil(y1 / downsample)))
        if mx1 <= mx0 or my1 <= my0:
            continue
        patch_small = patch.resize((mx1 - mx0, my1 - my0), RESAMPLE_NEAREST)
        patch_arr = np.array(patch_small, dtype=np.uint8)
        roi = slide_mask[my0:my1, mx0:mx1]
        slide_mask[my0:my1, mx0:mx1] = np.where(patch_arr > 0, patch_arr, roi)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(slide_mask).save(output_path)
    info = {
        "slide_id": rows[0]["slide_id"],
        "cohort": rows[0].get("cohort"),
        "slide_path": rows[0].get("slide_path"),
        "mask_path": str(output_path),
        "metadata_count": len(rows),
        "slide_width": slide_w,
        "slide_height": slide_h,
        "mask_width": mask_w,
        "mask_height": mask_h,
        "downsample": downsample,
        "source_mpp": rows[0].get("source_mpp"),
        "magnification": rows[0].get("magnification"),
        "target_mpp": rows[0].get("target_mpp"),
        "class_names": rows[0].get("class_names"),
    }
    save_json(output_path.with_suffix(".json"), info)
    return info
