from __future__ import annotations

import argparse
import fcntl
import json
import math
import queue
import sys
import threading
import time
from contextlib import contextmanager
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

import numpy as np
from PIL import Image

try:
    from .. import segment_tissue, slide_reader, utils
    from .cell_predictor import CellPrediction, Sam3CellPredictor, nms_predictions
    from . import compute_tme_readout, tme_utils
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from biomarker import segment_tissue, slide_reader, utils
    from biomarker.tme import compute_tme_readout, tme_utils
    from biomarker.tme.cell_predictor import CellPrediction, Sam3CellPredictor, nms_predictions


DEFAULT_TME_ROOT = Path("/jhcnas5/zq/data/PathSeg2/biomarker/tme_readout_wsi")
DEFAULT_TISSUE_CHECKPOINT = utils.CHECKPOINT_PATH
DEFAULT_CELL_CHECKPOINT = Path("/jhcnas5/zq/code/ckpt/finetune/ours/exps/pathseg_instance_ft_mix/checkpoints/best.pt")
DEFAULT_FORMAT_CSV = Path(__file__).with_name("TME_Readout_Format.csv")


class Level0Geometry(NamedTuple):
    x0: int
    y0: int
    width: int
    height: int
    scale: float


class CellTileJob(NamedTuple):
    x: int
    y: int
    core_x0: int
    core_y0: int
    core_x1: int
    core_y1: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end TME readout from WSI: 20x tissue segmentation, 40x "
            "7-class cell instance segmentation, and readout CSV generation."
        )
    )
    parser.add_argument("--cohort", default="BLCA", help="TCGA cohort, comma list, or all.")
    parser.add_argument("--slide-id", default="", help="Optional exact/substring slide id filter.")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
        help="Optional JSON mapping from slide stem to the exact source filename/path to use.",
    )
    parser.add_argument("--tcga-root", type=Path, default=utils.TCGA_ROOT)
    parser.add_argument("--histoqc-root", type=Path, default=utils.HISTOQC_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_TME_ROOT)
    parser.add_argument("--format-csv", type=Path, default=DEFAULT_FORMAT_CSV)
    parser.add_argument("--tissue-checkpoint", type=Path, default=DEFAULT_TISSUE_CHECKPOINT)
    parser.add_argument("--cell-checkpoint", type=Path, default=DEFAULT_CELL_CHECKPOINT)
    parser.add_argument("--bpe-path", type=Path, default=utils.BPE_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--stage",
        choices=("tissue", "cell", "readout", "all"),
        default="all",
        help="Pipeline stage to run. Legacy flags still map to these stages.",
    )
    parser.add_argument("--limit-slides", type=int, default=0)
    parser.add_argument("--random-sample", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prioritize-cases", action="store_true")
    parser.add_argument("--dynamic-slide-claim", action="store_true")
    parser.add_argument("--slide-shard-count", type=int, default=1)
    parser.add_argument("--slide-shard-index", type=int, default=0)

    parser.add_argument("--tissue-target-mpp", type=float, default=0.5, help="20x target MPP.")
    parser.add_argument("--tissue-patch-size", type=int, default=1024)
    parser.add_argument("--tissue-overlap", type=int, default=128)
    parser.add_argument("--tissue-batch-size", type=int, default=8)
    parser.add_argument("--tissue-model-resolution", type=int, default=1008)
    parser.add_argument("--tissue-prob-threshold", type=float, default=0.5)
    parser.add_argument("--tissue-threshold", type=float, default=0.2)
    parser.add_argument("--slide-mask-downsample", type=int, default=32)
    parser.add_argument("--limit-tissue-patches", type=int, default=0)

    parser.add_argument("--cell-target-mpp", type=float, default=0.25, help="40x target MPP.")
    parser.add_argument("--cell-tile-size", type=int, default=576)
    parser.add_argument("--cell-overlap", type=int, default=64)
    parser.add_argument("--cell-batch-size", type=int, default=64)
    parser.add_argument("--cell-model-resolution", type=int, default=1008)
    parser.add_argument("--cell-num-queries", type=int, default=300)
    parser.add_argument("--cell-detection-threshold", type=float, default=0.5)
    parser.add_argument("--cell-max-dets-per-tile", type=int, default=300)
    parser.add_argument("--cell-nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--cell-min-instance-area-px", type=int, default=50)
    parser.add_argument("--cell-dedupe-distance-um", type=float, default=2.0)
    parser.add_argument("--limit-cell-tiles", type=int, default=0)
    parser.add_argument("--cell-shard-count", type=int, default=1)
    parser.add_argument("--cell-shard-index", type=int, default=0)
    parser.add_argument("--cell-prefetch-workers", type=int, default=2)
    parser.add_argument(
        "--cell-postprocess-workers",
        type=int,
        default=0,
        help="CPU worker threads for async cell NMS/row conversion. 0 keeps synchronous behavior.",
    )
    parser.add_argument(
        "--cell-postprocess-max-pending",
        type=int,
        default=1,
        help="Maximum submitted cell postprocess batches waiting to be drained in order.",
    )
    parser.add_argument("--cell-tissue-filter-labels", default="Tumor,Epithelial Tissue,Stroma")
    parser.add_argument("--cell-tissue-filter-threshold", type=float, default=0.1)
    parser.add_argument("--tissue-only", action="store_true")
    parser.add_argument("--skip-tissue-stage", action="store_true")
    parser.add_argument("--merge-cell-shards-only", action="store_true")

    parser.add_argument("--radii-um", default="40,80")
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
    return parser.parse_args()


def normalize_stage_args(args: argparse.Namespace) -> argparse.Namespace:
    legacy_stages = []
    if getattr(args, "tissue_only", False):
        legacy_stages.append("tissue")
    if getattr(args, "skip_tissue_stage", False):
        legacy_stages.append("cell")
    if getattr(args, "merge_cell_shards_only", False):
        legacy_stages.append("readout")
    if len(set(legacy_stages)) > 1:
        raise ValueError("Use only one stage selector: --stage or one legacy stage flag")
    if legacy_stages:
        legacy_stage = legacy_stages[0]
        if getattr(args, "stage", "all") not in ("all", legacy_stage):
            raise ValueError(f"Conflicting stage selectors: --stage {args.stage} and legacy {legacy_stage}")
        args.stage = legacy_stage
    elif not hasattr(args, "stage"):
        args.stage = "all"
    return args


def load_source_manifest(path: Path) -> dict[str, str]:
    """Load an explicit mapping from slide stem to an available source file."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid source manifest JSON: {path}") from error
    if not isinstance(value, dict) or any(
        not isinstance(stem, str)
        or not stem
        or not isinstance(source, str)
        or not source
        for stem, source in value.items()
    ):
        raise ValueError("Source manifest must be a JSON object mapping non-empty strings to non-empty strings")
    return value


def validate_args(args: argparse.Namespace) -> None:
    normalize_stage_args(args)
    if args.tissue_batch_size <= 0 or args.cell_batch_size <= 0:
        raise ValueError("--tissue-batch-size and --cell-batch-size must be positive batch size values")
    if args.cell_shard_count <= 0:
        raise ValueError("--cell-shard-count must be positive")
    if not (0 <= args.cell_shard_index < args.cell_shard_count):
        raise ValueError("--cell-shard-index must be in [0, --cell-shard-count)")
    if args.cell_prefetch_workers < 0:
        raise ValueError("--cell-prefetch-workers must be non-negative")
    if args.cell_postprocess_workers < 0:
        raise ValueError("--cell-postprocess-workers must be non-negative")
    if args.cell_postprocess_max_pending <= 0:
        raise ValueError("--cell-postprocess-max-pending must be positive")
    if not (0.0 <= args.cell_tissue_filter_threshold <= 1.0):
        raise ValueError("--cell-tissue-filter-threshold must be in [0, 1]")
    if args.slide_shard_count <= 0:
        raise ValueError("--slide-shard-count must be positive")
    if not (0 <= args.slide_shard_index < args.slide_shard_count):
        raise ValueError("--slide-shard-index must be in [0, --slide-shard-count)")


def progress_iter(iterable, desc: str, total: int | None = None, enabled: bool = True, tqdm_factory=None):
    if not enabled:
        return iterable
    if tqdm_factory is None:
        try:
            from tqdm.auto import tqdm as tqdm_factory
        except ModuleNotFoundError:
            return iterable
    return tqdm_factory(iterable, desc=desc, total=total, dynamic_ncols=True)


def resolve_cohorts(value: str, tcga_root: Path) -> list[str]:
    if value.lower() == "all":
        return sorted(path.name for path in tcga_root.iterdir() if path.is_dir())
    return [item.strip() for item in value.split(",") if item.strip()]


def select_slide_shard(
    selected: Sequence[tuple[str, Path]],
    shard_index: int,
    shard_count: int,
) -> list[tuple[str, Path]]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if not (0 <= shard_index < shard_count):
        raise ValueError("shard_index must be in [0, shard_count)")
    return [item for index, item in enumerate(selected) if index % shard_count == shard_index]


def select_worker_slides(
    selected: Sequence[tuple[str, Path]],
    shard_index: int,
    shard_count: int,
    dynamic_slide_claim: bool,
) -> list[tuple[str, Path]]:
    if dynamic_slide_claim:
        return list(selected)
    return select_slide_shard(selected, shard_index=shard_index, shard_count=shard_count)


def iter_overlapping_target_windows(width: int, height: int, patch_size: int, overlap: int):
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if overlap < 0 or overlap >= patch_size:
        raise ValueError("overlap must be in [0, patch_size)")
    stride = patch_size - overlap
    for y in range(0, max(1, height), stride):
        for x in range(0, max(1, width), stride):
            yield x, y, min(patch_size, width - x), min(patch_size, height - y)
        if y + patch_size >= height:
            break


def target_window_to_level0(
    x: int,
    y: int,
    width: int,
    height: int,
    source_mpp: float,
    target_mpp: float,
) -> Level0Geometry:
    scale = target_mpp / source_mpp
    return Level0Geometry(
        x0=int(round(x * scale)),
        y0=int(round(y * scale)),
        width=max(1, int(round(width * scale))),
        height=max(1, int(round(height * scale))),
        scale=scale,
    )


def iter_tissue_patch_geometries(
    slide_width: int,
    slide_height: int,
    source_mpp: float,
    target_mpp: float,
    patch_size: int,
    overlap: int,
):
    scale = target_mpp / source_mpp
    target_width = int(math.ceil(slide_width / scale))
    target_height = int(math.ceil(slide_height / scale))
    stride = patch_size - overlap
    y_positions = list(range(0, max(1, target_height), stride))
    x_positions = list(range(0, max(1, target_width), stride))
    for y in y_positions:
        if y >= target_height:
            continue
        for x in x_positions:
            if x >= target_width:
                continue
            geom = target_window_to_level0(
                x,
                y,
                patch_size,
                patch_size,
                source_mpp=source_mpp,
                target_mpp=target_mpp,
            )
            yield geom, (x, y, min(patch_size, target_width - x), min(patch_size, target_height - y))


def count_tissue_patch_geometries(
    slide_width: int,
    slide_height: int,
    source_mpp: float,
    target_mpp: float,
    patch_size: int,
    overlap: int,
) -> int:
    return sum(
        1
        for _geom, _target_window in iter_tissue_patch_geometries(
            slide_width=slide_width,
            slide_height=slide_height,
            source_mpp=source_mpp,
            target_mpp=target_mpp,
            patch_size=patch_size,
            overlap=overlap,
        )
    )


def paste_tissue_patch(
    slide_mask: np.ndarray,
    patch_mask: np.ndarray,
    geom: Level0Geometry,
    slide_size: tuple[int, int],
    downsample: int,
) -> None:
    slide_w, slide_h = slide_size
    mask_h, mask_w = slide_mask.shape
    x1 = min(slide_w, geom.x0 + geom.width)
    y1 = min(slide_h, geom.y0 + geom.height)
    mx0 = int(math.floor(geom.x0 / downsample))
    my0 = int(math.floor(geom.y0 / downsample))
    mx1 = min(mask_w, int(math.ceil(x1 / downsample)))
    my1 = min(mask_h, int(math.ceil(y1 / downsample)))
    if mx1 <= mx0 or my1 <= my0:
        return
    patch_small = Image.fromarray(patch_mask.astype(np.uint8)).resize((mx1 - mx0, my1 - my0), utils.RESAMPLE_NEAREST)
    arr = np.asarray(patch_small, dtype=np.uint8)
    roi = slide_mask[my0:my1, mx0:mx1]
    slide_mask[my0:my1, mx0:mx1] = np.where(arr > 0, arr, roi)


def build_tissue_slide_mask(
    slide,
    slide_path: Path,
    cohort: str,
    valid_mask: np.ndarray,
    source_mpp: float,
    magnification: float | None,
    segmentor: segment_tissue.PromptSegmentor,
    args: argparse.Namespace,
    tissue_mask_path: Path,
) -> dict:
    slide_size = tuple(map(int, slide.dimensions))
    slide_mask = np.zeros(
        (
            int(math.ceil(slide_size[1] / args.slide_mask_downsample)),
            int(math.ceil(slide_size[0] / args.slide_mask_downsample)),
        ),
        dtype=np.uint8,
    )
    kept = 0
    skipped = 0
    pending: list[tuple[Level0Geometry, Image.Image]] = []

    def flush_pending() -> None:
        nonlocal pending
        if not pending:
            return
        patches = [patch for _geom, patch in pending]
        probs_batch = segmentor.predict_probabilities_batch(patches)
        for (geom, _patch), probs in zip(pending, probs_batch):
            class_mask = utils.combine_prompt_probabilities(probs, threshold=args.tissue_prob_threshold)
            paste_tissue_patch(
                slide_mask,
                class_mask,
                geom=geom,
                slide_size=slide_size,
                downsample=args.slide_mask_downsample,
            )
        pending = []

    tissue_jobs = iter_tissue_patch_geometries(
        slide_width=slide_size[0],
        slide_height=slide_size[1],
        source_mpp=source_mpp,
        target_mpp=args.tissue_target_mpp,
        patch_size=args.tissue_patch_size,
        overlap=args.tissue_overlap,
    )
    total_tissue_jobs = count_tissue_patch_geometries(
        slide_width=slide_size[0],
        slide_height=slide_size[1],
        source_mpp=source_mpp,
        target_mpp=args.tissue_target_mpp,
        patch_size=args.tissue_patch_size,
        overlap=args.tissue_overlap,
    )
    for geom, _target_window in progress_iter(
        tissue_jobs,
        desc=f"{slide_path.name} tissue",
        total=total_tissue_jobs,
    ):
        frac = utils.tissue_fraction_for_patch(valid_mask, slide_size, geom.x0, geom.y0, geom.width, geom.height)
        if frac < args.tissue_threshold:
            skipped += 1
            continue
        patch = segment_tissue.read_region_with_white_padding(
            slide,
            x0=geom.x0,
            y0=geom.y0,
            width=geom.width,
            height=geom.height,
            output_size=args.tissue_patch_size,
        )
        pending.append((geom, patch))
        kept += 1
        if len(pending) >= args.tissue_batch_size:
            flush_pending()
        if args.limit_tissue_patches and kept >= args.limit_tissue_patches:
            break
    flush_pending()

    tissue_mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(slide_mask).save(tissue_mask_path)
    metadata = {
        "slide_id": slide_path.name,
        "cohort": cohort,
        "slide_path": str(slide_path),
        "mask_path": str(tissue_mask_path),
        "slide_width": slide_size[0],
        "slide_height": slide_size[1],
        "mask_width": int(slide_mask.shape[1]),
        "mask_height": int(slide_mask.shape[0]),
        "downsample": args.slide_mask_downsample,
        "source_mpp": source_mpp,
        "magnification": magnification,
        "target_mpp": args.tissue_target_mpp,
        "class_names": list(utils.PROMPTS),
        "patch_size": args.tissue_patch_size,
        "overlap": args.tissue_overlap,
        "patches": kept,
        "skipped_low_tissue": skipped,
    }
    utils.save_json(tissue_mask_path.with_suffix(".json"), metadata)
    return metadata


def _axis_tile_jobs(length: int, tile_size: int, overlap: int) -> list[tuple[int, int, int]]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be in [0, tile_size)")
    stride = tile_size - overlap
    positions = list(range(0, max(1, length), stride))
    if positions and positions[-1] + tile_size < length:
        positions.append(max(0, length - tile_size))

    jobs: list[tuple[int, int, int]] = []
    prev_core_end = 0
    for idx, pos in enumerate(positions):
        next_pos = positions[idx + 1] if idx + 1 < len(positions) else length
        core_start = prev_core_end
        if idx + 1 < len(positions):
            core_end = min(length, (pos + tile_size + next_pos) // 2)
        else:
            core_end = length
        core_start = max(core_start, pos)
        core_end = max(core_start, core_end)
        jobs.append((pos, core_start, core_end))
        prev_core_end = core_end
        if pos + tile_size >= length:
            break
    return jobs


def iter_cell_tile_jobs(target_width: int, target_height: int, tile_size: int, overlap: int) -> list[CellTileJob]:
    x_jobs = _axis_tile_jobs(target_width, tile_size=tile_size, overlap=overlap)
    y_jobs = _axis_tile_jobs(target_height, tile_size=tile_size, overlap=overlap)
    return [
        CellTileJob(x=x, y=y, core_x0=cx0, core_y0=cy0, core_x1=cx1, core_y1=cy1)
        for y, cy0, cy1 in y_jobs
        for x, cx0, cx1 in x_jobs
    ]


def select_cell_tile_shard(
    tiles: Sequence[CellTileJob],
    shard_index: int,
    shard_count: int,
) -> list[CellTileJob]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if not (0 <= shard_index < shard_count):
        raise ValueError("shard_index must be in [0, shard_count)")
    return [tile for index, tile in enumerate(tiles) if index % shard_count == shard_index]


def parse_tissue_filter_labels(value: str) -> set[int]:
    if not value.strip():
        return set()
    name_to_label = {name.lower(): label for label, name in tme_utils.TISSUE_LABELS.items()}
    labels: set[int] = set()
    for item in value.split(","):
        key = item.strip()
        if not key:
            continue
        if key.isdigit():
            labels.add(int(key))
            continue
        label = name_to_label.get(key.lower())
        if label is None:
            valid = ", ".join(tme_utils.TISSUE_NAME_TO_LABEL)
            raise ValueError(f"Unknown tissue label for cell filter: {key!r}. Valid labels: {valid}")
        labels.add(label)
    return labels


def cell_tile_passes_tissue_filter(
    tissue_mask: np.ndarray,
    downsample: float,
    geom: Level0Geometry,
    allowed_labels: set[int],
    min_fraction: float,
) -> bool:
    if not allowed_labels:
        return True
    h, w = tissue_mask.shape[:2]
    mx0 = max(0, int(math.floor(geom.x0 / float(downsample))))
    my0 = max(0, int(math.floor(geom.y0 / float(downsample))))
    mx1 = min(w, int(math.ceil((geom.x0 + geom.width) / float(downsample))))
    my1 = min(h, int(math.ceil((geom.y0 + geom.height) / float(downsample))))
    if mx1 <= mx0 or my1 <= my0:
        return False
    roi = tissue_mask[my0:my1, mx0:mx1]
    if roi.size == 0:
        return False
    allowed = np.isin(roi, list(allowed_labels))
    return float(np.count_nonzero(allowed)) / float(roi.size) >= float(min_fraction)


def cell_shard_path(cell_table_path: Path, shard_index: int, shard_count: int) -> Path:
    return cell_table_path.with_name(f"{cell_table_path.stem}.shard{shard_index:03d}-of{shard_count:03d}.csv")


def cell_partial_path(cell_table_path: Path) -> Path:
    return cell_table_path.with_name(f"{cell_table_path.stem}.partial.csv")


def merge_cell_shards(
    cell_table_path: Path,
    tissue_mask_path: Path,
    tissue_metadata_path: Path,
    shard_count: int,
    dedupe_distance_um: float,
    source_mpp: float | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for shard_index in range(shard_count):
        shard_path = cell_shard_path(cell_table_path, shard_index=shard_index, shard_count=shard_count)
        if not shard_path.exists():
            raise FileNotFoundError(f"Cell shard CSV not found: {shard_path}")
        rows.extend(tme_utils.read_cell_table(shard_path))

    metadata = tme_utils.read_json(tissue_metadata_path)
    mpp = float(source_mpp if source_mpp is not None else metadata["source_mpp"])
    if dedupe_distance_um > 0:
        rows = tme_utils.drop_duplicate_cells(rows, distance_um=dedupe_distance_um, mpp=mpp)
    tissue_mask = tme_utils.read_tissue_mask(tissue_mask_path)
    rows = tme_utils.assign_tissue_to_cells(rows, tissue_mask, downsample=float(metadata["downsample"]))
    slide_id = rows[0].get("slide_id", "") if rows else ""
    for index, row in enumerate(rows, start=1):
        if slide_id:
            row["cell_id"] = f"{slide_id}:cell_{index:08d}"
    tme_utils.write_cell_table(cell_table_path, rows)
    return rows


def _mask_stats(mask: np.ndarray) -> dict | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    area_px = int(xs.size)
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[1:, :] |= mask[1:, :] & (mask[1:, :] != mask[:-1, :])
    boundary[:-1, :] |= mask[:-1, :] & (mask[:-1, :] != mask[1:, :])
    boundary[:, 1:] |= mask[:, 1:] & (mask[:, 1:] != mask[:, :-1])
    boundary[:, :-1] |= mask[:, :-1] & (mask[:, :-1] != mask[:, 1:])
    if xs.size > 1:
        coords = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        cov = np.cov(coords, rowvar=False, bias=True)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1].clip(min=0.0)
    else:
        eigvals = np.array([0.0, 0.0], dtype=np.float64)
    return {
        "centroid_x": float(xs.mean()),
        "centroid_y": float(ys.mean()),
        "bbox_x0": int(xs.min()),
        "bbox_y0": int(ys.min()),
        "bbox_x1": int(xs.max()) + 1,
        "bbox_y1": int(ys.max()) + 1,
        "area_px": area_px,
        "perimeter_px": int(boundary.sum()),
        "major_axis_px": float(4.0 * math.sqrt(float(eigvals[0]))) if eigvals[0] > 0 else 0.0,
        "minor_axis_px": float(4.0 * math.sqrt(float(eigvals[1]))) if eigvals[1] > 0 else 0.0,
        "eccentricity": (
            float(math.sqrt(max(0.0, 1.0 - float(eigvals[1]) / float(eigvals[0]))))
            if eigvals[0] > 0
            else 0.0
        ),
    }


def cell_prediction_to_row(
    pred: CellPrediction,
    tile: CellTileJob,
    slide_id: str,
    tile_geometry: Level0Geometry,
    target_mpp: float,
    row_index: int,
) -> dict | None:
    stats = _mask_stats(pred.mask)
    if stats is None:
        return None
    cx_target = tile.x + stats["centroid_x"]
    cy_target = tile.y + stats["centroid_y"]
    if not (tile.core_x0 <= cx_target < tile.core_x1 and tile.core_y0 <= cy_target < tile.core_y1):
        return None
    x_level0 = tile_geometry.x0 + stats["centroid_x"] * tile_geometry.width / float(pred.mask.shape[1])
    y_level0 = tile_geometry.y0 + stats["centroid_y"] * tile_geometry.height / float(pred.mask.shape[0])
    area_um2 = float(stats["area_px"] * target_mpp * target_mpp)
    perimeter_um = float(stats["perimeter_px"] * target_mpp)
    roundness = float(4.0 * math.pi * area_um2 / (perimeter_um * perimeter_um)) if perimeter_um > 0 else ""
    tumor_status = int(bool(pred.tumor_status) or pred.class_name == "Neoplastic cell")
    tumor_score = pred.tumor_score
    if tumor_status and tumor_score is None:
        tumor_score = pred.score
    bbox_x0 = tile_geometry.x0 + stats["bbox_x0"] * tile_geometry.width / float(pred.mask.shape[1])
    bbox_y0 = tile_geometry.y0 + stats["bbox_y0"] * tile_geometry.height / float(pred.mask.shape[0])
    bbox_x1 = tile_geometry.x0 + stats["bbox_x1"] * tile_geometry.width / float(pred.mask.shape[1])
    bbox_y1 = tile_geometry.y0 + stats["bbox_y1"] * tile_geometry.height / float(pred.mask.shape[0])
    return {
        "slide_id": slide_id,
        "cell_id": f"{slide_id}:cell_{row_index:08d}",
        "cell_type": pred.class_name,
        "score": float(pred.score),
        "tumor_status": tumor_status,
        "tumor_score": float(tumor_score) if tumor_score is not None else "",
        "x_level0": float(x_level0),
        "y_level0": float(y_level0),
        "area_um2": area_um2,
        "perimeter_um": perimeter_um,
        "roundness": roundness,
        "eccentricity": float(stats["eccentricity"]),
        "major_axis_um": float(stats["major_axis_px"] * target_mpp),
        "minor_axis_um": float(stats["minor_axis_px"] * target_mpp),
        "bbox_x0": float(bbox_x0),
        "bbox_y0": float(bbox_y0),
        "bbox_x1": float(bbox_x1),
        "bbox_y1": float(bbox_y1),
        "tile_x0": float(tile_geometry.x0),
        "tile_y0": float(tile_geometry.y0),
    }


def postprocess_cell_prediction_batch(
    batch_items: Sequence[tuple[CellTileJob, Level0Geometry]],
    predictions_batch: Sequence[Sequence[CellPrediction]],
    slide_id: str,
    target_mpp: float,
    nms_iou_threshold: float,
    min_instance_area_px: int = 50,
) -> list[dict]:
    rows: list[dict] = []
    local_row_index = 1
    min_area = max(int(min_instance_area_px), 0)
    for (tile, geom), predictions in zip(batch_items, predictions_batch):
        eligible_predictions = predictions
        if min_area:
            eligible_predictions = [
                pred for pred in predictions if (_mask_stats(pred.mask) or {}).get("area_px", 0) >= min_area
            ]
        for pred in nms_predictions(eligible_predictions, iou_threshold=nms_iou_threshold):
            row = cell_prediction_to_row(
                pred,
                tile=tile,
                slide_id=slide_id,
                tile_geometry=geom,
                target_mpp=target_mpp,
                row_index=local_row_index,
            )
            if row is None:
                continue
            rows.append(row)
            local_row_index += 1
    return rows


def iter_prefetched_cell_images(
    slide,
    jobs: Iterable[tuple[CellTileJob, Level0Geometry]],
    output_size: int,
    workers: int,
    max_queue_size: int,
) -> Iterable[tuple[CellTileJob, Level0Geometry, Image.Image]]:
    if workers <= 0:
        for tile, geom in jobs:
            image = segment_tissue.read_region_with_white_padding(
                slide,
                x0=geom.x0,
                y0=geom.y0,
                width=geom.width,
                height=geom.height,
                output_size=output_size,
            )
            yield tile, geom, image
        return

    job_queue: queue.Queue[tuple[CellTileJob, Level0Geometry] | None] = queue.Queue(maxsize=max_queue_size)
    result_queue: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=max_queue_size)
    sentinel = object()

    def producer() -> None:
        for item in jobs:
            job_queue.put(item)
        for _worker_index in range(workers):
            job_queue.put(None)

    def worker() -> None:
        while True:
            item = job_queue.get()
            try:
                if item is None:
                    result_queue.put(("done", sentinel))
                    return
                tile, geom = item
                image = segment_tissue.read_region_with_white_padding(
                    slide,
                    x0=geom.x0,
                    y0=geom.y0,
                    width=geom.width,
                    height=geom.height,
                    output_size=output_size,
                )
                result_queue.put(("ok", (tile, geom, image)))
            except BaseException as exc:
                result_queue.put(("error", exc))
                return
            finally:
                job_queue.task_done()

    threading.Thread(target=producer, daemon=True).start()
    for _worker_index in range(workers):
        threading.Thread(target=worker, daemon=True).start()

    finished_workers = 0
    while finished_workers < workers:
        status, payload = result_queue.get()
        if status == "done":
            finished_workers += 1
            continue
        if status == "error":
            raise payload
        yield payload


def finalize_cell_rows(
    rows: Sequence[dict],
    args: argparse.Namespace,
    source_mpp: float,
    tissue_mask: np.ndarray,
    tissue_downsample: float,
    timings: dict[str, float] | None = None,
) -> list[dict]:
    dedupe_started = time.perf_counter()
    if getattr(args, "cell_overlap", 0) > 0:
        rows = tme_utils.drop_duplicate_cells(
            rows,
            distance_um=args.cell_dedupe_distance_um,
            mpp=source_mpp,
        )
    dedupe_s = time.perf_counter() - dedupe_started

    tissue_assignment_started = time.perf_counter()
    rows = tme_utils.assign_tissue_to_cells(rows, tissue_mask, downsample=float(tissue_downsample))
    tissue_assignment_s = time.perf_counter() - tissue_assignment_started
    if timings is not None:
        timings["dedupe_s"] = dedupe_s
        timings["tissue_assignment_s"] = tissue_assignment_s
    return rows


def build_cell_table(
    slide,
    slide_path: Path,
    valid_mask: np.ndarray | None,
    source_mpp: float,
    predictor: Sam3CellPredictor,
    args: argparse.Namespace,
    tissue_mask_path: Path,
    tissue_metadata_path: Path,
    cell_table_path: Path,
) -> list[dict]:
    slide_id = slide_path.name
    slide_size = tuple(map(int, slide.dimensions))
    scale = args.cell_target_mpp / source_mpp
    target_w = int(math.ceil(slide_size[0] / scale))
    target_h = int(math.ceil(slide_size[1] / scale))
    rows: list[dict] = []
    row_index = 1
    processed_tiles = 0
    skipped_foreground = 0
    skipped_tissue_filter = 0
    pending: list[tuple[CellTileJob, Level0Geometry, Image.Image]] = []
    postprocess_workers = int(getattr(args, "cell_postprocess_workers", 0))
    postprocess_max_pending = int(getattr(args, "cell_postprocess_max_pending", 1))
    postprocess_executor: ThreadPoolExecutor | None = None
    postprocess_futures: list[Future[list[dict]]] = []
    tail_timings: dict[str, float] = {}
    output_table_path = (
        cell_shard_path(cell_table_path, args.cell_shard_index, args.cell_shard_count)
        if args.cell_shard_count > 1
        else cell_table_path
    )
    partial_table_path = cell_partial_path(output_table_path)
    tme_utils.append_cell_table_rows(partial_table_path, [], reset=True)
    tissue_mask = tme_utils.read_tissue_mask(tissue_mask_path)
    metadata = tme_utils.read_json(tissue_metadata_path)
    tissue_downsample = float(metadata["downsample"])
    allowed_tissue_labels = parse_tissue_filter_labels(args.cell_tissue_filter_labels)

    if postprocess_workers > 0:
        postprocess_executor = ThreadPoolExecutor(max_workers=postprocess_workers)

    def append_processed_rows(batch_rows: Sequence[dict]) -> None:
        nonlocal row_index
        if not batch_rows:
            return
        finalized_rows = [dict(row) for row in batch_rows]
        for row in finalized_rows:
            if args.cell_shard_count > 1:
                row["cell_id"] = f"{slide_id}:shard{args.cell_shard_index:03d}:cell_{row_index:08d}"
            else:
                row["cell_id"] = f"{slide_id}:cell_{row_index:08d}"
            row_index += 1
        rows.extend(finalized_rows)
        tme_utils.append_cell_table_rows(partial_table_path, finalized_rows)

    def drain_postprocess_one() -> None:
        if not postprocess_futures:
            return
        future = postprocess_futures.pop(0)
        append_processed_rows(future.result())

    def drain_postprocess_all() -> None:
        while postprocess_futures:
            drain_postprocess_one()

    def submit_postprocess(
        batch_items: Sequence[tuple[CellTileJob, Level0Geometry]],
        predictions_batch: Sequence[Sequence[CellPrediction]],
    ) -> None:
        min_instance_area_px = getattr(args, "cell_min_instance_area_px", 50)
        if postprocess_executor is None:
            batch_rows = postprocess_cell_prediction_batch(
                batch_items=batch_items,
                predictions_batch=predictions_batch,
                slide_id=slide_id,
                target_mpp=args.cell_target_mpp,
                nms_iou_threshold=args.cell_nms_iou_threshold,
                min_instance_area_px=min_instance_area_px,
            )
            append_processed_rows(batch_rows)
            return
        future = postprocess_executor.submit(
            postprocess_cell_prediction_batch,
            list(batch_items),
            list(predictions_batch),
            slide_id,
            args.cell_target_mpp,
            args.cell_nms_iou_threshold,
            min_instance_area_px,
        )
        postprocess_futures.append(future)
        while len(postprocess_futures) > postprocess_max_pending:
            drain_postprocess_one()

    def flush_pending() -> None:
        nonlocal pending
        if not pending:
            return
        batch_items = [(tile, geom) for tile, geom, _image in pending]
        images = [image for _tile, _geom, image in pending]
        predictions_batch = predictor.predict_tiles(images)
        submit_postprocess(batch_items, predictions_batch)
        pending = []

    cell_tiles = iter_cell_tile_jobs(target_w, target_h, tile_size=args.cell_tile_size, overlap=args.cell_overlap)
    shard_tiles = select_cell_tile_shard(
        cell_tiles,
        shard_index=args.cell_shard_index,
        shard_count=args.cell_shard_count,
    )
    shard_desc = (
        f" shard {args.cell_shard_index + 1}/{args.cell_shard_count}"
        if args.cell_shard_count > 1
        else ""
    )
    filter_names = ",".join(
        tme_utils.TISSUE_LABELS[label]
        for label in sorted(allowed_tissue_labels)
        if label in tme_utils.TISSUE_LABELS
    )
    print(
        f"[{slide_id}] cells{shard_desc}: {len(shard_tiles)}/{len(cell_tiles)} tiles, "
        f"tile={args.cell_tile_size}, overlap={args.cell_overlap}, "
        f"batch={args.cell_batch_size}, prefetch_workers={args.cell_prefetch_workers}, "
        f"postprocess_workers={postprocess_workers}, postprocess_max_pending={postprocess_max_pending}, "
        f"tissue_filter={filter_names or 'none'}, threshold={args.cell_tissue_filter_threshold}",
        flush=True,
    )

    def filtered_tile_jobs() -> Iterable[tuple[CellTileJob, Level0Geometry]]:
        nonlocal processed_tiles, skipped_foreground, skipped_tissue_filter
        for tile in progress_iter(
            shard_tiles,
            desc=f"{slide_id} cells{shard_desc}",
            total=len(shard_tiles),
        ):
            if args.limit_cell_tiles and processed_tiles >= args.limit_cell_tiles:
                break
            geom = target_window_to_level0(
                tile.x,
                tile.y,
                args.cell_tile_size,
                args.cell_tile_size,
                source_mpp=source_mpp,
                target_mpp=args.cell_target_mpp,
            )
            if not cell_tile_passes_tissue_filter(
                tissue_mask,
                downsample=tissue_downsample,
                geom=geom,
                allowed_labels=allowed_tissue_labels,
                min_fraction=args.cell_tissue_filter_threshold,
            ):
                skipped_tissue_filter += 1
                continue
            if valid_mask is not None:
                frac = utils.tissue_fraction_for_patch(valid_mask, slide_size, geom.x0, geom.y0, geom.width, geom.height)
                if frac < args.tissue_threshold:
                    skipped_foreground += 1
                    continue
            processed_tiles += 1
            yield tile, geom

    try:
        prefetch_queue_size = max(args.cell_batch_size * 2, args.cell_prefetch_workers * 2, 1)
        for tile, geom, image in iter_prefetched_cell_images(
            slide,
            filtered_tile_jobs(),
            output_size=args.cell_tile_size,
            workers=args.cell_prefetch_workers,
            max_queue_size=prefetch_queue_size,
        ):
            pending.append((tile, geom, image))
            if len(pending) >= args.cell_batch_size:
                flush_pending()

        tail_started = time.perf_counter()
        final_batch_started = time.perf_counter()
        flush_pending()
        tail_timings["final_batch_s"] = time.perf_counter() - final_batch_started

        postprocess_drain_started = time.perf_counter()
        drain_postprocess_all()
        tail_timings["postprocess_drain_s"] = time.perf_counter() - postprocess_drain_started
    finally:
        executor_shutdown_started = time.perf_counter()
        if postprocess_executor is not None:
            postprocess_executor.shutdown(wait=True)
        tail_timings["executor_shutdown_s"] = time.perf_counter() - executor_shutdown_started

    rows = finalize_cell_rows(
        rows,
        args=args,
        source_mpp=source_mpp,
        tissue_mask=tissue_mask,
        tissue_downsample=float(metadata["downsample"]),
        timings=tail_timings,
    )
    csv_write_started = time.perf_counter()
    tme_utils.write_cell_table(output_table_path, rows)
    tail_timings["csv_write_s"] = time.perf_counter() - csv_write_started
    tail_timings["total_s"] = time.perf_counter() - tail_started
    print(
        f"[{slide_id}] cell tail timings: "
        + ", ".join(
            f"{name}={tail_timings[name]:.3f}"
            for name in (
                "final_batch_s",
                "postprocess_drain_s",
                "executor_shutdown_s",
                "dedupe_s",
                "tissue_assignment_s",
                "csv_write_s",
                "total_s",
            )
        ),
        flush=True,
    )
    print(
        f"[{slide_id}] cells{shard_desc} done: predicted_tiles={processed_tiles}, "
        f"skipped_tissue_filter={skipped_tissue_filter}, skipped_foreground={skipped_foreground}, "
        f"cells={len(rows)}, output={output_table_path}",
        flush=True,
    )
    return rows


def output_paths(output_root: Path, cohort: str, slide_id: str) -> dict[str, Path]:
    slide_root = output_root / cohort / slide_id
    return {
        "slide_root": slide_root,
        "tissue_mask": slide_root / "tissue_mask.png",
        "tissue_metadata": slide_root / "tissue_mask.json",
        "cell_table": slide_root / "cells.csv",
        "readout": slide_root / "TME_Readout.csv",
    }


def stage_claim_path(slide_root: Path, stage: str) -> Path:
    return slide_root / f".{stage}.claim.lock"


@contextmanager
def try_stage_claim(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def parse_radii_um(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def compute_readout_from_outputs(paths: dict[str, Path], args: argparse.Namespace) -> dict:
    cell_shard_count = getattr(args, "cell_shard_count", 1)
    resume = getattr(args, "resume", False)
    if cell_shard_count > 1 and (not paths["cell_table"].exists() or not resume):
        merge_cell_shards(
            cell_table_path=paths["cell_table"],
            tissue_mask_path=paths["tissue_mask"],
            tissue_metadata_path=paths["tissue_metadata"],
            shard_count=cell_shard_count,
            dedupe_distance_um=getattr(args, "cell_dedupe_distance_um", 2.0),
        )
    values = compute_tme_readout.compute_slide_readout(
        format_csv=args.format_csv,
        tissue_mask_path=paths["tissue_mask"],
        tissue_metadata_path=paths["tissue_metadata"],
        cell_table_path=paths["cell_table"],
        output_csv=paths["readout"],
        radii_um=parse_radii_um(args.radii_um),
        min_area_px=getattr(args, "cell_min_instance_area_px", 50),
        cell_target_mpp=getattr(args, "cell_target_mpp", 0.25),
    )
    return {
        "cells": len(tme_utils.read_cell_table(paths["cell_table"])),
        "readout": str(paths["readout"]),
        "valid_tissue_area": values.get("Valid tissue area", ""),
    }


def stage_output_complete_for_resume(stage: str, paths: dict[str, Path]) -> bool:
    if stage == "tissue":
        return paths["tissue_mask"].exists() and paths["tissue_metadata"].exists()
    if stage == "cell":
        return paths["cell_table"].exists()
    if stage in {"readout", "all"}:
        return paths["readout"].exists()
    return False


def prioritize_pending_slides(
    selected: Sequence[tuple[str, Path]],
    cohort_slide_paths: dict[str, Sequence[Path]],
    output_root: Path,
    stage: str,
) -> list[tuple[str, Path]]:
    completed = [
        (cohort, slide_path)
        for cohort, slide_paths in cohort_slide_paths.items()
        for slide_path in slide_paths
        if stage_output_complete_for_resume(
            stage,
            output_paths(output_root, cohort, utils.slide_id_from_path(slide_path)),
        )
    ]
    pending = [
        (cohort, slide_path)
        for cohort, slide_path in selected
        if not stage_output_complete_for_resume(
            stage,
            output_paths(output_root, cohort, utils.slide_id_from_path(slide_path)),
        )
    ]
    return utils.prioritize_slide_paths_by_case(pending, completed)


def process_slide(
    slide_path: Path,
    cohort: str,
    tissue_segmentor: segment_tissue.PromptSegmentor | None,
    cell_predictor: Sam3CellPredictor | None,
    args: argparse.Namespace,
) -> dict:
    normalize_stage_args(args)
    slide_id = utils.slide_id_from_path(slide_path)
    paths = output_paths(args.output_root, cohort, slide_id)
    stage = args.stage
    if stage in {"all", "readout"} and args.resume and paths["readout"].exists():
        return {"slide_id": slide_id, "status": "skip_existing", "readout": str(paths["readout"])}

    cell_shard_count = getattr(args, "cell_shard_count", 1)
    cell_shard_index = getattr(args, "cell_shard_index", 0)

    if stage == "readout":
        result = compute_readout_from_outputs(paths, args)
        return {"slide_id": slide_id, "status": "ok", **result}

    if stage == "tissue" and args.resume and paths["tissue_mask"].exists() and paths["tissue_metadata"].exists():
        return {"slide_id": slide_id, "status": "skip_existing_tissue", "tissue_mask": str(paths["tissue_mask"])}

    if stage == "cell" and cell_shard_count > 1:
        if args.resume and paths["cell_table"].exists():
            return {
                "slide_id": slide_id,
                "status": "skip_existing_cell_table",
                "cell_table": str(paths["cell_table"]),
            }
        shard_path = cell_shard_path(paths["cell_table"], cell_shard_index, cell_shard_count)
        if args.resume and shard_path.exists():
            return {
                "slide_id": slide_id,
                "status": "skip_existing_cell_shard",
                "shard": f"{cell_shard_index + 1}/{cell_shard_count}",
                "cell_table": str(shard_path),
            }
    if stage == "cell" and cell_shard_count == 1 and args.resume and paths["cell_table"].exists():
        return {"slide_id": slide_id, "status": "skip_existing_cell_table", "cell_table": str(paths["cell_table"])}

    if (
        stage == "all"
        and cell_shard_count == 1
        and args.resume
        and paths["tissue_mask"].exists()
        and paths["tissue_metadata"].exists()
        and paths["cell_table"].exists()
    ):
        result = compute_readout_from_outputs(paths, args)
        return {"slide_id": slide_id, "status": "ok", **result}

    paths["slide_root"].mkdir(parents=True, exist_ok=True)
    with slide_reader.open_slide(slide_path) as slide:
        magnification = utils.get_slide_magnification(slide)
        source_mpp = utils.get_slide_mpp(slide)
        if source_mpp is None:
            source_mpp = utils.infer_mpp_from_magnification(magnification)
        if source_mpp is None:
            return {"slide_id": slide_id, "status": "skip_no_mpp"}

        tissue_exists = paths["tissue_mask"].exists() and paths["tissue_metadata"].exists()
        if stage == "cell":
            if not tissue_exists:
                raise FileNotFoundError(f"Tissue mask is required for cell shard stage: {paths['tissue_mask']}")
            valid_mask = None
        elif not (args.resume and tissue_exists):
            valid_mask, foreground_metadata = segment_tissue.load_foreground_mask(slide, slide_id, cohort, args)
            if valid_mask is None:
                return {"slide_id": slide_id, "status": foreground_metadata.get("status", "skip_no_foreground")}
            if tissue_segmentor is None:
                raise RuntimeError("tissue_segmentor is required when building tissue masks")
            build_tissue_slide_mask(
                slide,
                slide_path=slide_path,
                cohort=cohort,
                valid_mask=valid_mask,
                source_mpp=source_mpp,
                magnification=magnification,
                segmentor=tissue_segmentor,
                args=args,
                tissue_mask_path=paths["tissue_mask"],
            )
            valid_mask = None
        else:
            valid_mask = None

        if stage == "tissue":
            return {"slide_id": slide_id, "status": "ok_tissue", "tissue_mask": str(paths["tissue_mask"])}

        if cell_predictor is None:
            raise RuntimeError("cell_predictor is required when building cell tables")

        if cell_shard_count > 1:
            build_cell_table(
                slide,
                slide_path=slide_path,
                valid_mask=valid_mask,
                source_mpp=source_mpp,
                predictor=cell_predictor,
                args=args,
                tissue_mask_path=paths["tissue_mask"],
                tissue_metadata_path=paths["tissue_metadata"],
                cell_table_path=paths["cell_table"],
            )
            return {
                "slide_id": slide_id,
                "status": "ok_cell_shard",
                "shard": f"{cell_shard_index + 1}/{cell_shard_count}",
                "cell_table": str(cell_shard_path(paths["cell_table"], cell_shard_index, cell_shard_count)),
            }

        if not (args.resume and paths["cell_table"].exists()):
            build_cell_table(
                slide,
                slide_path=slide_path,
                valid_mask=valid_mask,
                source_mpp=source_mpp,
                predictor=cell_predictor,
                args=args,
                tissue_mask_path=paths["tissue_mask"],
                tissue_metadata_path=paths["tissue_metadata"],
                cell_table_path=paths["cell_table"],
            )

    if stage == "cell":
        return {"slide_id": slide_id, "status": "ok_cell", "cell_table": str(paths["cell_table"])}

    result = compute_readout_from_outputs(paths, args)
    return {"slide_id": slide_id, "status": "ok", **result}


def main() -> None:
    args = normalize_stage_args(parse_args())
    validate_args(args)
    required_paths = []
    if args.stage in {"tissue", "cell", "all"}:
        required_paths.append((args.bpe_path, "BPE file"))
    if args.stage in {"tissue", "all"}:
        required_paths.append((args.tissue_checkpoint, "tissue checkpoint"))
    if args.stage in {"cell", "all"}:
        required_paths.append((args.cell_checkpoint, "cell checkpoint"))
    if args.stage in {"readout", "all"}:
        required_paths.append((args.format_csv, "format CSV"))
    for path, name in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    source_manifest = load_source_manifest(args.source_manifest) if args.source_manifest else None
    cohorts = resolve_cohorts(args.cohort, args.tcga_root)
    cohort_slide_paths = {
        cohort: list(utils.iter_slide_paths(cohort, args.tcga_root, source_manifest=source_manifest))
        for cohort in cohorts
    }
    selected = utils.select_slide_paths_by_cohort(
        cohort_slide_paths,
        slide_id=args.slide_id,
        total_limit=args.limit_slides,
        random_sample=args.random_sample and not bool(args.slide_id),
        seed=args.seed,
    )
    if not selected:
        raise RuntimeError("No slides selected. Check --cohort, --tcga-root, or --slide-id.")
    resumable_stage = args.stage in {"tissue", "cell", "readout"}
    if args.resume and args.prioritize_cases and resumable_stage:
        before_prefilter = len(selected)
        selected = prioritize_pending_slides(
            selected,
            cohort_slide_paths,
            output_root=args.output_root,
            stage=args.stage,
        )
        print(
            f"[case-first {args.stage}] {len(selected)}/{before_prefilter} slides pending "
            "with uncovered cases scheduled first",
            flush=True,
        )
        if not selected:
            return
    elif args.resume and args.slide_shard_count > 1 and resumable_stage:
        before_prefilter = len(selected)
        selected = [
            (cohort, slide_path)
            for cohort, slide_path in selected
            if not stage_output_complete_for_resume(
                args.stage,
                output_paths(args.output_root, cohort, utils.slide_id_from_path(slide_path)),
            )
        ]
        print(
            f"[resume prefilter {args.stage}] {len(selected)}/{before_prefilter} slides pending "
            f"before {args.slide_shard_count}-way sharding",
            flush=True,
        )
        if not selected:
            return
    if args.slide_shard_count > 1:
        before_shard = len(selected)
        selected = select_worker_slides(
            selected,
            shard_index=args.slide_shard_index,
            shard_count=args.slide_shard_count,
            dynamic_slide_claim=args.dynamic_slide_claim,
        )
        if args.dynamic_slide_claim:
            print(
                f"[dynamic worker {args.slide_shard_index + 1}/{args.slide_shard_count}] "
                f"shared queue with {len(selected)} slides; fixed sharding disabled",
                flush=True,
            )
        else:
            print(
                f"[slide shard {args.slide_shard_index + 1}/{args.slide_shard_count}] "
                f"{len(selected)}/{before_shard} selected slides",
                flush=True,
            )
        if not selected:
            return

    needs_tissue_model = args.stage in {"tissue", "all"}
    needs_cell_model = args.stage in {"cell", "all"}

    tissue_segmentor = None
    if needs_tissue_model:
        tissue_model = segment_tissue.load_model(
            argparse.Namespace(
                bpe_path=args.bpe_path,
                device=args.device,
                checkpoint=args.tissue_checkpoint,
            )
        )
        tissue_segmentor = segment_tissue.PromptSegmentor(
            tissue_model,
            utils.PROMPTS,
            args.device,
            resolution=args.tissue_model_resolution,
        )

    cell_predictor = None
    if needs_cell_model:
        cell_predictor = Sam3CellPredictor(
            checkpoint=args.cell_checkpoint,
            bpe_path=args.bpe_path,
            device=args.device,
            model_resolution=args.cell_model_resolution,
            num_queries=args.cell_num_queries,
            detection_threshold=args.cell_detection_threshold,
            max_dets_per_tile=args.cell_max_dets_per_tile,
        )

    for index, (cohort, slide_path) in enumerate(selected, start=1):
        if not args.dynamic_slide_claim:
            result = process_slide(slide_path, cohort, tissue_segmentor, cell_predictor, args)
            print(f"[{cohort} {index}/{len(selected)}] {result}")
            continue

        paths = output_paths(args.output_root, cohort, utils.slide_id_from_path(slide_path))
        lock_path = stage_claim_path(paths["slide_root"], args.stage)
        with try_stage_claim(lock_path) as claimed:
            if not claimed:
                continue
            if args.resume and stage_output_complete_for_resume(args.stage, paths):
                continue
            result = process_slide(slide_path, cohort, tissue_segmentor, cell_predictor, args)
        print(f"[{cohort} {index}/{len(selected)}] {result}")


if __name__ == "__main__":
    main()
