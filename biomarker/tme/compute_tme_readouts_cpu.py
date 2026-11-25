from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    from . import compute_tme_readout, tme_utils
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from biomarker.tme import compute_tme_readout, tme_utils


DEFAULT_OUTPUT_ROOT = Path("/jhcnas3/MRI_jhcnas3/biomarker/tme_readout")
DEFAULT_FORMAT_CSV = Path(__file__).with_name("TME_Readout_Format.csv")


@dataclass(frozen=True)
class ReadoutJob:
    slide_id: str
    slide_root: Path
    tissue_mask: Path
    tissue_metadata: Path
    cell_table: Path
    readout: Path


def parse_radii_um(value: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(value, str):
        return tuple(float(item) for item in value.split(",") if item.strip())
    return tuple(float(item) for item in value)


def iter_slide_roots(output_root: Path, cohort: str, slide_id: str = "") -> list[Path]:
    cohort_root = output_root / cohort
    if not cohort_root.exists():
        return []
    roots = [path for path in cohort_root.iterdir() if path.is_dir()]
    if slide_id:
        roots = [path for path in roots if slide_id in path.name]
    return sorted(roots)


def shard_slide_roots(
    roots: Sequence[Path], *, num_shards: int = 1, shard_index: int = 0
) -> list[Path]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    return list(roots)[shard_index::num_shards]


def make_job(
    slide_root: Path,
    tissue_mask_stem: str = "tissue_mask",
    readout_filename: str = "TME_Readout.csv",
) -> ReadoutJob:
    return ReadoutJob(
        slide_id=slide_root.name,
        slide_root=slide_root,
        tissue_mask=slide_root / f"{tissue_mask_stem}.png",
        tissue_metadata=slide_root / f"{tissue_mask_stem}.json",
        cell_table=slide_root / "cells.csv",
        readout=slide_root / readout_filename,
    )


def missing_inputs(job: ReadoutJob) -> list[Path]:
    required = [job.tissue_mask, job.tissue_metadata, job.cell_table]
    return [path for path in required if not path.exists()]


def _read_long_readout(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["metric"]: row["results"] for row in csv.DictReader(f)}


def readout_has_metrics(path: Path, metrics: Sequence[str]) -> bool:
    existing = _read_long_readout(path)
    return set(metrics).issubset(existing)


def write_matrix(matrix_csv: Path, readout_paths: Iterable[Path], metrics: Sequence[str]) -> None:
    matrix_csv.parent.mkdir(parents=True, exist_ok=True)
    with matrix_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["slide_id", *metrics])
        writer.writeheader()
        for path in sorted(readout_paths, key=lambda item: item.parent.name):
            values = _read_long_readout(path)
            writer.writerow({"slide_id": path.parent.name, **{metric: values.get(metric, "") for metric in metrics}})


def _compute_one(
    job: ReadoutJob,
    format_csv: Path,
    radii_um: tuple[float, ...],
    min_cell_area_px: int,
    cell_target_mpp: float,
    spatial_working_memory_mb: float,
    reassign_tissue: bool,
) -> dict[str, object]:
    values = compute_tme_readout.compute_slide_readout(
        format_csv=format_csv,
        tissue_mask_path=job.tissue_mask,
        tissue_metadata_path=job.tissue_metadata,
        cell_table_path=job.cell_table,
        output_csv=job.readout,
        radii_um=radii_um,
        min_area_px=min_cell_area_px,
        cell_target_mpp=cell_target_mpp,
        spatial_working_memory_mb=spatial_working_memory_mb,
        reassign_tissue=reassign_tissue,
    )
    return {
        "slide_id": job.slide_id,
        "status": "processed",
        "readout": str(job.readout),
        "valid_tissue_area": values.get("Valid tissue area", ""),
    }


def _compute_ready_jobs(
    jobs: Sequence[ReadoutJob],
    format_csv: Path,
    radii_um: tuple[float, ...],
    workers: int,
    min_cell_area_px: int,
    cell_target_mpp: float,
    spatial_working_memory_mb: float,
    reassign_tissue: bool,
) -> list[dict[str, object]]:
    if workers <= 1:
        results = []
        for job in jobs:
            result = _compute_one(
                job,
                format_csv,
                radii_um,
                min_cell_area_px,
                cell_target_mpp,
                spatial_working_memory_mb,
                reassign_tissue,
            )
            print(result, flush=True)
            results.append(result)
        return results

    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_job = {
            executor.submit(
                _compute_one,
                job,
                format_csv,
                radii_um,
                min_cell_area_px,
                cell_target_mpp,
                spatial_working_memory_mb,
                reassign_tissue,
            ): job
            for job in jobs
        }
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                result = future.result()
            except Exception as exc:
                print({"slide_id": job.slide_id, "status": "error", "error": repr(exc)}, flush=True)
                raise
            print(result, flush=True)
            results.append(result)
    return sorted(results, key=lambda item: str(item["slide_id"]))


def compute_cohort_readouts(
    output_root: Path,
    cohort: str,
    format_csv: Path,
    radii_um: str | Sequence[float] = (40, 80),
    resume: bool = True,
    workers: int = 1,
    limit_slides: int = 0,
    slide_id: str = "",
    matrix_csv: Path | None = None,
    min_cell_area_px: int = 50,
    cell_target_mpp: float = 0.25,
    spatial_working_memory_mb: float = 256.0,
    tissue_mask_stem: str = "tissue_mask",
    readout_filename: str = "TME_Readout.csv",
    matrix_filename: str = "TME_Readout_Matrix.csv",
    reassign_tissue: bool = False,
    num_shards: int = 1,
    shard_index: int = 0,
    write_matrix_output: bool = True,
) -> dict[str, int | str]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    metrics = tme_utils.load_format_metrics(format_csv)
    roots = iter_slide_roots(output_root, cohort, slide_id=slide_id)
    roots = shard_slide_roots(roots, num_shards=num_shards, shard_index=shard_index)
    if limit_slides:
        roots = roots[:limit_slides]

    jobs = [
        make_job(root, tissue_mask_stem=tissue_mask_stem, readout_filename=readout_filename)
        for root in roots
    ]
    ready_jobs: list[ReadoutJob] = []
    skipped_existing = 0
    skipped_missing_inputs = 0
    for job in jobs:
        missing = missing_inputs(job)
        if missing:
            skipped_missing_inputs += 1
            print(
                {
                    "slide_id": job.slide_id,
                    "status": "skip_missing_inputs",
                    "missing": [str(path.name) for path in missing],
                },
                flush=True,
            )
            continue
        if resume and job.readout.exists():
            if readout_has_metrics(job.readout, metrics):
                skipped_existing += 1
                print(
                    {"slide_id": job.slide_id, "status": "skip_existing", "readout": str(job.readout)},
                    flush=True,
                )
                continue
            print(
                {"slide_id": job.slide_id, "status": "recompute_incomplete", "readout": str(job.readout)},
                flush=True,
            )
        ready_jobs.append(job)

    radii = parse_radii_um(radii_um)
    results = _compute_ready_jobs(
        ready_jobs,
        format_csv=format_csv,
        radii_um=radii,
        workers=workers,
        min_cell_area_px=min_cell_area_px,
        cell_target_mpp=cell_target_mpp,
        spatial_working_memory_mb=spatial_working_memory_mb,
        reassign_tissue=reassign_tissue,
    )

    matrix_path = matrix_csv or (output_root / cohort / matrix_filename)
    if write_matrix_output:
        readout_paths = [
            make_job(root, tissue_mask_stem=tissue_mask_stem, readout_filename=readout_filename).readout
            for root in roots
            if make_job(root, tissue_mask_stem=tissue_mask_stem, readout_filename=readout_filename).readout.exists()
        ]
        write_matrix(matrix_path, readout_paths, metrics)
    summary = {
        "slides": len(jobs),
        "processed": len(results),
        "skipped_existing": skipped_existing,
        "skipped_missing_inputs": skipped_missing_inputs,
        "matrix": str(matrix_path) if write_matrix_output else "",
        "num_shards": num_shards,
        "shard_index": shard_index,
    }
    print({"status": "done", **summary}, flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only TME readout computation from existing tissue_mask.png/json and cells.csv outputs."
    )
    parser.add_argument("--cohort", default="BLCA")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--format-csv", type=Path, default=DEFAULT_FORMAT_CSV)
    parser.add_argument("--radii-um", default="40,80")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit-slides", type=int, default=0)
    parser.add_argument("--slide-id", default="")
    parser.add_argument("--matrix-csv", type=Path, default=None)
    parser.add_argument("--min-cell-area-px", type=int, default=50)
    parser.add_argument("--cell-target-mpp", type=float, default=0.25)
    parser.add_argument("--spatial-working-mb", type=float, default=256.0)
    parser.add_argument("--tissue-mask-stem", default="tissue_mask")
    parser.add_argument("--readout-filename", default="TME_Readout.csv")
    parser.add_argument("--matrix-filename", default="TME_Readout_Matrix.csv")
    parser.add_argument("--reassign-tissue", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-matrix", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compute_cohort_readouts(
        output_root=args.output_root,
        cohort=args.cohort,
        format_csv=args.format_csv,
        radii_um=args.radii_um,
        resume=args.resume,
        workers=args.workers,
        limit_slides=args.limit_slides,
        slide_id=args.slide_id,
        matrix_csv=args.matrix_csv,
        min_cell_area_px=args.min_cell_area_px,
        cell_target_mpp=args.cell_target_mpp,
        spatial_working_memory_mb=args.spatial_working_mb,
        tissue_mask_stem=args.tissue_mask_stem,
        readout_filename=args.readout_filename,
        matrix_filename=args.matrix_filename,
        reassign_tissue=args.reassign_tissue,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        write_matrix_output=not args.skip_matrix,
    )


if __name__ == "__main__":
    main()
