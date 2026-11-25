import argparse
from bisect import bisect_left
from contextlib import contextmanager
import fcntl
import gc
import os
import re
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import h5py
import numpy as np
import torch
from PIL import Image as PILImage
from torchvision.datasets.vision import VisionDataset

try:
    import openslide
except ImportError:
    openslide = None

from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image,
    InferenceMetadata,
    Object,
)
from sam3.train.transforms.point_sampling import (
    sample_negative_points_from_mask,
    sample_points_from_mask,
)


from sam3.train.data.dataset_paths import (
    DATA_ROOT, DATASETS, EXTERNAL_DATASETS, SEMANTIC_ROOTS, WSI_DATASETS,
)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
WSI_EXTS = (".svs", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
CACHE_ROOT = Path(__file__).parent / "cache" / "custom_image_dataset"


def _ds_base_dir(ds_name: str) -> Path:
    return SEMANTIC_ROOTS[ds_name]


def _find_path(directory: Path, sample_id: str, exts: Sequence[str]) -> Optional[Path]:
    for ext in exts:
        candidate = directory / f"{sample_id}{ext}"
        if candidate.exists():
            return candidate
        matches = sorted(directory.glob(f"{sample_id}*{ext}"))
        if matches:
            return matches[0]
    return None


def _sanitize_cache_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _read_manifest_rows(manifest_path: Path, expected_fields: int) -> Optional[List[Tuple[str, ...]]]:
    if not manifest_path.exists():
        return None

    rows: List[Tuple[str, ...]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = tuple(line.split("\t"))
            if len(parts) != expected_fields:
                return None
            rows.append(parts)
    return rows


def _write_manifest_rows(manifest_path: Path, rows: Sequence[Sequence[str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f"{manifest_path.name}.", dir=manifest_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write("\t".join(row))
                f.write("\n")
        os.replace(tmp_path, manifest_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@contextmanager
def _manifest_lock(manifest_path: Path):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = manifest_path.with_suffix(f"{manifest_path.suffix}.lock")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _iter_valid_files(directory: Path, exts: Sequence[str]) -> List[Path]:
    ext_set = {ext.lower() for ext in exts}
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in ext_set
    )


def _build_exact_index(paths: Sequence[Path]) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for path in paths:
        index.setdefault(path.stem, str(path))
    return index


def _build_candidate_entries(paths: Sequence[Path]) -> List[Tuple[str, str, str]]:
    return sorted((path.stem, path.suffix.lower(), str(path)) for path in paths)


def _prefix_upper_bound(value: str) -> str:
    if not value:
        return chr(0x10FFFF)
    return value[:-1] + chr(ord(value[-1]) + 1)


def _resolve_sample_path(
    sample_id: str,
    exact_index: Dict[str, str],
    candidate_entries: Sequence[Tuple[str, str, str]],
    ext_priority: Dict[str, int],
    default_priority: int,
) -> Optional[str]:
    exact = exact_index.get(sample_id)
    if exact is not None:
        return exact

    start = bisect_left(candidate_entries, (sample_id, "", ""))
    end = bisect_left(candidate_entries, (_prefix_upper_bound(sample_id), "", ""))
    best_match: Optional[str] = None
    best_priority = default_priority
    for stem, suffix, path_str in candidate_entries[start:end]:
        if not stem.startswith(sample_id):
            continue
        priority = ext_priority.get(suffix, default_priority)
        if best_match is None or priority < best_priority or (
            priority == best_priority and path_str < best_match
        ):
            best_match = path_str
            best_priority = priority

    return best_match


def _resolve_sample_paths(
    sample_ids: Sequence[str],
    candidate_paths: Sequence[Path],
    exts: Sequence[str],
    *,
    item_label: str,
    directory: Path,
) -> List[str]:
    exact_index = _build_exact_index(candidate_paths)
    ext_priority = {ext: i for i, ext in enumerate(exts)}
    candidate_entries = _build_candidate_entries(candidate_paths)

    resolved: List[str] = []
    missing: List[str] = []
    for sample_id in sample_ids:
        resolved_path = _resolve_sample_path(
            sample_id,
            exact_index,
            candidate_entries,
            ext_priority,
            len(exts),
        )
        if resolved_path is None:
            missing.append(sample_id)
        else:
            resolved.append(resolved_path)

    if missing:
        preview = ", ".join(missing[:3])
        raise RuntimeError(f"{item_label} file not found for sample_id(s)={preview} in {directory}")

    return resolved


def _path_manifest_path(ds: str, split: str, image_dir: Path, mask_dir: Path) -> Path:
    cache_dir = CACHE_ROOT / "path" / _sanitize_cache_component(ds) / _sanitize_cache_component(split)
    return cache_dir / "pairs.tsv"


def _wsi_manifest_path(ds: str, split: str, image_dir: Path, mask_dir: Path, coord_dir: Path) -> Path:
    cache_dir = CACHE_ROOT / "wsi" / _sanitize_cache_component(ds) / _sanitize_cache_component(split)
    return cache_dir / "triples.tsv"


def _load_or_build_path_manifest(ds: str, split: str) -> Tuple[List[str], List[str]]:
    base_dir = _ds_base_dir(ds)
    image_dir = base_dir / split
    mask_dir = base_dir / f"{split}_mask"
    if not image_dir.exists() or not mask_dir.exists():
        print(f"Warning: Dataset paths not found for {ds}, skipping: {image_dir}, {mask_dir}")
        return [], []

    manifest_path = _path_manifest_path(ds, split, image_dir, mask_dir)
    cached_rows = _read_manifest_rows(manifest_path, expected_fields=2)
    if cached_rows is not None:
        image_paths = [row[0] for row in cached_rows]
        mask_paths = [row[1] for row in cached_rows]
        return image_paths, mask_paths

    with _manifest_lock(manifest_path):
        cached_rows = _read_manifest_rows(manifest_path, expected_fields=2)
        if cached_rows is not None:
            image_paths = [row[0] for row in cached_rows]
            mask_paths = [row[1] for row in cached_rows]
            return image_paths, mask_paths

        mask_files = _iter_valid_files(mask_dir, IMG_EXTS)
        sample_ids = [_parse_mask_filename(mask_path.name)[0] for mask_path in mask_files]
        image_files = _iter_valid_files(image_dir, IMG_EXTS)
        image_lookup = _build_exact_index(image_files)
        image_candidates = _build_candidate_entries(image_files)
        ext_priority = {ext: i for i, ext in enumerate(IMG_EXTS)}
        image_paths = []
        mask_paths = []
        missing = []

        for sample_id, mask_path in zip(sample_ids, mask_files):
            image_path = _resolve_sample_path(
                sample_id,
                image_lookup,
                image_candidates,
                ext_priority,
                len(IMG_EXTS),
            )
            if image_path is None:
                missing.append(sample_id)
                continue

            image_paths.append(image_path)
            mask_paths.append(str(mask_path))

        if missing:
            preview = ", ".join(missing[:3])
            print(
                f"Warning: Skipping {len(missing)} mask(s) without matching image "
                f"for {ds} split={split}: {preview}"
            )

        _write_manifest_rows(manifest_path, zip(image_paths, mask_paths))
        return image_paths, mask_paths


def _load_or_build_wsi_manifest(ds: str, split: str) -> Tuple[List[str], List[str], List[str]]:
    base_dir = _ds_base_dir(ds)
    image_dir = base_dir / split
    mask_dir = base_dir / f"{split}_mask"
    coord_dir = base_dir / f"{split}_coord"
    if not image_dir.exists() or not mask_dir.exists() or not coord_dir.exists():
        print(f"Warning: Dataset paths not found for {ds}, skipping: {image_dir}, {mask_dir}, {coord_dir}")
        return [], [], []

    manifest_path = _wsi_manifest_path(ds, split, image_dir, mask_dir, coord_dir)
    cached_rows = _read_manifest_rows(manifest_path, expected_fields=3)
    if cached_rows is not None:
        wsi_paths = [row[0] for row in cached_rows]
        mask_paths = [row[1] for row in cached_rows]
        coord_paths = [row[2] for row in cached_rows]
        return wsi_paths, mask_paths, coord_paths

    with _manifest_lock(manifest_path):
        cached_rows = _read_manifest_rows(manifest_path, expected_fields=3)
        if cached_rows is not None:
            wsi_paths = [row[0] for row in cached_rows]
            mask_paths = [row[1] for row in cached_rows]
            coord_paths = [row[2] for row in cached_rows]
            return wsi_paths, mask_paths, coord_paths

        mask_files = _iter_valid_files(mask_dir, WSI_EXTS)
        sample_ids = [_parse_mask_filename(mask_path.name)[0] for mask_path in mask_files]
        wsi_files = _iter_valid_files(image_dir, WSI_EXTS)
        coord_files = _iter_valid_files(coord_dir, (".h5",))

        wsi_paths = _resolve_sample_paths(
            sample_ids,
            wsi_files,
            WSI_EXTS,
            item_label="WSI",
            directory=image_dir,
        )
        coord_paths = _resolve_sample_paths(
            sample_ids,
            coord_files,
            (".h5",),
            item_label="Coord",
            directory=coord_dir,
        )
        mask_paths = [str(mask_path) for mask_path in mask_files]

        _write_manifest_rows(manifest_path, zip(wsi_paths, mask_paths, coord_paths))
        return wsi_paths, mask_paths, coord_paths


def _resolve_path_dataset_names(dataset_name: str) -> List[str]:
    if dataset_name == "all":
        return list(DATASETS)
    return [dataset_name]


def _resolve_wsi_dataset_names(dataset_name: str) -> List[str]:
    if dataset_name == "all":
        return list(WSI_DATASETS)
    return [dataset_name]


def build_manifests(dataset_type: str, dataset_name: str, split: Optional[str] = None) -> None:
    splits = [split] if split is not None else ["train", "test"]

    if dataset_type == "path":
        datasets = _resolve_path_dataset_names(dataset_name)
        for current_split in splits:
            for ds in datasets:
                image_paths, mask_paths = _load_or_build_path_manifest(ds, current_split)
                print(
                    f"[path] dataset={ds} split={current_split} "
                    f"pairs={len(mask_paths)} cache={_path_manifest_path(ds, current_split, _ds_base_dir(ds) / current_split, _ds_base_dir(ds) / f'{current_split}_mask')}"
                )
    elif dataset_type == "wsi":
        datasets = _resolve_wsi_dataset_names(dataset_name)
        for current_split in splits:
            for ds in datasets:
                wsi_paths, mask_paths, coord_paths = _load_or_build_wsi_manifest(ds, current_split)
                print(
                    f"[wsi] dataset={ds} split={current_split} "
                    f"triples={len(mask_paths)} cache={_wsi_manifest_path(ds, current_split, _ds_base_dir(ds) / current_split, _ds_base_dir(ds) / f'{current_split}_mask', _ds_base_dir(ds) / f'{current_split}_coord')}"
                )
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def _parse_mask_filename(mask_file: str):
    stem = mask_file.rsplit(".", 1)[0]
    left, right = stem.split("_pathology_", 1)
    sample_id = left + "_pathology"
    fields = right.split("_")
    organ_name = " ".join(fields[0].split("+"))
    rest = "_".join(fields[1:])
    rest_parts = rest.split("+")
    level_name = rest_parts[0]
    object_name = " ".join(rest_parts[1:])
    return sample_id, organ_name, level_name, object_name


def _pack_strings_to_shared_tensors(values: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    offsets = np.empty(len(values) + 1, dtype=np.int64)
    offsets[0] = 0

    buffer = bytearray()
    for i, value in enumerate(values, start=1):
        buffer.extend(value.encode("utf-8"))
        offsets[i] = len(buffer)

    data = np.frombuffer(bytes(buffer), dtype=np.uint8).copy()
    data_tensor = torch.from_numpy(data)
    offsets_tensor = torch.from_numpy(offsets)
    data_tensor.share_memory_()
    offsets_tensor.share_memory_()
    return data_tensor, offsets_tensor


class _PackedStringMixin:
    def _get_numpy_view(self, tensor_attr: str):
        cache_attr = f"_{tensor_attr}_np"
        cached = getattr(self, cache_attr, None)
        if cached is None:
            cached = getattr(self, tensor_attr).numpy()
            setattr(self, cache_attr, cached)
        return cached

    def _decode_packed_string(self, data_attr: str, offsets_attr: str, idx: int) -> str:
        data_np = self._get_numpy_view(data_attr)
        offsets_np = self._get_numpy_view(offsets_attr)
        start = int(offsets_np[idx])
        end = int(offsets_np[idx + 1])
        return data_np[start:end].tobytes().decode("utf-8")


LEVEL_MARKER = re.compile(r"\b(?:tissue|cells?|nuclei|nucleus)\b", re.IGNORECASE)

def find_level(object_name: str) -> bool:
    return bool(LEVEL_MARKER.search(object_name))

def build_text_prompt(
    organ_name: str,
    object_name: str,
    level_name: str,
    *,
    training: bool,
) -> str:
    level = None
    if level_name and level_name.strip().lower() != "unspecified":
        level = level_name.strip()
    if find_level(object_name): 
        level = None
    prompt = f"{object_name} {level}" if level else object_name
    return " ".join(prompt.split())


def select_text_prompt(
    text_prompt_override: Optional[str],
    organ_name: str,
    object_name: str,
    level_name: str,
    training: bool,
) -> str:
    if text_prompt_override:
        return " ".join(text_prompt_override.split())
    return build_text_prompt(organ_name, object_name, level_name, training=training)



# def build_text_prompt(
#     organ_name: str,
#     object_name: str,
#     level_name: str,
#     *,
#     training: bool,
# ) -> str:
#     # domain_terms = ["pathology image", "histology image", "H&E slide", "histopathology"]
#     # verbs = ["segment", "mask", "outline", "delineate", "find", "locate", "identify"]
#     # preps = ["in", "on", "within"]
#     domain_terms = ["pathology image"]
#     verbs = ["segment"]
#     preps = ["in"]

#     organ = None if (organ_name is None or organ_name.strip().lower() == "unspecified") else organ_name.strip()
#     level = None if (level_name is None or level_name.strip().lower() == "unspecified") else level_name.strip()

#     if training:
#         domain = random.choice(domain_terms)
#         verb = random.choice(verbs)
#         prep = random.choice(preps)
#     else:
#         domain = domain_terms[0]
#         verb = verbs[0]
#         prep = preps[0]

#     level_phrase = f"{level}-level" if level else None
#     organ_phrase = organ if organ else None

#     templates = [
#         "{verb} {level_phrase} {obj} {prep} {organ_phrase} {domain}",
#         "{verb} {level_phrase} {obj}",
#     ]
#     if level and level.lower() == "tissue":
#         templates.extend(
#             [
#                 "{verb} {obj} {prep} {organ_phrase} {domain}",
#                 "{verb} {obj}",
#             ]
#         )

#     tmpl = random.choice(templates) if training else templates[0]
#     prompt = tmpl.format(
#         verb=verb,
#         obj=object_name,
#         prep=prep,
#         organ_phrase=organ_phrase or "",
#         level_phrase=level_phrase or "",
#         domain=domain,
#     )
#     return " ".join(prompt.split())


class PathDataset(VisionDataset, _PackedStringMixin):
    """
    Optimized patch-level dataset:
    - replaces `self.samples: List[(dataset_name, Path)]` with packed shared-memory strings
    - eagerly resolves image/mask paths once, but stores them compactly
    - keeps behavior close to the original `PathDataset`
    """

    def __init__(
        self,
        dataset_name: str,
        training: bool,
        transforms: Optional[List[Any]] = None,
        use_caching: bool = False,
        return_raw_images: bool = True,
        load_segmentation: bool = True,
        return_geometry: bool = True,
        max_input_points: int = 5,
        return_box_prob: float = 0.3,
        return_point_prob: float = 0.3,
        limit_per_dataset: Optional[int] = None,
        text_prompt_override: Optional[str] = None,
        split: Optional[str] = None,
        split_seed: int = 123,
    ) -> None:
        super().__init__(root=str(DATA_ROOT))

        valid_datasets = DATASETS + EXTERNAL_DATASETS
        if dataset_name != "all" and dataset_name not in valid_datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {valid_datasets} or 'all'")

        self.dataset_name = dataset_name
        self.training = training
        self.split = split if split is not None else ("train" if training else "test")
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {self.split}")
        self.split_seed = split_seed
        self.transforms = transforms or []
        self.cache_images = use_caching
        self.return_raw_images = return_raw_images
        self.load_segmentation = load_segmentation
        self.return_geometry = return_geometry
        self.max_input_points = max_input_points
        self.return_box_prob = return_box_prob
        self.return_point_prob = return_point_prob
        self.limit_per_dataset = limit_per_dataset
        self.text_prompt_override = text_prompt_override
        self._image_cache: Dict[str, PILImage.Image] = {}

        image_paths, mask_paths = self._collect_samples(dataset_name)
        if not mask_paths:
            raise RuntimeError(f"No mask files found for dataset_name={dataset_name}, split={self.split}")

        self._num_samples = len(mask_paths)
        self.image_bytes, self.image_offsets = _pack_strings_to_shared_tensors(image_paths)
        self.mask_bytes, self.mask_offsets = _pack_strings_to_shared_tensors(mask_paths)

        self._image_bytes_np = None
        self._image_offsets_np = None
        self._mask_bytes_np = None
        self._mask_offsets_np = None

        del image_paths
        del mask_paths
        gc.collect()

    def _collect_samples(self, dataset_name: str) -> Tuple[List[str], List[str]]:
        ds_list = _resolve_path_dataset_names(dataset_name)

        image_paths: List[str] = []
        mask_paths: List[str] = []

        for ds in ds_list:
            source_split = "train" if self.split == "val" else self.split
            current_images, current_masks = _load_or_build_path_manifest(ds, source_split)
            if self.split in {"train", "val"}:
                # Keep all concept masks of one image in the same partition.
                unique_images = sorted(set(current_images))
                random.Random(f"{self.split_seed}:{ds}").shuffle(unique_images)
                n_val = (
                    min(max(1, round(0.05 * len(unique_images))), len(unique_images) - 1)
                    if len(unique_images) > 1 else 0
                )
                val_images = set(unique_images[:n_val])
                selected = [
                    (image, mask) for image, mask in zip(current_images, current_masks)
                    if (image in val_images) == (self.split == "val")
                ]
                current_images = [image for image, _ in selected]
                current_masks = [mask for _, mask in selected]
            if self.limit_per_dataset is not None and self.limit_per_dataset > 0:
                current_images = current_images[:self.limit_per_dataset]
                current_masks = current_masks[:self.limit_per_dataset]
            image_paths.extend(current_images)
            mask_paths.extend(current_masks)

        return image_paths, mask_paths

    def __len__(self) -> int:
        return self._num_samples

    def _get_image_path(self, idx: int) -> str:
        return self._decode_packed_string("image_bytes", "image_offsets", idx)

    def _get_mask_path(self, idx: int) -> str:
        return self._decode_packed_string("mask_bytes", "mask_offsets", idx)

    def _load_pil_image(self, image_path: str) -> PILImage.Image:
        if self.cache_images and image_path in self._image_cache:
            return self._image_cache[image_path]

        with PILImage.open(image_path) as pil:
            image = pil.convert("RGB")

        if self.cache_images:
            self._image_cache[image_path] = image
        return image

    def __getitem__(self, idx: int) -> Datapoint:
        image_path = self._get_image_path(idx)
        mask_path = self._get_mask_path(idx)

        mask_name = Path(mask_path).name
        _, organ_name, level_name, object_name = _parse_mask_filename(mask_name)
        text_prompt = select_text_prompt(
            self.text_prompt_override,
            organ_name,
            object_name,
            level_name,
            self.training,
        )

        pil_img = self._load_pil_image(image_path)
        width, height = pil_img.size

        with PILImage.open(mask_path) as mask_img:
            gt_mask_np = np.array(mask_img.convert("L"), dtype=np.uint8)
        gt_mask_bin = (gt_mask_np > 0).astype(np.uint8)

        semantic_target = torch.from_numpy(gt_mask_bin) if self.load_segmentation else None
        objects = []
        ys, xs = np.where(gt_mask_bin > 0)
        if len(xs) > 0:
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            bbox_t = torch.tensor([x0, y0, x1, y1], dtype=torch.float32)
            objects.append(
                Object(
                    bbox=bbox_t,
                    area=float(gt_mask_bin.sum()),
                    object_id=0,
                    frame_index=-1,
                    segment=semantic_target,
                    is_crowd=False,
                    source=Path(mask_path).parent.parent.name,
                )
            )
        object_ids_output = [obj.object_id for obj in objects]

        input_bbox = None
        input_bbox_label = None
        input_points = None
        if self.return_geometry:
            if objects and random.random() < self.return_box_prob:
                input_bbox = bbox_t.view(1, 4)
                input_bbox_label = torch.ones((1,), dtype=torch.long)

            if random.random() < self.return_point_prob:
                n_points = max(1, self.max_input_points // 2) if self.training else self.max_input_points
                positive_points = (
                    sample_points_from_mask(gt_mask_bin, n_points, mode="centered")
                    if objects else np.empty((0, 3), dtype=np.float64)
                )
                pts = positive_points
                if self.training:
                    negative_points = sample_negative_points_from_mask(
                        gt_mask_bin, n_points, mode="random_mask",
                    )
                    pts = np.concatenate([positive_points, negative_points], axis=0)
                input_points = torch.as_tensor(pts).view(1, -1, 3)

        image = Image(
            data=pil_img,
            objects=objects,
            size=(height, width),
            blurring_mask=None,
        )

        query = FindQueryLoaded(
            query_text=text_prompt,
            image_id=0,
            object_ids_output=object_ids_output,
            is_exhaustive=True,
            is_pixel_exhaustive=True,
            query_processing_order=0,
            input_bbox=input_bbox,
            input_bbox_label=input_bbox_label,
            input_points=input_points,
            semantic_target=semantic_target,
            inference_metadata=InferenceMetadata(
                coco_image_id=-1 if self.training else idx,
                original_image_id=-1 if self.training else idx,
                original_category_id=-1,
                original_size=(height, width),
                object_id=-1,
                frame_index=-1,
                is_conditioning_only=False,
            ),
        )

        datapoint = Datapoint(
            find_queries=[query],
            images=[image],
            raw_images=[pil_img] if self.return_raw_images else None,
        )
        datapoint.image_path = mask_path

        for transform in self.transforms:
            try:
                datapoint = transform(datapoint)
            except TypeError:
                datapoint = transform(datapoint, epoch=0)

        return datapoint


class WSIDataset(VisionDataset, _PackedStringMixin):
    """
    Optimized WSI dataset:
    - stores pre-resolved wsi/mask/coord paths in packed shared-memory buffers
    - avoids `self.samples: List[(dataset_name, Path)]`
    - keeps runtime logic close to the original implementation
    """

    def __init__(
        self,
        dataset_name: str,
        training: bool,
        transforms: Optional[List[Any]] = None,
        use_caching: bool = False,
        return_raw_images: bool = False,
        load_segmentation: bool = True,
        return_geometry: bool = True,
        max_input_points: int = 5,
        return_box_prob: float = 0.1,
        return_point_prob: float = 0.1,
        patch_size: int = 1024,
    ) -> None:
        super().__init__(root=str(DATA_ROOT))

        if dataset_name != "all" and dataset_name not in WSI_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {WSI_DATASETS} or 'all'")

        self.dataset_name = dataset_name
        self.training = training
        self.split = "train" if training else "test"
        self.transforms = transforms or []
        self.cache_images = use_caching
        self.return_raw_images = return_raw_images
        self.load_segmentation = load_segmentation
        self.return_geometry = return_geometry
        self.max_input_points = max_input_points
        self.return_box_prob = return_box_prob
        self.return_point_prob = return_point_prob
        self.patch_size = patch_size

        wsi_paths, mask_paths, coord_paths = self._collect_samples(dataset_name)
        if not mask_paths:
            raise RuntimeError(f"No mask files found for dataset_name={dataset_name}, split={self.split}")

        self._num_samples = len(mask_paths)
        self.wsi_bytes, self.wsi_offsets = _pack_strings_to_shared_tensors(wsi_paths)
        self.mask_bytes, self.mask_offsets = _pack_strings_to_shared_tensors(mask_paths)
        self.coord_bytes, self.coord_offsets = _pack_strings_to_shared_tensors(coord_paths)

        self._wsi_bytes_np = None
        self._wsi_offsets_np = None
        self._mask_bytes_np = None
        self._mask_offsets_np = None
        self._coord_bytes_np = None
        self._coord_offsets_np = None

        del wsi_paths
        del mask_paths
        del coord_paths
        gc.collect()

    def _collect_samples(self, dataset_name: str) -> Tuple[List[str], List[str], List[str]]:
        ds_list = _resolve_wsi_dataset_names(dataset_name)

        wsi_paths: List[str] = []
        mask_paths: List[str] = []
        coord_paths: List[str] = []

        for ds in ds_list:
            current_wsis, current_masks, current_coords = _load_or_build_wsi_manifest(ds, self.split)
            wsi_paths.extend(current_wsis)
            mask_paths.extend(current_masks)
            coord_paths.extend(current_coords)

        return wsi_paths, mask_paths, coord_paths

    def __len__(self) -> int:
        return self._num_samples

    def _get_wsi_path(self, idx: int) -> str:
        return self._decode_packed_string("wsi_bytes", "wsi_offsets", idx)

    def _get_mask_path(self, idx: int) -> str:
        return self._decode_packed_string("mask_bytes", "mask_offsets", idx)

    def _get_coord_path(self, idx: int) -> str:
        return self._decode_packed_string("coord_bytes", "coord_offsets", idx)

    def _read_region(self, path: Path, x: int, y: int, size: int = 1024, is_mask: bool = False):
        path_str = str(path)
        if path.suffix.lower() in [".svs", ".tif", ".tiff", ".ndpi"]:
            if openslide is None:
                raise ImportError("Please install openslide-python to read WSI files.")
            slide = openslide.OpenSlide(path_str)
            region = slide.read_region((x, y), 0, (size, size))
            return region.convert("L") if is_mask else region.convert("RGB")

        with PILImage.open(path_str) as image:
            region = image.crop((x, y, x + size, y + size))
            return region.convert("L") if is_mask else region.convert("RGB")

    def __getitem__(self, idx: int):
        wsi_path = Path(self._get_wsi_path(idx))
        mask_path = Path(self._get_mask_path(idx))
        coord_path = Path(self._get_coord_path(idx))

        _, organ_name, level_name, object_name = _parse_mask_filename(mask_path.name)
        text_prompt = build_text_prompt(organ_name, object_name, level_name, training=self.training)

        with h5py.File(coord_path, "r") as f:
            coords = f["coords"][:]

        images_list = []
        query_list = []
        raw_images_list = []

        for x_start, y_start in coords:
            pil_img = self._read_region(wsi_path, int(x_start), int(y_start), size=self.patch_size, is_mask=False)
            mask_patch = self._read_region(mask_path, int(x_start), int(y_start), size=self.patch_size, is_mask=True)

            gt_mask_np = np.array(mask_patch, dtype=np.uint8)
            gt_mask_bin = (gt_mask_np > 0).astype(np.uint8)
            semantic_target = torch.from_numpy(gt_mask_bin) if self.load_segmentation else None

            if gt_mask_bin.max() == 0:
                objects = []
                object_ids_output = []
                input_bbox = None
                input_bbox_label = None
                input_points = None
            else:
                ys, xs = np.where(gt_mask_bin > 0)
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
                bbox_t = torch.tensor([x0, y0, x1, y1], dtype=torch.float32)
                area = float(gt_mask_bin.sum())
                seg_t = torch.from_numpy(gt_mask_bin) if self.load_segmentation else None

                objects = [
                    Object(
                        bbox=bbox_t,
                        area=area,
                        object_id=0,
                        frame_index=-1,
                        segment=seg_t,
                        is_crowd=False,
                        source=mask_path.parent.parent.name,
                    )
                ]
                object_ids_output = [0]

                input_bbox = None
                input_bbox_label = None
                input_points = None
                if self.return_geometry:
                    if random.random() < self.return_box_prob:
                        input_bbox = bbox_t.view(1, 4)
                        input_bbox_label = torch.ones((1,), dtype=torch.long)
                    if random.random() < self.return_point_prob:
                        pts = sample_points_from_mask(
                            gt_mask_bin,
                            n_points=self.max_input_points,
                            mode="centered",
                        )
                        input_points = torch.as_tensor(pts).view(1, -1, 3)

            image = Image(
                data=pil_img,
                objects=objects,
                size=(self.patch_size, self.patch_size),
                blurring_mask=None,
            )
            images_list.append(image)

            query = FindQueryLoaded(
                query_text=text_prompt,
                image_id=len(images_list) - 1,
                object_ids_output=object_ids_output,
                is_exhaustive=True,
                is_pixel_exhaustive=True,
                query_processing_order=0,
                input_bbox=input_bbox,
                input_bbox_label=input_bbox_label,
                input_points=input_points,
                semantic_target=semantic_target,
                inference_metadata=InferenceMetadata(
                    coco_image_id=-1 if self.training else idx,
                    original_image_id=-1 if self.training else idx,
                    original_category_id=-1,
                    original_size=(self.patch_size, self.patch_size),
                    object_id=-1,
                    frame_index=-1,
                    is_conditioning_only=False,
                ),
            )
            query_list.append(query)

            if self.return_raw_images:
                raw_images_list.append(pil_img)

        datapoint = Datapoint(
            find_queries=query_list,
            images=images_list,
            raw_images=raw_images_list if self.return_raw_images else None,
        )
        datapoint.image_path = str(mask_path)

        for transform in self.transforms:
            try:
                datapoint = transform(datapoint)
            except TypeError:
                datapoint = transform(datapoint, epoch=0)

        return datapoint


def debug():
    parser = argparse.ArgumentParser(description="Debug custom_image_dataset metadata parsing.")
    parser.add_argument("--dataset-type", choices=["path", "wsi"], default="path")
    parser.add_argument("--dataset-name", default="all")
    parser.add_argument("--show", type=int, default=2)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--build-manifests", action="store_true")
    parser.add_argument("--split", choices=["train", "test"], default=None)
    args = parser.parse_args()

    if args.build_manifests:
        build_manifests(args.dataset_type, args.dataset_name, split=args.split)
        return

    if args.dataset_type == "path":
        dataset = PathDataset(
            dataset_name=args.dataset_name,
            training=True,
            transforms=[],
            use_caching=False,
            return_raw_images=True,
            load_segmentation=True,
            return_geometry=True,
        )
        print(f"dataset_type      : path")
        print(f"dataset_name      : {args.dataset_name}")
        print(f"num_samples       : {len(dataset)}")
        print(f"image_bytes_MB    : {dataset.image_bytes.numel() / 1024 / 1024:.2f}")
        print(f"mask_bytes_MB     : {dataset.mask_bytes.numel() / 1024 / 1024:.2f}")
        inspect_indices = [args.index] if args.index is not None else list(range(min(args.show, len(dataset))))
        for idx in inspect_indices:
            print("\n" + "=" * 80)
            print(f"idx              : {idx}")
            print(f"image_path       : {dataset._get_image_path(idx)}")
            print(f"mask_path        : {dataset._get_mask_path(idx)}")
            dp = dataset[idx]
            print(f"query_text       : {dp.find_queries[0].query_text}")
            print(f"image_size       : {dp.images[0].size}")
            print(f"num_objects      : {len(dp.images[0].objects)}")
            if dp.images[0].objects:
                print(f"bbox             : {dp.images[0].objects[0].bbox.tolist()}")
    else:
        dataset = WSIDataset(
            dataset_name=args.dataset_name,
            training=True,
            transforms=[],
            use_caching=False,
            return_raw_images=False,
            load_segmentation=True,
            return_geometry=False,
            patch_size=args.patch_size,
        )
        print(f"dataset_type      : wsi")
        print(f"dataset_name      : {args.dataset_name}")
        print(f"num_samples       : {len(dataset)}")
        print(f"wsi_bytes_MB      : {dataset.wsi_bytes.numel() / 1024 / 1024:.2f}")
        print(f"mask_bytes_MB     : {dataset.mask_bytes.numel() / 1024 / 1024:.2f}")
        print(f"coord_bytes_MB    : {dataset.coord_bytes.numel() / 1024 / 1024:.2f}")
        inspect_indices = [args.index] if args.index is not None else list(range(min(args.show, len(dataset))))
        for idx in inspect_indices:
            print("\n" + "=" * 80)
            print(f"idx              : {idx}")
            print(f"wsi_path         : {dataset._get_wsi_path(idx)}")
            print(f"mask_path        : {dataset._get_mask_path(idx)}")
            print(f"coord_path       : {dataset._get_coord_path(idx)}")


if __name__ == "__main__":
    debug()
