import argparse
import gc
import os
import re
import random
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Collection, Dict, List, Optional, Sequence, Tuple

import fcntl
import numpy as np
import torch
from PIL import Image as PILImage
from torchvision.datasets.vision import VisionDataset

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


from sam3.train.data.dataset_paths import INSTANCE_ROOT as DATA_ROOT, INSTANCE_VAL_DATASETS

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
CACHE_ROOT = Path(__file__).parent / "cache" / "custom_instance_dataset"


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


def _parse_mask_filename(mask_file: str) -> Tuple[str, str, str, str]:
    stem = mask_file.rsplit(".", 1)[0]
    left, right = stem.split("_pathology_", 1)
    sample_id = left + "_pathology"
    fields = right.split("_")
    organ_name = " ".join(fields[0].split("+"))
    rest = "_".join(fields[1:])
    rest_parts = rest.split("+")
    level_name = rest_parts[0]
    object_name = " ".join(rest_parts[1:]) if len(rest_parts) > 1 else ""
    return sample_id, organ_name, level_name, object_name


def _build_image_lookup(image_dir: Path) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for image_path in _iter_valid_files(image_dir, IMG_EXTS):
        stem = image_path.stem
        path_str = str(image_path)
        lookup.setdefault(stem, path_str)
        if "_pathology" in stem:
            sample_id = stem.split("_pathology", 1)[0] + "_pathology"
            lookup.setdefault(sample_id, path_str)
    return lookup


def _resolve_image_path(image_dir: Path, sample_id: str, image_lookup: Optional[Dict[str, str]] = None) -> str:
    if image_lookup is not None and sample_id in image_lookup:
        return image_lookup[sample_id]

    for ext in IMG_EXTS:
        exact = image_dir / f"{sample_id}{ext}"
        if exact.exists():
            return str(exact)

    candidates = []
    for ext in IMG_EXTS:
        candidates.extend(image_dir.glob(f"{sample_id}*{ext}"))
    if candidates:
        return str(sorted(candidates)[0])

    raise RuntimeError(f"Image file not found for sample_id={sample_id} in {image_dir}")


def _manifest_path(root: Path, dataset_name: str, split: str) -> Path:
    cache_dir = (
        CACHE_ROOT
        / "path"
        / _sanitize_cache_component(dataset_name)
        / _sanitize_cache_component(split)
    )
    return cache_dir / "pairs.tsv"


def _manifest_rows_are_current(
    rows: Optional[Sequence[Tuple[str, ...]]],
    instance_paths: Sequence[Path],
) -> bool:
    if rows is None:
        return False
    expected_masks = [str(path) for path in instance_paths]
    cached_masks = [row[1] for row in rows]
    if cached_masks != expected_masks:
        return False
    return all(Path(row[0]).exists() for row in rows)


def _available_datasets(root: Path, split: str) -> List[str]:
    datasets = []
    if not root.exists():
        return datasets
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        if split == "test" and path.name not in INSTANCE_VAL_DATASETS:
            continue
        if (path / split).exists() and (path / f"{split}_inst_mask").exists():
            datasets.append(path.name)
    return datasets


def _resolve_dataset_names(root: Path, dataset_name: str, split: str) -> List[str]:
    if dataset_name == "all":
        return _available_datasets(root, split)
    return [dataset_name]


def _load_or_build_manifest(root: Path, dataset_name: str, split: str) -> Tuple[List[str], List[str]]:
    base_dir = root / dataset_name
    image_dir = base_dir / split
    instance_dir = base_dir / f"{split}_inst_mask"
    if not image_dir.exists() or not instance_dir.exists():
        print(f"Warning: instance dataset paths not found, skipping: {image_dir}, {instance_dir}")
        return [], []

    manifest_path = _manifest_path(root, dataset_name, split)
    instance_paths = _iter_valid_files(instance_dir, IMG_EXTS)
    cached_rows = _read_manifest_rows(manifest_path, expected_fields=2)
    if _manifest_rows_are_current(cached_rows, instance_paths):
        return [r[0] for r in cached_rows], [r[1] for r in cached_rows]

    with _manifest_lock(manifest_path):
        instance_paths = _iter_valid_files(instance_dir, IMG_EXTS)
        cached_rows = _read_manifest_rows(manifest_path, expected_fields=2)
        if _manifest_rows_are_current(cached_rows, instance_paths):
            return [r[0] for r in cached_rows], [r[1] for r in cached_rows]

        rows = []
        image_lookup = _build_image_lookup(image_dir)
        for instance_path in instance_paths:
            sample_id, _, _, _ = _parse_mask_filename(instance_path.name)
            image_path = _resolve_image_path(image_dir, sample_id, image_lookup)
            rows.append(
                (
                    image_path,
                    str(instance_path),
                )
            )

        _write_manifest_rows(manifest_path, rows)
        return [r[0] for r in rows], [r[1] for r in rows]


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


def _read_label_map(path: str) -> np.ndarray:
    with PILImage.open(path) as pil:
        arr = np.array(pil)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int64, copy=False)


def _patch_starts(length: int, patch_size: int, stride: int) -> List[int]:
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if stride <= 0:
        raise ValueError(f"patch_stride must be positive, got {stride}")
    if stride > patch_size:
        raise ValueError(
            f"patch_stride must be <= patch_size to cover the full image, got "
            f"stride={stride}, patch_size={patch_size}"
        )
    if length <= patch_size:
        return [0]

    last = length - patch_size
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def _patch_grid(height: int, width: int, patch_size: int, stride: int) -> List[Tuple[int, int]]:
    ys = _patch_starts(height, patch_size, stride)
    xs = _patch_starts(width, patch_size, stride)
    return [(y, x) for y in ys for x in xs]


def _axis_patch_regions(length: int, patch_size: int, stride: int) -> List[Dict[str, int]]:
    starts = _patch_starts(length, patch_size, stride)
    real_ends = [min(start + patch_size, length) for start in starts]
    centers = [(start + end) / 2.0 for start, end in zip(starts, real_ends)]

    boundaries = [0]
    for left, right in zip(centers[:-1], centers[1:]):
        boundary = int(round((left + right) / 2.0))
        boundaries.append(max(boundaries[-1], min(length, boundary)))
    boundaries.append(length)

    regions: List[Dict[str, int]] = []
    for i, start in enumerate(starts):
        abs_valid_start = boundaries[i]
        abs_valid_end = boundaries[i + 1]
        crop_valid_start = max(0, abs_valid_start - start)
        crop_valid_end = max(crop_valid_start, min(patch_size, abs_valid_end - start))
        regions.append(
            {
                "start": int(start),
                "crop_valid_start": int(crop_valid_start),
                "crop_valid_end": int(crop_valid_end),
                "paste_start": int(abs_valid_start),
                "valid_length": int(abs_valid_end - abs_valid_start),
            }
        )
    return regions


def _patch_grid_records(
    height: int,
    width: int,
    patch_size: int,
    stride: int,
) -> List[Dict[str, int]]:
    y_regions = _axis_patch_regions(height, patch_size, stride)
    x_regions = _axis_patch_regions(width, patch_size, stride)
    records: List[Dict[str, int]] = []
    for y_region in y_regions:
        for x_region in x_regions:
            records.append(
                {
                    "x": x_region["start"],
                    "y": y_region["start"],
                    "width": int(patch_size),
                    "height": int(patch_size),
                    "crop_valid_x": x_region["crop_valid_start"],
                    "crop_valid_y": y_region["crop_valid_start"],
                    "valid_width": x_region["valid_length"],
                    "valid_height": y_region["valid_length"],
                    "paste_x": x_region["paste_start"],
                    "paste_y": y_region["paste_start"],
                }
            )
    return records


class InstancePathDataset(VisionDataset, _PackedStringMixin):
    """
    PathSeg2 instance segmentation dataset.

    Each sample is one image and one text query. The query targets all instances
    in the corresponding integer instance-id mask, where 0 is background and
    positive values are independent object ids.
    """

    def __init__(
        self,
        dataset_name: str,
        training: bool,
        root: str = str(DATA_ROOT),
        transforms: Optional[List[Any]] = None,
        use_caching: bool = False,
        return_raw_images: bool = True,
        load_segmentation: bool = True,
        return_geometry: bool = False,
        max_input_points: int = 5,
        return_box_prob: float = 0.0,
        return_point_prob: float = 0.0,
        min_instance_area: int = 1,
        max_instances_per_image: Optional[int] = None,
        limit_per_dataset: Optional[int] = None,
        patchify_eval: bool = False,
        patch_size: int = 256,
        patch_stride: int = 192,
        split: Optional[str] = None,
        split_seed: int = 123,
    ) -> None:
        root_path = Path(root)
        super().__init__(root=str(root_path))
        self.root = root_path

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
        self.min_instance_area = int(min_instance_area)
        self.max_instances_per_image = max_instances_per_image
        self.limit_per_dataset = limit_per_dataset
        self.patchify_eval = bool(patchify_eval and not training)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self._image_cache: Dict[str, PILImage.Image] = {}

        image_paths, instance_paths = self._collect_samples()
        if not instance_paths:
            raise RuntimeError(
                f"No instance mask files found for dataset_name={dataset_name}, "
                f"split={self.split}, root={self.root}"
            )

        self._num_samples = len(instance_paths)
        self.image_bytes, self.image_offsets = _pack_strings_to_shared_tensors(image_paths)
        self.instance_bytes, self.instance_offsets = _pack_strings_to_shared_tensors(instance_paths)

        self._image_bytes_np = None
        self._image_offsets_np = None
        self._instance_bytes_np = None
        self._instance_offsets_np = None

        del image_paths
        del instance_paths
        gc.collect()

    def _collect_samples(self) -> Tuple[List[str], List[str]]:
        image_paths: List[str] = []
        instance_paths: List[str] = []

        source_split = "train" if self.split == "val" else self.split
        for ds in _resolve_dataset_names(self.root, self.dataset_name, source_split):
            current_images, current_instances = _load_or_build_manifest(
                self.root, ds, source_split
            )
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
                    (image, mask) for image, mask in zip(current_images, current_instances)
                    if (image in val_images) == (self.split == "val")
                ]
                current_images = [image for image, _ in selected]
                current_instances = [mask for _, mask in selected]
            if self.limit_per_dataset is not None and self.limit_per_dataset > 0:
                current_images = current_images[: self.limit_per_dataset]
                current_instances = current_instances[: self.limit_per_dataset]
            image_paths.extend(current_images)
            instance_paths.extend(current_instances)

        return image_paths, instance_paths

    def __len__(self) -> int:
        return self._num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _get_image_path(self, idx: int) -> str:
        return self._decode_packed_string("image_bytes", "image_offsets", idx)

    def _get_instance_path(self, idx: int) -> str:
        return self._decode_packed_string("instance_bytes", "instance_offsets", idx)

    def _load_pil_image(self, image_path: str) -> PILImage.Image:
        if self.cache_images and image_path in self._image_cache:
            return self._image_cache[image_path]

        with PILImage.open(image_path) as pil:
            image = pil.convert("RGB")

        if self.cache_images:
            self._image_cache[image_path] = image
        return image

    def _select_instance_ids(self, inst_map: np.ndarray) -> List[int]:
        ids = [int(x) for x in np.unique(inst_map) if int(x) > 0]
        if self.min_instance_area <= 1 and self.max_instances_per_image is None:
            return ids

        ids_and_areas = []
        for inst_id in ids:
            area = int((inst_map == inst_id).sum())
            if area >= self.min_instance_area:
                ids_and_areas.append((inst_id, area))

        if self.max_instances_per_image is not None and len(ids_and_areas) > self.max_instances_per_image:
            ids_and_areas = sorted(ids_and_areas, key=lambda x: x[1], reverse=True)[
                : self.max_instances_per_image
            ]
            ids_and_areas = sorted(ids_and_areas, key=lambda x: x[0])

        return [inst_id for inst_id, _ in ids_and_areas]

    def _eval_patch_records(
        self,
        height: int,
        width: int,
    ) -> Optional[List[Dict[str, int]]]:
        if not self.patchify_eval:
            return None
        # Validation and test use the same image-only grid, including empty tiles.
        return _patch_grid_records(
            height,
            width,
            self.patch_size,
            self.patch_stride,
        )

    def _objects_from_instance_patch(
        self,
        inst_patch: np.ndarray,
        selected_instance_ids: Collection[int],
        source: str,
    ) -> List[Object]:
        objects: List[Object] = []
        patch_instance_ids = [
            int(inst_id) for inst_id in np.unique(inst_patch) if int(inst_id) > 0
        ]

        for inst_id in patch_instance_ids:
            if inst_id not in selected_instance_ids:
                continue
            mask_np = inst_patch == inst_id
            area = int(mask_np.sum())
            if area < self.min_instance_area:
                continue

            ys, xs = np.where(mask_np)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            bbox_t = torch.tensor([x0, y0, x1 + 1, y1 + 1], dtype=torch.float32)
            segment = torch.from_numpy(mask_np.astype(np.uint8)) if self.load_segmentation else None
            objects.append(
                Object(
                    bbox=bbox_t,
                    area=float(area),
                    object_id=len(objects),
                    frame_index=-1,
                    segment=segment,
                    is_crowd=False,
                    source=source,
                )
            )
        return objects

    def _make_query(
        self,
        *,
        text_prompt: str,
        image_id: int,
        objects: List[Object],
        foreground_mask: np.ndarray,
        original_image_id: int,
        original_size: Tuple[int, int],
        patch_x: int = -1,
        patch_y: int = -1,
    ) -> FindQueryLoaded:
        object_ids_output = [obj.object_id for obj in objects]
        input_bbox = None
        input_bbox_label = None
        input_points = None
        if self.return_geometry:
            selected_mask = None
            if objects:
                selected_obj = random.choice(objects)
                selected_mask = selected_obj.segment.numpy().astype(np.uint8) if selected_obj.segment is not None else None
                if random.random() < self.return_box_prob:
                    input_bbox = selected_obj.bbox.view(1, 4)
                    input_bbox_label = torch.ones((1,), dtype=torch.long)
            if self.load_segmentation and random.random() < self.return_point_prob:
                n_points = max(1, self.max_input_points // 2)
                positive_points = (
                    sample_points_from_mask(selected_mask, n_points, mode="centered")
                    if selected_mask is not None else np.empty((0, 3), dtype=np.float64)
                )
                negative_points = sample_negative_points_from_mask(
                    foreground_mask, n_points, mode="random_mask",
                )
                pts = np.concatenate([positive_points, negative_points], axis=0)
                input_points = torch.as_tensor(pts).view(1, -1, 3)

        return FindQueryLoaded(
            query_text=text_prompt,
            image_id=image_id,
            object_ids_output=object_ids_output,
            is_exhaustive=True,
            is_pixel_exhaustive=True,
            query_processing_order=0,
            input_bbox=input_bbox,
            input_bbox_label=input_bbox_label,
            input_points=input_points,
            semantic_target=None,
            inference_metadata=InferenceMetadata(
                coco_image_id=-1 if self.training else original_image_id,
                original_image_id=-1 if self.training else original_image_id,
                original_category_id=1,
                original_size=original_size,
                object_id=patch_x,
                frame_index=patch_y,
                is_conditioning_only=False,
            ),
        )

    def _getitem_patchified_eval(
        self,
        idx: int,
        image_path: str,
        instance_path: str,
        text_prompt: str,
        pil_img: PILImage.Image,
        inst_map: np.ndarray,
        selected_instance_ids: Sequence[int],
        patch_records: Sequence[Dict[str, int]],
    ) -> Datapoint:
        width, height = pil_img.size
        source = Path(instance_path).parent.parent.name
        selected_instance_id_set = set(selected_instance_ids)

        images: List[Image] = []
        queries: List[FindQueryLoaded] = []
        raw_images: List[PILImage.Image] = []
        output_patch_records: List[Dict[str, int]] = []

        for patch_record in patch_records:
            patch_x = patch_record["x"]
            patch_y = patch_record["y"]
            patch_width = patch_record["width"]
            patch_height = patch_record["height"]
            patch_img = pil_img.crop(
                (
                    patch_x,
                    patch_y,
                    patch_x + patch_width,
                    patch_y + patch_height,
                )
            )
            image_valid_h = max(0, min(patch_height, height - patch_y))
            image_valid_w = max(0, min(patch_width, width - patch_x))

            inst_patch = np.zeros((patch_height, patch_width), dtype=inst_map.dtype)
            if image_valid_h > 0 and image_valid_w > 0:
                inst_patch[:image_valid_h, :image_valid_w] = inst_map[
                    patch_y : patch_y + image_valid_h,
                    patch_x : patch_x + image_valid_w,
                ]

            objects = self._objects_from_instance_patch(
                inst_patch,
                selected_instance_id_set,
                source=source,
            )
            image = Image(
                data=patch_img,
                objects=objects,
                size=(patch_height, patch_width),
                blurring_mask=None,
            )
            images.append(image)

            queries.append(
                self._make_query(
                    text_prompt=text_prompt,
                    image_id=len(images) - 1,
                    objects=objects,
                    foreground_mask=inst_patch > 0,
                    original_image_id=idx,
                    original_size=(patch_height, patch_width),
                    patch_x=patch_x,
                    patch_y=patch_y,
                )
            )
            output_patch_records.append(patch_record)
            if self.return_raw_images:
                raw_images.append(patch_img)

        datapoint = Datapoint(
            find_queries=queries,
            images=images,
            raw_images=raw_images if self.return_raw_images else None,
        )
        datapoint.image_path = instance_path
        patch_sizes = {
            (record["height"], record["width"]) for record in output_patch_records
        }
        uniform_patch_size = next(iter(patch_sizes))[0] if len(patch_sizes) == 1 else 0
        datapoint.patch_info = {
            "image_id": int(idx),
            "image_path": image_path,
            "instance_path": instance_path,
            "full_height": int(height),
            "full_width": int(width),
            "patch_size": int(uniform_patch_size),
            "patch_stride": self.patch_stride,
            "patches": output_patch_records,
        }
        return datapoint

    def __getitem__(self, idx: int) -> Datapoint:
        image_path = self._get_image_path(idx)
        instance_path = self._get_instance_path(idx)

        _, organ_name, level_name, object_name = _parse_mask_filename(Path(instance_path).name)
        text_prompt = build_text_prompt(
            organ_name,
            object_name,
            level_name,
            training=self.training,
        )

        pil_img = self._load_pil_image(image_path)
        width, height = pil_img.size

        inst_map = _read_label_map(instance_path)
        if inst_map.shape != (height, width):
            raise RuntimeError(
                f"Instance mask shape {inst_map.shape} does not match image size "
                f"{(height, width)} for {instance_path}"
            )

        selected_instance_ids = self._select_instance_ids(inst_map)
        patch_records = self._eval_patch_records(height, width)
        if patch_records is not None:
            datapoint = self._getitem_patchified_eval(
                idx=idx,
                image_path=image_path,
                instance_path=instance_path,
                text_prompt=text_prompt,
                pil_img=pil_img,
                inst_map=inst_map,
                selected_instance_ids=selected_instance_ids,
                patch_records=patch_records,
            )
            for transform in self.transforms:
                try:
                    datapoint = transform(datapoint)
                except TypeError:
                    datapoint = transform(datapoint, epoch=getattr(self, "epoch", 0))
            return datapoint

        objects: List[Object] = []
        for local_id, inst_id in enumerate(selected_instance_ids):
            mask_np = inst_map == inst_id
            if not mask_np.any():
                continue

            ys, xs = np.where(mask_np)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            bbox_t = torch.tensor([x0, y0, x1 + 1, y1 + 1], dtype=torch.float32)
            segment = torch.from_numpy(mask_np.astype(np.uint8)) if self.load_segmentation else None

            objects.append(
                Object(
                    bbox=bbox_t,
                    area=float(mask_np.sum()),
                    object_id=local_id,
                    frame_index=-1,
                    segment=segment,
                    is_crowd=False,
                    source=Path(instance_path).parent.parent.name,
                )
            )

        image = Image(
            data=pil_img,
            objects=objects,
            size=(height, width),
            blurring_mask=None,
        )

        query = self._make_query(
            text_prompt=text_prompt,
            image_id=0,
            objects=objects,
            foreground_mask=inst_map > 0,
            original_image_id=idx,
            original_size=(height, width),
        )

        datapoint = Datapoint(
            find_queries=[query],
            images=[image],
            raw_images=[pil_img] if self.return_raw_images else None,
        )
        datapoint.image_path = instance_path

        for transform in self.transforms:
            try:
                datapoint = transform(datapoint)
            except TypeError:
                datapoint = transform(datapoint, epoch=getattr(self, "epoch", 0))

        return datapoint


def debug() -> None:
    parser = argparse.ArgumentParser(description="Debug PathSeg2 instance dataset.")
    parser.add_argument("--root", default=str(DATA_ROOT))
    parser.add_argument("--dataset-name", default="all")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--show", type=int, default=2)
    args = parser.parse_args()

    dataset = InstancePathDataset(
        root=args.root,
        dataset_name=args.dataset_name,
        training=args.split == "train",
        transforms=[],
        return_raw_images=True,
        load_segmentation=True,
    )
    print(f"num_samples: {len(dataset)}")
    for idx in range(min(args.show, len(dataset))):
        dp = dataset[idx]
        print("=" * 80)
        print(f"idx: {idx}")
        print(f"image_path: {dataset._get_image_path(idx)}")
        print(f"instance_path: {dataset._get_instance_path(idx)}")
        print(f"query_text: {dp.find_queries[0].query_text}")
        print(f"image_size: {dp.images[0].size}")
        print(f"num_objects: {len(dp.images[0].objects)}")
        print(f"first_bbox: {dp.images[0].objects[0].bbox.tolist()}")


if __name__ == "__main__":
    debug()
