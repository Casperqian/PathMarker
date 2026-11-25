import argparse
import csv
import gc
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
from PIL import Image as PILImage
from torch.utils.data import Dataset
from torchvision import transforms

from sam3.train.data.hed_augmentation import HedLighterColorAugmenter
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image,
    InferenceMetadata,
)


from sam3.train.data.dataset_paths import PRETRAIN_ROOTS, ENTITY_TABLE_DATASETS

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]



class StainAugmentTransform:
    def __init__(self, augmenter):
        self.augmenter = augmenter

    def __call__(self, img):
        if isinstance(img, PILImage.Image):
            img = np.array(img)

        if hasattr(self.augmenter, "randomize"):
            self.augmenter.randomize()

        out = self.augmenter.transform(img)
        if isinstance(out, np.ndarray):
            out = PILImage.fromarray(out)
        return out


TRANSFORMS = {
    "train": transforms.Compose(
        [
            transforms.Resize((1008, 1008)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            StainAugmentTransform(HedLighterColorAugmenter()),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize((1008, 1008)),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    ),
    "demo": transforms.Compose(
        [
            transforms.Resize((1008, 1008)),
        ]
    ),
}


@dataclass(frozen=True)
class LabelInfo:
    organ_name: str
    level_name: str
    object_name: str


def _ds_base_dir(dataset_name: str) -> Path:
    return PRETRAIN_ROOTS[dataset_name]


def _metadata_path(dataset_name: str) -> Path:
    return PRETRAIN_ROOTS[dataset_name] / "all" / "metadata.csv"


def _normalize_image_stem(dataset_name: str, image_stem: str) -> str:
    if dataset_name in ENTITY_TABLE_DATASETS:
        return image_stem.replace("_pathology_unspecified", "")
    return image_stem


def _label_to_payload(label: str) -> str:
    """
    Strip the sample-specific prefix because training only needs organ/level/object.
    Input:
        <sample>_pathology_<organ>_<level>+<object>
    Output:
        <organ>_<level>+<object>
    """
    if "_pathology_" not in label:
        raise ValueError(f"Invalid label format: {label}")
    return label.split("_pathology_", 1)[1]


def _parse_label_payload(payload: str) -> LabelInfo:
    fields = payload.split("_")
    if len(fields) < 2:
        raise ValueError(f"Invalid label payload format: {payload}")

    organ_token = fields[0]
    rest = "_".join(fields[1:])

    organ_name = " ".join(organ_token.split("+"))
    rest_parts = rest.split("+")

    level_name = rest_parts[0]
    object_name = " ".join(rest_parts[1:]) if len(rest_parts) > 1 else ""

    return LabelInfo(
        organ_name=organ_name,
        level_name=level_name,
        object_name=object_name,
    )


# def _build_text_prompt(label_info: LabelInfo) -> str:
#     organ_part = (
#         f"in {label_info.organ_name} pathology image"
#         if label_info.organ_name != "unspecified"
#         else "in pathology image"
#     )
#     text_prompt = f"{label_info.level_name}-level {label_info.object_name} {organ_part}"
#     return " ".join(text_prompt.split())

def _build_text_prompt(label_info: LabelInfo) -> str:
    text_prompt = f"{label_info.object_name}"
    return " ".join(text_prompt.split())


def _pack_strings_to_shared_tensors(values: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    offsets = np.empty(len(values) + 1, dtype=np.int64)
    offsets[0] = 0

    buffer = bytearray()
    for i, value in enumerate(values, start=1):
        buffer.extend(value.encode("utf-8"))
        offsets[i] = len(buffer)

    # copy() is required because frombuffer(bytes/bytearray) can be read-only or retain
    # references to temporary Python objects.
    data = np.frombuffer(bytes(buffer), dtype=np.uint8).copy()

    data_tensor = torch.from_numpy(data)
    offsets_tensor = torch.from_numpy(offsets)
    data_tensor.share_memory_()
    offsets_tensor.share_memory_()
    return data_tensor, offsets_tensor


def _share_numpy_int_array(values: Sequence[int], dtype=np.int32) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(values, dtype=dtype))
    tensor.share_memory_()
    return tensor


class PretrainDataset(Dataset):
    """
    An optimized variant of the pretraining dataset that keeps resident metadata in
    compact shared-memory tensors instead of large Python object lists.

    Key optimizations:
    - No long-lived `List[SampleSpec]`
    - No long-lived `List[Path]`
    - Image paths are deduplicated
    - Label strings drop the sample-specific prefix before storage
    - Metadata buffers are stored in shared-memory CPU tensors, which is friendlier
      to multiprocessing DataLoader workers than arbitrary Python object graphs
    """

    def __init__(
        self,
        dataset_name: str,
        transform: Optional[Callable] = None,
        num_copy: int = 1,
        split: str = "train",
        val_fraction: float = 0.0,
        split_seed: int = 123,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split={split}. Expected 'train' or 'val'.")
        if not 0 <= val_fraction < 1:
            raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}.")

        self.dataset_name = dataset_name
        self.transform = transform or transforms.ToTensor()
        self.num_copy = num_copy
        self.split = split
        self.val_fraction = val_fraction
        self.split_seed = split_seed

        unique_paths, sample_path_indices, label_payloads = self._collect_metadata(dataset_name)
        if not label_payloads:
            raise RuntimeError(f"No samples found for dataset={dataset_name}, split={split}")

        self._num_samples = len(label_payloads)

        self.path_bytes, self.path_offsets = _pack_strings_to_shared_tensors(unique_paths)
        self.sample_path_indices = _share_numpy_int_array(sample_path_indices, dtype=np.int32)
        self.label_bytes, self.label_offsets = _pack_strings_to_shared_tensors(label_payloads)

        self._path_bytes_np = None
        self._path_offsets_np = None
        self._label_bytes_np = None
        self._label_offsets_np = None

        # Release temporary Python containers aggressively before workers are created.
        del unique_paths
        del sample_path_indices
        del label_payloads
        gc.collect()

    def __len__(self) -> int:
        return self._num_samples

    def _select_split_paths(self, dataset_name: str, image_paths) -> set:
        image_paths = sorted(image_paths)
        if self.val_fraction <= 0 or not image_paths:
            return set(image_paths) if self.split == "train" else set()

        if len(image_paths) == 1:
            n_val = 0
        else:
            n_val = int(round(len(image_paths) * self.val_fraction))
            n_val = min(max(1, n_val), len(image_paths) - 1)

        rng = random.Random(f"{self.split_seed}:{dataset_name}")
        rng.shuffle(image_paths)
        val_paths = set(image_paths[:n_val])

        if self.split == "val":
            return val_paths
        return set(image_paths) - val_paths

    def _collect_metadata(
        self, dataset_name: str
    ) -> Tuple[List[str], List[int], List[str]]:
        if dataset_name == "all":
            dataset_list = list(PRETRAIN_ROOTS)
        else:
            dataset_list = [dataset_name]

        unique_paths: List[str] = []
        path_to_index: Dict[str, int] = {}
        sample_path_indices: List[int] = []
        label_payloads: List[str] = []

        for ds_name in dataset_list:
            md_path = _metadata_path(ds_name)
            if not md_path.exists():
                print(f"[WARN] Metadata not found: {md_path}")
                continue

            if ds_name in ENTITY_TABLE_DATASETS:
                rows = self._iter_entity_table_rows(ds_name, md_path)
            else:
                rows = self._iter_standard_rows(ds_name, md_path)

            labels_by_path: Dict[str, List[str]] = {}
            for image_path, label_payload in rows:
                labels_by_path.setdefault(image_path, []).append(label_payload)

            split_paths = self._select_split_paths(ds_name, labels_by_path.keys())
            for image_path, current_label_payloads in labels_by_path.items():
                if image_path not in split_paths:
                    continue

                path_idx = path_to_index.get(image_path)
                if path_idx is None:
                    path_idx = len(unique_paths)
                    unique_paths.append(image_path)
                    path_to_index[image_path] = path_idx

                sample_path_indices.extend([path_idx] * len(current_label_payloads))
                label_payloads.extend(current_label_payloads)

        return unique_paths, sample_path_indices, label_payloads

    def _iter_standard_rows(self, dataset_name: str, md_path: Path):
        # with md_path.open("r", newline="") as f:
        with md_path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            required_cols = {"image", "label"}
            missing = required_cols - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{dataset_name} metadata missing columns: {missing}, columns found: {reader.fieldnames}")

            resolve_cache: Dict[str, str] = {}
            for row in reader:
                image_stem = str(row["image"])
                label_stem = str(row["label"])

                image_key = _normalize_image_stem(dataset_name, image_stem)
                image_path = resolve_cache.get(image_key)
                if image_path is None:
                    image_path = str(self._find_image_path(dataset_name, image_key))
                    resolve_cache[image_key] = image_path

                yield image_path, _label_to_payload(label_stem)

    def _iter_entity_table_rows(self, dataset_name: str, md_path: Path):
        with md_path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            # required_cols = {"filename", "tissue", "cell", "nuclei"}
            required_cols = {"filename", "tissue", "cell"}
            missing = required_cols - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{dataset_name} metadata missing columns: {missing}")

            for row in reader:
                filename = str(row["filename"])
                if not filename:
                    continue

                file_stem = Path(filename).stem
                image_id = f"{file_stem}_pathology_unspecified"
                image_path = str(PRETRAIN_ROOTS[dataset_name] / "images" / filename)

                # for level in ("tissue", "cell", "nuclei"):
                for level in ("tissue", "cell"):
                    items = row.get(level, "")
                    if not items:
                        continue

                    entities = {entity.strip() for entity in items.split(";") if entity.strip()}
                    for entity in entities:
                        label_id = f"{image_id}_{level}+{entity.replace(' ', '+')}"
                        yield image_path, _label_to_payload(label_id)

    def _find_image_path(self, dataset_name: str, image_stem: str) -> Path:
        base_dir = _ds_base_dir(dataset_name)

        candidate_dirs = [
            base_dir / "all",
            base_dir / "train",
            base_dir / "images",
        ]

        checked_dirs = []

        for image_dir in candidate_dirs:
            if not image_dir.exists():
                continue

            checked_dirs.append(str(image_dir))

            for ext in IMG_EXTS:
                exact_path = image_dir / f"{image_stem}{ext}"
                if exact_path.exists():
                    return exact_path

                fuzzy_matches = sorted(image_dir.glob(f"{image_stem}*{ext}"))
                if fuzzy_matches:
                    return fuzzy_matches[0]

        raise FileNotFoundError(
            f"Image not found for dataset={dataset_name}, image_stem={image_stem}. "
            f"Checked dirs: {checked_dirs}, exts={IMG_EXTS}"
        )

    @staticmethod
    def _build_image(image_data, h: int, w: int):
        return Image(
            data=image_data,
            objects=[],
            size=(h, w),
            blurring_mask=None,
        )

    @staticmethod
    def _build_find_query(text_prompt: str, idx: int, copy_idx: int, h: int, w: int):
        return FindQueryLoaded(
            query_text=text_prompt,
            image_id=copy_idx,
            object_ids_output=[],
            is_exhaustive=False,
            is_pixel_exhaustive=False,
            query_processing_order=0,
            input_bbox=None,
            input_bbox_label=None,
            input_points=None,
            semantic_target=None,
            inference_metadata=InferenceMetadata(
                coco_image_id=idx,
                original_image_id=idx,
                original_category_id=-1,
                original_size=(h, w),
                object_id=-1,
                frame_index=-1,
                is_conditioning_only=False,
            ),
        )

    def _get_numpy_view(self, tensor_attr: str):
        cache_attr = f"_{tensor_attr}_np"
        cached = getattr(self, cache_attr)
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

    def _get_image_path(self, idx: int) -> str:
        path_idx = int(self.sample_path_indices[idx])
        return self._decode_packed_string("path_bytes", "path_offsets", path_idx)

    def _get_label_payload(self, idx: int) -> str:
        return self._decode_packed_string("label_bytes", "label_offsets", idx)

    def __getitem__(self, idx: int) -> Datapoint:
        label_payload = self._get_label_payload(idx)
        label_info = _parse_label_payload(label_payload)
        text_prompt = _build_text_prompt(label_info)
        image_path = self._get_image_path(idx)

        with PILImage.open(image_path) as im:
            pil_img = im.convert("RGB")

        w, h = pil_img.size

        images = []
        find_queries = []

        for copy_idx in range(self.num_copy):
            src_img = pil_img if self.num_copy == 1 else pil_img.copy()
            image_data = self.transform(src_img) if self.transform is not None else src_img

            images.append(self._build_image(image_data, h, w))
            find_queries.append(self._build_find_query(text_prompt, idx, copy_idx, h, w))

        datapoint = Datapoint(
            find_queries=find_queries,
            images=images,
            raw_images=None,
        )

        datapoint.label = {
            "organ": label_info.organ_name,
            "level": label_info.level_name,
            "object": label_info.object_name,
        }
        datapoint.image_path = image_path

        return datapoint


def debug():
    parser = argparse.ArgumentParser(description="Debug PretrainDataset metadata parsing.")
    parser.add_argument(
        "--dataset-name",
        default="DatasetD",
        help="Dataset name to inspect. Use 'all' to scan every configured dataset.",
    )
    parser.add_argument(
        "--num-copy",
        type=int,
        default=1,
        help="num_copy passed to the dataset.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=3,
        help="Number of samples to print.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="If set, only inspect this sample index.",
    )
    args = parser.parse_args()

    dataset = PretrainDataset(
        dataset_name=args.dataset_name,
        transform=TRANSFORMS["test"],
        num_copy=args.num_copy,
    )

    num_unique_paths = int(dataset.path_offsets.numel() - 1)
    print(f"dataset_name      : {args.dataset_name}")
    print(f"num_samples       : {len(dataset)}")
    print(f"num_unique_paths  : {num_unique_paths}")
    print(f"path_bytes_MB     : {dataset.path_bytes.numel() / 1024 / 1024:.2f}")
    print(f"label_bytes_MB    : {dataset.label_bytes.numel() / 1024 / 1024:.2f}")
    print(f"num_copy          : {dataset.num_copy}")

    if args.index is not None:
        inspect_indices = [args.index]
    else:
        inspect_indices = list(range(min(args.show, len(dataset))))

    for idx in inspect_indices:
        label_payload = dataset._get_label_payload(idx)
        label_info = _parse_label_payload(label_payload)
        text_prompt = _build_text_prompt(label_info)
        image_path = dataset._get_image_path(idx)

        print("\n" + "=" * 80)
        print(f"idx              : {idx}")
        print(f"image_path       : {image_path}")
        print(f"image_exists     : {Path(image_path).exists()}")
        print(f"label_payload    : {label_payload}")
        print(f"label            : {label_info}")
        print(f"text_prompt      : {text_prompt}")

        datapoint = dataset[idx]
        image_tensor = datapoint.images[0].data
        print(f"tensor_shape      : {tuple(image_tensor.shape)}")
        print(f"tensor_dtype      : {image_tensor.dtype}")
        print(f"tensor_minmax     : ({float(image_tensor.min()):.4f}, {float(image_tensor.max()):.4f})")
        print(f"datapoint_label   : {datapoint.label}")
        print(f"datapoint_path    : {datapoint.image_path}")
        print(f"query_text        : {datapoint.find_queries[0].query_text}")
        print(f"original_size     : {datapoint.find_queries[0].inference_metadata.original_size}")


if __name__ == "__main__":
    debug()
