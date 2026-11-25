import copy
import gc
import json
import logging
import os
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from iopath.common.file_io import g_pathmgr
from PIL import Image as PILImage
from scipy.optimize import linear_sum_assignment

from sam3.model import box_ops
from sam3.model.data_misc import (
    BatchedDatapoint,
    BatchedFindTarget,
    BatchedInferenceMetadata,
    FindStage,
    interpolate,
)
from sam3.perflib.connected_components import connected_components
from sam3.train.utils.distributed import (
    all_gather,
    gather_to_rank_0_via_filesys,
    get_rank,
    get_world_size,
    is_main_process,
)


def _read_instance_label_map(path: str) -> np.ndarray:
    with PILImage.open(path) as pil:
        arr = np.array(pil)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int64, copy=False)


def _source_instance_paths(batch: Optional[Any]) -> Optional[List[str]]:
    paths = getattr(batch, "img_path_batch", None)
    if not paths or any(path is None for path in paths):
        return None
    if not all(Path(path).parent.name.endswith("_inst_mask") for path in paths):
        return None
    return [str(path) for path in paths]


def _instance_masks_from_label_map(
    inst_map: np.ndarray,
    min_area: int = 1,
    max_instances: int = -1,
) -> torch.Tensor:
    ids, counts = np.unique(inst_map, return_counts=True)
    ids_and_areas = [
        (int(inst_id), int(area))
        for inst_id, area in zip(ids, counts)
        if int(inst_id) > 0 and int(area) >= min_area
    ]
    if max_instances > 0 and len(ids_and_areas) > max_instances:
        ids_and_areas = sorted(ids_and_areas, key=lambda x: x[1], reverse=True)[
            :max_instances
        ]
        ids_and_areas = sorted(ids_and_areas, key=lambda x: x[0])

    if not ids_and_areas:
        h, w = inst_map.shape
        return torch.zeros((0, h, w), dtype=torch.bool)

    masks = [torch.from_numpy(inst_map == inst_id) for inst_id, _ in ids_and_areas]
    return torch.stack(masks, dim=0).bool()


def _as_nhw_masks(masks: torch.Tensor) -> torch.Tensor:
    if masks.ndim == 4 and masks.shape[1] == 1:
        return masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(f"Expected masks [N,H,W] or [N,1,H,W], got {tuple(masks.shape)}")
    return masks


def _boxes_from_binary_masks(
    masks: torch.Tensor,
    offset_x: int,
    offset_y: int,
) -> torch.Tensor:
    boxes = torch.zeros((masks.shape[0], 4), dtype=torch.float32)
    for i, mask in enumerate(masks):
        ys, xs = torch.where(mask)
        boxes[i] = torch.tensor(
            [
                int(xs.min().item()) + offset_x,
                int(ys.min().item()) + offset_y,
                int(xs.max().item()) + 1 + offset_x,
                int(ys.max().item()) + 1 + offset_y,
            ],
            dtype=torch.float32,
        )
    return boxes


def _boxes_from_full_binary_masks(masks: torch.Tensor) -> torch.Tensor:
    boxes = torch.zeros((masks.shape[0], 4), dtype=torch.float32)
    for i, mask in enumerate(masks):
        ys, xs = torch.where(mask)
        if ys.numel() == 0:
            continue
        boxes[i] = torch.tensor(
            [
                int(xs.min().item()),
                int(ys.min().item()),
                int(xs.max().item()) + 1,
                int(ys.max().item()) + 1,
            ],
            dtype=torch.float32,
        )
    return boxes


def _box_iou_one_to_many(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(box[:2], boxes[:, :2])
    rb = torch.minimum(box[2:], boxes[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area1 = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    area2 = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp(min=0)
    union = (area1 + area2 - inter).clamp(min=1.0)
    return inter / union


def _box_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    if iou_threshold <= 0:
        return torch.argsort(scores, descending=True)

    order = torch.argsort(scores, descending=True)
    keep = []
    while order.numel() > 0:
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = _box_iou_one_to_many(boxes[current], boxes[rest])
        order = rest[ious <= iou_threshold]
    return torch.stack(keep).long()


def _mask_nms(masks: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if masks.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    if iou_threshold <= 0:
        return torch.argsort(scores, descending=True)

    masks = masks.bool().cpu()
    scores = scores.cpu()
    areas = masks.flatten(1).sum(dim=1).float()
    boxes = _boxes_from_masks_for_pairwise(masks)

    order = torch.argsort(scores, descending=True)
    keep = []
    while order.numel() > 0:
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break

        rest = order[1:]
        current_box = boxes[current]
        rest_boxes = boxes[rest]
        overlap = (
            (current_box[0] < rest_boxes[:, 2])
            & (current_box[2] > rest_boxes[:, 0])
            & (current_box[1] < rest_boxes[:, 3])
            & (current_box[3] > rest_boxes[:, 1])
        )
        ious = torch.zeros((rest.numel(),), dtype=torch.float32)
        overlap_pos = torch.nonzero(overlap, as_tuple=False).flatten()
        if overlap_pos.numel() > 0:
            overlap_idx = rest[overlap_pos]
            x0, y0, x1, y1 = current_box.tolist()
            current_crop = masks[current, y0:y1, x0:x1]
            rest_crop = masks[overlap_idx, y0:y1, x0:x1]
            inter = (rest_crop & current_crop).flatten(1).sum(dim=1).float()
            union = areas[current] + areas[overlap_idx] - inter
            ious[overlap_pos] = inter / union.clamp(min=1.0)

        order = rest[ious <= iou_threshold]
    return torch.stack(keep).long()


def _mask_nms_with_offsets(
    patch_masks: List[torch.Tensor],
    boxes: torch.Tensor,
    offsets: torch.Tensor,
    order: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Full-image mask IoU NMS without allocating a full canvas per candidate."""
    if iou_threshold <= 0 or order.numel() == 0:
        return order

    areas = [int(mask.sum().item()) for mask in patch_masks]
    keep: List[int] = []
    while order.numel() > 0:
        current = int(order[0].item())
        keep.append(current)
        rest = order[1:]
        current_box = boxes[current]
        rest_boxes = boxes[rest]
        overlap = (
            (current_box[0] < rest_boxes[:, 2])
            & (current_box[2] > rest_boxes[:, 0])
            & (current_box[1] < rest_boxes[:, 3])
            & (current_box[3] > rest_boxes[:, 1])
        )
        suppress = torch.zeros(rest.numel(), dtype=torch.bool)
        current_x, current_y = offsets[current].tolist()
        for position in torch.nonzero(overlap, as_tuple=False).flatten().tolist():
            other = int(rest[position].item())
            x0, y0 = torch.maximum(current_box[:2], boxes[other, :2]).long().tolist()
            x1, y1 = torch.minimum(current_box[2:], boxes[other, 2:]).long().tolist()
            other_x, other_y = offsets[other].tolist()
            current_crop = patch_masks[current][
                y0 - current_y : y1 - current_y, x0 - current_x : x1 - current_x
            ]
            other_crop = patch_masks[other][
                y0 - other_y : y1 - other_y, x0 - other_x : x1 - other_x
            ]
            intersection = int((current_crop & other_crop).sum().item())
            union = areas[current] + areas[other] - intersection
            suppress[position] = intersection / max(union, 1) > iou_threshold
        order = rest[~suppress]

    return torch.tensor(keep, dtype=torch.long)


class InstancePostProcessImage(torch.nn.Module):
    """
    Post-process SAM3 detector outputs for class-agnostic instance segmentation.

    This intentionally lives outside postprocessors.py so the semantic PathSeg
    evaluation path remains unchanged.
    """

    def __init__(
        self,
        max_dets_per_img: int,
        iou_type: str = "segm",
        to_cpu: bool = True,
        use_original_ids: bool = True,
        use_original_sizes_box: bool = True,
        use_original_sizes_mask: bool = True,
        detection_threshold: float = 0.5,
        use_presence: bool = True,
        always_interpolate_masks_on_gpu: bool = True,
        aggregate_patches: bool = False,
        max_dets_per_patch: int = 128,
        patch_pre_nms_topk: int = 2000,
        patch_nms_iou_threshold: float = 0.7,
        patch_min_mask_area: int = 1,
        target_min_instance_area: int = 1,
        target_max_instances_per_image: int = -1,
        enable_morph_postprocess: bool = False,
        postprocess_min_instance_area: int = 16,
        postprocess_hole_area: int = 16,
        postprocess_split_components: bool = True,
    ) -> None:
        super().__init__()
        if iou_type != "segm":
            raise ValueError(f"InstancePostProcessImage only supports iou_type='segm', got {iou_type}")
        self.max_dets_per_img = max_dets_per_img
        self.iou_type = iou_type
        self.to_cpu = to_cpu
        self.use_original_ids = use_original_ids
        self.use_original_sizes_box = use_original_sizes_box
        self.use_original_sizes_mask = use_original_sizes_mask
        self.detection_threshold = detection_threshold
        self.use_presence = use_presence
        self.always_interpolate_masks_on_gpu = always_interpolate_masks_on_gpu
        self.aggregate_patches = aggregate_patches
        self.max_dets_per_patch = int(max_dets_per_patch)
        self.patch_pre_nms_topk = int(patch_pre_nms_topk)
        self.patch_nms_iou_threshold = float(patch_nms_iou_threshold)
        self.patch_min_mask_area = int(patch_min_mask_area)
        self.target_min_instance_area = int(target_min_instance_area)
        self.target_max_instances_per_image = int(target_max_instances_per_image)
        self.enable_morph_postprocess = bool(enable_morph_postprocess)
        self.postprocess_min_instance_area = int(postprocess_min_instance_area)
        self.postprocess_hole_area = int(postprocess_hole_area)
        self.postprocess_split_components = bool(postprocess_split_components)

    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target_sizes_boxes: torch.Tensor,
        target_sizes_masks: torch.Tensor,
        forced_labels: Optional[torch.Tensor] = None,
        consistent: bool = False,
        mask_threshold: float = 0.5,
    ) -> List[Dict[str, torch.Tensor]]:
        out_bbox = outputs.get("pred_boxes", None)
        out_logits = outputs["pred_logits"]
        pred_masks = outputs.get("pred_masks", None)
        out_probs = out_logits.sigmoid()

        if self.use_presence and "presence_logit_dec" in outputs:
            out_probs = out_probs * outputs["presence_logit_dec"].sigmoid().unsqueeze(1)

        boxes, scores, labels, keep = self._process_boxes_and_labels(
            target_sizes_boxes,
            forced_labels,
            out_bbox,
            out_probs,
        )
        masks = self._process_masks(
            target_sizes_masks,
            pred_masks,
            consistent=consistent,
            keep=keep,
            mask_threshold=mask_threshold,
        )

        results = {
            "scores": scores,
            "labels": labels,
            "boxes": boxes,
            "masks": masks,
        }
        return [dict(zip(results.keys(), res_tuple)) for res_tuple in zip(*results.values())]

    def _process_boxes_and_labels(
        self,
        target_sizes: torch.Tensor,
        forced_labels: Optional[torch.Tensor],
        out_bbox: Optional[torch.Tensor],
        out_probs: torch.Tensor,
    ):
        if out_bbox is None:
            raise ValueError("Instance segmentation postprocessing requires pred_boxes")

        if self.to_cpu:
            out_probs = out_probs.cpu()
        scores, labels = out_probs.max(-1)
        if forced_labels is None:
            labels = torch.ones_like(labels, dtype=torch.long)
        else:
            if self.to_cpu:
                forced_labels = forced_labels.cpu()
            labels = forced_labels[:, None].expand_as(labels)

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        if self.to_cpu:
            boxes = boxes.cpu()

        keep = None
        if self.detection_threshold > 0:
            keep = scores > self.detection_threshold
            boxes = [b[k.to(b.device)] for b, k in zip(boxes, keep)]
            scores = [s[k.to(s.device)] for s, k in zip(scores, keep)]
            labels = [l[k.to(l.device)] for l, k in zip(labels, keep)]

        return boxes, scores, labels, keep

    def _process_masks(
        self,
        target_sizes: torch.Tensor,
        pred_masks: Optional[torch.Tensor],
        consistent: bool = False,
        keep: Optional[List[torch.Tensor]] = None,
        mask_threshold: float = 0.5,
    ):
        if pred_masks is None:
            raise ValueError("Instance segmentation postprocessing requires pred_masks")

        if self.always_interpolate_masks_on_gpu and target_sizes.device.type == "cuda":
            pred_masks = pred_masks.to(device=target_sizes.device)

        if consistent:
            assert keep is None, "consistent=True with score filtering is not implemented"
            target_size = target_sizes.unique(dim=0)
            assert target_size.size(0) == 1, "Expecting all target sizes to be equal"
            out_masks = (
                interpolate(
                    pred_masks,
                    target_size.squeeze().tolist(),
                    mode="bilinear",
                    align_corners=False,
                ).sigmoid()
                > mask_threshold
            )
            return out_masks.cpu() if self.to_cpu else out_masks

        out_masks = [[]] * len(pred_masks)
        for i, mask in enumerate(pred_masks):
            h, w = target_sizes[i]
            if keep is not None:
                mask = mask[keep[i].to(mask.device)]

            interpolated = (
                interpolate(
                    mask.unsqueeze(1),
                    (int(h.item()), int(w.item())),
                    mode="bilinear",
                    align_corners=False,
                ).sigmoid()
                > mask_threshold
            )
            out_masks[i] = interpolated.cpu() if self.to_cpu else interpolated
        return out_masks

    def process_results(
        self,
        find_stages,
        find_metadatas: List[BatchedInferenceMetadata],
        **kwargs,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        if find_stages.loss_stages is not None:
            find_metadatas = [find_metadatas[i] for i in find_stages.loss_stages]

        batch = kwargs.get("batch", None)
        patch_infos = self._flatten_patch_infos(batch)
        if self.aggregate_patches and patch_infos is not None:
            return self._process_patch_results(find_stages, find_metadatas, patch_infos)
        if self.aggregate_patches and self._looks_like_patchified_eval(find_metadatas):
            raise RuntimeError(
                "Patchified instance eval is missing batch.patch_info_batch. "
                "This would merge patch predictions but keep only patch-local targets; "
                "make sure collate_fn_api returns patch_info_batch and trainer._step "
                "attaches it to the BatchedDatapoint."
            )

        results: Dict[int, Dict[str, torch.Tensor]] = {}
        for outputs, meta in zip(find_stages, find_metadatas):
            img_size_for_boxes = (
                meta.original_size
                if self.use_original_sizes_box
                else torch.ones_like(meta.original_size)
            )
            img_size_for_masks = (
                meta.original_size
                if self.use_original_sizes_mask
                else torch.ones_like(meta.original_size)
            )
            detection_results = self(
                outputs,
                img_size_for_boxes,
                img_size_for_masks,
                forced_labels=meta.original_category_id if self.use_original_ids else None,
            )
            ids = meta.original_image_id if self.use_original_ids else meta.coco_image_id
            assert len(detection_results) == len(ids)

            for img_id, result in zip(ids, detection_results):
                img_id_int = int(img_id.item())
                if img_id_int not in results:
                    results[img_id_int] = result
                    continue

                for key in result.keys():
                    results[img_id_int][key] = torch.cat(
                        [results[img_id_int][key], result[key]],
                        dim=0,
                    )

        for img_id, result in results.items():
            if (
                self.max_dets_per_img > 0
                and len(result["scores"]) > self.max_dets_per_img
            ):
                _, topk = torch.topk(result["scores"], self.max_dets_per_img, dim=0)
                for key in result.keys():
                    results[img_id][key] = result[key][topk.to(result[key].device)]

        self._postprocess_results_inplace(results)
        return results

    def _looks_like_patchified_eval(
        self,
        find_metadatas: List[BatchedInferenceMetadata],
    ) -> bool:
        for meta in find_metadatas:
            ids = getattr(meta, "original_image_id", None)
            patch_x = getattr(meta, "object_id", None)
            patch_y = getattr(meta, "frame_index", None)
            if ids is None or patch_x is None or patch_y is None:
                continue
            if not torch.is_tensor(ids):
                ids = torch.as_tensor(ids)
            if not torch.is_tensor(patch_x):
                patch_x = torch.as_tensor(patch_x)
            if not torch.is_tensor(patch_y):
                patch_y = torch.as_tensor(patch_y)
            if ids.numel() <= 1:
                continue
            has_duplicate_image_ids = torch.unique(ids.detach().cpu()).numel() < ids.numel()
            has_patch_coords = bool(
                ((patch_x.detach().cpu() >= 0) & (patch_y.detach().cpu() >= 0))
                .any()
                .item()
            )
            if has_duplicate_image_ids and has_patch_coords:
                return True
        return False

    def _flatten_patch_infos(self, batch: Optional[Any]) -> Optional[List[Dict[str, Any]]]:
        patch_info_batch = getattr(batch, "patch_info_batch", None)
        if not patch_info_batch or all(info is None for info in patch_info_batch):
            return None

        flat_patch_infos: List[Dict[str, Any]] = []
        for info in patch_info_batch:
            if info is None:
                return None
            for patch in info["patches"]:
                patch_info = dict(patch)
                patch_info["image_id"] = int(info["image_id"])
                patch_info["image_path"] = info["image_path"]
                patch_info["instance_path"] = info["instance_path"]
                patch_info["full_height"] = int(info["full_height"])
                patch_info["full_width"] = int(info["full_width"])
                flat_patch_infos.append(patch_info)
        return flat_patch_infos

    def _process_patch_results(
        self,
        find_stages,
        find_metadatas: List[BatchedInferenceMetadata],
        patch_infos: List[Dict[str, Any]],
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        expected_queries = sum(len(meta.original_image_id) for meta in find_metadatas)
        if expected_queries != len(patch_infos):
            raise RuntimeError(
                "Patch metadata/query count mismatch: "
                f"metadatas={expected_queries}, patch_infos={len(patch_infos)}"
            )

        pending: Dict[int, Dict[str, Any]] = {}
        patch_cursor = 0
        for outputs, meta in zip(find_stages, find_metadatas):
            count = len(meta.original_image_id)
            stage_patch_infos = patch_infos[patch_cursor : patch_cursor + count]
            patch_cursor += count

            detection_results = self(
                outputs,
                meta.original_size,
                meta.original_size,
                forced_labels=meta.original_category_id if self.use_original_ids else None,
            )
            ids = meta.original_image_id if self.use_original_ids else meta.coco_image_id
            assert len(detection_results) == len(ids) == len(stage_patch_infos)

            for img_id, result, patch_info in zip(ids, detection_results, stage_patch_infos):
                img_id_int = int(img_id.item())
                self._append_patch_prediction(pending, img_id_int, result, patch_info)

        return self._finalize_patch_predictions(pending)

    def _append_patch_prediction(
        self,
        pending: Dict[int, Dict[str, Any]],
        img_id: int,
        result: Dict[str, torch.Tensor],
        patch_info: Dict[str, Any],
    ) -> None:
        entry = pending.setdefault(
            img_id,
            {
                "full_height": int(patch_info["full_height"]),
                "full_width": int(patch_info["full_width"]),
                "scores": [],
                "labels": [],
                "boxes": [],
                "masks": [],
                "offsets": [],
                "valid_sizes": [],
            },
        )

        x0 = int(patch_info["x"])
        y0 = int(patch_info["y"])
        crop_valid_x = int(patch_info.get("crop_valid_x", 0))
        crop_valid_y = int(patch_info.get("crop_valid_y", 0))
        core_x0 = int(patch_info.get("paste_x", x0 + crop_valid_x))
        core_y0 = int(patch_info.get("paste_y", y0 + crop_valid_y))
        full_h = int(patch_info["full_height"])
        full_w = int(patch_info["full_width"])
        core_x1 = min(core_x0 + int(patch_info["valid_width"]), full_w)
        core_y1 = min(core_y0 + int(patch_info["valid_height"]), full_h)

        scores = result["scores"].float().cpu()
        if scores.numel() == 0:
            return

        labels = result["labels"].long().cpu()
        masks = _as_nhw_masks(result["masks"]).bool().cpu()

        mask_h, mask_w = int(masks.shape[-2]), int(masks.shape[-1])
        valid_h = max(0, min(mask_h, full_h - y0))
        valid_w = max(0, min(mask_w, full_w - x0))
        if valid_h <= 0 or valid_w <= 0:
            return

        # As in TME cell inference, the centroid selects the owning core.
        # Keep the complete mask, clipping only padding outside the source image.
        valid_masks = masks[:, :valid_h, :valid_w]
        mask_area = valid_masks.flatten(1).sum(dim=1)
        centroid_x = (
            valid_masks.sum(dim=1) * torch.arange(valid_w, dtype=torch.float32)
        ).sum(dim=1) / mask_area.clamp(min=1) + x0
        centroid_y = (
            valid_masks.sum(dim=2) * torch.arange(valid_h, dtype=torch.float32)
        ).sum(dim=1) / mask_area.clamp(min=1) + y0
        keep = (
            (mask_area >= max(1, self.patch_min_mask_area))
            & (centroid_x >= core_x0)
            & (centroid_x < core_x1)
            & (centroid_y >= core_y0)
            & (centroid_y < core_y1)
        )

        if not keep.any():
            return

        keep_idx = torch.nonzero(keep, as_tuple=False).flatten()
        scores = scores[keep_idx]
        labels = labels[keep_idx]
        valid_masks = valid_masks[keep_idx]
        if self.max_dets_per_patch > 0 and scores.numel() > self.max_dets_per_patch:
            _, topk = torch.topk(scores, self.max_dets_per_patch, dim=0)
            scores = scores[topk]
            labels = labels[topk]
            valid_masks = valid_masks[topk]

        num_kept = int(scores.numel())
        boxes = _boxes_from_binary_masks(valid_masks, x0, y0)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=full_w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=full_h)

        entry["scores"].append(scores)
        entry["labels"].append(labels)
        entry["boxes"].append(boxes)
        entry["masks"].extend(mask.clone() for mask in valid_masks)
        entry["offsets"].append(
            torch.tensor([[x0, y0]] * num_kept, dtype=torch.long)
        )
        entry["valid_sizes"].append(
            torch.tensor([[valid_h, valid_w]] * num_kept, dtype=torch.long)
        )

    def _finalize_patch_predictions(
        self, pending: Dict[int, Dict[str, Any]]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        results: Dict[int, Dict[str, torch.Tensor]] = {}
        for img_id, entry in pending.items():
            full_h = int(entry["full_height"])
            full_w = int(entry["full_width"])
            if not entry["scores"]:
                continue

            scores = torch.cat(entry["scores"], dim=0)
            labels = torch.cat(entry["labels"], dim=0)
            boxes = torch.cat(entry["boxes"], dim=0)
            patch_masks = entry["masks"]
            offsets = torch.cat(entry["offsets"], dim=0)
            valid_sizes = torch.cat(entry["valid_sizes"], dim=0)

            order = torch.argsort(scores, descending=True)
            if self.patch_pre_nms_topk > 0 and order.numel() > self.patch_pre_nms_topk:
                order = order[: self.patch_pre_nms_topk]

            if self.patch_nms_iou_threshold > 0 and order.numel() > 0:
                order = _mask_nms_with_offsets(
                    patch_masks,
                    boxes,
                    offsets,
                    order,
                    self.patch_nms_iou_threshold,
                )

            if self.max_dets_per_img > 0 and order.numel() > self.max_dets_per_img:
                order = order[: self.max_dets_per_img]

            selected_scores = scores[order]
            selected_labels = labels[order]
            selected_boxes = boxes[order]
            selected_masks = self._paste_patch_masks(
                [patch_masks[index] for index in order.tolist()],
                offsets[order],
                valid_sizes[order],
                full_h,
                full_w,
            )
            results[img_id] = {
                "scores": selected_scores,
                "labels": selected_labels,
                "boxes": selected_boxes,
                "masks": selected_masks,
            }
        self._postprocess_results_inplace(results)
        return results

    def _postprocess_results_inplace(
        self,
        results: Dict[int, Dict[str, torch.Tensor]],
    ) -> None:
        if not self.enable_morph_postprocess:
            return
        for img_id, result in results.items():
            processed = self.postprocess_instance_masks(
                masks=result["masks"],
                scores=result["scores"],
                labels=result["labels"],
            )
            results[img_id] = processed

    def postprocess_instance_masks(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        masks = _as_nhw_masks(masks).bool().cpu()
        scores = scores.float().cpu()
        labels = labels.long().cpu()
        if masks.numel() == 0:
            empty_boxes = torch.zeros((0, 4), dtype=torch.float32)
            return {
                "scores": scores[:0],
                "labels": labels[:0],
                "boxes": empty_boxes,
                "masks": masks,
            }

        min_area = max(int(self.postprocess_min_instance_area), self.patch_min_mask_area, 1)
        max_hole_area = max(int(self.postprocess_hole_area), 0)

        processed_masks: List[torch.Tensor] = []
        processed_scores: List[torch.Tensor] = []
        processed_labels: List[torch.Tensor] = []
        for mask, score, label in zip(masks, scores, labels):
            cleaned_components = self._clean_single_mask(
                mask,
                min_area=min_area,
                max_hole_area=max_hole_area,
            )
            for component in cleaned_components:
                processed_masks.append(component)
                processed_scores.append(score)
                processed_labels.append(label)

        if not processed_masks:
            return {
                "scores": scores[:0],
                "labels": labels[:0],
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "masks": masks[:0],
            }

        out_masks = torch.stack(processed_masks, dim=0).bool()
        out_scores = torch.stack(processed_scores, dim=0).float()
        out_labels = torch.stack(processed_labels, dim=0).long()
        order = torch.argsort(out_scores, descending=True)
        if self.max_dets_per_img > 0 and order.numel() > self.max_dets_per_img:
            order = order[: self.max_dets_per_img]
        out_masks = out_masks[order]
        out_scores = out_scores[order]
        out_labels = out_labels[order]
        return {
            "scores": out_scores,
            "labels": out_labels,
            "boxes": _boxes_from_full_binary_masks(out_masks),
            "masks": out_masks,
        }

    def _clean_single_mask(
        self,
        mask: torch.Tensor,
        min_area: int,
        max_hole_area: int,
    ) -> List[torch.Tensor]:
        mask = mask.bool().cpu()
        if int(mask.sum().item()) < min_area:
            return []

        filled = self._fill_small_holes(mask, max_hole_area=max_hole_area)
        components = self._foreground_components(filled)
        if not components:
            return []

        if not self.postprocess_split_components:
            merged = torch.zeros_like(filled)
            for component in components:
                if int(component.sum().item()) >= min_area:
                    merged |= component
            return [merged] if int(merged.sum().item()) >= min_area else []

        return [
            component
            for component in components
            if int(component.sum().item()) >= min_area
        ]

    @staticmethod
    def _fill_small_holes(mask: torch.Tensor, max_hole_area: int) -> torch.Tensor:
        if max_hole_area <= 0:
            return mask
        background = ~mask.bool()
        labels, counts = connected_components(background.unsqueeze(0))
        labels = labels.squeeze(0).squeeze(0).cpu()
        counts = counts.squeeze(0).squeeze(0).cpu()
        if labels.numel() == 0:
            return mask

        border_labels = torch.cat(
            [
                labels[0, :],
                labels[-1, :],
                labels[:, 0],
                labels[:, -1],
            ],
            dim=0,
        ).unique()
        hole = (labels > 0) & (counts <= max_hole_area)
        if border_labels.numel() > 0:
            touches_border = torch.isin(labels, border_labels)
            hole &= ~touches_border
        if not hole.any():
            return mask
        return mask | hole

    @staticmethod
    def _foreground_components(mask: torch.Tensor) -> List[torch.Tensor]:
        labels, _ = connected_components(mask.bool().unsqueeze(0))
        labels = labels.squeeze(0).squeeze(0).cpu()
        component_ids = [int(x) for x in torch.unique(labels).tolist() if int(x) > 0]
        return [(labels == component_id) for component_id in component_ids]

    def _paste_patch_masks(
        self,
        patch_masks: Sequence[torch.Tensor],
        offsets: torch.Tensor,
        valid_sizes: torch.Tensor,
        full_h: int,
        full_w: int,
    ) -> torch.Tensor:
        if len(patch_masks) == 0:
            return torch.zeros((0, full_h, full_w), dtype=torch.bool)

        full_masks = torch.zeros(
            (len(patch_masks), full_h, full_w),
            dtype=torch.bool,
        )
        for i, (mask, offset, valid_size) in enumerate(
            zip(patch_masks, offsets, valid_sizes)
        ):
            x0 = int(offset[0].item())
            y0 = int(offset[1].item())
            valid_h = int(valid_size[0].item())
            valid_w = int(valid_size[1].item())
            full_masks[i, y0 : y0 + valid_h, x0 : x0 + valid_w] = mask[
                :valid_h, :valid_w
            ]
        return full_masks

    def process_targets(
        self,
        find_targets,
        find_metadatas: List[BatchedInferenceMetadata],
        **kwargs,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        batch = kwargs.get("batch", None)
        source_paths = _source_instance_paths(batch)
        if source_paths is not None:
            image_ids: List[int] = []
            seen_image_ids = set()
            for meta in find_metadatas:
                original_ids = meta.original_image_id
                if torch.is_tensor(original_ids):
                    original_ids = original_ids.detach().cpu().tolist()
                for image_id in original_ids:
                    image_id = int(image_id)
                    if image_id not in seen_image_ids:
                        seen_image_ids.add(image_id)
                        image_ids.append(image_id)
            if len(image_ids) != len(source_paths):
                raise RuntimeError(
                    "Instance source path/image-id count mismatch: "
                    f"paths={len(source_paths)}, image_ids={len(image_ids)}"
                )

            return {
                image_id: {
                    "label_map": _compact_instance_label_map(
                        _read_instance_label_map(instance_path),
                        min_area=self.target_min_instance_area,
                        max_instances=self.target_max_instances_per_image,
                    )
                }
                for image_id, instance_path in zip(image_ids, source_paths)
            }

        patch_info_batch = getattr(batch, "patch_info_batch", None)
        if self.aggregate_patches and patch_info_batch and any(
            info is not None for info in patch_info_batch
        ):
            return self._process_patch_targets(patch_info_batch)
        if self.aggregate_patches and self._looks_like_patchified_eval(find_metadatas):
            raise RuntimeError(
                "Patchified instance eval is missing batch.patch_info_batch, "
                "so full-image targets cannot be reconstructed."
            )

        results: Dict[int, Dict[str, torch.Tensor]] = {}
        for targets, meta in zip(find_targets, find_metadatas):
            masks = targets["masks"]
            if masks is None:
                raise ValueError("InstanceSegDumper requires targets['masks']; set with_seg_masks=True")

            img_size_for_masks = (
                meta.original_size
                if self.use_original_sizes_mask
                else torch.ones_like(meta.original_size)
            )
            ids = meta.original_image_id if self.use_original_ids else meta.coco_image_id
            num_boxes = targets["num_boxes"].to("cpu").tolist()
            masks_per_image = torch.split(masks, num_boxes, dim=0)

            for img_id, img_masks, size in zip(ids, masks_per_image, img_size_for_masks):
                h, w = int(size[0].item()), int(size[1].item())
                if img_masks.numel() == 0:
                    resized = torch.zeros((0, h, w), dtype=torch.bool)
                else:
                    resized = (
                        interpolate(
                            img_masks.float().unsqueeze(1),
                            size=(h, w),
                            mode="nearest",
                        ).squeeze(1)
                        > 0.5
                    ).cpu()
                results[int(img_id.item())] = {"masks": resized}
        return results

    def _process_patch_targets(
        self,
        patch_info_batch: List[Optional[Dict[str, Any]]],
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        results: Dict[int, Dict[str, torch.Tensor]] = {}
        for info in patch_info_batch:
            if info is None:
                continue
            img_id = int(info["image_id"])
            if img_id in results:
                continue
            inst_map = _read_instance_label_map(info["instance_path"])
            masks = _instance_masks_from_label_map(
                inst_map,
                min_area=self.target_min_instance_area,
                max_instances=self.target_max_instances_per_image,
            )
            results[img_id] = {"masks": masks}
        return results



def _boxes_from_masks_for_pairwise(masks: torch.Tensor) -> torch.Tensor:
    boxes = torch.zeros((masks.shape[0], 4), dtype=torch.long)
    for i, mask in enumerate(masks):
        ys, xs = torch.where(mask)
        if ys.numel() == 0:
            continue
        boxes[i, 0] = xs.min()
        boxes[i, 1] = ys.min()
        boxes[i, 2] = xs.max() + 1
        boxes[i, 3] = ys.max() + 1
    return boxes


def _pairwise_mask_intersection_and_union(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    chunk_size: int = 64,
):
    if pred_masks.numel() == 0 or gt_masks.numel() == 0:
        shape = (pred_masks.shape[0], gt_masks.shape[0])
        empty = torch.zeros(shape, dtype=torch.float32)
        return empty, empty

    pred_masks = pred_masks.bool().cpu()
    gt_masks = gt_masks.bool().cpu()
    num_pred = pred_masks.shape[0]
    num_gt = gt_masks.shape[0]

    pred_area = pred_masks.flatten(1).sum(dim=1).float()
    gt_area = gt_masks.flatten(1).sum(dim=1).float()

    intersections = torch.zeros((num_pred, num_gt), dtype=torch.float32)
    unions = pred_area[:, None] + gt_area[None, :]

    pred_boxes = _boxes_from_masks_for_pairwise(pred_masks)
    gt_boxes = _boxes_from_masks_for_pairwise(gt_masks)
    bbox_overlap = (
        (pred_boxes[:, None, 0] < gt_boxes[None, :, 2])
        & (pred_boxes[:, None, 2] > gt_boxes[None, :, 0])
        & (pred_boxes[:, None, 1] < gt_boxes[None, :, 3])
        & (pred_boxes[:, None, 3] > gt_boxes[None, :, 1])
    )

    for pred_idx in range(num_pred):
        x0, y0, x1, y1 = pred_boxes[pred_idx].tolist()
        if x1 <= x0 or y1 <= y0:
            continue

        gt_idx = torch.nonzero(bbox_overlap[pred_idx], as_tuple=False).flatten()
        if gt_idx.numel() == 0:
            continue

        pred_crop = pred_masks[pred_idx, y0:y1, x0:x1]
        for start in range(0, gt_idx.numel(), chunk_size):
            cur_gt_idx = gt_idx[start : start + chunk_size]
            gt_crop = gt_masks[cur_gt_idx, y0:y1, x0:x1]
            inter = (gt_crop & pred_crop).flatten(1).sum(dim=1).float()
            intersections[pred_idx, cur_gt_idx] = inter
            unions[pred_idx, cur_gt_idx] = pred_area[pred_idx] + gt_area[cur_gt_idx] - inter

    return intersections, unions


INSTANCE_METRIC_KEYS = ("Dice", "AJI", "DQ", "SQ", "PQ", "AJI+")


def _empty_instance_metrics(num_pred: int, num_gt: int, pred_area: float, gt_area: float) -> Dict[str, float]:
    aji_union = float(pred_area + gt_area)
    return {
        "Dice": 0.0,
        "AJI": 0.0,
        "DQ": 0.0,
        "SQ": 0.0,
        "PQ": 0.0,
        "AJI+": 0.0,
        "aji_intersection": 0.0,
        "aji_union": aji_union,
        "num_gt": float(num_gt),
        "num_pred": float(num_pred),
    }


def _make_non_overlapping_masks(masks: torch.Tensor) -> torch.Tensor:
    if masks.shape[0] == 0:
        return masks.bool().cpu()

    occupied = torch.zeros_like(masks[0], dtype=torch.bool)
    kept = []
    for mask in masks.bool().cpu():
        current = mask & ~occupied
        if current.any():
            kept.append(current)
            occupied |= current
    if not kept:
        return masks[:0].bool().cpu()
    return torch.stack(kept, dim=0)


def _compact_instance_label_map(
    label_map: np.ndarray,
    min_area: int = 1,
    max_instances: int = -1,
) -> np.ndarray:
    label_map = np.asarray(label_map)
    if label_map.ndim != 2:
        raise ValueError(f"Expected label map [H,W], got {label_map.shape}")

    labels, inverse, counts = np.unique(
        label_map,
        return_inverse=True,
        return_counts=True,
    )
    selected = [
        (index, int(label), int(area))
        for index, (label, area) in enumerate(zip(labels, counts))
        if int(label) > 0 and int(area) >= min_area
    ]
    if max_instances > 0 and len(selected) > max_instances:
        selected = sorted(selected, key=lambda item: item[2], reverse=True)[
            :max_instances
        ]
        selected.sort(key=lambda item: item[1])

    compact_ids = np.zeros(len(labels), dtype=np.int32)
    for compact_id, (index, _, _) in enumerate(selected, start=1):
        compact_ids[index] = compact_id
    return compact_ids[inverse].reshape(label_map.shape)


def _ordered_masks_to_label_map(masks: torch.Tensor) -> np.ndarray:
    masks = _as_nhw_masks(masks).bool().cpu()
    height, width = masks.shape[-2:]
    label_map = np.zeros((height, width), dtype=np.int32)
    masks_np = masks.numpy()

    # Assign in reverse so the earliest, highest-priority mask owns overlaps.
    for mask_index in range(masks_np.shape[0] - 1, -1, -1):
        label_map[masks_np[mask_index]] = mask_index + 1
    return _compact_instance_label_map(label_map)


def _compute_instance_metrics_from_label_maps(
    pred_label_map: np.ndarray,
    gt_label_map: np.ndarray,
) -> Dict[str, float]:
    pred_label_map = _compact_instance_label_map(pred_label_map)
    gt_label_map = _compact_instance_label_map(gt_label_map)
    if pred_label_map.shape != gt_label_map.shape:
        raise ValueError(
            "Prediction and ground-truth label maps must have the same shape, "
            f"got {pred_label_map.shape} and {gt_label_map.shape}"
        )

    num_pred = int(pred_label_map.max(initial=0))
    num_gt = int(gt_label_map.max(initial=0))
    pair_ids = (
        pred_label_map.astype(np.int64, copy=False).ravel() * (num_gt + 1)
        + gt_label_map.astype(np.int64, copy=False).ravel()
    )
    contingency = np.bincount(
        pair_ids,
        minlength=(num_pred + 1) * (num_gt + 1),
    ).reshape(num_pred + 1, num_gt + 1)

    pred_area = torch.from_numpy(contingency[1:, :].sum(axis=1)).float()
    gt_area = torch.from_numpy(contingency[:, 1:].sum(axis=0)).float()
    total_pred_area = float(pred_area.sum().item())
    total_gt_area = float(gt_area.sum().item())

    if num_gt == 0 and num_pred == 0:
        return _empty_instance_metrics(num_pred, num_gt, 0.0, 0.0)
    if num_gt == 0:
        return _empty_instance_metrics(num_pred, num_gt, total_pred_area, 0.0)
    if num_pred == 0:
        return _empty_instance_metrics(num_pred, num_gt, 0.0, total_gt_area)

    intersections = torch.from_numpy(contingency[1:, 1:]).float()
    unions = pred_area[:, None] + gt_area[None, :] - intersections
    ious = intersections / unions.clamp(min=1.0)

    foreground_intersection = float(contingency[1:, 1:].sum())
    dice_denom = total_gt_area + total_pred_area
    dice = (
        2.0 * foreground_intersection / dice_denom
        if dice_denom > 0
        else 0.0
    )

    best_pred_for_gt = torch.argmax(ious, dim=0)
    best_iou_for_gt = ious[best_pred_for_gt, torch.arange(num_gt)]
    paired_gt = best_iou_for_gt > 0.0
    paired_pred = best_pred_for_gt[paired_gt]
    paired_gt_idx = torch.nonzero(paired_gt, as_tuple=False).flatten()
    aji_inter = (
        float(intersections[paired_pred, paired_gt_idx].sum().item())
        if paired_gt_idx.numel() > 0
        else 0.0
    )
    aji_union = (
        float(unions[paired_pred, paired_gt_idx].sum().item())
        if paired_gt_idx.numel() > 0
        else 0.0
    )
    paired_pred_mask = torch.zeros(num_pred, dtype=torch.bool)
    if paired_pred.numel() > 0:
        paired_pred_mask[paired_pred.unique()] = True
    aji_union += float(gt_area[~paired_gt].sum().item())
    aji_union += float(pred_area[~paired_pred_mask].sum().item())
    aji = aji_inter / aji_union if aji_union > 0 else 0.0

    pairwise_iou = ious.numpy()
    pq_pred, pq_gt = np.nonzero(pairwise_iou > 0.5)
    paired_iou = pairwise_iou[pq_pred, pq_gt]
    paired_true_ids = set((pq_gt + 1).tolist())
    paired_pred_ids = set((pq_pred + 1).tolist())
    tp = len(paired_true_ids)
    fp = num_pred - len(paired_pred_ids)
    fn = num_gt - len(paired_true_ids)
    dq_denom = tp + 0.5 * fp + 0.5 * fn
    dq = tp / dq_denom if dq_denom > 0 else 0.0
    sq = float(paired_iou.sum()) / (tp + 1.0e-6) if tp > 0 else 0.0
    pq = dq * sq

    plus_pred, plus_gt = linear_sum_assignment(-pairwise_iou)
    plus_iou = pairwise_iou[plus_pred, plus_gt]
    valid_plus = plus_iou > 0.0
    plus_pred = plus_pred[valid_plus]
    plus_gt = plus_gt[valid_plus]
    aji_plus_inter = (
        float(intersections[plus_pred, plus_gt].sum().item())
        if len(plus_gt) > 0
        else 0.0
    )
    aji_plus_union = (
        float(unions[plus_pred, plus_gt].sum().item())
        if len(plus_gt) > 0
        else 0.0
    )
    plus_gt_mask = torch.zeros(num_gt, dtype=torch.bool)
    plus_pred_mask = torch.zeros(num_pred, dtype=torch.bool)
    if len(plus_gt) > 0:
        plus_gt_mask[torch.from_numpy(plus_gt)] = True
        plus_pred_mask[torch.from_numpy(plus_pred)] = True
    aji_plus_union += float(gt_area[~plus_gt_mask].sum().item())
    aji_plus_union += float(pred_area[~plus_pred_mask].sum().item())
    aji_plus = aji_plus_inter / aji_plus_union if aji_plus_union > 0 else 0.0

    return {
        "Dice": float(dice),
        "AJI": float(aji),
        "DQ": float(dq),
        "SQ": float(sq),
        "PQ": float(pq),
        "AJI+": float(aji_plus),
        "aji_intersection": float(aji_inter),
        "aji_union": float(aji_union),
        "num_gt": float(num_gt),
        "num_pred": float(num_pred),
    }


def _compute_instance_metrics_for_image(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
) -> Dict[str, float]:
    return _compute_instance_metrics_from_label_maps(
        _ordered_masks_to_label_map(pred_masks),
        _ordered_masks_to_label_map(gt_masks),
    )


def _compute_aji_for_image(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> Dict[str, float]:
    return _compute_instance_metrics_for_image(pred_masks, gt_masks)


def _build_instance_metric_row(
    image_id: int,
    score_max: Optional[float],
    pred_label_map: np.ndarray,
    gt_label_map: np.ndarray,
) -> Dict[str, float]:
    row = {
        "image_id": int(image_id),
        **_compute_instance_metrics_from_label_maps(
            pred_label_map,
            gt_label_map,
        ),
    }
    if score_max is not None:
        row["score_max"] = float(score_max)
    return row


class InstanceSegDumper:
    """
    Distributed meter for class-agnostic instance segmentation metrics.

    It reports per-image mean Dice, AJI, DQ, SQ, PQ, and AJI+ using the same
    definitions as HoverNet's instance metric script.
    """

    def __init__(
        self,
        dump_dir: str,
        postprocessor: InstancePostProcessImage,
        maxdets: int = 100,
        iou_type: str = "segm",
        gather_pred_via_filesys: bool = False,
        merge_predictions: bool = True,
        iou_thresholds: Optional[List[float]] = None,
        adaptive_inference: bool = False,
        adaptive_threshold: int = 250,
        adaptive_min_size: int = 256,
        adaptive_max_depth: int = 2,
        adaptive_context: int = 0,
        metric_workers: int = 0,
        metric_queue_size: Optional[int] = None,
    ) -> None:
        if iou_type != "segm":
            raise ValueError(f"InstanceSegDumper only supports iou_type='segm', got {iou_type}")
        self.dump_dir = dump_dir
        self.postprocessor = postprocessor
        self.maxdets = int(maxdets)
        self.iou_type = iou_type
        self.gather_pred_via_filesys = gather_pred_via_filesys
        self.merge_predictions = merge_predictions
        self.iou_thresholds = iou_thresholds
        self.adaptive_inference = bool(adaptive_inference)
        self.adaptive_threshold = int(adaptive_threshold)
        self.adaptive_min_size = int(adaptive_min_size)
        self.adaptive_max_depth = int(adaptive_max_depth)
        self.adaptive_context = int(adaptive_context)
        self.metric_workers = int(metric_workers)
        if self.metric_workers < 0:
            raise ValueError(f"metric_workers must be non-negative, got {metric_workers}")
        default_queue_size = max(1, 2 * self.metric_workers)
        self.metric_queue_size = int(metric_queue_size or default_queue_size)
        if self.metric_queue_size <= 0:
            raise ValueError(
                f"metric_queue_size must be positive, got {metric_queue_size}"
            )
        self._metric_executor = (
            ThreadPoolExecutor(
                max_workers=self.metric_workers,
                thread_name_prefix="instance-metric",
            )
            if self.metric_workers > 0
            else None
        )
        self._pending_metric_rows: Deque[Future] = deque()

        if is_main_process():
            os.makedirs(self.dump_dir, exist_ok=True)
            logging.info(f"Created instance prediction dump directory: {self.dump_dir}")

        self.reset()

    @staticmethod
    def is_better(current: float, previous: float) -> bool:
        return current > previous

    @torch.no_grad()
    def update(self, *args, **kwargs) -> None:
        if self.adaptive_inference and self._can_use_adaptive_inference(kwargs.get("batch", None)):
            predictions = self._process_adaptive_results(kwargs["model"], kwargs["batch"])
        else:
            predictions = self.postprocessor.process_results(*args, **kwargs)
        targets = self.postprocessor.process_targets(*args, **kwargs)
        rows, total_gt, total_pred = self._prepare_aji_rows(predictions, targets)
        self.dump.extend(rows)
        self.total_gt += total_gt
        self.total_pred += total_pred

    def _submit_metric_row(
        self,
        *,
        image_id: int,
        score_max: Optional[float],
        pred_label_map: np.ndarray,
        gt_label_map: np.ndarray,
    ) -> None:
        if self._metric_executor is None:
            self.dump.append(
                _build_instance_metric_row(
                    image_id,
                    score_max,
                    pred_label_map,
                    gt_label_map,
                )
            )
            return

        if len(self._pending_metric_rows) >= self.metric_queue_size:
            self._collect_oldest_metric_row()
        self._pending_metric_rows.append(
            self._metric_executor.submit(
                _build_instance_metric_row,
                image_id,
                score_max,
                pred_label_map,
                gt_label_map,
            )
        )

    def _collect_oldest_metric_row(self) -> None:
        future = self._pending_metric_rows.popleft()
        self.dump.append(future.result())

    def _drain_metric_rows(self) -> None:
        while self._pending_metric_rows:
            self._collect_oldest_metric_row()

    def _can_use_adaptive_inference(self, batch: Optional[Any]) -> bool:
        if get_world_size() != 1:
            raise ValueError("adaptive_inference currently requires single-GPU eval to avoid uneven DDP forwards")
        if batch is None:
            return False
        patch_info_batch = getattr(batch, "patch_info_batch", None)
        if patch_info_batch and any(info is not None for info in patch_info_batch):
            return False
        return True

    def _make_empty_find_target(
        self,
        *,
        device: torch.device,
        height: int,
        width: int,
    ) -> BatchedFindTarget:
        del height, width
        return BatchedFindTarget(
            num_boxes=torch.zeros((1,), dtype=torch.long, device=device),
            boxes=torch.zeros((0, 4), dtype=torch.float32, device=device),
            boxes_padded=torch.zeros((1, 0, 4), dtype=torch.float32, device=device),
            target_points=None,
            target_points_padded=None,
            valid_boxes=None,
            valid_boxes_padded=None,
            valid_points=None,
            valid_points_padded=None,
            supervision_modes=None,
            valid_dense_target=None,
            is_exhaustive=torch.ones((1,), dtype=torch.bool, device=device),
            segments=torch.zeros((0, 0, 0), dtype=torch.bool, device=device),
            semantic_segments=None,
            is_valid_segment=torch.zeros((0,), dtype=torch.bool, device=device),
            repeated_boxes=torch.zeros((0, 4), dtype=torch.float32, device=device),
            object_ids=torch.zeros((0,), dtype=torch.long, device=device),
            object_ids_padded=torch.full((1, 0), -1, dtype=torch.long, device=device),
        )

    def _make_adaptive_batch(
        self,
        *,
        image: torch.Tensor,
        text_batch: List[str],
        text_id: int,
        image_id: int,
        category_id: int,
        crop_h: int,
        crop_w: int,
    ) -> BatchedDatapoint:
        device = image.device
        return BatchedDatapoint(
            img_batch=image.unsqueeze(0),
            find_text_batch=text_batch,
            find_inputs=[
                FindStage(
                    img_ids=torch.zeros((1,), dtype=torch.long, device=device),
                    text_ids=torch.tensor([text_id], dtype=torch.long, device=device),
                    input_boxes=torch.zeros((0, 1, 4), dtype=torch.float32, device=device),
                    input_boxes_label=torch.zeros((0, 1), dtype=torch.long, device=device),
                    input_boxes_mask=torch.zeros((1, 0), dtype=torch.bool, device=device),
                    input_points=torch.zeros((1, 0, 3), dtype=torch.float32, device=device),
                    input_points_mask=torch.zeros((1, 0), dtype=torch.bool, device=device),
                    object_ids=[[]],
                )
            ],
            find_targets=[
                self._make_empty_find_target(
                    device=device,
                    height=crop_h,
                    width=crop_w,
                )
            ],
            find_metadatas=[
                BatchedInferenceMetadata(
                    coco_image_id=torch.full((1,), -1, dtype=torch.long, device=device),
                    original_image_id=torch.tensor([image_id], dtype=torch.long, device=device),
                    original_category_id=torch.tensor([category_id], dtype=torch.int, device=device),
                    original_size=torch.tensor([[crop_h, crop_w]], dtype=torch.long, device=device),
                    object_id=torch.full((1,), -1, dtype=torch.long, device=device),
                    frame_index=torch.full((1,), -1, dtype=torch.long, device=device),
                    is_conditioning_only=[False],
                )
            ],
            raw_images=None,
        )

    def _run_adaptive_region(
        self,
        model,
        *,
        full_image: torch.Tensor,
        text_batch: List[str],
        text_id: int,
        image_id: int,
        category_id: int,
        region: Dict[str, int],
    ) -> Dict[str, torch.Tensor]:
        crop = full_image[
            :,
            region["crop_y0"] : region["crop_y1"],
            region["crop_x0"] : region["crop_x1"],
        ]
        crop_h, crop_w = int(crop.shape[-2]), int(crop.shape[-1])
        crop_batch = self._make_adaptive_batch(
            image=crop,
            text_batch=text_batch,
            text_id=text_id,
            image_id=image_id,
            category_id=category_id,
            crop_h=crop_h,
            crop_w=crop_w,
        )
        outputs = model(crop_batch)
        predictions = self.postprocessor.process_results(
            find_stages=outputs,
            find_metadatas=crop_batch.find_metadatas,
            batch=crop_batch,
        )
        return predictions.get(
            image_id,
            {
                "scores": torch.zeros((0,), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.long),
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "masks": torch.zeros((0, crop_h, crop_w), dtype=torch.bool),
            },
        )

    def _patch_info_from_region(
        self,
        region: Dict[str, int],
        *,
        full_h: int,
        full_w: int,
    ) -> Dict[str, int]:
        return {
            "x": int(region["crop_x0"]),
            "y": int(region["crop_y0"]),
            "width": int(region["crop_x1"] - region["crop_x0"]),
            "height": int(region["crop_y1"] - region["crop_y0"]),
            "crop_valid_x": int(region["valid_x0"] - region["crop_x0"]),
            "crop_valid_y": int(region["valid_y0"] - region["crop_y0"]),
            "valid_width": int(region["valid_x1"] - region["valid_x0"]),
            "valid_height": int(region["valid_y1"] - region["valid_y0"]),
            "paste_x": int(region["valid_x0"]),
            "paste_y": int(region["valid_y0"]),
            "full_height": int(full_h),
            "full_width": int(full_w),
        }

    def _count_valid_predictions(
        self,
        result: Dict[str, torch.Tensor],
        patch_info: Dict[str, int],
    ) -> int:
        scores = result["scores"]
        if scores.numel() == 0:
            return 0
        masks = _as_nhw_masks(result["masks"]).bool().cpu()
        crop_valid_x = int(patch_info["crop_valid_x"])
        crop_valid_y = int(patch_info["crop_valid_y"])
        valid_w = int(patch_info["valid_width"])
        valid_h = int(patch_info["valid_height"])
        valid_masks = masks[
            :,
            crop_valid_y : crop_valid_y + valid_h,
            crop_valid_x : crop_valid_x + valid_w,
        ]
        return int((valid_masks.flatten(1).sum(dim=1) >= self.postprocessor.patch_min_mask_area).sum().item())

    def _split_region(
        self,
        region: Dict[str, int],
        *,
        full_h: int,
        full_w: int,
    ) -> List[Dict[str, int]]:
        x0, y0 = int(region["valid_x0"]), int(region["valid_y0"])
        x1, y1 = int(region["valid_x1"]), int(region["valid_y1"])
        if x1 - x0 <= 1 or y1 - y0 <= 1:
            return []

        xm = (x0 + x1) // 2
        ym = (y0 + y1) // 2
        children = []
        for child_y0, child_y1 in ((y0, ym), (ym, y1)):
            for child_x0, child_x1 in ((x0, xm), (xm, x1)):
                if child_x1 <= child_x0 or child_y1 <= child_y0:
                    continue
                crop_x0 = max(0, child_x0 - self.adaptive_context)
                crop_y0 = max(0, child_y0 - self.adaptive_context)
                crop_x1 = min(full_w, child_x1 + self.adaptive_context)
                crop_y1 = min(full_h, child_y1 + self.adaptive_context)
                children.append(
                    {
                        "valid_x0": int(child_x0),
                        "valid_y0": int(child_y0),
                        "valid_x1": int(child_x1),
                        "valid_y1": int(child_y1),
                        "crop_x0": int(crop_x0),
                        "crop_y0": int(crop_y0),
                        "crop_x1": int(crop_x1),
                        "crop_y1": int(crop_y1),
                        "depth": int(region["depth"] + 1),
                    }
                )
        return children

    def _process_adaptive_results(self, model, batch: BatchedDatapoint):
        find_meta = batch.find_metadatas[0]
        if batch.img_batch.shape[0] != 1 or len(find_meta.original_image_id) != 1:
            raise ValueError("adaptive_inference currently requires val_batch_size=1")

        full_image = batch.img_batch[0]
        full_h, full_w = int(full_image.shape[-2]), int(full_image.shape[-1])
        image_id = int(find_meta.original_image_id[0].item())
        category_id = int(find_meta.original_category_id[0].item())
        text_id = int(batch.find_inputs[0].text_ids[0].item())

        pending: Dict[int, Dict[str, Any]] = {}
        queue = [
            {
                "valid_x0": 0,
                "valid_y0": 0,
                "valid_x1": full_w,
                "valid_y1": full_h,
                "crop_x0": 0,
                "crop_y0": 0,
                "crop_x1": full_w,
                "crop_y1": full_h,
                "depth": 0,
            }
        ]

        while queue:
            region = queue.pop(0)
            patch_info = self._patch_info_from_region(
                region,
                full_h=full_h,
                full_w=full_w,
            )
            result = self._run_adaptive_region(
                model,
                full_image=full_image,
                text_batch=batch.find_text_batch,
                text_id=text_id,
                image_id=image_id,
                category_id=category_id,
                region=region,
            )
            valid_count = self._count_valid_predictions(result, patch_info)
            valid_w = int(region["valid_x1"] - region["valid_x0"])
            valid_h = int(region["valid_y1"] - region["valid_y0"])
            should_split = (
                valid_count > self.adaptive_threshold
                and max(valid_w, valid_h) > self.adaptive_min_size
                and int(region["depth"]) < self.adaptive_max_depth
            )
            if should_split:
                queue.extend(
                    self._split_region(
                        region,
                        full_h=full_h,
                        full_w=full_w,
                    )
                )
                continue

            self.postprocessor._append_patch_prediction(
                pending,
                image_id,
                result,
                patch_info,
            )

        return self.postprocessor._finalize_patch_predictions(pending)

    @torch.no_grad()
    def _prepare_aji_rows(
        self,
        predictions: Dict[int, Dict[str, torch.Tensor]],
        targets: Dict[int, Dict[str, torch.Tensor]],
    ):
        total_gt = 0
        total_pred = 0

        for image_id, target in targets.items():
            if "label_map" in target:
                gt_label_map = _compact_instance_label_map(target["label_map"])
            else:
                gt_label_map = _ordered_masks_to_label_map(target["masks"])
            total_gt += int(gt_label_map.max(initial=0))

            pred = predictions.get(image_id)
            if pred is None or pred["scores"].numel() == 0:
                self._submit_metric_row(
                    image_id=int(image_id),
                    score_max=None,
                    pred_label_map=np.zeros_like(gt_label_map, dtype=np.int32),
                    gt_label_map=gt_label_map,
                )
                continue

            pred_masks = pred["masks"]
            if pred_masks.ndim == 4 and pred_masks.shape[1] == 1:
                pred_masks = pred_masks[:, 0]
            elif pred_masks.ndim != 3:
                raise ValueError(f"Expected pred masks [N,H,W] or [N,1,H,W], got {tuple(pred_masks.shape)}")

            scores = pred["scores"].float().cpu()
            order = torch.argsort(scores, descending=True)
            if self.maxdets > 0:
                order = order[: self.maxdets]
            scores = scores[order]
            pred_masks = pred_masks[order].bool()
            total_pred += int(pred_masks.shape[0])

            self._submit_metric_row(
                image_id=int(image_id),
                score_max=(
                    float(scores.max().item()) if scores.numel() > 0 else 0.0
                ),
                pred_label_map=_ordered_masks_to_label_map(pred_masks),
                gt_label_map=gt_label_map,
            )

        return [], total_gt, total_pred

    def _gather_state(self):
        self._drain_metric_rows()
        payload = {
            "rows": copy.deepcopy(self.dump),
            "total_gt": int(self.total_gt),
            "total_pred": int(self.total_pred),
        }
        if self.gather_pred_via_filesys:
            gathered = gather_to_rank_0_via_filesys(payload)
        else:
            gathered = all_gather(payload, force_cpu=True)
        return gathered

    def compute_synced(self):
        gathered = self._gather_state()
        if not is_main_process():
            self.reset()
            return {"": 0.0}

        rows = []
        total_gt = 0
        total_pred = 0
        for item in gathered:
            rows.extend(item["rows"])
            total_gt += int(item["total_gt"])
            total_pred += int(item.get("total_pred", 0))

        summary = self.summarize(rows, total_gt, total_pred)
        dumped_file = Path(self.dump_dir) / f"instance_predictions_{self.iou_type}_{get_rank()}.json"
        if self.merge_predictions:
            dumped_file = Path(self.dump_dir) / f"instance_predictions_{self.iou_type}.json"
        with g_pathmgr.open(str(dumped_file), "w") as f:
            json.dump({"summary": summary, "matches": rows}, f)

        self.reset()
        return summary

    def compute(self):
        self._drain_metric_rows()
        return self.summarize(self.dump, self.total_gt, self.total_pred)

    def summarize(self, rows: List[Dict], total_gt: int, total_pred: int = 0) -> Dict[str, float]:
        summary = {
            "num_images": float(len(rows)),
            "num_gt": float(total_gt),
            "num_pred": float(total_pred),
        }
        for key in INSTANCE_METRIC_KEYS:
            values = [float(row[key]) for row in rows if key in row]
            summary[key] = float(sum(values) / len(values)) if values else 0.0
        return summary

    def reset(self) -> None:
        if hasattr(self, "_pending_metric_rows"):
            self._drain_metric_rows()
        self.dump = []
        self.total_gt = 0
        self.total_pred = 0
        gc.collect()
