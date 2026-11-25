# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
COCO prediction dumper for distributed training.

Handles collection and dumping of COCO-format predictions from models.
Supports distributed processing with multiple GPUs/processes.
"""

import copy
import gc
import heapq
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import pycocotools.mask as mask_utils
import torch
from iopath.common.file_io import g_pathmgr
from sam3.eval.coco_eval_offline import convert_to_xywh
from sam3.train.masks_ops import rle_encode
from sam3.train.utils.distributed import (
    all_gather,
    gather_to_rank_0_via_filesys,
    get_rank,
    is_main_process,
)


### Helper functions and classes


class HeapElement:
    """Utility class to make a heap with a custom comparator based on score."""

    def __init__(self, val):
        self.val = val

    def __lt__(self, other):
        return self.val["score"] < other.val["score"]


def _gather_dump(dump, gather_pred_via_filesys: bool):
    logging.info("CustomSegWriter: Gathering predictions from all processes")
    gc.collect()
    if gather_pred_via_filesys:
        return gather_to_rank_0_via_filesys(dump)
    return all_gather(dump, force_cpu=True)


def _flatten_gathered_dump(gathered_dump):
    merged = []
    for rank_dump in gathered_dump:
        merged.extend(rank_dump)
    return merged


def _mean(values):
    return float(sum(values) / (len(values) + 1e-12))


def _summarize_dice_iou_rows(rows):
    if not rows:
        return {"dice_mean": 0.0, "iou_mean": 0.0, "n_preds": 0.0}

    meters = {
        "dice_mean": _mean([x["dice"] for x in rows]),
        "iou_mean": _mean([x["iou"] for x in rows]),
        "n_preds": float(len(rows)),
    }

    by_cat = defaultdict(list)
    for row in rows:
        by_cat[int(row["category_id"])].append(float(row["dice"]))
    meters.update({f"dice_cat_{k}": _mean(v) for k, v in by_cat.items()})
    return meters


def _append_no_pred_row(out, image_id: int, gt_area: int):
    out.append(
        {
            "image_id": int(image_id),
            "category_id": -1,
            "score": 0.0,
            "dice": 0.0,
            "iou": 0.0,
            "pred_area": 0,
            "gt_area": int(gt_area),
            "union_area": int(gt_area),
            "matched_gt_index": 0,
        }
    )


def _squeeze_prediction_masks(masks):
    if masks.ndim == 4 and masks.shape[1] == 1:
        return masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(
            f"pred masks expected [N,H,W] or [N,1,H,W], got {tuple(masks.shape)}"
        )
    return masks


def _filename_by_image_id(postprocessor, batch, find_metadatas):
    filename_by_image_id = {}
    img_path_batch = getattr(batch, "img_path_batch", None)
    if img_path_batch is None or find_metadatas is None:
        return filename_by_image_id

    for meta in find_metadatas:
        ids = (
            meta.original_image_id
            if postprocessor.use_original_ids
            else meta.coco_image_id
        )
        if len(ids) != len(img_path_batch):
            continue
        return {
            int(img_id.item()): os.path.basename(path) if path is not None else None
            for img_id, path in zip(ids, img_path_batch)
        }
    return filename_by_image_id


class _CustomSegMetricMixin:
    def gather_and_merge_predictions(self):
        return _flatten_gathered_dump(
            _gather_dump(self.dump, self.gather_pred_via_filesys)
        )

    def compute_synced(self):
        dumped_file = self.synchronize_between_processes()
        if not is_main_process():
            return {"": 0.0}

        with g_pathmgr.open(str(dumped_file), "r") as f:
            merged = json.load(f)
        return self.summarize_metrics(merged)

    def summarize_metrics(self, merged):
        return _summarize_dice_iou_rows(merged)


class PredictionDumper:
    """
    Handles collection and dumping of COCO-format predictions from a model.

    This class processes model outputs through a postprocessor, converts them to COCO format,
    and saves them to disk. It supports distributed processing with multiple GPUs/processes.
    """

    def __init__(
        self,
        dump_dir: str,
        postprocessor,
        maxdets: int,
        iou_type: str,
        gather_pred_via_filesys: bool = False,
        merge_predictions: bool = False,
        pred_file_evaluators: Optional[Any] = None,
    ):
        """
        Initialize the PredictionDumper.

        Args:
            dump_dir: Directory to dump predictions.
            postprocessor: Module to convert the model's output into COCO format.
            maxdets: Maximum number of detections per image.
            iou_type: IoU type to evaluate. Can include "bbox", "segm"
            gather_pred_via_filesys: If True, use the filesystem for collective gathers across
                processes (requires a shared filesystem). Otherwise, use torch collective ops.
            merge_predictions: If True, merge predictions from all processes and dump to a single file.
        """
        self.iou_type = iou_type # bbox
        self.maxdets = maxdets # 100
        self.dump_dir = dump_dir
        self.postprocessor = postprocessor
        self.gather_pred_via_filesys = gather_pred_via_filesys
        self.merge_predictions = merge_predictions
        self.pred_file_evaluators = pred_file_evaluators
        if self.pred_file_evaluators is not None:
            assert (
                merge_predictions
            ), "merge_predictions must be True if pred_file_evaluators are provided"
        assert self.dump_dir is not None, "dump_dir must be provided"

        if is_main_process():
            os.makedirs(self.dump_dir, exist_ok=True)
            logging.info(f"Created prediction dump directory: {self.dump_dir}")

        # Initialize state
        self.reset()

    def update(self, *args, **kwargs):
        """
        Process and accumulate predictions from model outputs.

        Args:
            *args, **kwargs: Arguments passed to postprocessor.process_results()
        """
        predictions = self.postprocessor.process_results(*args, **kwargs)
        results = self.prepare(predictions, self.iou_type)
        self._dump(results)

    def _dump(self, results):
        """
        Add results to the dump list with precision rounding.

        Args:
            results: List of prediction dictionaries in COCO format.
        """
        dumped_results = copy.deepcopy(results)
        for r in dumped_results:
            if "bbox" in r:
                r["bbox"] = [round(coord, 5) for coord in r["bbox"]]
            r["score"] = round(r["score"], 5)
        self.dump.extend(dumped_results)

    def synchronize_between_processes(self):
        """
        Synchronize predictions across all processes and save to disk.

        If gather_pred_via_filesys is True, uses filesystem for gathering.
        Otherwise, uses torch distributed collective operations.
        Saves per-rank predictions to separate JSON files.
        """
        logging.info("Prediction Dumper: Synchronizing between processes")

        if not self.merge_predictions:
            dumped_file = (
                Path(self.dump_dir)
                / f"coco_predictions_{self.iou_type}_{get_rank()}.json"
            )
            logging.info(
                f"Prediction Dumper: Dumping local predictions to {dumped_file}"
            )
            with g_pathmgr.open(str(dumped_file), "w") as f:
                json.dump(self.dump, f)
        else:
            self.dump = self.gather_and_merge_predictions()
            dumped_file = Path(self.dump_dir) / f"coco_predictions_{self.iou_type}.json"
            if is_main_process():
                logging.info(
                    f"Prediction Dumper: Dumping merged predictions to {dumped_file}"
                )
                with g_pathmgr.open(str(dumped_file), "w") as f:
                    json.dump(self.dump, f)

        self.reset()
        return dumped_file

    def gather_and_merge_predictions(self):
        """
        Gather predictions from all processes and merge them, keeping top predictions per image.

        This method collects predictions from all processes, then keeps only the top maxdets
        predictions per image based on score. It also deduplicates predictions by (image_id, category_id).

        Returns:
            List of merged prediction dictionaries.
        """
        logging.info("Prediction Dumper: Gathering predictions from all processes")
        gc.collect()

        if self.gather_pred_via_filesys:
            dump = gather_to_rank_0_via_filesys(self.dump)
        else:
            dump = all_gather(self.dump, force_cpu=True)

        # Combine predictions, keeping only top maxdets per image
        preds_by_image = defaultdict(list)
        seen_img_cat = set()

        for cur_dump in dump:
            cur_seen_img_cat = set()
            for p in cur_dump:
                image_id = p["image_id"]
                cat_id = p["category_id"]

                # Skip if we've already seen this image/category pair in a previous dump
                if (image_id, cat_id) in seen_img_cat:
                    continue

                cur_seen_img_cat.add((image_id, cat_id))

                # Use a min-heap to keep top predictions
                if len(preds_by_image[image_id]) < self.maxdets:
                    heapq.heappush(preds_by_image[image_id], HeapElement(p))
                else:
                    heapq.heappushpop(preds_by_image[image_id], HeapElement(p))

            seen_img_cat.update(cur_seen_img_cat)

        # Flatten the heap elements back to a list
        merged_dump = sum(
            [[h.val for h in cur_preds] for cur_preds in preds_by_image.values()], []
        )

        return merged_dump

    def compute_synced(self):
        """
        Synchronize predictions across processes and compute summary.

        Returns:
            Summary dictionary from summarize().
        """
        dumped_file = self.synchronize_between_processes()
        if not is_main_process():
            return {"": 0.0}

        meters = {}
        if self.pred_file_evaluators is not None:
            for evaluator in self.pred_file_evaluators:
                results = evaluator.evaluate(dumped_file)
                meters.update(results)

        if len(meters) == 0:
            meters = {"": 0.0}
        return meters

    def compute(self):
        """
        Compute without synchronization.

        Returns:
            Empty metric dictionary.
        """
        return {"": 0.0}

    def reset(self):
        """Reset internal state for a new evaluation round."""
        self.dump = []

    def prepare(self, predictions, iou_type):
        """
        Route predictions to the appropriate preparation method based on iou_type.

        Args:
            predictions: Dictionary mapping image IDs to prediction dictionaries.
            iou_type: Type of evaluation ("bbox", "segm").

        Returns:
            List of COCO-format prediction dictionaries.
        """
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        else:
            raise ValueError(f"Unknown iou type: {iou_type}")

    def prepare_for_coco_detection(self, predictions):
        """
        Convert predictions to COCO detection format.

        Args:
            predictions: Dictionary mapping image IDs to prediction dictionaries
                containing "boxes", "scores", and "labels".

        Returns:
            List of COCO-format detection dictionaries.
        """
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    @torch.no_grad()
    def prepare_for_coco_segmentation(self, predictions):
        """
        Convert predictions to COCO segmentation format.

        Args:
            predictions: Dictionary mapping image IDs to prediction dictionaries
                containing "masks" or "masks_rle", "scores", and "labels".
                Optionally includes "boundaries" and "dilated_boundaries".

        Returns:
            List of COCO-format segmentation dictionaries with RLE-encoded masks.
        """
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue
            # print(f"prediction keys: {prediction.keys()}")
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            boxes = None
            if "boxes" in prediction:
                boxes = prediction["boxes"]
                boxes = convert_to_xywh(boxes).tolist()
                assert len(boxes) == len(scores)

            if "masks_rle" in prediction:
                rles = prediction["masks_rle"]
                areas = []
                for rle in rles:
                    cur_area = mask_utils.area(rle)
                    h, w = rle["size"]
                    areas.append(cur_area / (h * w))
            else:
                masks = prediction["masks"]
                masks = masks > 0.5
                h, w = masks.shape[-2:]

                areas = masks.flatten(1).sum(1) / (h * w)
                areas = areas.tolist()

                rles = rle_encode(masks.squeeze(1))

                # Memory cleanup
                del masks
                del prediction["masks"]

            assert len(areas) == len(rles) == len(scores)

            for k, rle in enumerate(rles):
                payload = {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "segmentation": rle,
                    "score": scores[k],
                    "area": areas[k],
                }
                if boxes is not None:
                    payload["bbox"] = boxes[k]

                coco_results.append(payload)

        return coco_results

class CustomSegDumper(_CustomSegMetricMixin, PredictionDumper):
    """
    Inherit PredictionDumper but:
    - NOT COCO format
    - dump per-instance segmentation metrics (dice/iou/areas + match info)
    - compute_synced returns aggregated meters
    """

    def __init__(
        self,
        *args,
        mask_threshold: float = 0.5,
        match_strategy: str = "best_dice",  # "best_dice" | "best_iou"
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mask_threshold = float(mask_threshold)
        self.match_strategy = match_strategy

    # -------- override: update 需要拿到 GT --------
    def update(self, *args, **kwargs):
        """
        Require gt_by_image in kwargs:
          gt_by_image[image_id] = {"masks": Tensor[M,H,W] or [M,1,H,W], "labels": Tensor[M]}
        """
        predictions = self.postprocessor.process_results(*args, **kwargs)
        targets = self.postprocessor.process_targets(*args, **kwargs)
        results = self.prepare(predictions, targets, self.iou_type)
        self._dump(results)

    # -------- override: prepare route（bbox/segm）--------
    def prepare(self, predictions, targets, iou_type, **kwargs):
        if iou_type != "segm":
            raise ValueError(f"CustomSegWriter only supports iou_type='segm', got {iou_type}")
        return self.prepare_for_seg_metrics(predictions, targets, **kwargs)
    
    @torch.no_grad()
    def prepare_for_seg_metrics(self, predictions, targets):
        eps = 1e-6
        out = []

        common_ids = sorted(set(predictions.keys()) & set(targets.keys()))
        for image_id in common_ids:
            pred = predictions[image_id]
            target = targets[image_id]

            # ---- GT: single mask ----
            gt_mask = target["masks"]
            if not torch.is_tensor(gt_mask):
                gt_mask = torch.as_tensor(gt_mask)
                
            assert gt_mask.ndim == 2, f"gt masks expected 2D tensor, got {gt_mask.ndim}D"

            gt_mask = gt_mask.to(torch.bool)
            g_area = int(gt_mask.sum().item())
            if g_area == 0:
                continue

            p_masks = pred["masks"]
            p_scores = pred["scores"]
            p_labels = pred["labels"]

            p_masks = _squeeze_prediction_masks(p_masks)

            if p_masks.shape[0] == 0:
                _append_no_pred_row(out, image_id, gt_area=g_area)
                continue

            p_masks = (p_masks > self.mask_threshold).to(torch.bool)

            # top-k by score
            if p_masks.shape[0] > self.maxdets:
                topk = torch.topk(p_scores, k=self.maxdets, largest=True).indices
                p_masks = p_masks[topk]
                p_labels = p_labels[topk]
                p_scores = p_scores[topk]

            p_area = p_masks.flatten(1).sum(dim=1)  # [N]
            valid = p_area > 0

            if not torch.any(valid):
                _append_no_pred_row(out, image_id, gt_area=g_area)
                continue

            pv = p_masks[valid]                    # [Nv,H,W]
            scores_v = p_scores[valid]             # [Nv]
            labels_v = p_labels[valid]             # [Nv]
            merged = pv.any(dim=0)  # [H,W] bool

            pred_area = merged.sum().to(torch.float32)          # scalar
            inter = (merged & gt_mask).sum().to(torch.float32)  # scalar
            g_area_f = float(g_area)
            union = pred_area + g_area_f - inter                # scalar
            iou_m = inter / (union + eps)
            dice_m = (2.0 * inter) / (pred_area + g_area_f + eps)
            score_m = float(scores_v.max().item())
            cat_m = int(labels_v[torch.argmax(scores_v)].item())  

            out.append({
                "image_id": int(image_id),
                "category_id": int(cat_m),
                "score": score_m,
                "dice": float(dice_m.item()),
                "iou": float(iou_m.item()),
                "pred_area": int(pred_area.item()),
                "gt_area": int(g_area),
                "union_area": int(union.item()),
                "matched_gt_index": 0,
            })

        return out

class SemanticSegmentationMetricsWriter(_CustomSegMetricMixin, PredictionDumper):
    """Write per-image semantic mask metrics and aggregate them across ranks."""

    def __init__(
        self,
        *args,
        mask_threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mask_threshold = float(mask_threshold)

    def update(self, *args, **kwargs):
        predictions = self.postprocessor.process_results(*args, **kwargs)
        targets = self.postprocessor.prepare_semantic_targets(*args, **kwargs)
        filename_by_image_id = _filename_by_image_id(
            self.postprocessor,
            kwargs.get("batch", None),
            kwargs.get("find_metadatas", None),
        )

        results = self.prepare(
            predictions,
            targets,
            self.iou_type,
            filename_by_image_id=filename_by_image_id,
        )
        self._dump(results)

    def prepare(self, predictions, targets, iou_type, **kwargs):
        if iou_type != "segm":
            raise ValueError(f"CustomSegWriter only supports iou_type='segm', got {iou_type}")
        return self.prepare_for_seg_metrics(predictions, targets, **kwargs)
    
    @torch.no_grad()
    def prepare_for_seg_metrics(
        self,
        predictions: dict,
        targets: dict,
        filename_by_image_id: Optional[dict] = None,
    ):
        summary = self.compute_semantic_metrics(
            predictions,
            targets,
            eps=1e-6,
            filename_by_image_id=filename_by_image_id,
        )

        return summary
    
    @torch.no_grad()
    def compute_instance_metrics(
        self,
        predictions: dict,
        targets: dict,
        *,
        iou_thresholds=None,
        maxdets: int = 100,
        eps: float = 1e-6,
    ) -> dict:
        if iou_thresholds is None:
            iou_thresholds = [x / 100 for x in range(50, 100, 5)]  # 0.50..0.95

        # store global lists (aligned per prediction)
        all_scores = []
        all_tps = {float(thr): [] for thr in iou_thresholds}
        total_gt = 0

        # iterate over shared ids (safer than targets.keys())
        ids = sorted(set(predictions.keys()) & set(targets.keys()))

        for img_id in ids:
            gt_m = targets[img_id]["masks"]          # [M,1,H,W] or bool
            pd_m = predictions[img_id]["masks"]      # [N,1,H,W]
            pd_s = predictions[img_id].get("scores", None)

            # sanitize
            gt_m = (gt_m > 0) if gt_m.dtype != torch.bool else gt_m
            pd_m = (pd_m > 0) if pd_m.dtype != torch.bool else pd_m

            gt_m = gt_m.squeeze(1)  # [M,H,W]
            pd_m = pd_m.squeeze(1)  # [N,H,W]

            M = gt_m.size(0)
            N = pd_m.size(0)
            total_gt += M

            if N == 0:
                continue

            if pd_s is None:
                pd_s = torch.ones((N,), device=pd_m.device, dtype=torch.float32)
            else:
                pd_s = pd_s.to(device=pd_m.device, dtype=torch.float32).view(-1)
                assert pd_s.numel() == N

            # per-image maxdets and per-image score sort (IMPORTANT)
            if maxdets > 0 and N > maxdets:
                pd_s, top_idx = pd_s.topk(maxdets, largest=True, sorted=True)
                pd_m = pd_m[top_idx]
            else:
                order = torch.argsort(pd_s, descending=True)
                pd_s = pd_s[order]
                pd_m = pd_m[order]

            N = pd_m.size(0)

            # IoU matrix: [N,M]
            if M > 0:
                inter = torch.einsum("nhw,mhw->nm", pd_m.float(), gt_m.float())
                area_p = pd_m.sum((1, 2), dtype=torch.float32).view(-1, 1)
                area_g = gt_m.sum((1, 2), dtype=torch.float32).view(1, -1)
                union = area_p + area_g - inter
                ious = inter / (union + eps)
            else:
                ious = None

            # append scores (cpu ok)
            all_scores.append(pd_s.detach().cpu())

            # matching per threshold (score-ordered greedy; allow second-best GT)
            for thr in iou_thresholds:
                thr = float(thr)
                tp = torch.zeros((N,), dtype=torch.float32, device=pd_m.device)

                if M == 0:
                    all_tps[thr].append(tp.cpu())
                    continue

                # mutable IoU copy for this threshold
                ious_thr = ious.clone()  # [N,M]
                matched = torch.zeros((M,), dtype=torch.bool, device=ious_thr.device)

                for p_idx in range(N):  # already score-sorted
                    # mask out already matched GT
                    ious_thr[p_idx, matched] = -1.0
                    max_iou, g_idx = torch.max(ious_thr[p_idx], dim=0)
                    if max_iou >= thr:
                        tp[p_idx] = 1.0
                        matched[g_idx] = True

                all_tps[thr].append(tp.cpu())

        if total_gt == 0:
            return {"mAP": 0.0, "AP50": 0.0, "AP75": 0.0, "num_gt": 0, "num_pred": 0}

        # concat global
        scores_cat = torch.cat(all_scores, dim=0)
        num_pred = int(scores_cat.numel())
        sort_idx = torch.argsort(scores_cat, descending=True)

        final_aps = {}
        for thr in iou_thresholds:
            thr = float(thr)
            tp_cat = torch.cat(all_tps[thr], dim=0)[sort_idx]
            fp_cat = 1.0 - tp_cat

            tp_cum = torch.cumsum(tp_cat, dim=0)
            fp_cum = torch.cumsum(fp_cat, dim=0)

            recall = tp_cum / (float(total_gt) + eps)
            precision = tp_cum / (tp_cum + fp_cum + eps)

            # 101-pt interpolated AP
            ap = 0.0
            for r in torch.linspace(0, 1, 101):
                mask = recall >= r
                p = precision[mask].max() if mask.any() else torch.tensor(0.0)
                ap += float(p.item())
            final_aps[thr] = ap / 101.0

        mAP = sum(final_aps.values()) / len(final_aps)
        return {
            "mAP": float(mAP),
            "AP50": float(final_aps.get(0.5, 0.0)),
            "AP75": float(final_aps.get(0.75, 0.0)),
            "num_gt": int(total_gt),
            "num_pred": int(num_pred),
        }   
    
    @torch.no_grad()
    def compute_semantic_metrics(
        self,
        predictions: dict,
        targets: dict,
        *,
        eps: float = 1e-6,
        filename_by_image_id: Optional[dict] = None,
    ) -> list:
        out = []
        filename_by_image_id = filename_by_image_id or {}

        for image_id in targets.keys():
            assert image_id in predictions, f"Image ID {image_id} found in targets but not in predictions"
            pred = predictions[image_id]
            target = targets[image_id]

            gt_masks = target["semantic_masks"]
            p_masks  = pred["semantic_masks"].squeeze()

            assert torch.is_tensor(gt_masks) and torch.is_tensor(p_masks), "semantic_masks must be tensors"
            assert gt_masks.shape == p_masks.shape, f"Expect same shape, got {gt_masks.shape=}, {p_masks.shape=}"

            pred_area = p_masks.sum(dtype=torch.float32)                # scalar
            gt_area   = gt_masks.sum(dtype=torch.float32)                  # scalar
            inter     = (p_masks & gt_masks).sum(dtype=torch.float32)     # scalar
            union     = pred_area + gt_area - inter                      # scalar
            iou  = inter / (union + eps)
            dice = (2.0 * inter) / (pred_area + gt_area + eps)

            category_id = int(pred.get("category_id", -1))  
            score = float(pred.get("score", 1.0))           

            row = {
                "image_id": int(image_id),
                "category_id": category_id,
                "score": score,
                "dice": float(dice.item()),
                "iou": float(iou.item()),
                "pred_area": int(pred_area.item()),
                "gt_area": int(gt_area.item()),
                "union_area": int(union.item()),
            }
            filename = filename_by_image_id.get(int(image_id))
            if filename is not None:
                row["filename"] = filename
            out.append(row)

        return out


class SemanticSegmentationMetricsWriterWithMasks(SemanticSegmentationMetricsWriter):
    """Spot-check writer that retains compressed semantic masks for review."""

    @torch.no_grad()
    def compute_semantic_metrics(
        self,
        predictions: dict,
        targets: dict,
        *,
        eps: float = 1e-6,
        filename_by_image_id: Optional[dict] = None,
    ) -> list:
        rows = super().compute_semantic_metrics(
            predictions,
            targets,
            eps=eps,
            filename_by_image_id=filename_by_image_id,
        )
        rows_by_id = {int(row["image_id"]): row for row in rows}
        for image_id, row in rows_by_id.items():
            pred_mask = predictions[image_id]["semantic_masks"].squeeze().to(torch.bool)
            gt_mask = targets[image_id]["semantic_masks"].squeeze().to(torch.bool)
            row["pred_mask_rle"] = rle_encode(pred_mask.unsqueeze(0))[0]
            row["gt_mask_rle"] = rle_encode(gt_mask.unsqueeze(0))[0]
        return rows
