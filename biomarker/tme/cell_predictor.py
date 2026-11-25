"""Batched SAM3 cell inference for the seven-class TME pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
from PIL import Image

from biomarker import segment_tissue, utils
from .tme_utils import CELL_TYPES


@dataclass
class CellPrediction:
    class_name: str
    score: float
    mask: np.ndarray
    tumor_status: bool = False
    tumor_score: float | None = None


def nms_predictions(
    predictions: Sequence[CellPrediction], iou_threshold: float = 0.5,
) -> list[CellPrediction]:
    """Keep the highest-scoring mask per cell across prompts, retaining tumor evidence."""
    candidates = []
    for prediction in sorted(predictions, key=lambda item: item.score, reverse=True):
        mask = np.asarray(prediction.mask, dtype=bool)
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        tumor = prediction.tumor_status or prediction.class_name == "Neoplastic cell"
        prediction = replace(
            prediction, mask=mask, tumor_status=bool(tumor),
            tumor_score=prediction.tumor_score if prediction.tumor_score is not None
            else prediction.score if tumor else None,
        )
        candidates.append((prediction, len(xs), (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)))

    kept = []
    for prediction, area, box in candidates:
        duplicate = None
        for index, (previous, previous_area, previous_box) in enumerate(kept):
            if iou_threshold <= 0:
                break
            x0, y0 = max(box[0], previous_box[0]), max(box[1], previous_box[1])
            x1, y1 = min(box[2], previous_box[2]), min(box[3], previous_box[3])
            if x1 <= x0 or y1 <= y0:
                continue
            intersection = np.count_nonzero(
                prediction.mask[y0:y1, x0:x1] & previous.mask[y0:y1, x0:x1]
            )
            if intersection / float(area + previous_area - intersection) > iou_threshold:
                duplicate = index
                break
        if duplicate is None:
            kept.append((prediction, area, box))
        elif prediction.tumor_status:
            previous, previous_area, previous_box = kept[duplicate]
            scores = [s for s in (previous.tumor_score, prediction.tumor_score) if s is not None]
            kept[duplicate] = (
                replace(previous, tumor_status=True, tumor_score=max(scores)),
                previous_area, previous_box,
            )
    return [prediction for prediction, _, _ in kept]


class Sam3CellPredictor:
    def __init__(
        self, checkpoint: Path, bpe_path: Path, device: str = "cuda",
        model_resolution: int = 1008, num_queries: int = 300,
        detection_threshold: float = 0.5, max_dets_per_tile: int = 300,
    ):
        import torch

        sys.path.insert(0, str(utils.SAM3_ROOT))
        from sam3.model_builder import build_sam3_image_model
        from sam3.eval.instance_seg_writer import InstancePostProcessImage

        if device.startswith("cuda"):
            torch.cuda.set_device(torch.device(device) if ":" in device else torch.cuda.current_device())
        self.model = build_sam3_image_model(
            bpe_path=str(bpe_path), checkpoint_path=str(checkpoint),
            load_from_HF=False, device=device, eval_mode=True,
            enable_segmentation=True, enable_matcher=True, num_queries=num_queries,
            separate_instance_pixel_decoder=True,
        ).to(device).eval()
        self.segmentor = segment_tissue.PromptSegmentor(
            self.model, CELL_TYPES, device, resolution=model_resolution,
        )
        self.postprocessor = InstancePostProcessImage(
            max_dets_per_img=max_dets_per_tile, detection_threshold=detection_threshold,
            to_cpu=True, use_presence=False, aggregate_patches=False,
        )
        self.device = device
        self.resolution = model_resolution
        self.max_dets_per_tile = max_dets_per_tile

    def predict_tiles(self, images: Sequence[Image.Image]) -> list[list[CellPrediction]]:
        import torch

        if not images:
            return []
        num_prompts = len(CELL_TYPES)
        with torch.inference_mode():
            tensor = segment_tissue.preprocess_images(images, self.resolution, self.device)
            backbone_out = self.model.backbone.forward_image(tensor)
            backbone_out.update(dict(self.segmentor.text_features))
            output = self.model.forward_grounding(
                backbone_out=backbone_out,
                find_input=self.segmentor._find_stage_for_batch(len(images)),
                find_target=None,
                geometric_prompt=self.segmentor._geometric_prompt_for_batch(len(images)),
            )
            sizes = torch.tensor(
                [[image.height, image.width] for image in images for _ in CELL_TYPES],
                dtype=torch.long, device=self.device,
            )
            results = self.postprocessor(output, sizes, sizes)
        batches = [[] for _ in images]
        for index, result in enumerate(results):
            class_name = CELL_TYPES[index % num_prompts]
            image = images[index // num_prompts]
            for score, mask in zip(result["scores"], result["masks"]):
                mask_array = mask.detach().cpu().numpy().reshape(image.height, image.width).astype(bool)
                if mask_array.any():
                    batches[index // num_prompts].append(CellPrediction(class_name, float(score), mask_array))
        if self.max_dets_per_tile > 0:
            batches = [
                sorted(predictions, key=lambda item: item.score, reverse=True)[:self.max_dets_per_tile]
                for predictions in batches
            ]
        return batches
