# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from sam3.model.model_misc import SAM3Output

from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
from sam3.model.vl_combiner import SAM3VLBackbone
from sam3.perflib.nms import nms_masks

from sam3.train.data.collator import BatchedDatapoint

from .act_ckpt_utils import activation_ckpt_wrapper

from .box_ops import box_cxcywh_to_xyxy

from .geometry_encoders import Prompt
from .model_misc import inverse_sigmoid


def _update_out(out, out_name, out_value, auxiliary=True, update_aux=True):
    out[out_name] = out_value[-1] if auxiliary else out_value
    if auxiliary and update_aux:
        if "aux_outputs" not in out:
            out["aux_outputs"] = [{} for _ in range(len(out_value) - 1)]
        assert len(out["aux_outputs"]) == len(out_value) - 1
        for aux_output, aux_value in zip(out["aux_outputs"], out_value[:-1]):
            aux_output[out_name] = aux_value


class Sam3ImageEncoder(torch.nn.Module):
    TEXT_ID_FOR_TEXT = 0
    TEXT_ID_FOR_VISUAL = 1
    TEXT_ID_FOR_GEOMETRIC = 2

    def __init__(
        self,
        backbone: SAM3VLBackbone,
        **kwargs,
    ):
        super().__init__()
        self.backbone = backbone

    @property
    def device(self):
        self._device = getattr(self, "_device", None) or next(self.parameters()).device
        return self._device

    def to(self, *args, **kwargs):
        # clear cached _device in case the model is moved to a different device
        self._device = None
        return super().to(*args, **kwargs)

    def forward(self, input: BatchedDatapoint):
        device = self.device
        backbone_out = {"img_batch_all_stages": input.img_batch}
        backbone_out.update(self.backbone.forward_image(input.img_batch))
        num_frames = len(input.find_inputs)
        assert num_frames == 1
        find_input = input.find_inputs[0]
        text_outputs = self.backbone.forward_text(input.find_text_batch, device=device)
        backbone_out.update(text_outputs)
        txt_ids = find_input.text_ids
        out = {}
        out['backbone_fpn'] = backbone_out['backbone_fpn']
        out["language_features"] = backbone_out["language_features"][:, txt_ids]
        out["language_mask"] = backbone_out["language_mask"][txt_ids]
        # out["text_ids"] = txt_ids
        return out
