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


class PathMarkerSemanticModel(torch.nn.Module):
    TEXT_ID_FOR_TEXT = 0
    TEXT_ID_FOR_VISUAL = 1
    TEXT_ID_FOR_GEOMETRIC = 2

    def __init__(
        self,
        backbone: SAM3VLBackbone,
        transformer,
        input_geometry_encoder,
        segmentation_head=None,
        num_feature_levels=1,
        o2m_mask_predict=True,
        dot_prod_scoring=None,
        use_instance_query: bool = True,
        multimask_output: bool = True,
        use_act_checkpoint_seg_head: bool = True,
        interactivity_in_encoder: bool = True,
        matcher=None,
        use_dot_prod_scoring=True,
        supervise_joint_box_scores: bool = False,  # only relevant if using presence token/score
        detach_presence_in_joint_score: bool = False,  # only relevant if using presence token/score
        separate_scorer_for_instance: bool = False,
        num_interactive_steps_val: int = 0,
        inst_interactive_predictor: SAM3InteractiveImagePredictor = None,
        **kwargs,
    ):
        super().__init__()
        self.backbone = backbone
        self.geometry_encoder = input_geometry_encoder
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.segmentation_head = segmentation_head
        self.use_act_checkpoint_seg_head = use_act_checkpoint_seg_head
        self.interactivity_in_encoder = interactivity_in_encoder
        self.matcher = matcher

        self.num_interactive_steps_val = num_interactive_steps_val
        self.multimask_output = multimask_output

        self.inst_interactive_predictor = inst_interactive_predictor

    @property
    def device(self):
        self._device = getattr(self, "_device", None) or next(self.parameters()).device
        return self._device

    def to(self, *args, **kwargs):
        # clear cached _device in case the model is moved to a different device
        self._device = None
        return super().to(*args, **kwargs)

    def _get_img_feats(self, backbone_out, img_ids):
        """Retrieve correct image features from backbone output."""
        if "backbone_fpn" in backbone_out:
            if "id_mapping" in backbone_out and backbone_out["id_mapping"] is not None:
                img_ids = backbone_out["id_mapping"][img_ids]
                # If this assert fails, it likely means we're requesting different img_ids (perhaps a different frame?)
                # We currently don't expect this to happen. We could technically trigger a recompute here,
                # but likely at the cost of a cpu<->gpu sync point, which would deteriorate perf
                torch._assert_async((img_ids >= 0).all())

            vis_feats = backbone_out["backbone_fpn"][-self.num_feature_levels :]
            vis_pos_enc = backbone_out["vision_pos_enc"][-self.num_feature_levels :]
            vis_feat_sizes = [x.shape[-2:] for x in vis_pos_enc]  # (H, W) shapes
            # index and flatten visual features NxCxHxW => HWxNxC (batch-first => seq-first)
            img_feats = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_feats]
            img_pos_embeds = [
                x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_pos_enc
            ]
            return backbone_out, img_feats, img_pos_embeds, vis_feat_sizes

        # Image features not available in backbone output, so we compute them on the fly
        # This case likely occurs for video. In that case, we want to forward only the current frame
        img_batch = backbone_out["img_batch_all_stages"]
        if img_ids.numel() > 1:
            # Only forward backbone on unique image ids to avoid repetitive computation
            unique_ids, _ = torch.unique(img_ids, return_inverse=True)
        else:
            unique_ids, _ = img_ids, slice(None)
        # Compute the image features on those unique image ids
        # note: we allow using a list (or other indexable types) of tensors as img_batch
        # (e.g. for async frame loading in demo). In this case we index img_batch.tensors directly
        if isinstance(img_batch, torch.Tensor):
            image = img_batch[unique_ids]
        elif unique_ids.numel() == 1:
            image = img_batch[unique_ids.item()].unsqueeze(0)
        else:
            image = torch.stack([img_batch[i] for i in unique_ids.tolist()])
        # `img_batch` might be fp16 and offloaded to CPU
        image = image.to(dtype=torch.float32, device=self.device)
        # Next time we call this function, we want to remember which indices we computed
        id_mapping = torch.full(
            (len(img_batch),), -1, dtype=torch.long, device=self.device
        )
        id_mapping[unique_ids] = torch.arange(len(unique_ids), device=self.device)
        backbone_out = {
            **backbone_out,
            **self.backbone.forward_image(image),
            "id_mapping": id_mapping,
        }
        assert "backbone_fpn" in backbone_out
        return self._get_img_feats(backbone_out, img_ids=img_ids)

    def _encode_prompt(
        self,
        backbone_out,
        find_input,
        geometric_prompt,
        visual_prompt_embed=None,
        visual_prompt_mask=None,
        encode_text=True,
        prev_mask_pred=None,
    ):
        # index text features (note that regardless of early or late fusion, the batch size of
        # `txt_feats` is always the number of *prompts* in the encoder)
        txt_ids = find_input.text_ids
        txt_feats = backbone_out["language_features"][:, txt_ids]
        txt_masks = backbone_out["language_mask"][txt_ids]

        feat_tuple = self._get_img_feats(backbone_out, find_input.img_ids)
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple

        if prev_mask_pred is not None:
            img_feats = [img_feats[-1] + prev_mask_pred]
        # Encode geometry
        geo_feats, geo_masks = self.geometry_encoder(
            geo_prompt=geometric_prompt,
            img_feats=img_feats,
            img_sizes=vis_feat_sizes,
            img_pos_embeds=img_pos_embeds,
        )
        if visual_prompt_embed is None:
            visual_prompt_embed = torch.zeros(
                (0, *geo_feats.shape[1:]), device=geo_feats.device
            )
            visual_prompt_mask = torch.zeros(
                (*geo_masks.shape[:-1], 0),
                device=geo_masks.device,
                dtype=geo_masks.dtype,
            )
        if encode_text:
            prompt = torch.cat([txt_feats, geo_feats, visual_prompt_embed], dim=0)
            prompt_mask = torch.cat([txt_masks, geo_masks, visual_prompt_mask], dim=1)
        else:
            prompt = torch.cat([geo_feats, visual_prompt_embed], dim=0)
            prompt_mask = torch.cat([geo_masks, visual_prompt_mask], dim=1)
        return prompt, prompt_mask, backbone_out

    def _run_encoder(
        self,
        backbone_out,
        find_input,
        prompt,
        prompt_mask,
        encoder_extra_kwargs: Optional[Dict] = None,
    ):
        feat_tuple = self._get_img_feats(backbone_out, find_input.img_ids)
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple

        # Run the encoder
        prompt_pos_embed = torch.zeros_like(prompt)
        # make a copy of the image feature lists since the encoder may modify these lists in-place
        memory = self.transformer.encoder(
            src=img_feats.copy(),
            src_key_padding_mask=None,
            src_pos=img_pos_embeds.copy(),
            prompt=prompt,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes,
            encoder_extra_kwargs=encoder_extra_kwargs,
        )
        encoder_out = {
            # encoded image features
            "encoder_hidden_states": memory["memory"],
            "pos_embed": memory["pos_embed"],
            "padding_mask": memory["padding_mask"],
            "level_start_index": memory["level_start_index"],
            "spatial_shapes": memory["spatial_shapes"],
            "valid_ratios": memory["valid_ratios"],
            "vis_feat_sizes": vis_feat_sizes,
            # encoded text features (or other prompts)
            "prompt_before_enc": prompt,
            "prompt_after_enc": memory.get("memory_text", prompt),
            "prompt_mask": prompt_mask,
        }
        if "moe_aux_loss" in memory:
            encoder_out["moe_aux_loss"] = memory["moe_aux_loss"]
        return backbone_out, encoder_out, feat_tuple

    def _run_segmentation_heads(
        self,
        out,
        backbone_out,
        img_ids,
        vis_feat_sizes,
        encoder_hidden_states,
        prompt,
        prompt_mask,
    ):
        if self.segmentation_head is not None:
            seg_head_outputs = activation_ckpt_wrapper(self.segmentation_head)(
                backbone_feats=backbone_out["backbone_fpn"],
                # obj_queries=obj_queries,
                image_ids=img_ids,
                encoder_hidden_states=encoder_hidden_states,
                act_ckpt_enable=self.training and self.use_act_checkpoint_seg_head,
                prompt=prompt,
                prompt_mask=prompt_mask,
            )
            for k, v in seg_head_outputs.items():
                out[k] = v
        else:
            backbone_out.pop("backbone_fpn", None)

    def _get_best_mask(self, out):
        prev_mask_idx = out["pred_logits"].argmax(dim=1).squeeze(1)
        batch_idx = torch.arange(
            out["pred_logits"].shape[0], device=prev_mask_idx.device
        )
        prev_mask_pred = out["pred_masks"][batch_idx, prev_mask_idx][:, None]
        # Downsample mask to match image resolution.
        prev_mask_pred = self.geometry_encoder.mask_encoder.mask_downsampler(
            prev_mask_pred
        )
        prev_mask_pred = prev_mask_pred.flatten(-2).permute(2, 0, 1)

        return prev_mask_pred

    def forward_grounding(
        self,
        backbone_out,
        find_input,
        find_target,
        geometric_prompt: Prompt,
    ):
        with torch.profiler.record_function("SAM3ImageV2._encode_prompt"):
            prompt, prompt_mask, backbone_out = self._encode_prompt(
                backbone_out, find_input, geometric_prompt
            )
        # Run the encoder
        with torch.profiler.record_function("SAM3ImageV2._run_encoder"):
            backbone_out, encoder_out, _ = self._run_encoder(
                backbone_out, find_input, prompt, prompt_mask
            )
        out = {
            "encoder_hidden_states": encoder_out["encoder_hidden_states"],
            "prev_encoder_out": {
                "encoder_out": encoder_out,
                "backbone_out": backbone_out,
            },
        }
        if "moe_aux_loss" in encoder_out:
            out["moe_aux_loss"] = encoder_out["moe_aux_loss"]

        # Run segmentation heads
        with torch.profiler.record_function("SAM3ImageV2._run_segmentation_heads"):
            self._run_segmentation_heads(
                out=out,
                backbone_out=backbone_out,
                img_ids=find_input.img_ids,
                vis_feat_sizes=encoder_out["vis_feat_sizes"],
                encoder_hidden_states=out["encoder_hidden_states"], # semantic
                prompt=prompt,
                prompt_mask=prompt_mask,
            )

        return out

    def _postprocess_out(self, out: Dict, multimask_output: bool = False):
        # For multimask output, during eval we return the single best mask with the dict keys expected by the evaluators, but also return the multimasks output with new keys.
        num_mask_boxes = out["pred_boxes"].size(1)
        if not self.training and multimask_output and num_mask_boxes > 1:
            out["multi_pred_logits"] = out["pred_logits"]
            if "pred_masks" in out:
                out["multi_pred_masks"] = out["pred_masks"]
            out["multi_pred_boxes"] = out["pred_boxes"]
            out["multi_pred_boxes_xyxy"] = out["pred_boxes_xyxy"]

            best_mask_idx = out["pred_logits"].argmax(1).squeeze(1)
            batch_idx = torch.arange(len(best_mask_idx), device=best_mask_idx.device)

            out["pred_logits"] = out["pred_logits"][batch_idx, best_mask_idx].unsqueeze(
                1
            )
            if "pred_masks" in out:
                out["pred_masks"] = out["pred_masks"][
                    batch_idx, best_mask_idx
                ].unsqueeze(1)
            out["pred_boxes"] = out["pred_boxes"][batch_idx, best_mask_idx].unsqueeze(1)
            out["pred_boxes_xyxy"] = out["pred_boxes_xyxy"][
                batch_idx, best_mask_idx
            ].unsqueeze(1)

        return out

    def _get_dummy_prompt(self, num_prompts=1):
        device = self.device
        geometric_prompt = Prompt(
            box_embeddings=torch.zeros(0, num_prompts, 4, device=device),
            box_mask=torch.zeros(num_prompts, 0, device=device, dtype=torch.bool),
        )
        return geometric_prompt

    def forward(self, input: BatchedDatapoint):
        device = self.device
        backbone_out = {"img_batch_all_stages": input.img_batch}
        backbone_out.update(self.backbone.forward_image(input.img_batch))
        num_frames = len(input.find_inputs)
        assert num_frames == 1

        text_outputs = self.backbone.forward_text(input.find_text_batch, device=device)
        backbone_out.update(text_outputs)

        previous_stages_out = SAM3Output(
            iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE
        )

        find_input = input.find_inputs[0]
        find_target = input.find_targets[0]

        # if find_input.input_points is not None and find_input.input_points.numel() > 0:
        #     print("Warning: Point prompts are ignored in PCS.")

        num_interactive_steps = 0 if self.training else self.num_interactive_steps_val
        
        #  Here we support points as prompts for image as well
        if find_input.input_points is not None and find_input.input_points.numel() > 0:
            # Here we put the num_boxes/points dimension first for embeddings and labels
            geometric_prompt = Prompt(
                box_embeddings=find_input.input_boxes,
                box_mask=find_input.input_boxes_mask,
                box_labels=find_input.input_boxes_label,
                point_embeddings=find_input.input_points[..., :2].transpose(0, 1),
                point_mask=find_input.input_points_mask,
                point_labels=find_input.input_points[..., -1].transpose(0, 1),
            )
        else:
            geometric_prompt = Prompt(
                box_embeddings=find_input.input_boxes,
                box_mask=find_input.input_boxes_mask,
                box_labels=find_input.input_boxes_label,
            )

        # Init vars that are shared across the loop.
        stage_outs = []
        for cur_step in range(num_interactive_steps + 1):
            if cur_step > 0:
                # We sample interactive geometric prompts (boxes, points)
                geometric_prompt, _ = self.interactive_prompt_sampler.sample(
                    geo_prompt=geometric_prompt,
                    find_target=find_target,
                    previous_out=stage_outs[-1],
                )
            out = self.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_input,
                find_target=find_target,
                geometric_prompt=geometric_prompt.clone(),
            )
            stage_outs.append(out)

        previous_stages_out.append(stage_outs)
        return previous_stages_out

    def forward_with_backbone_features(self, input: BatchedDatapoint):
        device = self.device
        backbone_out = {"img_batch_all_stages": input.img_batch}
        backbone_out.update(self.backbone.forward_image(input.img_batch))
        num_frames = len(input.find_inputs)
        assert num_frames == 1

        text_outputs = self.backbone.forward_text(input.find_text_batch, device=device)
        backbone_out.update(text_outputs)

        previous_stages_out = SAM3Output(
            iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE
        )

        find_input = input.find_inputs[0]
        find_target = input.find_targets[0]

        # if find_input.input_points is not None and find_input.input_points.numel() > 0:
        #     print("Warning: Point prompts are ignored in PCS.")

        num_interactive_steps = 0 if self.training else self.num_interactive_steps_val
        
        #  Here we support points as prompts for image as well
        if find_input.input_points is not None and find_input.input_points.numel() > 0:
            # Here we put the num_boxes/points dimension first for embeddings and labels
            geometric_prompt = Prompt(
                box_embeddings=find_input.input_boxes,
                box_mask=find_input.input_boxes_mask,
                box_labels=find_input.input_boxes_label,
                point_embeddings=find_input.input_points[..., :2].transpose(0, 1),
                point_mask=find_input.input_points_mask,
                point_labels=find_input.input_points[..., -1].transpose(0, 1),
            )
        else:
            geometric_prompt = Prompt(
                box_embeddings=find_input.input_boxes,
                box_mask=find_input.input_boxes_mask,
                box_labels=find_input.input_boxes_label,
            )

        # Init vars that are shared across the loop.
        stage_outs = []
        for cur_step in range(num_interactive_steps + 1):
            if cur_step > 0:
                # We sample interactive geometric prompts (boxes, points)
                geometric_prompt, _ = self.interactive_prompt_sampler.sample(
                    geo_prompt=geometric_prompt,
                    find_target=find_target,
                    previous_out=stage_outs[-1],
                )
            out = self.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_input,
                find_target=find_target,
                geometric_prompt=geometric_prompt.clone(),
            )
            txt_ids = find_input.text_ids
            out['backbone_fpn'] = backbone_out['backbone_fpn']
            out["language_features"] = backbone_out["language_features"][:, txt_ids]
            out["language_mask"] = backbone_out["language_mask"][txt_ids]
            stage_outs.append(out)

        previous_stages_out.append(stage_outs)
        return previous_stages_out

    def _compute_matching(self, out, targets):
        out["indices"] = self.matcher(out, targets)
        for aux_out in out.get("aux_outputs", []):
            aux_out["indices"] = self.matcher(aux_out, targets)

    def back_convert(self, targets):
        batched_targets = {
            "boxes": targets.boxes.view(-1, 4),
            "boxes_xyxy": box_cxcywh_to_xyxy(targets.boxes.view(-1, 4)),
            "boxes_padded": targets.boxes_padded,
            "positive_map": targets.boxes.new_ones(len(targets.boxes), 1),
            "num_boxes": targets.num_boxes,
            "masks": targets.segments,
            "semantic_masks": targets.semantic_segments,
            "is_valid_mask": targets.is_valid_segment,
            "is_exhaustive": targets.is_exhaustive,
            "object_ids_packed": targets.object_ids,
            "object_ids_padded": targets.object_ids_padded,
        }
        return batched_targets

    def predict_inst(
        self,
        inference_state,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        orig_h, orig_w = (
            inference_state["original_height"],
            inference_state["original_width"],
        )
        backbone_out = inference_state["backbone_out"]["sam2_backbone_out"]
        (
            _,
            vision_feats,
            _,
            _,
        ) = self.inst_interactive_predictor.model._prepare_backbone_features(
            backbone_out
        )
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        vision_feats[-1] = (
            vision_feats[-1] + self.inst_interactive_predictor.model.no_mem_embed
        )
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(
                vision_feats[::-1], self.inst_interactive_predictor._bb_feat_sizes[::-1]
            )
        ][::-1]
        self.inst_interactive_predictor._features = {
            "image_embed": feats[-1],
            "high_res_feats": feats[:-1],
        }
        self.inst_interactive_predictor._is_image_set = True
        self.inst_interactive_predictor._orig_hw = [(orig_h, orig_w)]
        res = self.inst_interactive_predictor.predict(**kwargs)
        self.inst_interactive_predictor._features = None
        self.inst_interactive_predictor._is_image_set = False
        return res

    def predict_inst_batch(
        self,
        inference_state,
        *args,
        **kwargs,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        backbone_out = inference_state["backbone_out"]["sam2_backbone_out"]
        (
            _,
            vision_feats,
            _,
            _,
        ) = self.inst_interactive_predictor.model._prepare_backbone_features(
            backbone_out
        )
        # Add no_mem_embed, which is added to the lowest res feat. map during training on videos
        vision_feats[-1] = (
            vision_feats[-1] + self.inst_interactive_predictor.model.no_mem_embed
        )
        batch_size = vision_feats[-1].shape[1]
        orig_heights, orig_widths = (
            inference_state["original_heights"],
            inference_state["original_widths"],
        )
        assert (
            batch_size == len(orig_heights) == len(orig_widths)
        ), f"Batch size mismatch in predict_inst_batch. Got {batch_size}, {len(orig_heights)}, {len(orig_widths)}"
        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(
                vision_feats[::-1], self.inst_interactive_predictor._bb_feat_sizes[::-1]
            )
        ][::-1]
        self.inst_interactive_predictor._features = {
            "image_embed": feats[-1],
            "high_res_feats": feats[:-1],
        }
        self.inst_interactive_predictor._is_image_set = True
        self.inst_interactive_predictor._is_batch = True
        self.inst_interactive_predictor._orig_hw = [
            (orig_h, orig_w) for orig_h, orig_w in zip(orig_heights, orig_widths)
        ]
        res = self.inst_interactive_predictor.predict_batch(*args, **kwargs)
        self.inst_interactive_predictor._features = None
        self.inst_interactive_predictor._is_image_set = False
        self.inst_interactive_predictor._is_batch = False
        return res
