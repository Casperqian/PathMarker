# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import torch
import re

from sam3.model.model_misc import SAM3Output

from sam3.train.utils.distributed import get_world_size

from .loss_fns import CORE_LOSS_KEY, Det2TrkAssoc, Masks


class DummyLoss(torch.nn.Module):
    """A dummy loss that always returns 0 (as a placeholder for eval)"""

    def __init__(
        self,
        core_loss_key: str = CORE_LOSS_KEY,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()
        self.core_loss_key = core_loss_key
        self.device = torch.device(device)

    def forward(self, *args, **kwargs):
        return {self.core_loss_key: torch.tensor(0.0, device=self.device)}

    def accumulate(self, out_dict):
        """
        Called by iterative losses.
        """
        if self.core_loss_key not in out_dict:
            out_dict[self.core_loss_key] = torch.tensor(0.0, device=self.device)
        return out_dict


class Sam3LossWrapper(torch.nn.Module):
    def __init__(
        self,
        loss_fns_find,
        normalization="global",
        matcher=None,
        o2m_matcher=None,
        o2m_weight=1.0,
        use_o2m_matcher_on_o2m_aux=True,
        loss_fn_semantic_seg=None,
        normalize_by_valid_object_num=False,
        normalize_by_stage_num=False,
        scale_by_find_batch_size=False,
        weight_dict=None,
        moe_aux_loss_weight=None,
    ):
        super().__init__()
        self.loss_fns_find = loss_fns_find
        assert normalization in ["global", "local", "none"]
        self.normalization = normalization
        self.normalize_by_valid_object_num = normalize_by_valid_object_num
        self.normalize_by_stage_num = normalize_by_stage_num
        self.matcher = matcher
        self.o2m_matcher = o2m_matcher
        self.o2m_weight = o2m_weight
        # whether to use the o2m matcher on the o2m queries in auxiliary outputs
        self.use_o2m_matcher_on_o2m_aux = use_o2m_matcher_on_o2m_aux
        self.loss_fn_semantic_seg = loss_fn_semantic_seg
        self.scale_by_find_batch_size = scale_by_find_batch_size
        self.weight_dict = weight_dict if weight_dict is not None else {}
        if moe_aux_loss_weight is not None:
            self.weight_dict.setdefault("loss_moe_balance", moe_aux_loss_weight)

    def _get_num_boxes(self, targets):
        # the average number of target boxes for loss normalization
        if self.normalize_by_valid_object_num:
            # valid boxes are those with non-zero height and width
            # (while padded invisible boxes are )
            boxes_hw = targets["boxes"].view(-1, 4)  # cx, cy, w, h
            num_boxes = (boxes_hw[:, 2:] > 0).all(dim=-1).sum().float()
        else:
            num_boxes = targets["num_boxes"].sum().float()
        if self.normalization == "global":
            torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1)
        elif self.normalization == "local":
            num_boxes = torch.clamp(num_boxes, min=1)
        elif self.normalization == "none":
            num_boxes = 1
        return num_boxes

    def compute_loss(self, nested_out, targets):
        num_boxes = self._get_num_boxes(targets)
        o2m_out_is_valid = nested_out.get("o2m_out_is_valid", None)
        o2m_target_is_valid_padded = nested_out.get("o2m_target_is_valid_padded", None)

        # Get a list of outputs, including auxiliary and first stage outputs
        output_list = [(nested_out, "", False)]  # (out, suffix, is_aux)
        if "aux_outputs" in nested_out:
            output_list.extend(
                (aux_out, f"_aux_{i}", True)
                for i, aux_out in enumerate(nested_out["aux_outputs"])
            )
        if "first_stage" in nested_out:
            output_list.append((nested_out["first_stage"], "_fs", True))
        # print("*"*32 + "[DBG_WRAPPER]" + "*"*32)
        # for (out, suffix, is_aux) in output_list:
        #     print(f"[DBG_WRAPPER] output suffix: {suffix}, is_aux: {is_aux}")
        #     print(f"[DBG_WRAPPER] output keys: {list(out.keys())}")
        # print("*"*32 + "[DBG_WRAPPER]" + "*"*32)
        # Compute all the requested losses
        losses = {}
        total_core_loss = 0.0
        for out, suffix, is_aux in output_list:
            # o2o matcher indices need to be computed by the model (as the video model requires
            # a specific way of matching free and locked indices beyond just calling the matcher)
            indices = out["indices"]
            has_o2m_out = "pred_logits_o2m" in out
            if has_o2m_out:
                o2m_out = {
                    k[: -len("_o2m")]: v for k, v in out.items() if k.endswith("_o2m")
                }
                # o2m targets are the same as the o2o targets (assuming repeat=1)
                o2m_targets = targets
                if self.use_o2m_matcher_on_o2m_aux or not is_aux:
                    o2m_indices = self.o2m_matcher(
                        o2m_out,
                        o2m_targets,
                        out_is_valid=o2m_out_is_valid,
                        target_is_valid_padded=o2m_target_is_valid_padded,
                    )
                else:
                    o2m_indices = self.matcher(
                        o2m_out,
                        o2m_targets,
                        out_is_valid=o2m_out_is_valid,
                        target_is_valid_padded=o2m_target_is_valid_padded,
                    )

            for loss_fn in self.loss_fns_find:
                l_dict = loss_fn(
                    outputs=out,
                    targets=targets,
                    indices=indices,
                    num_boxes=num_boxes,
                    is_aux=is_aux,
                )
                total_core_loss += l_dict.pop(CORE_LOSS_KEY)
                # print(f"[DBG_WRAPPER] suffix: {suffix}, l_dict keys: {list(l_dict.keys())}")
                losses.update({f"{k}{suffix}": v for k, v in l_dict.items()})

                compute_o2m_loss = has_o2m_out and getattr(
                    loss_fn, "supports_o2m_loss", True
                )
                # a special handling to allow turning off mask loss in o2m
                # (to be compatible with the original implementation)
                if isinstance(loss_fn, Masks):
                    compute_o2m_loss = compute_o2m_loss and "pred_masks" in o2m_out
                if isinstance(loss_fn, Det2TrkAssoc):
                    compute_o2m_loss = False  # Det2TrkAssoc does not support o2m
                if compute_o2m_loss:
                    l_dict = loss_fn(
                        outputs=o2m_out,
                        targets=o2m_targets,
                        indices=o2m_indices,
                        num_boxes=num_boxes,
                        is_aux=is_aux,
                    )
                    for k in l_dict:
                        l_dict[k] *= self.o2m_weight
                    total_core_loss += l_dict.pop(CORE_LOSS_KEY)
                    losses.update({f"{k}{suffix}_o2m": v for k, v in l_dict.items()})

        moe_aux_loss = nested_out.get("moe_aux_loss", None)
        if moe_aux_loss is not None:
            losses["loss_moe_balance"] = moe_aux_loss
            moe_aux_loss_weight = self.weight_dict.get("loss_moe_balance", 0.0)
            if moe_aux_loss_weight != 0:
                total_core_loss = total_core_loss + moe_aux_loss_weight * moe_aux_loss

        losses[CORE_LOSS_KEY] = total_core_loss
        return losses

    def forward(self, find_stages: SAM3Output, find_targets):
        if find_stages.loss_stages is not None:
            find_targets = [find_targets[i] for i in find_stages.loss_stages]
        with SAM3Output.iteration_mode(
            find_stages, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
        ) as find_stages:
            assert len(find_stages) == len(find_targets)
            total_losses = {}
            for stage_outputs, stage_targets in zip(find_stages, find_targets):
                stage_targets = [stage_targets] * len(stage_outputs)
                # If there are multiple steps within a stage, compute the loss for all of them (e.g. interactivity)
                for outputs, targets in zip(stage_outputs, stage_targets):
                    
                    cur_losses = self.compute_loss(outputs, targets)

                    if self.loss_fn_semantic_seg is not None:
                        cur_losses_semantic = self.loss_fn_semantic_seg(
                            outputs, targets
                        )
                        cur_losses[CORE_LOSS_KEY] += cur_losses_semantic.pop(
                            CORE_LOSS_KEY
                        )
                        # make sure the semantic losses don't overlap with the find losses
                        assert set(cur_losses).isdisjoint(set(cur_losses_semantic))
                        cur_losses.update(cur_losses_semantic)

                    # Optionally, normalize the loss by the number of find stages (training video frames) so that
                    # image batches and video batches have similar loss scales. (Otherwise video batches would
                    # have a much higher loss scale due to summing the losses over all the find stages.)
                    if self.normalize_by_stage_num:
                        cur_losses[CORE_LOSS_KEY] /= len(find_stages)

                    if self.scale_by_find_batch_size:
                        bs = targets["num_boxes"].shape[0]
                        # sqrt scaling based on the "effective" batch size
                        cur_losses[CORE_LOSS_KEY] *= bs**0.5

                    for k, v in cur_losses.items():
                        if k not in total_losses:
                            total_losses[k] = v
                        else:
                            total_losses[k] += v

        return total_losses
    

class SemanticLossWrapper(torch.nn.Module):
    def __init__(
        self,
        loss_fn_semantic_seg=None,
        normalize_by_stage_num=False,
        scale_by_find_batch_size=False,
        weight_dict=None,
        moe_aux_loss_weight=None,
    ):
        super().__init__()
        # whether to use the o2m matcher on the o2m queries in auxiliary outputs
        self.loss_fn_semantic_seg = loss_fn_semantic_seg
        self.normalize_by_stage_num = normalize_by_stage_num
        self.scale_by_find_batch_size = scale_by_find_batch_size
        self.weight_dict = weight_dict if weight_dict is not None else {}
        if moe_aux_loss_weight is not None:
            self.weight_dict.setdefault("loss_moe_balance", moe_aux_loss_weight)

    def forward(self, find_stages: SAM3Output, find_targets):
        if find_stages.loss_stages is not None:
            find_targets = [find_targets[i] for i in find_stages.loss_stages]
        with SAM3Output.iteration_mode(
            find_stages, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
        ) as find_stages:
            assert len(find_stages) == len(find_targets)
            total_losses = {}
            for stage_outputs, stage_targets in zip(find_stages, find_targets):
                stage_targets = [stage_targets] * len(stage_outputs)
                # If there are multiple steps within a stage, compute the loss for all of them (e.g. interactivity)
                for outputs, targets in zip(stage_outputs, stage_targets):

                    cur_losses = {}

                    if self.loss_fn_semantic_seg is not None:
                        cur_losses_semantic = self.loss_fn_semantic_seg(
                            outputs, targets
                        )
                        if not cur_losses.get(CORE_LOSS_KEY, None):
                            cur_losses[CORE_LOSS_KEY] = cur_losses_semantic.pop(
                                CORE_LOSS_KEY
                            )
                        else:
                            cur_losses[CORE_LOSS_KEY] += cur_losses_semantic.pop(
                                CORE_LOSS_KEY
                            )
                        # make sure the semantic losses don't overlap with the find losses
                        assert set(cur_losses).isdisjoint(set(cur_losses_semantic))
                        cur_losses.update(cur_losses_semantic)

                    moe_aux_loss = outputs.get("moe_aux_loss", None)
                    if moe_aux_loss is not None:
                        cur_losses["loss_moe_balance"] = moe_aux_loss
                        moe_aux_loss_weight = self.weight_dict.get(
                            "loss_moe_balance", 0.0
                        )
                        if moe_aux_loss_weight != 0:
                            if CORE_LOSS_KEY not in cur_losses:
                                cur_losses[CORE_LOSS_KEY] = (
                                    moe_aux_loss_weight * moe_aux_loss
                                )
                            else:
                                cur_losses[CORE_LOSS_KEY] = cur_losses[
                                    CORE_LOSS_KEY
                                ] + moe_aux_loss_weight * moe_aux_loss

                    # Optionally, normalize the loss by the number of find stages (training video frames) so that
                    # image batches and video batches have similar loss scales. (Otherwise video batches would
                    # have a much higher loss scale due to summing the losses over all the find stages.)
                    if self.normalize_by_stage_num:
                        cur_losses[CORE_LOSS_KEY] /= len(find_stages)

                    if self.scale_by_find_batch_size:
                        bs = targets["num_boxes"].shape[0]
                        # sqrt scaling based on the "effective" batch size
                        cur_losses[CORE_LOSS_KEY] *= bs**0.5

                    for k, v in cur_losses.items():
                        if k not in total_losses:
                            total_losses[k] = v
                        else:
                            total_losses[k] += v

        return total_losses
    

class DebugSam3LossWrapper(Sam3LossWrapper):
    """
    Enhanced Loss Wrapper for Debugging and Fixing Validation Issues.

    Purpose:
        1)  During training, 'outputs["indices"]' might already exist.
            However, during validation, it is often missing.
            This wrapper uses the matcher to calculate indices on-the-fly if they are absent.
        2)  After patching, it prints the target object counts and indices query distribution signature.
            If the signature differs for each batch, it confirms that indices are calculated dynamically
            for the current batch.
    """

    def __init__(
        self,
        *args,
        debug_print_every: int = 1,
        debug_print_first_n: int = 50,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._debug_step = 0
        self._debug_prev_sig = None
        self.debug_print_every = int(debug_print_every)
        self.debug_print_first_n = int(debug_print_first_n)

    def forward(self, find_stages, find_targets):
        # The 'step' here refers to the "number of times loss is called",
        # which is sufficient for debugging.
        self._debug_step += 1
        return super().forward(find_stages, find_targets)

    def _should_print(self) -> bool:
        if self.debug_print_every <= 0:
            return False
        if self._debug_step <= self.debug_print_first_n:
            return True
        return (self._debug_step % self.debug_print_every) == 0

    @torch.no_grad()
    def _ensure_indices(self, outputs: dict, targets: dict) -> dict:
        """
        If 'indices' are missing in outputs, calculate them using
        self.matcher(outputs, targets) and write them back to outputs.
        """
        if isinstance(outputs, dict) and ("indices" in outputs) and (outputs["indices"] is not None):
            return outputs

        # Perform matching directly using the current batch's outputs/targets
        indices = self.matcher(outputs, targets)
        outputs["indices"] = indices

        # If aux_outputs exist and lack indices, DETR training usually share the same indices.
        # (If the matcher/criterion has different requirements for aux, this logic may need adjustment)
        if "aux_outputs" in outputs and isinstance(outputs["aux_outputs"], (list, tuple)):
            for aux in outputs["aux_outputs"]:
                if isinstance(aux, dict) and (("indices" not in aux) or (aux["indices"] is None)):
                    aux["indices"] = indices

        if self._should_print():
            self._log_indices_summary(indices, targets)

        return outputs

    def _log_indices_summary(self, indices, targets):
        """
        Prints:
            - Target object count (n_obj)
            - Match count (n_match)
            - Query ID range, unique count, top histogram
            - Whether the signature matches the previous batch
        """
        # 1) Target object count
        n_obj = None
        if isinstance(targets, dict):
            if "num_boxes" in targets:
                nb = targets["num_boxes"]
                if torch.is_tensor(nb):
                    n_obj = int(nb.sum().item())
                else:
                    try:
                        n_obj = int(nb)
                    except Exception:
                        n_obj = None
            elif "boxes" in targets and torch.is_tensor(targets["boxes"]):
                n_obj = int(targets["boxes"].shape[0])

        # 2) Indices structure: The current matcher returns a tuple (len=3 or similar),
        #    where typically index 0 contains batch image ids, index 1 contains query ids and index 2 is None.
        q_ids = None
        if isinstance(indices, (list, tuple)) and len(indices) >= 2 and torch.is_tensor(indices[1]):
            q_ids = indices[1]
        elif torch.is_tensor(indices):
            q_ids = indices

        if q_ids is None:
            # print(
            #     f"[DEBUG][idx] step={self._debug_step} grad={torch.is_grad_enabled()} "
            #     "cannot-parse-indices type={type(indices)}"
            #     )
            return

        q_cpu = q_ids.detach().to("cpu")
        n_match = int(q_cpu.numel())
        uniq = torch.unique(q_cpu) if n_match > 0 else torch.empty((0,), dtype=torch.long)
        uniq_n = int(uniq.numel())
        q_min = int(q_cpu.min().item()) if n_match > 0 else -1
        q_max = int(q_cpu.max().item()) if n_match > 0 else -1

        # 3) Query histogram top-k (Usually each query is used once, so most counts are 1)
        top_pairs = []
        if n_match > 0:
            bc = torch.bincount(q_cpu, minlength=(q_max + 1))
            k = min(5, int(bc.numel()))
            if k > 0:
                vals, idxs = torch.topk(bc, k=k)
                for i, v in zip(idxs, vals):
                    vi = int(v.item())
                    if vi > 0:
                        top_pairs.append((int(i.item()), vi))

        # 4) Signature: Used to determine if recalculation happens every batch.
        #    - Take partial unique (first 10) to avoid excessive length.
        uniq_head = tuple(sorted([int(x) for x in uniq.tolist()])[:10])
        sig = (n_obj, n_match, q_min, q_max, uniq_head)
        same_as_prev = (sig == self._debug_prev_sig)
        self._debug_prev_sig = sig

        # print(
        #     f"[DEBUG][idx] step={self._debug_step} grad={torch.is_grad_enabled()} "
        #     f"n_obj={n_obj} n_match={n_match} q_range=[{q_min},{q_max}] uniq={uniq_n} "
        #     f"top={top_pairs} same_as_prev={same_as_prev} sig={sig}"
        # )

    def compute_loss(self, outputs, targets):
        # compute_loss typically receives a dict (single stage)
        if isinstance(outputs, dict):
            outputs = self._ensure_indices(outputs, targets)
        return super().compute_loss(outputs, targets)
