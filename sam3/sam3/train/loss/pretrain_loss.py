# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.distributed.nn as dist_nn
import torch.nn.functional as F

from . import hyperbolic
from .concept_aliases import ConceptAliases
from .hierarchy_data import load_hierarchy_data, normalize_name
from .loss_fns import CORE_LOSS_KEY

def _normalize_label_name(label) -> str:
    if isinstance(label, dict):
        label = label.get("object", "")
    if label is None:
        return ""
    text = str(label)
    return normalize_name(text) if text.strip() else ""


class CustomContrastiveLoss(torch.nn.Module):
    def __init__(self, tau: float = 0.07, soft_label_tau: float = 0.1, threshold=0.95):
        super().__init__()
        self._tau = tau
        self.soft_label_tau = soft_label_tau
        self.eps = 1e-8
        self.threshold = threshold

    @staticmethod
    def _soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, B]
        soft_targets: [B, B], each row sums to 1
        """
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(soft_targets * log_probs).sum(dim=-1).mean()
        return loss

    def _build_targets(self, target_emb: torch.Tensor) -> torch.Tensor:
        B = target_emb.size(0)
        sim = torch.matmul(target_emb, target_emb.T) 
        device = sim.device
        eye = torch.eye(B, device=device, dtype=torch.bool)
        pos_mask = sim > self.threshold
        mask = torch.logical_or(eye, pos_mask) 
        neg_mask = ~mask                       
        neg_only_sim = sim.masked_fill(mask, float("inf"))
        row_min = neg_only_sim.min(dim=-1, keepdim=True).values # [B, 1]
        row_min = torch.where(torch.isinf(row_min), torch.zeros_like(row_min), row_min)
        targets_logits = torch.where(neg_mask, sim - row_min, sim)
        targets = F.softmax(targets_logits.float() / self.soft_label_tau, dim=-1).to(sim.dtype)
        return targets

    def _match_by_semantic_center(
        self,
        semantic_center: torch.Tensor,   # [B, K, D]
        language_embed: torch.Tensor,    # [B, D]
    ):
        # [B, K]
        # 第 i 张图第 k 个 center 与第 i 个文本的相似度
        center_text_sim = torch.einsum("bkd,bd->bk", semantic_center, language_embed)
        best_idx = center_text_sim.argmax(dim=-1)

        return best_idx, center_text_sim

    def forward(self, inputs, target_emb):
        total_losses = {}

        vision_embed = inputs["semantic_feature"]   # [B, K, D]
        language_embed = inputs["text_feature"]     # [B, D]
        semantic_center = inputs["semantic_center"]  # [B, K, D]
        vision_embed = F.normalize(vision_embed, dim=-1)
        language_embed = F.normalize(language_embed, dim=-1)
        semantic_center = F.normalize(semantic_center, dim=-1)
        
        # 1. soft labels from text-text similarity
        with torch.no_grad():
            target_emb = F.normalize(target_emb, dim=-1)
            target_embs = dist_nn.all_gather(target_emb)
            target_emb_global = torch.cat(target_embs, dim=0)
            soft_targets = self._build_targets(target_emb_global)  # [B, B]
            # 2. select best semantic group token for each text prompt
            best_idx, center_text_sim = self._match_by_semantic_center(
                semantic_center, language_embed
            )

        batch_idx = torch.arange(vision_embed.size(0), device=vision_embed.device)
        selected_vision_emb = vision_embed[batch_idx, best_idx] # [B, D]
        language_embeds = dist_nn.all_gather(language_embed)
        selected_vision_embs = dist_nn.all_gather(selected_vision_emb)
        language_embed_global = torch.cat(language_embeds, dim=0)
        selected_vision_emb_global = torch.cat(selected_vision_embs, dim=0)
    
        # 3. image-text logits
        logits_v2t = torch.matmul(selected_vision_emb_global, language_embed_global.T) / self._tau
        # logits_v2t = torch.matmul(selected_vision_emb, language_embed.T) / self._tau  # [B, B]
        logits_t2v = logits_v2t.T  # [B, B]

        # 4. soft-label contrastive loss
        loss_vision2lang = self._soft_cross_entropy(logits_v2t, soft_targets)
        loss_lang2vision = self._soft_cross_entropy(logits_t2v, soft_targets.T)

        total_contrastive_loss = 0.5 * (loss_vision2lang + loss_lang2vision)

        total_losses[CORE_LOSS_KEY] = total_contrastive_loss

        return total_losses


class _AliasLoss(torch.nn.Module):
    def __init__(self, aliases_path=None):
        super().__init__()
        self.aliases = ConceptAliases(aliases_path)

    @staticmethod
    def _all_gather_tensor(tensor):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return dist_nn.all_gather(tensor)
        return (tensor,)

    @staticmethod
    def _all_gather_label_batch(label_batch):
        if label_batch is None:
            return None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(gathered, label_batch)
            return [label for rank_labels in gathered for label in rank_labels]
        return list(label_batch)

    def _canonical_label(self, label):
        name = _normalize_label_name(label)
        if not name:
            raise ValueError("label_batch must contain nonempty concept names")
        return self.aliases.canonicalize(name)

    def _select_query_indices(self, text_features, global_labels):
        if global_labels is None:
            return torch.arange(text_features.size(0), device=text_features.device)
        if len(global_labels) != text_features.size(0):
            raise ValueError("label_batch length must match the gathered text features")
        seen = set()
        indices = []
        for index, label in enumerate(global_labels):
            name = self._canonical_label(label)
            if name not in seen:
                seen.add(name)
                indices.append(index)
        return torch.tensor(indices, device=text_features.device, dtype=torch.long)

    def _alias_positive_mask(self, text_features, global_labels, query_indices):
        # Unlabelled calls know only the paired diagonal; never infer aliases from vectors.
        if global_labels is None:
            return torch.eye(text_features.size(0), device=text_features.device, dtype=torch.bool)
        names = [self._canonical_label(label) for label in global_labels]
        query_names = [names[i] for i in query_indices.tolist()]
        return torch.tensor([[a == b for b in query_names] for a in names],
                            device=text_features.device, dtype=torch.bool)


class ClipContrastiveLoss(_AliasLoss):
    """Bidirectional CLIP loss over images and unique canonical text queries."""

    def __init__(self, tau: float = 0.07, aliases_path=None):
        super().__init__(aliases_path)
        self._tau = tau

    def forward(self, inputs, label_batch=None):
        vision = inputs["vision_feature"] if "vision_feature" in inputs else inputs["vision_feat"]
        images = torch.cat(self._all_gather_tensor(F.normalize(vision, dim=-1)))
        texts = torch.cat(self._all_gather_tensor(F.normalize(inputs["text_feature"], dim=-1)))
        labels = self._all_gather_label_batch(label_batch)
        indices = self._select_query_indices(texts, labels)
        positive = self._alias_positive_mask(texts, labels, indices)
        logits = (images @ texts[indices].T) / self._tau
        image_loss = -(positive * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        text_loss = -(positive * F.log_softmax(logits, dim=0)).sum(dim=0) / positive.sum(dim=0)
        return {CORE_LOSS_KEY: (image_loss + text_loss.mean()) / 2}


class SigmoidLoss(_AliasLoss):
    """Sigmoid contrastive loss with fixed alias positives and query identities."""

    def __init__(self, aliases_path=None):
        super().__init__(aliases_path)
        self.scale = torch.nn.Parameter(torch.tensor(math.log(10.0)))
        self.bias = torch.nn.Parameter(torch.tensor(-10.0))

    def forward(self, inputs, label_batch=None):
        v = F.normalize(inputs["vision_feat"], dim=-1)
        t = F.normalize(inputs["text_feature"], dim=-1)
        v_global = torch.cat(self._all_gather_tensor(v), dim=0)
        t_global = torch.cat(self._all_gather_tensor(t), dim=0)
        labels = self._all_gather_label_batch(label_batch)
        indices = self._select_query_indices(t_global, labels)
        positive = self._alias_positive_mask(t_global, labels, indices)
        target = positive.to(t_global.dtype).mul(2).sub(1)
        logits = (v_global @ t_global[indices].T) * self.scale.exp() + self.bias
        return {CORE_LOSS_KEY: F.softplus(-target * logits).sum() / target.size(1)}


class HierarchyLoss(_AliasLoss):
    """Hyperbolic sigmoid contrastive loss with hierarchy cone regularization."""

    def __init__(
        self,
        logit_scale_init: float = 1.0,
        logit_bias_init: float = 1.0,
        curv_init: float = 1.0,
        learn_curv: bool = True,
        tangent_scale_init: float = 0.05,
        learn_tangent_scale: bool = True,
        max_tangent_scale: float = 1.0,
        entailment_loss_weight: float = 0.1,
        entailment_aperture_scale: float = 1.0,
        entailment_min_radius: float = 0.1,
        seed_hierarchy_path=None,
        concept_hierarchy_path=None,
        aliases_path=None,
        hierarchy_calibration_steps: int = 1000,
    ):
        super().__init__(aliases_path)
        self.scale = torch.nn.Parameter(torch.tensor(math.log(logit_scale_init)))
        self.bias = torch.nn.Parameter(torch.tensor(logit_bias_init))

        hierarchy = load_hierarchy_data(
            seed_hierarchy_path,
            concept_hierarchy_path,
        )
        self.seed_edges = hierarchy.seed_edges
        self.seed_parents = hierarchy.seed_parents
        self.hierarchy_concepts = hierarchy.concepts

        log_curv = torch.tensor(math.log(curv_init))
        if learn_curv:
            self.log_curv = torch.nn.Parameter(log_curv)
        else:
            self.register_buffer("log_curv", log_curv)

        log_tangent_scale = torch.tensor(math.log(tangent_scale_init))
        if learn_tangent_scale:
            self.log_tangent_scale = torch.nn.Parameter(log_tangent_scale)
        else:
            self.register_buffer("log_tangent_scale", log_tangent_scale)

        self.max_tangent_scale = max_tangent_scale
        self.entailment_loss_weight = entailment_loss_weight
        self.entailment_aperture_scale = entailment_aperture_scale
        self.entailment_min_radius = entailment_min_radius

        if (
            not isinstance(hierarchy_calibration_steps, int)
            or isinstance(hierarchy_calibration_steps, bool)
            or hierarchy_calibration_steps <= 0
        ):
            raise ValueError("hierarchy_calibration_steps must be a positive integer")

        self.calibration_steps = hierarchy_calibration_steps
        count = len(self.hierarchy_concepts)
        self.concept_index = {
            name: index for index, name in enumerate(self.hierarchy_concepts)
        }
        self.register_buffer("calibration_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("graph_ready", torch.tensor(False))
        self.register_buffer("rel_threshold", torch.tensor(0.0))
        self.register_buffer(
            "hierarchy_graph",
            torch.zeros((count, count), dtype=torch.bool),
        )

    def _build_asymmetric_positive_mask(self, image_labels, query_labels, device):
        image_names = [self._canonical_label(label) for label in image_labels]
        query_names = [self._canonical_label(label) for label in query_labels]
        # Aliases inherit the same directed seed relations, not new synonym hierarchies.
        parents = {}
        for child, parent in self.seed_edges:
            child = self.aliases.canonicalize(child)
            parent = self.aliases.canonicalize(parent)
            if child != parent:
                parents.setdefault(child, set()).add(parent)
        return torch.tensor(
            [[parent in parents.get(child, ()) for parent in query_names] for child in image_names],
            device=device, dtype=torch.bool,
        ).reshape(len(image_names), len(query_names))

    def _curvature(self) -> torch.Tensor:
        return self.log_curv.exp().clamp(min=1e-4, max=10.0)

    def _tangent_scale(self) -> torch.Tensor:
        return self.log_tangent_scale.exp().clamp(min=1e-4, max=self.max_tangent_scale)

    def _to_hyperbolic(self, tensor: torch.Tensor, curv: torch.Tensor) -> torch.Tensor:
        tangent = tensor.float() * self._tangent_scale()
        return hyperbolic.exp_map0(tangent, curv)

    def _seed_mask(
        self,
        image_labels: Optional[Sequence[dict]],
        query_labels: Optional[Sequence[dict]],
        child_text_h: torch.Tensor,
        parent_text_h: torch.Tensor,
        curv: torch.Tensor,
    ) -> torch.Tensor:
        del curv
        if image_labels is None or query_labels is None:
            mask = torch.zeros(
                (child_text_h.size(0), parent_text_h.size(0)),
                device=child_text_h.device,
                dtype=torch.bool,
            )
        else:
            mask = self._build_asymmetric_positive_mask(
                image_labels,
                query_labels,
                device=child_text_h.device,
            )
        return mask

    def _build_semantic_target_and_hierarchy_weight(
        self,
        alias_pos: torch.Tensor,
        hierarchy_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target = alias_pos.to(torch.float32).mul(2).sub(1)
        target = torch.where(hierarchy_mask, torch.ones_like(target), target)
        weight = torch.ones_like(target)
        return target, weight

    def _entailment_loss_for_pairs(
        self,
        parent_h: torch.Tensor,
        child_h: torch.Tensor,
        curv: torch.Tensor,
    ) -> torch.Tensor:
        if parent_h.numel() == 0:
            return child_h.new_zeros(())
        loss = hyperbolic.entailment_cone_loss(
            parent=parent_h,
            child=child_h,
            curv=curv,
            aperture_scale=self.entailment_aperture_scale,
            min_radius=self.entailment_min_radius,
        )
        return loss.mean()

    def _assemble_entailment_losses(
        self,
        image_h: torch.Tensor,
        child_text_h: torch.Tensor,
        parent_text_h: torch.Tensor,
        hierarchy_mask: torch.Tensor,
        curv: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        has_hierarchy_pairs = bool(hierarchy_mask.any())
        text_image_parent_h = child_text_h
        text_image_child_h = image_h
        text_text_entailment_loss = image_h.new_zeros(())

        if has_hierarchy_pairs:
            child_idx, parent_idx = hierarchy_mask.nonzero(as_tuple=True)
            hierarchy_parent_h = parent_text_h[parent_idx]
            text_image_parent_h = torch.cat(
                [text_image_parent_h, hierarchy_parent_h], dim=0
            )
            text_image_child_h = torch.cat(
                [text_image_child_h, image_h[child_idx]], dim=0
            )
            text_text_entailment_loss = self._entailment_loss_for_pairs(
                parent_h=hierarchy_parent_h,
                child_h=child_text_h[child_idx],
                curv=curv,
            )

        text_image_entailment_loss = self._entailment_loss_for_pairs(
            parent_h=text_image_parent_h,
            child_h=text_image_child_h,
            curv=curv,
        )
        if has_hierarchy_pairs:
            hierarchy_loss = (
                text_image_entailment_loss + text_text_entailment_loss
            ) / 2
        else:
            hierarchy_loss = text_image_entailment_loss

        return (
            text_image_entailment_loss,
            text_text_entailment_loss,
            hierarchy_loss,
        )

    def forward(self, inputs, label_batch=None):
        snapshot = inputs.get("hierarchy_text_feature")
        if snapshot is not None:
            self._calibrate_graph(snapshot)

        # Gather features and labels, then select unique text queries.
        v_raw = inputs["vision_feat"]
        t_raw = inputs["text_feature"]
        v_h_source = torch.cat(self._all_gather_tensor(v_raw), dim=0)  # [N, D]
        t_h_source = torch.cat(self._all_gather_tensor(t_raw), dim=0)  # [N, D]
        v_global = F.normalize(v_h_source, dim=-1)  # [N, D]
        t_global = F.normalize(t_h_source, dim=-1)  # [N, D]

        global_labels = self._all_gather_label_batch(label_batch)  # [N]
        if global_labels is not None:
            assert len(global_labels) == t_global.size(0), (
                f"label_batch has {len(global_labels)} labels but "
                f"features have {t_global.size(0)} rows"
            )

        u_idx = self._select_query_indices(t_global, global_labels)
        alias_pos = self._alias_positive_mask(t_global, global_labels, u_idx)

        query_labels = None
        if global_labels is not None:
            query_labels = [
                global_labels[int(idx)] for idx in u_idx.detach().cpu().tolist()
            ]  # [U]

        # Project image and text features into hyperbolic space.
        curv = self._curvature().to(v_h_source.device)
        v_h = self._to_hyperbolic(v_h_source, curv)
        t_h = self._to_hyperbolic(t_h_source, curv)
        parent_text_h = t_h[u_idx]  # [U, D]

        # Build hierarchy pairs through the subclass extension point.
        hierarchy_mask, hierarchy_diagnostics = self._build_hierarchy_mask(
            image_labels=global_labels,
            query_labels=query_labels,
            child_text_h=t_h,
            parent_text_h=parent_text_h,
            curv=curv,
        )

        # Compute the weighted sigmoid contrastive objective.
        with torch.no_grad():
            target, weight = self._build_semantic_target_and_hierarchy_weight(
                alias_pos,
                hierarchy_mask,
            )
        query_text = t_global[u_idx]  # [U, D]
        logits = (v_global @ query_text.T) * self.scale.exp() + self.bias  # [N, U]
        contrastive_loss = (F.softplus(-target * logits) * weight).mean()

        # Assemble text-image and text-text entailment objectives.
        (
            text_image_entailment_loss,
            text_text_entailment_loss,
            hierarchy_loss,
        ) = self._assemble_entailment_losses(
            image_h=v_h,
            child_text_h=t_h,
            parent_text_h=parent_text_h,
            hierarchy_mask=hierarchy_mask,
            curv=curv,
        )

        total_loss = contrastive_loss + self.entailment_loss_weight * hierarchy_loss

        total_losses = dict(hierarchy_diagnostics)
        total_losses["contrastive_loss"] = contrastive_loss
        total_losses["text_image_entailment_loss"] = text_image_entailment_loss
        total_losses["text_text_entailment_loss"] = text_text_entailment_loss
        total_losses["hierarchy_loss"] = hierarchy_loss
        total_losses[CORE_LOSS_KEY] = total_loss
        return total_losses


    def get_extra_state(self):
        return {
            "hierarchy_concepts": self.hierarchy_concepts,
            "seed_edges": self.seed_edges,
        }

    def set_extra_state(self, state):
        concepts = tuple(state.get("hierarchy_concepts", ()))
        if concepts != self.hierarchy_concepts:
            raise RuntimeError(
                "checkpoint concept vocabulary does not match hierarchy data"
            )
        seed_edges = tuple(tuple(edge) for edge in state.get("seed_edges", ()))
        if seed_edges != self.seed_edges:
            raise RuntimeError(
                "checkpoint seed relations do not match hierarchy data"
            )

    @property
    def needs_hierarchy_snapshot(self) -> bool:
        return (
            self.training
            and not self.graph_ready.item()
            and self.calibration_updates.item() >= self.calibration_steps
        )

    @torch.no_grad()
    def _pairwise_relative_angles(
        self,
        text_h: torch.Tensor,
        curv: torch.Tensor,
    ) -> torch.Tensor:
        return hyperbolic.relative_cone_angle(
            parent=text_h.detach().unsqueeze(0),
            child=text_h.detach().unsqueeze(1),
            curv=curv.detach(),
            aperture_scale=self.entailment_aperture_scale,
            min_radius=self.entailment_min_radius,
        ).detach()

    def _seed_graph(self, device) -> torch.Tensor:
        graph = torch.zeros_like(self.hierarchy_graph, device=device)
        for child, parent in self.seed_edges:
            if self.aliases.canonicalize(child) != self.aliases.canonicalize(parent):
                graph[self.concept_index[child], self.concept_index[parent]] = True
        return graph

    @torch.no_grad()
    def _calibrate_graph(self, text_feature: torch.Tensor) -> None:
        if self.graph_ready.item() or not self.needs_hierarchy_snapshot:
            return
        if text_feature.ndim != 2 or text_feature.size(0) != len(
            self.hierarchy_concepts
        ):
            raise ValueError(
                "hierarchy_text_feature must match the ordered hierarchy vocabulary"
            )

        curv = self._curvature().to(text_feature.device)
        text_h = self._to_hyperbolic(text_feature.detach(), curv)
        rel = self._pairwise_relative_angles(text_h, curv)
        seeds = self._seed_graph(rel.device)
        seed_rel = rel[seeds]
        if seed_rel.numel() == 0:
            raise RuntimeError("hierarchy calibration requires at least one seed edge")

        # Calibrate with the linearly interpolated seed Q95, capped at the cone boundary.
        threshold = torch.quantile(
            seed_rel.float(), 0.95, interpolation="linear"
        ).clamp(max=1.0)
        radius = torch.linalg.vector_norm(text_h, dim=-1)
        radial_flag = radius[:, None] > radius[None, :]
        canonical = [self.aliases.canonicalize(name) for name in self.hierarchy_concepts]
        distinct = torch.tensor([[a != b for b in canonical] for a in canonical],
                                dtype=torch.bool, device=rel.device)
        graph = (
            (rel <= threshold)
            & (rel.T > threshold)
            & radial_flag
            & distinct
        )
        graph |= seeds
        # Distinct raw alias nodes can imply opposite directions after merging.
        # Resolve those directions at the concept level before freezing the graph.
        edges = self._alias_graph_edges(graph)
        valid = torch.tensor([[(child, parent) in edges for parent in canonical]
                              for child in canonical], dtype=torch.bool, device=graph.device)
        graph &= valid
        self.rel_threshold.copy_(threshold.to(self.rel_threshold))
        self.hierarchy_graph.copy_(graph.to(self.hierarchy_graph.device))
        self.graph_ready.fill_(True)

    def _alias_graph_edges(self, graph):
        canonical = [self.aliases.canonicalize(name) for name in self.hierarchy_concepts]
        # One transfer, rather than a GPU synchronization per label pair.
        edges = {(canonical[child], canonical[parent])
                 for child, parent in graph.nonzero().tolist()
                 if canonical[child] != canonical[parent]}
        seeds = {(self.aliases.canonicalize(child), self.aliases.canonicalize(parent))
                 for child, parent in self.seed_edges}
        # A seed fixes the direction; ambiguous discovered pairs activate neither way.
        return {(child, parent) for child, parent in edges
                if (child, parent) in seeds or
                ((parent, child) not in edges and (parent, child) not in seeds)}

    def _graph_mask(
        self,
        image_labels: Sequence[dict],
        query_labels: Sequence[dict],
        device,
    ) -> torch.Tensor:
        image_names = [self._canonical_label(label) for label in image_labels]
        query_names = [self._canonical_label(label) for label in query_labels]
        edges = self._alias_graph_edges(self.hierarchy_graph)
        return torch.tensor(
            [[(child, parent) in edges for parent in query_names] for child in image_names],
            dtype=torch.bool, device=device,
        ).reshape(len(image_names), len(query_names))

    def _diagnostics(self) -> Dict[str, torch.Tensor]:
        seed_count = self.rel_threshold.new_tensor(float(sum(
            self.aliases.canonicalize(child) != self.aliases.canonicalize(parent)
            for child, parent in self.seed_edges
        )))
        total_count = self.hierarchy_graph.sum().to(self.rel_threshold)
        return {
            "hierarchy_threshold": self.rel_threshold.detach().clone(),
            "hierarchy_calibration_progress": (
                self.calibration_updates.float() / self.calibration_steps
            ).clamp(max=1.0).detach().clone(),
            "seed_hierarchy_pairs": seed_count.detach(),
            "discovered_hierarchy_pairs": (total_count - seed_count).clamp_min(0).detach(),
            "total_hierarchy_pairs": total_count.detach(),
            "hierarchy_graph_ready": self.graph_ready.float().detach().clone(),
        }

    def _build_hierarchy_mask(
        self,
        image_labels: Optional[Sequence[dict]],
        query_labels: Optional[Sequence[dict]],
        child_text_h: torch.Tensor,
        parent_text_h: torch.Tensor,
        curv: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        seed_mask = self._seed_mask(
            image_labels=image_labels,
            query_labels=query_labels,
            child_text_h=child_text_h,
            parent_text_h=parent_text_h,
            curv=curv,
        )
        mask = seed_mask
        if image_labels is not None and query_labels is not None:
            if self.graph_ready.item():
                mask = self._graph_mask(
                    image_labels,
                    query_labels,
                    device=child_text_h.device,
                )
            elif (
                self.training
                and seed_mask.any()
                and self.calibration_updates.item() < self.calibration_steps
            ):
                self.calibration_updates.add_(1)
        return mask, self._diagnostics()
