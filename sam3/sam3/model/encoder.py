# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# Based on https://github.com/IDEA-Research/GroundingDINO

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .act_ckpt_utils import activation_ckpt_wrapper
from .model_misc import get_activation_fn, get_clones, get_valid_ratio


class SparseMoEFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        activation,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")

        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, d_model),
                )
                for _ in range(num_experts)
            ]
        )
        self.activation = activation

    @staticmethod
    def _balance_loss(
        router_probs: Tensor,
        selected_experts: Tensor,
        num_experts: int,
    ) -> Tensor:
        importance = router_probs.mean(dim=0)
        selected_mask = F.one_hot(selected_experts, num_classes=num_experts).float()
        load = selected_mask.mean(dim=(0, 1))
        return num_experts * torch.sum(importance * load)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        input_shape = x.shape
        flat_x = x.reshape(-1, input_shape[-1])

        router_logits = self.router(flat_x)
        router_probs = torch.softmax(router_logits.float(), dim=-1)
        top_probs, top_idx = torch.topk(router_probs, self.top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        top_probs = top_probs.to(flat_x.dtype)

        flat_out = torch.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.experts):
            token_idx, slot_idx = torch.where(top_idx == expert_id)
            if token_idx.numel() == 0:
                continue
            expert_input = flat_x[token_idx]
            expert_hidden = expert[0](expert_input)
            expert_hidden = self.activation(expert_hidden)
            expert_hidden = expert[1](expert_hidden)
            expert_out = expert[2](expert_hidden)
            expert_out = expert_out * top_probs[token_idx, slot_idx].unsqueeze(-1)
            flat_out = flat_out.index_add(0, token_idx, expert_out)

        balance_loss = self._balance_loss(router_probs, top_idx, self.num_experts)
        return flat_out.reshape(input_shape), balance_loss


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer that performs self-attention followed by cross-attention.

    This layer was previously called TransformerDecoderLayer but was renamed to better
    reflect its role in the architecture. It processes input sequences through self-attention
    and then cross-attention with another input (typically image features).

    The layer supports both pre-norm and post-norm configurations, as well as
    positional encoding at different stages of the attention mechanism.
    """

    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        pre_norm: bool, # True
        self_attention: nn.Module,
    ):
        """
        Initialize a transformer encoder layer.

        Args:
            activation: Activation function to use in the feedforward network
            cross_attention: Cross-attention module for attending to image features
            d_model: Model dimension/hidden size
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout probability
            pos_enc_at_attn: Whether to add positional encodings at self-attention
            pos_enc_at_cross_attn_keys: Whether to add positional encodings to keys in cross-attention
            pos_enc_at_cross_attn_queries: Whether to add positional encodings to queries in cross-attention
            pre_norm: Whether to use pre-norm (True) or post-norm (False) architecture
            self_attention: Self-attention module
        """
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.moe_ffn = None
        self.last_moe_aux_loss = None

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)
        self.pre_norm = pre_norm

        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

        self.layer_idx = None

    def enable_moe_ffn(
        self,
        num_experts: int,
        top_k: int,
        dim_feedforward: int,
    ) -> None:
        self.moe_ffn = SparseMoEFFN(
            d_model=self.d_model,
            dim_feedforward=dim_feedforward,
            dropout=self.dropout_value,
            activation=self.activation,
            num_experts=num_experts,
            top_k=top_k,
        )

    def _forward_ffn(self, x: Tensor) -> Tensor:
        if self.moe_ffn is None:
            self.last_moe_aux_loss = None
            return self.linear2(self.dropout(self.activation(self.linear1(x))))

        out, aux_loss = self.moe_ffn(x)
        self.last_moe_aux_loss = aux_loss
        return out

    def forward_post(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Forward pass for post-norm architecture.

        In post-norm architecture, normalization is applied after attention and feedforward operations.

        Args:
            tgt: Input tensor to be processed
            memory: Memory tensor for cross-attention
            tgt_mask: Mask for self-attention
            memory_mask: Mask for cross-attention
            tgt_key_padding_mask: Key padding mask for self-attention
            memory_key_padding_mask: Key padding mask for cross-attention
            pos: Positional encoding for memory
            query_pos: Positional encoding for query (image)
            **kwargs: Additional keyword arguments

        Returns:
            Processed tensor
        """
        q = k = tgt + query_pos if self.pos_enc_at_attn else tgt

        # Self attention
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention to image
        tgt2 = self.cross_attn_image(
            query=tgt + query_pos if self.pos_enc_at_cross_attn_queries else tgt,
            key=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self._forward_ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt: Tensor,
        memory: Tensor,
        dac: bool = False,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        # attn_bias: Optional[Tensor] = None,
        # **kwargs,
    ) -> Tensor:
        """
        Forward pass for pre-norm architecture.

        In pre-norm architecture, normalization is applied before attention and feedforward operations.

        Args:
            tgt: Input tensor to be processed
            memory: Memory tensor for cross-attention
            dac: Whether to use Divide-and-Conquer attention
            tgt_mask: Mask for self-attention
            memory_mask: Mask for cross-attention
            tgt_key_padding_mask: Key padding mask for self-attention
            memory_key_padding_mask: Key padding mask for cross-attention
            pos: Positional encoding for memory
            query_pos: Positional encoding for query
            attn_bias: Optional attention bias tensor
            **kwargs: Additional keyword arguments

        Returns:
            Processed tensor
        """
        if dac:
            # we only apply self attention to the first half of the queries
            assert tgt.shape[0] % 2 == 0
            other_tgt = tgt[tgt.shape[0] // 2 :]
            tgt = tgt[: tgt.shape[0] // 2]
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        if dac:
            # Recombine
            tgt = torch.cat((tgt, other_tgt), dim=0)
            
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            query=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            key=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            # attn_bias=attn_bias,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self._forward_ffn(tgt2)
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        dac: bool = False,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        # attn_bias: Optional[Tensor] = None,
        # **kwds: Any,
    ) -> torch.Tensor:
        """
        Forward pass for the transformer encoder layer.

        Args:
            tgt: Input tensor to be processed
            memory: Memory tensor (e.g., image features) for cross-attention
            dac: Whether to use Divide-and-Conquer attention (only apply self-attention to first half)
            tgt_mask: Mask for self-attention
            memory_mask: Mask for cross-attention
            tgt_key_padding_mask: Key padding mask for self-attention
            memory_key_padding_mask: Key padding mask for cross-attention
            pos: Positional encoding for memory
            query_pos: Positional encoding for query
            attn_bias: Optional attention bias tensor
            **kwds: Additional keyword arguments

        Returns:
            Processed tensor after self-attention, cross-attention, and feedforward network
        """
        fwd_fn = self.forward_pre if self.pre_norm else self.forward_post
        return fwd_fn(
            tgt,
            memory,
            dac=dac,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos,
            query_pos=query_pos,
            # attn_bias=attn_bias,
            # **kwds,
        )


class TransformerEncoder(nn.Module):
    """
    Transformer encoder that processes multi-level features.

    This encoder takes multi-level features (e.g., from a backbone network) and processes
    them through a stack of transformer encoder layers. It supports features from multiple
    levels (e.g., different resolutions) and can apply activation checkpointing for memory
    efficiency during training.

    Args:
        layer: The encoder layer to be stacked multiple times
        num_layers: Number of encoder layers to stack
        d_model: Model dimension/hidden size
        num_feature_levels: Number of feature levels to process
        frozen: Whether to freeze the parameters of this module
        use_act_checkpoint: Whether to use activation checkpointing during training
    """

    def __init__(
        self,
        layer: nn.Module,
        num_layers: int,
        d_model: int,
        num_feature_levels: int,
        frozen: bool = False,
        use_act_checkpoint: bool = False,
        moe_num_experts: int = 0,
        moe_top_k: int = 2,
        moe_dim_feedforward: Optional[int] = None,
        moe_layers: int = 0,
        moe_layer_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers

        self.num_feature_levels = num_feature_levels
        self.level_embed = None
        if num_feature_levels > 1:
            self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if moe_layer_indices is None:
            if moe_layers < 0 or moe_layers > num_layers:
                raise ValueError(
                    f"moe_layers must be in [0, {num_layers}], got {moe_layers}"
                )
            moe_layer_indices = (
                list(range(num_layers - moe_layers, num_layers))
                if moe_layers > 0
                else []
            )
        else:
            normalized_moe_layer_indices = []
            for idx in moe_layer_indices:
                idx = idx + num_layers if idx < 0 else idx
                if idx < 0 or idx >= num_layers:
                    raise ValueError(
                        f"MoE layer index {idx} is out of range for {num_layers} layers"
                    )
                normalized_moe_layer_indices.append(idx)
            moe_layer_indices = normalized_moe_layer_indices
        moe_layer_indices = set(moe_layer_indices)

        # assign layer index to each layer so that some layers can decide what to do
        # based on which layer index they are (e.g. cross attention to memory bank only
        # in selected layers)
        for layer_idx, layer in enumerate(self.layers):
            layer.layer_idx = layer_idx
            if moe_num_experts > 0 and layer_idx in moe_layer_indices:
                layer.enable_moe_ffn(
                    num_experts=moe_num_experts,
                    top_k=moe_top_k,
                    dim_feedforward=moe_dim_feedforward or layer.dim_feedforward,
                )

        if frozen:
            for p in self.parameters():
                p.requires_grad_(False)

        self.use_act_checkpoint = use_act_checkpoint

        self.last_moe_aux_loss = None

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        with torch.no_grad():
            reference_points_list = []
            for lvl, (H_, W_) in enumerate(spatial_shapes):
                ref_y, ref_x = torch.meshgrid(
                    torch.linspace(
                        0.5, H_ - 0.5, H_, dtype=torch.float32, device=device
                    ),
                    torch.linspace(
                        0.5, W_ - 0.5, W_, dtype=torch.float32, device=device
                    ),
                )
                ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
                ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
                ref = torch.stack((ref_x, ref_y), -1)
                reference_points_list.append(ref)
            reference_points = torch.cat(reference_points_list, 1)
            reference_points = reference_points[:, :, None] * valid_ratios[:, None]

        return reference_points

    def _prepare_multilevel_features(self, srcs, masks, pos_embeds):
        assert (
            len(srcs) == self.num_feature_levels
        ), "mismatch between expected and received # of feature levels"

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        has_mask = masks is not None and masks[0] is not None
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            if has_mask:
                mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            if self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            if has_mask:
                mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1) if has_mask else None  # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # bs, \sum{hxw}, c
        spatial_shapes = torch.tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )
        if has_mask:
            valid_ratios = torch.stack([get_valid_ratio(m) for m in masks], 1)
        else:
            valid_ratios = torch.ones(
                (src_flatten.shape[0], self.num_feature_levels, 2),
                device=src_flatten.device,
            )

        return (
            src_flatten,
            mask_flatten,
            lvl_pos_embed_flatten,
            level_start_index,
            valid_ratios,
            spatial_shapes,
        )

    def forward(
        self,
        src: List[Tensor],
        src_key_padding_masks: Optional[List[Tensor]] = None,
        pos: Optional[List[Tensor]] = None,
        prompt: Optional[Tensor] = None,
        prompt_key_padding_mask: Optional[Tensor] = None,
        encoder_extra_kwargs: Optional[Dict] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor, Tensor, Tensor]:
        """
        Process multi-level features through the transformer encoder.

        Args:
            src: List of multi-level features, each with shape (batch_size, channels, height, width)
            src_key_padding_masks: List of padding masks for each feature level, each with shape (batch_size, height, width)
            pos: List of positional embeddings for each feature level, each with shape (batch_size, channels, height, width)
            prompt: Optional text/prompt features to attend to, with shape (seq_len, batch_size, d_model)
            prompt_key_padding_mask: Optional padding mask for prompt, with shape (batch_size, seq_len)
            encoder_extra_kwargs: Optional additional arguments to pass to each encoder layer

        Returns:
            A tuple containing:
            - output: Processed features with shape (seq_len, batch_size, d_model)
            - key_padding_masks_flatten: Flattened padding masks
            - lvl_pos_embed_flatten: Flattened positional embeddings
            - level_start_index: Starting indices for each feature level
            - spatial_shapes: Spatial dimensions of each feature level
            - valid_ratios: Valid ratios for each feature level
        """
        assert (
            len(src) == self.num_feature_levels
        ), "must be equal to num_feature_levels"
        # src_key_padding_masks is None num_feature_levels is 1
        if src_key_padding_masks is not None:
            assert len(src_key_padding_masks) == self.num_feature_levels
        if pos is not None:
            assert len(pos) == self.num_feature_levels
        # Flatten multilevel feats and add level pos embeds
        (
            src_flatten,
            key_padding_masks_flatten,
            lvl_pos_embed_flatten,
            level_start_index,
            valid_ratios,
            spatial_shapes,
        ) = self._prepare_multilevel_features(src, src_key_padding_masks, pos)

        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src_flatten.device
        )

        output = src_flatten
        moe_aux_losses = []
        for layer in self.layers:
            layer_kwargs = {}

            assert isinstance(layer, TransformerEncoderLayer)
            layer_kwargs["memory"] = prompt
            layer_kwargs["memory_key_padding_mask"] = prompt_key_padding_mask
            layer_kwargs["query_pos"] = lvl_pos_embed_flatten
            layer_kwargs["tgt"] = output
            layer_kwargs["tgt_key_padding_mask"] = key_padding_masks_flatten

            if self.training:
                assert self.use_act_checkpoint, "activation ckpt not enabled in encoder"
            if encoder_extra_kwargs is not None:
                layer_kwargs.update(encoder_extra_kwargs)
            output = activation_ckpt_wrapper(layer)(
                **layer_kwargs,
                act_ckpt_enable=self.training and self.use_act_checkpoint,
            )
            if layer.last_moe_aux_loss is not None:
                moe_aux_losses.append(layer.last_moe_aux_loss)

        self.last_moe_aux_loss = (
            torch.stack(moe_aux_losses).mean() if moe_aux_losses else None
        )
        # return as seq first
        return (
            output.transpose(0, 1),
            (
                key_padding_masks_flatten.transpose(0, 1)
                if key_padding_masks_flatten is not None
                else None
            ),
            lvl_pos_embed_flatten.transpose(0, 1),
            level_start_index,
            spatial_shapes,
            valid_ratios,
        )


class TransformerEncoderFusion(TransformerEncoder):
    """
    Transformer encoder that fuses text and image features.

    This encoder extends TransformerEncoder to handle both text and image features,
    with the ability to add pooled text features to image features for better
    cross-modal fusion. It supports torch.compile for performance optimization.

    Args:
        layer: The encoder layer to be stacked multiple times
        num_layers: Number of encoder layers to stack
        d_model: Model dimension/hidden size
        num_feature_levels: Number of feature levels to process
        add_pooled_text_to_img_feat: Whether to add pooled text features to image features
        pool_text_with_mask: Whether to use the mask when pooling text features
        compile_mode: Mode for torch.compile, or None to disable compilation
        **kwargs: Additional arguments to pass to the parent class
    """

    def __init__(
        self,
        layer: nn.Module,
        num_layers: int,
        d_model: int,
        num_feature_levels: int,
        add_pooled_text_to_img_feat: bool = True,
        pool_text_with_mask: bool = False,
        compile_mode: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            layer,
            num_layers,
            d_model,
            num_feature_levels,
            **kwargs,
        )
        self.add_pooled_text_to_img_feat = add_pooled_text_to_img_feat
        if self.add_pooled_text_to_img_feat:
            self.text_pooling_proj = nn.Linear(d_model, d_model)
        self.pool_text_with_mask = pool_text_with_mask
        if compile_mode is not None:
            self.forward = torch.compile(
                self.forward, mode=compile_mode, fullgraph=True
            )

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        # Not needed here
        return None

    def forward(
        self,
        src: List[Tensor],
        prompt: Tensor,
        src_key_padding_mask: Optional[List[Tensor]] = None,
        src_pos: Optional[List[Tensor]] = None,
        prompt_key_padding_mask: Optional[Tensor] = None,
        prompt_pos: Optional[Tensor] = None,
        feat_sizes: Optional[List[int]] = None,
        encoder_extra_kwargs: Optional[Dict] = None,
    ):
        # Restore spatial shapes of vision
        bs = src[0].shape[1]  # seq first
        if feat_sizes is not None:
            assert len(feat_sizes) == len(src)
            if src_key_padding_mask is None:
                src_key_padding_mask = [None] * len(src)
            for i, (h, w) in enumerate(feat_sizes):
                src[i] = src[i].reshape(h, w, bs, -1).permute(2, 3, 0, 1)
                src_pos[i] = src_pos[i].reshape(h, w, bs, -1).permute(2, 3, 0, 1)
                src_key_padding_mask[i] = (
                    src_key_padding_mask[i].reshape(h, w, bs).permute(2, 0, 1)
                    if src_key_padding_mask[i] is not None
                    else None
                )
        else:
            assert all(
                x.dim == 4 for x in src
            ), "expected list of (bs, c, h, w) tensors"

        if self.add_pooled_text_to_img_feat:
            # Fusion: Add mean pooled text to image features
            pooled_text = pool_text_feat(
                prompt, prompt_key_padding_mask, self.pool_text_with_mask
            )
            pooled_text = self.text_pooling_proj(pooled_text)[
                ..., None, None
            ]  # prompt is seq first
            src = [x.add_(pooled_text) for x in src]

        (
            out,
            key_padding_masks_flatten,
            lvl_pos_embed_flatten,
            level_start_index,
            spatial_shapes,
            valid_ratios,
        ) = super().forward(
            src,
            src_key_padding_masks=src_key_padding_mask,
            pos=src_pos,
            prompt=prompt.transpose(0, 1),
            prompt_key_padding_mask=prompt_key_padding_mask,
            encoder_extra_kwargs=encoder_extra_kwargs,
        )

        result = {
            "memory": out,
            "padding_mask": key_padding_masks_flatten,
            "pos_embed": lvl_pos_embed_flatten,
            "memory_text": prompt,
            "level_start_index": level_start_index,
            "spatial_shapes": spatial_shapes,
            "valid_ratios": valid_ratios,
        }
        if self.last_moe_aux_loss is not None:
            result["moe_aux_loss"] = self.last_moe_aux_loss
        return result


def pool_text_feat(prompt, prompt_mask, pool_with_mask):
    # prompt has shape (seq, bs, dim)
    if not pool_with_mask:
        return prompt.mean(dim=0)

    # prompt_mask has shape (bs, seq), where False is valid and True is padding
    assert prompt_mask.dim() == 2
    # is_valid has shape (seq, bs, 1), where 1 is valid and 0 is padding
    is_valid = (~prompt_mask).float().permute(1, 0)[..., None]
    # num_valid has shape (bs, 1)
    num_valid = torch.clamp(torch.sum(is_valid, dim=0), min=1.0)

    # mean pool over all the valid tokens
    pooled_text = (prompt * is_valid).sum(dim=0) / num_valid
    return pooled_text # (B, 256)
