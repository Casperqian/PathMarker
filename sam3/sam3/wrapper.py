import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Literal
from sam3.model_builder import build_pathmarker_semantic_model, build_sam3_encoder


def gumbel_sigmoid(logits: torch.Tensor, tau:float = 1, hard: bool = False):
    """Samples from the Gumbel-Sigmoid distribution and optionally discretizes.

    References:
        - https://github.com/yandexdataschool/gumbel_dpg/blob/master/gumbel.py
        - https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax

    Note:
        X - Y ~ Logistic(0,1) s.t. X, Y ~ Gumbel(0, 1).
        That is, we can implement gumbel_sigmoid using Logistic distribution.
    """
    logistic = torch.rand_like(logits)
    logistic = logistic.div_(1. - logistic).log_()  # ~Logistic(0,1)

    gumbels = (logits + logistic) / tau  # ~Logistic(logits, tau)
    y_soft = gumbels.sigmoid_()

    if hard:
        # Straight through.
        y_hard = y_soft.gt(0.5).type(y_soft.dtype)
        # gt_ break gradient flow
        #  y_hard = y_soft.gt_(0.5)  # gt_() maintain dtype, different to gt()
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft

    return ret


class AttentionPooling(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_probe: int = 1,
        mlp_ratio: int = 4,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert (
            self.embed_dim % num_heads == 0
        ), "embed_dim must be divisible by num_heads"

        self.probe = nn.Parameter(torch.randn(1, num_probe, self.embed_dim))
        self.attn = nn.MultiheadAttention(
            self.embed_dim, self.num_heads, batch_first=True
        )

        self.layernorm = norm_layer(embed_dim)
        self.mlp_width = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(self.embed_dim, self.mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(self.mlp_width, self.embed_dim)),
                ]
            )
        )

    def forward(self, x: torch.Tensor):
        batch, _, _ = x.shape

        q = self.probe.repeat((batch, 1, 1)).to(x.dtype)
        x = self.attn(q, x, x, need_weights=False)[0]
        x = x + self.mlp(self.layernorm(x))

        return x
    

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)
    

class PostProcess(nn.Module):
    def __init__(
            self,
            context_dim: int,
            output_dim,
            norm_layer: Callable = LayerNorm,
            normalize_final: bool = True,
            skip_ln: bool = False
    ):
        super().__init__()
        self.skip_ln = skip_ln
        self.scale = context_dim ** -0.5
        self.proj = nn.Parameter(self.scale * torch.randn(context_dim, output_dim))
        if not self.skip_ln:
            self.ln_post = norm_layer(context_dim)
        self.normalize_final = normalize_final

    def forward(self, x):
        if self.skip_ln:
            out = x @ self.proj
        else:
            out = self.ln_post(x) @ self.proj

        if self.normalize_final:
            return F.normalize(out, dim=-1)
        else:
            return out
        
    
class PureAttentionPoolingBlock(nn.Module):
    """
    Just a pure attn_pooling implementation, without ln_post, without projection, no mormalized_final
    """

    def __init__(
            self,
            context_dim: int,
            n_head: int = 8,
            norm_layer: Callable = LayerNorm,
            need_weights: bool = False
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(context_dim, n_head, kdim=context_dim, vdim=context_dim, batch_first=True,
                                          add_zero_attn=True)
        #self.attn = nn.MultiheadAttention(context_dim, n_head, kdim=context_dim, vdim=context_dim, batch_first=True)
        self.ln_q = norm_layer(context_dim)
        self.ln_k = norm_layer(context_dim)
        self.ln_v = norm_layer(context_dim)
        self.need_weights=need_weights

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, output_attn_weights=False, average_attn_weights=True):
        batch_size, seg_length, embed_dim = k.size()
        q = self.ln_q(q)
        k = self.ln_k(k)
        v = self.ln_v(v)

        if self.need_weights or output_attn_weights:
            out, attn_weights = self.attn(q, k, v, need_weights=True, average_attn_weights=average_attn_weights)
            return out, attn_weights
        else:
            out = self.attn(q, k, v, need_weights=False)[0]
        # we can directly normalize the output, without setting a flag
        #return F.normalize(out, dim=-1)
            return out


class EncoderWrapper(nn.Module):
    def __init__(
        self,
        bpe_path=None,
        d_model=256,
        device=None,
        eval_mode=True,
        checkpoint_path=None,
        load_from_HF=True,
        enable_inst_interactivity=False,
        compile=False,
        attn_pooler_heads=8,
    ):
        super().__init__()

        self.sam3 = build_sam3_encoder(
            bpe_path=bpe_path,
            device=device,
            eval_mode=eval_mode,
            checkpoint_path=checkpoint_path,
            load_from_HF=load_from_HF,
            enable_inst_interactivity=enable_inst_interactivity,
            compile=compile,
        )
        self.vision_pool = PureAttentionPoolingBlock(
            context_dim=d_model,
            n_head=attn_pooler_heads,
        )
        self.vision_post = PostProcess(context_dim=d_model, output_dim=d_model, normalize_final=False)
        self.text_post = PostProcess(context_dim=d_model, output_dim=d_model, normalize_final=False)

    def _encode_hierarchy_texts(self, texts, device):
        backbone = self.sam3.backbone
        was_training = backbone.training
        try:
            backbone.eval()
            with torch.no_grad():
                output = backbone.forward_text(list(texts), device=device)
                text_feat = pool_text_feat(
                    output['language_features'],
                    output['language_mask'],
                )
                return self.text_post(text_feat).detach()
        finally:
            backbone.train(was_training)

    def forward(self, batch, num_level=2, hierarchy_texts=None):
        output = self.sam3(batch)

        language_feat = output['language_features']
        language_mask = output['language_mask']
        text_feat = pool_text_feat(language_feat, language_mask) # [B, C]
        text_feat = self.text_post(text_feat)
        text_query = text_feat.unsqueeze(1)

        vision_feat_list = output['backbone_fpn'][-num_level:]

        # [B, C, H, W] -> [B, N, C]
        tokens = [feat.flatten(2).transpose(1, 2) for feat in vision_feat_list]
        vision_feat = torch.cat(tokens, dim=1)  # [B, N, C]
        vision_feat = self.vision_post(vision_feat)

        tc_vision_feat = self.vision_pool(text_query, vision_feat, vision_feat)
        tc_vision_feat  = tc_vision_feat.squeeze(1)

        result = {
            'vision_feat': tc_vision_feat,
            'text_feature': text_feat,
        }
        if hierarchy_texts:
            result['hierarchy_text_feature'] = self._encode_hierarchy_texts(
                hierarchy_texts,
                text_feat.device,
            )
        return result

def pool_text_feat(prompt, prompt_mask):
    # prompt_mask has shape (bs, seq), where False is valid and True is padding
    assert prompt_mask.dim() == 2
    # is_valid has shape (seq, bs, 1), where 1 is valid and 0 is padding
    is_valid = (~prompt_mask).float().permute(1, 0)[..., None]
    # num_valid has shape (bs, 1)
    num_valid = torch.clamp(torch.sum(is_valid, dim=0), min=1.0)
    # mean pool over all the valid tokens
    pooled_text = (prompt * is_valid).sum(dim=0) / num_valid
    return pooled_text # (B, 256)


# class EncoderWrapper(nn.Module):
#     def __init__(
#         self,
#         bpe_path=None,
#         d_model=256,
#         device=None,
#         eval_mode=True,
#         checkpoint_path=None,
#         load_from_HF=True,
#         enable_inst_interactivity=False,
#         compile=False,
#         attn_pooler_heads=8,
#         act_layer=nn.GELU,
#         norm_layer=partial(nn.LayerNorm, eps=1e-5),
#         num_semantic_center=8,
#     ):
#         super().__init__()

#         self.sam3 = build_sam3_encoder(
#             bpe_path=bpe_path,
#             device=device,
#             eval_mode=eval_mode,
#             checkpoint_path=checkpoint_path,
#             load_from_HF=load_from_HF,
#             enable_inst_interactivity=enable_inst_interactivity,
#             compile=compile,
#         )
#         self.semantic_layer = AttentionPooling(
#             embed_dim=d_model,
#             num_heads=attn_pooler_heads,
#             act_layer=act_layer,
#             norm_layer=norm_layer,
#             num_probe=num_semantic_center
#         )
#         self.num_semantic_center = num_semantic_center
#         self.text_pool = nn.Linear(d_model, d_model)
#         self.proj = nn.Linear(d_model, d_model)

#     def forward(self, batch, num_level=2):
#         output = self.sam3(batch)

#         language_feat = output['language_features']
#         language_mask = output['language_mask']
#         # text_ids = output['text_ids']
#         text_feat = pool_text_feat(language_feat, language_mask)

#         vision_feat_list = output['backbone_fpn'][-num_level:]

#         # [B, C, H, W] -> [B, N, C]
#         tokens = [feat.flatten(2).transpose(1, 2) for feat in vision_feat_list]
#         v_h = torch.cat(tokens, dim=1)  # [B, N, C]

#         s_c = self.semantic_layer(v_h)  # [B, K, C]

#         assign_logits = torch.matmul(v_h, s_c.transpose(-1, -2))  # [B, N, K]
#         group_metrics = F.gumbel_softmax(assign_logits, tau=1.0, hard=True, dim=-1)

#         group_sum = torch.einsum('bnk,bnc->bkc', group_metrics, v_h)
#         group_cnt = group_metrics.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)
#         s_h = group_sum / group_cnt

#         s_h = self.proj(s_h) + s_c
#         t_h = self.text_pool(text_feat)

#         return {
#             'semantic_feature': s_h,
#             'text_feature': t_h,
#             'semantic_center': s_c,
#         }


# class Sam3Wrapper(nn.Module):
#     def __init__(
#         self,
#         bpe_path=None,
#         d_model=256,
#         device=None,
#         eval_mode=True,
#         checkpoint_path=None,
#         load_from_HF=True,
#         enable_segmentation=True,
#         enable_inst_interactivity=False,
#         compile=False,
#         attn_pooler_heads=8,
#         act_layer=nn.GELU,
#         norm_layer=partial(nn.LayerNorm, eps=1e-5),
#     ):
#         super().__init__()

#         self.sam3 = build_pathmarker_semantic_model(
#             bpe_path=bpe_path,
#             device=device,
#             eval_mode=eval_mode,
#             checkpoint_path=checkpoint_path,
#             load_from_HF=load_from_HF,
#             enable_segmentation=enable_segmentation,
#             enable_inst_interactivity=enable_inst_interactivity,
#             compile=compile,
#         )
#         self.vision_pool = AttentionPooling(
#             embed_dim=d_model,
#             num_heads=attn_pooler_heads,
#             act_layer=act_layer,
#             norm_layer=norm_layer,
#         )
#         self.text_pool = nn.Linear(d_model, d_model)
#         self.proj = nn.Linear(d_model, d_model)
#         self.attn = nn.Linear(d_model, 1)


#     def forward(self, batch, num_level=2, return_attn=False):
#         output = self.sam3.forward_with_backbone_features(batch)

#         language_feat, language_mask = output[0]['language_features'], output[0]['language_mask']
#         text_feat = pool_text_feat(language_feat, language_mask)

#         # [B, 256, 288, 288], [B, 256, 144, 144], [B, 256, 72, 72]
#         vision_feat_list = output[0]['backbone_fpn'][-num_level:]  # no iterative, only one stage
#         vision_feat_size = [feat.shape[-2:] for feat in vision_feat_list]
        
#         # Create mask for semantic segmentation
#         mask = output[0]['semantic_seg']  # [B, 1, 288, 288]
#         mask_list = [F.interpolate(mask, size, mode='bilinear') for size in vision_feat_size]
#         mask_list = [gumbel_sigmoid(mask, tau=1.0, hard=True) for mask in mask_list]
        
#         # Flatten feature maps and masks for easier manipulation
#         vision_feat_list = [feat.flatten(2) for feat in vision_feat_list]
#         mask_list = [mask.flatten(2) for mask in mask_list]
#         vision_feat_list = [feat * mask for feat, mask in zip(vision_feat_list, mask_list)]
        
#         # Concatenate the masked feature maps
#         v_h = torch.cat(vision_feat_list, dim=-1)  # (B, dim, N)
#         v_h = self.vision_pool(v_h.transpose(1, 2)).squeeze(1) # (B, dim)
        
#         t_h = self.text_pool(text_feat)
        
#         return {'vision': v_h, 'language': t_h}


class ClipEncoderWrapper(nn.Module):
    def __init__(
        self,
        bpe_path=None,
        d_model=256,
        device=None,
        eval_mode=True,
        checkpoint_path=None,
        load_from_HF=True,
        enable_inst_interactivity=False,
        compile=False,
        attn_pooler_heads=8,
        act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-5),
    ):
        super().__init__()

        self.sam3 = build_sam3_encoder(
            bpe_path=bpe_path,
            device=device,
            eval_mode=eval_mode,
            checkpoint_path=checkpoint_path,
            load_from_HF=load_from_HF,
            enable_inst_interactivity=enable_inst_interactivity,
            compile=compile,
        )
        self.attn = AttentionPooling(
            embed_dim=d_model,
            num_heads=attn_pooler_heads,
            act_layer=act_layer,
            norm_layer=norm_layer,
            num_probe=1
        )
        self.text_pool = nn.Linear(d_model, d_model)

    def forward(self, batch, num_level=2):
        output = self.sam3(batch)

        language_feat = output['language_features']
        language_mask = output['language_mask']
        # text_ids = output['text_ids']
        text_feat = pool_text_feat(language_feat, language_mask)

        vision_feat_list = output['backbone_fpn'][-num_level:]

        # [B, C, H, W] -> [B, N, C]
        tokens = [feat.flatten(2).transpose(1, 2) for feat in vision_feat_list]
        v_h = torch.cat(tokens, dim=1)  # [B, N, C]
        v_h = self.attn(v_h).squeeze(1)  # [B, C]
        t_h = self.text_pool(text_feat)

        return {
            'vision_feature': v_h,
            'text_feature': t_h,
        }
