# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Various utility models."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn


class DotProductScoring(torch.nn.Module):
    """A module that computes dot-product scores between a set of query features and a."""

    def __init__(
        self,
        d_model,
        d_proj,
        prompt_mlp=None,
        clamp_logits=True,
        clamp_max_val=12.0,
    ):
        """Initialize the DotProductScoring module."""
        super().__init__()
        self.d_proj = d_proj
        assert isinstance(prompt_mlp, torch.nn.Module) or prompt_mlp is None
        self.prompt_mlp = prompt_mlp
        self.prompt_proj = torch.nn.Linear(d_model, d_proj)
        self.hs_proj = torch.nn.Linear(d_model, d_proj)
        self.scale = float(1.0 / np.sqrt(d_proj))
        self.clamp_logits = clamp_logits
        if self.clamp_logits:
            self.clamp_max_val = clamp_max_val

    @staticmethod
    def mean_pool_text(prompt, prompt_mask):
        """Mean-pool the prompt embeddings over the valid tokens only."""

        is_valid = (~prompt_mask).to(prompt.dtype).permute(1, 0)[..., None]

        num_valid = torch.clamp(torch.sum(is_valid, dim=0), min=1.0)

        pooled_prompt = (prompt * is_valid).sum(dim=0) / num_valid
        return pooled_prompt

    def forward(self, hs, prompt, prompt_mask):
        """Compute dot-product scores between hs and prompt."""



        assert hs.dim() == 4 and prompt.dim() == 3 and prompt_mask.dim() == 2


        if self.prompt_mlp is not None:
            prompt = self.prompt_mlp(prompt.to(hs.dtype))


        pooled_prompt = self.mean_pool_text(prompt, prompt_mask)


        proj_pooled_prompt = self.prompt_proj(pooled_prompt)
        proj_hs = self.hs_proj(hs)


        scores = torch.matmul(proj_hs, proj_pooled_prompt.unsqueeze(-1))
        scores *= self.scale


        if self.clamp_logits:
            scores.clamp_(min=-self.clamp_max_val, max=self.clamp_max_val)

        return scores


class LayerScale(nn.Module):
    """LayerScale module as introduced in "Meta Pseudo Labels" and used in."""

    def __init__(
        self,
        dim: int,
        init_values: float | Tensor = 1e-5,
        inplace: bool = False,
    ) -> None:
        """Initialize the LayerScale module."""
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Apply LayerScale to the input tensor."""
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class TransformerWrapper(nn.Module):
    """A wrapper for the transformer consisting of an encoder and a decoder."""

    def __init__(
        self,
        encoder,
        decoder,
        d_model: int,
        two_stage_type="none",
        pos_enc_at_input_dec=True,
    ):
        """Initialize the TransformerWrapper."""
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_queries = decoder.num_queries if decoder is not None else None
        self.pos_enc_at_input_dec = pos_enc_at_input_dec


        assert two_stage_type in ["none"], f"unknown param {two_stage_type} of two_stage_type"
        self.two_stage_type = two_stage_type

        self._reset_parameters()
        self.d_model = d_model

    def _reset_parameters(self):
        """Initialize the parameters of the model."""
        for n, p in self.named_parameters():
            if p.dim() > 1:
                if "box_embed" not in n and "query_embed" not in n and "reference_points" not in n:
                    nn.init.xavier_uniform_(p)


def get_valid_ratio(mask):
    """Compute the valid ratio of height and width from the mask."""
    _, H, W = mask.shape
    valid_H = torch.sum(~mask[:, :, 0], 1)
    valid_W = torch.sum(~mask[:, 0, :], 1)
    valid_ratio_h = valid_H.float() / H
    valid_ratio_w = valid_W.float() / W
    valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
    return valid_ratio


def gen_sineembed_for_position(pos_tensor: torch.Tensor, num_feats: int = 256):
    """Generate sinusoidal position embeddings for 2D or 4D coordinate tensors.

    This function creates sinusoidal embeddings using sine and cosine functions at different frequencies, similar to the
    positional encoding used in Transformer models. It supports both 2D position tensors (x, y) and 4D tensors (x, y, w,
    h) for bounding box coordinates.

    Args:
        pos_tensor (torch.Tensor): Input position tensor of shape (n_query, bs, 2) for 2D coordinates or (n_query, bs,
            4) for 4D coordinates (bounding boxes).
        num_feats (int): Number of feature dimensions for the output embedding. Must be even. Defaults to 256.

    Returns:
        (torch.Tensor): Sinusoidal position embeddings of shape (n_query, bs, num_feats) for 2D input or (n_query, bs,
            num_feats * 2) for 4D input.

    Raises:
        AssertionError: If num_feats is not even.
        ValueError: If pos_tensor.size(-1) is not 2 or 4.

    Examples:
        >>> pos_2d = torch.rand(100, 8, 2)  # 100 queries, batch size 8, 2D coordinates
        >>> embeddings_2d = gen_sineembed_for_position(pos_2d, num_feats=256)
        >>> embeddings_2d.shape
        torch.Size([100, 8, 256])
        >>> pos_4d = torch.rand(50, 4, 4)  # 50 queries, batch size 4, 4D coordinates
        >>> embeddings_4d = gen_sineembed_for_position(pos_4d, num_feats=128)
        >>> embeddings_4d.shape
        torch.Size([50, 4, 256])
    """
    assert num_feats % 2 == 0
    num_feats = num_feats // 2


    scale = 2 * math.pi
    dim_t = torch.arange(num_feats, dtype=pos_tensor.dtype, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (torch.div(dim_t, 2, rounding_mode="floor")) / num_feats)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError(f"Unknown pos_tensor shape(-1):{pos_tensor.size(-1)}")
    return pos
