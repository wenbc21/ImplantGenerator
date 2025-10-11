# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from functools import partial
# from einops import rearrange, reduce, repeat

from monai.networks.blocks import Convolution, UpSample, CrossAttentionBlock
from monai.networks.blocks.mednext_block import MedNeXtBlock, MedNeXtDownBlock, MedNeXtOutBlock, MedNeXtUpBlock
from monai.networks.layers.factories import Conv, Pool
from monai.utils import ensure_tuple_rep


class TwoConv(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: str | tuple,
        norm: str | tuple,
        bias: bool,
        dropout: float | tuple = 0.0,
    ):
        super().__init__()
        emb_chns = 64*4
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            torch.nn.Linear(
                emb_chns,
                out_chns,
            ),
        )

        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(
            spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)
    
    def forward(self, x, emb=None):
        x = self.conv_0(x)
        if emb is not None :
            emb_out = self.emb_layers(emb)
            while len(emb_out.shape) < len(x.shape):
                emb_out = emb_out[..., None]
            x = x + emb_out
        x = self.conv_1(x)
        return x 


class ResidualBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int | None = None,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.channels = in_channels
        self.out_channels = out_channels or in_channels
        
        emb_chns = 64*4
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            torch.nn.Linear(
                emb_chns,
                out_channels,
            ),
        )

        self.norm1 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=out_channels, eps=norm_eps, affine=True)
        self.nonlinearity = nn.SiLU()
        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        self.norm2 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=self.out_channels, eps=norm_eps, affine=True)
        self.conv2 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )
        self.skip_connection: nn.Module
        if self.out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        h = x
        h = self.conv1(h)
        h = self.norm1(h)
        h = self.nonlinearity(h)

        if emb is not None :
            emb_out = self.emb_layers(emb)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
            h = h + emb_out

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.nonlinearity(h)
        output: torch.Tensor = self.skip_connection(x) + h
        return output


class Down(nn.Sequential):

    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: str | tuple,
        norm: str | tuple,
        bias: bool,
        dropout: float | tuple = 0.0,
    ):
        super().__init__()
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        # convs = ResidualBlock(spatial_dims, in_chns, out_chns)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)
    
    def forward(self, x, emb=None):
        x = self.max_pooling(x)
        x = self.convs(x, emb)
        return x 


class UpCat(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: str | tuple,
        norm: str | tuple,
        bias: bool,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
        pre_conv: nn.Module | str | None = "default",
        interp_mode: str = "linear",
        align_corners: bool | None = True,
        halves: bool = True,
        is_pad: bool = True,
    ):
        super().__init__()
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)
        # self.convs = ResidualBlock(spatial_dims, cat_chns + up_chns, out_chns)
        self.is_pad = is_pad

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor], emb=None):
        x_0 = self.upsample(x)

        if x_e is not None and torch.jit.isinstance(x_e, torch.Tensor):
            if self.is_pad:
                # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
                dimensions = len(x.shape) - 2
                sp = [0] * (dimensions * 2)
                for i in range(dimensions):
                    if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                        sp[i * 2 + 1] = 1
                x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1), emb)  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0, emb)

        return x


def Fourier_filter_3d(x, threshold: int = 1, scale: float = 1.0):
    """
    3D Fourier filter for feature maps.
    x: [B, C, D, H, W]
    """
    # FFT
    x_freq = fft.fftn(x, dim=(-3, -2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))

    B, C, D, H, W = x_freq.shape
    mask = torch.ones((B, C, D, H, W), device=x.device, dtype=x.dtype)

    cd, ch, cw = D // 2, H // 2, W // 2
    mask[..., cd - threshold:cd + threshold,
         ch - threshold:ch + threshold,
         cw - threshold:cw + threshold] = scale

    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-3, -2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-3, -2, -1)).real

    return x_filtered


class FreeU(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv"
    ):
        super().__init__()
        fea = ensure_tuple_rep(features, 6)
        print(f"UNet features: {fea}.")

        self.conv_0 = TwoConv(spatial_dims, in_channels, fea[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

        # self.bottleneck = TwoConv(spatial_dims, fea[4], fea[4], act, norm, bias, dropout)

        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)
        
        # FreeU parameters
        self.b1, self.b2 = 1.3, 1.4
        self.s1, self.s2 = 0.9, 0.2

    def forward(self, x: torch.Tensor):
        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)
        
        # Stage 1 (deepest)
        hidden_mean = x4.mean(1, keepdim=True)
        hidden_mean = (hidden_mean - hidden_mean.min()) / (hidden_mean.max() - hidden_mean.min() + 1e-6)
        scale = ((self.b1 - 1) * hidden_mean + 1)
        x4 = torch.cat([
            x4[:, : x4.shape[1] // 2] * scale,
            x4[:, x4.shape[1] // 2:]
        ], dim=1)
        x3 = Fourier_filter_3d(x3, threshold=1, scale=self.s1)
        u4 = self.upcat_4(x4, x3)

        # Stage 2
        hidden_mean = u4.mean(1, keepdim=True)
        hidden_mean = (hidden_mean - hidden_mean.min()) / (hidden_mean.max() - hidden_mean.min() + 1e-6)
        scale = ((self.b2 - 1) * hidden_mean + 1)
        u4 = torch.cat([
            u4[:, : u4.shape[1] // 2] * scale,
            u4[:, u4.shape[1] // 2:]
        ], dim=1)
        x2 = Fourier_filter_3d(x2, threshold=1, scale=self.s2)
        u3 = self.upcat_3(u4, x2)
        
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)

        logits = self.final_conv(u1)

        return logits
