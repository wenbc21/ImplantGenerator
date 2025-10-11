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

import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, UpSample
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


class UNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv"
    ):
        super().__init__()
        fea = ensure_tuple_rep(features, 6)

        self.conv_0 = TwoConv(spatial_dims, in_channels, fea[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        u4 = self.upcat_4(x4, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)

        logits = self.final_conv(u1)

        return logits
