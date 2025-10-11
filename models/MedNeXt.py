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
from functools import partial
# from einops import rearrange, reduce, repeat

from monai.networks.blocks import Convolution, UpSample, CrossAttentionBlock
from monai.networks.blocks.mednext_block import MedNeXtBlock, MedNeXtDownBlock, MedNeXtOutBlock, MedNeXtUpBlock
from monai.networks.layers.factories import Conv, Pool
from monai.utils import ensure_tuple_rep


class MedNeXt(nn.Module):
    """
    MedNeXt model class from paper: https://arxiv.org/pdf/2303.09975

    Args:
        spatial_dims: spatial dimension of the input data. Defaults to 3.
        init_filters: number of output channels for initial convolution layer. Defaults to 32.
        in_channels: number of input channels for the network. Defaults to 1.
        out_channels: number of output channels for the network. Defaults to 2.
        encoder_expansion_ratio: expansion ratio for encoder blocks. Defaults to 2.
        decoder_expansion_ratio: expansion ratio for decoder blocks. Defaults to 2.
        bottleneck_expansion_ratio: expansion ratio for bottleneck blocks. Defaults to 2.
        kernel_size: kernel size for convolutions. Defaults to 7.
        deep_supervision: whether to use deep supervision. Defaults to False.
        use_residual_connection: whether to use residual connections in standard, down and up blocks. Defaults to False.
        blocks_down: number of blocks in each encoder stage. Defaults to [2, 2, 2, 2].
        blocks_bottleneck: number of blocks in bottleneck stage. Defaults to 2.
        blocks_up: number of blocks in each decoder stage. Defaults to [2, 2, 2, 2].
        norm_type: type of normalization layer. Defaults to 'group'.
        global_resp_norm: whether to use Global Response Normalization. Defaults to False. Refer: https://arxiv.org/abs/2301.00808
    """
    
    """
    MedNeXt S Config
        - encoder_expansion_ratio: (2, 2, 2, 2)
        - decoder_expansion_ratio: (2, 2, 2, 2)
        - bottleneck_expansion_ratio: 2
        - kernel_size: 3
        - deep_supervision: False
        - use_residual_connection: True
        - blocks_down: (2, 2, 2, 2)
        - blocks_bottleneck: 2
        - blocks_up: (2, 2, 2, 2)
        - norm_type: 'group'
        - global_resp_norm: False
    
    MedNeXt B config
        - encoder_expansion_ratio: (2, 3, 4, 4)
        - decoder_expansion_ratio: (4, 4, 3, 2)
        - bottleneck_expansion_ratio: 4
        - kernel_size: 3
        - deep_supervision: False
        - use_residual_connection: True
        - blocks_down: (2, 2, 2, 2)
        - blocks_bottleneck: 2
        - blocks_up: (2, 2, 2, 2)
        - norm_type: 'group'
    
    MedNeXt M Config
        - encoder_expansion_ratio: (2, 3, 4, 4)
        - decoder_expansion_ratio: (4, 4, 3, 2)
        - bottleneck_expansion_ratio: 4
        - kernel_size: 3
        - deep_supervision: False
        - use_residual_connection: True
        - blocks_down: (3, 4, 4, 4)
        - blocks_bottleneck: 4
        - blocks_up: (4, 4, 4, 3)
        - norm_type: 'group'
        - global_resp_norm: False
    
    MedNeXt L config
        - encoder_expansion_ratio: (3, 4, 8, 8)
        - decoder_expansion_ratio: (8, 8, 4, 3)
        - bottleneck_expansion_ratio: 8
        - kernel_size: 3
        - deep_supervision: False
        - use_residual_connection: True
        - blocks_down: (3, 4, 8, 8)
        - blocks_bottleneck: 8
        - blocks_up: (8, 8, 4, 3)
        - norm_type: 'group'
        - global_resp_norm: False
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 32,
        in_channels: int = 1,
        out_channels: int = 2,
        encoder_expansion_ratio: Sequence[int] = (2, 2, 2, 2),
        decoder_expansion_ratio: Sequence[int] = (2, 2, 2, 2),
        bottleneck_expansion_ratio: int = 2,
        kernel_size: int = 5,
        use_residual_connection: bool = True,
        blocks_down: Sequence[int] = (2, 2, 2, 2),
        blocks_bottleneck: int = 2,
        blocks_up: Sequence[int] = (2, 2, 2, 2),
        norm_type: str = "group",
        global_resp_norm: bool = False,
    ):
        super().__init__()

        assert spatial_dims in [2, 3], "`spatial_dims` can only be 2 or 3."
        spatial_dims_str = f"{spatial_dims}d"
        enc_kernel_size = dec_kernel_size = kernel_size

        if isinstance(encoder_expansion_ratio, int):
            encoder_expansion_ratio = [encoder_expansion_ratio] * len(blocks_down)

        if isinstance(decoder_expansion_ratio, int):
            decoder_expansion_ratio = [decoder_expansion_ratio] * len(blocks_up)

        conv = nn.Conv2d if spatial_dims_str == "2d" else nn.Conv3d

        self.stem = conv(in_channels, init_filters, kernel_size=1)

        enc_stages = []
        down_blocks = []

        for i, num_blocks in enumerate(blocks_down):
            enc_stages.append(
                nn.Sequential(
                    *[
                        MedNeXtBlock(
                            in_channels=init_filters * (2**i),
                            out_channels=init_filters * (2**i),
                            expansion_ratio=encoder_expansion_ratio[i],
                            kernel_size=enc_kernel_size,
                            use_residual_connection=use_residual_connection,
                            norm_type=norm_type,
                            dim=spatial_dims_str,
                            global_resp_norm=global_resp_norm,
                        )
                        for _ in range(num_blocks)
                    ]
                )
            )

            down_blocks.append(
                MedNeXtDownBlock(
                    in_channels=init_filters * (2**i),
                    out_channels=init_filters * (2 ** (i + 1)),
                    expansion_ratio=encoder_expansion_ratio[i],
                    kernel_size=enc_kernel_size,
                    use_residual_connection=use_residual_connection,
                    norm_type=norm_type,
                    dim=spatial_dims_str,
                )
            )

        self.enc_stages = nn.ModuleList(enc_stages)
        self.down_blocks = nn.ModuleList(down_blocks)

        self.bottleneck = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=init_filters * (2 ** len(blocks_down)),
                    out_channels=init_filters * (2 ** len(blocks_down)),
                    expansion_ratio=bottleneck_expansion_ratio,
                    kernel_size=dec_kernel_size,
                    use_residual_connection=use_residual_connection,
                    norm_type=norm_type,
                    dim=spatial_dims_str,
                    global_resp_norm=global_resp_norm,
                )
                for _ in range(blocks_bottleneck)
            ]
        )

        up_blocks = []
        dec_stages = []
        for i, num_blocks in enumerate(blocks_up):
            up_blocks.append(
                MedNeXtUpBlock(
                    in_channels=init_filters * (2 ** (len(blocks_up) - i)),
                    out_channels=init_filters * (2 ** (len(blocks_up) - i - 1)),
                    expansion_ratio=decoder_expansion_ratio[i],
                    kernel_size=dec_kernel_size,
                    use_residual_connection=use_residual_connection,
                    norm_type=norm_type,
                    dim=spatial_dims_str,
                    global_resp_norm=global_resp_norm,
                )
            )

            dec_stages.append(
                nn.Sequential(
                    *[
                        MedNeXtBlock(
                            in_channels=init_filters * (2 ** (len(blocks_up) - i - 1)),
                            out_channels=init_filters * (2 ** (len(blocks_up) - i - 1)),
                            expansion_ratio=decoder_expansion_ratio[i],
                            kernel_size=dec_kernel_size,
                            use_residual_connection=use_residual_connection,
                            norm_type=norm_type,
                            dim=spatial_dims_str,
                            global_resp_norm=global_resp_norm,
                        )
                        for _ in range(num_blocks)
                    ]
                )
            )

        self.up_blocks = nn.ModuleList(up_blocks)
        self.dec_stages = nn.ModuleList(dec_stages)

        self.out_0 = MedNeXtOutBlock(in_channels=init_filters, n_classes=out_channels, dim=spatial_dims_str)

    def forward(self, x: torch.Tensor) -> torch.Tensor | Sequence[torch.Tensor]:
        # Apply stem convolution
        x = self.stem(x)

        # Encoder forward pass
        enc_outputs = []
        for enc_stage, down_block in zip(self.enc_stages, self.down_blocks):
            x = enc_stage(x)
            enc_outputs.append(x)
            x = down_block(x)

        # Bottleneck forward pass
        x = self.bottleneck(x)

        # Decoder forward pass with skip connections
        for i, (up_block, dec_stage) in enumerate(zip(self.up_blocks, self.dec_stages)):
            x = up_block(x)
            x = x + enc_outputs[-(i + 1)]
            x = dec_stage(x)

        # Final output block
        x = self.out_0(x)
        return x