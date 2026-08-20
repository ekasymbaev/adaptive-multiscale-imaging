"""A compact full-frame U-Net for coarse M-A island segmentation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class DoubleConv(nn.Sequential):
    """Two 3 x 3 convolutions with batch normalization and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DownBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.convolutions = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            raise ValueError(
                f"U-Net skip mismatch: decoder {x.shape[-2:]}, encoder {skip.shape[-2:]}"
            )
        return self.convolutions(torch.cat([skip, x], dim=1))


class CompactUNet(nn.Module):
    """Four-stage U-Net returning one foreground logit per input pixel."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        encoder_channels: Sequence[int] = (16, 32, 64, 128),
        bottleneck_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        channels = tuple(int(value) for value in encoder_channels)
        if len(channels) != 4 or any(value <= 0 for value in channels):
            raise ValueError("encoder_channels must contain four positive widths")
        if not 0.0 <= bottleneck_dropout < 1.0:
            raise ValueError("bottleneck_dropout must be in [0, 1)")

        self.input_block = DoubleConv(in_channels, channels[0])
        self.down1 = DownBlock(channels[0], channels[1])
        self.down2 = DownBlock(channels[1], channels[2])
        self.down3 = DownBlock(channels[2], channels[3])
        self.bottleneck_dropout = nn.Dropout2d(bottleneck_dropout)
        self.up1 = UpBlock(channels[3], channels[2], channels[2])
        self.up2 = UpBlock(channels[2], channels[1], channels[1])
        self.up3 = UpBlock(channels[1], channels[0], channels[0])
        self.output = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        level0 = self.input_block(x)
        level1 = self.down1(level0)
        level2 = self.down2(level1)
        level3 = self.bottleneck_dropout(self.down3(level2))
        decoded = self.up1(level3, level2)
        decoded = self.up2(decoded, level1)
        decoded = self.up3(decoded, level0)
        return self.output(decoded)
