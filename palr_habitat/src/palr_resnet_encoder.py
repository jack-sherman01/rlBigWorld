"""
palr_resnet_encoder.py
======================
ResNet-18 visual encoder with PALR's two mechanisms:

  1. Per-block gradient scaling  — multiply gradients of each ResNet block's
     parameters by lr_scales[k] after backward(), before optimizer.step().

  2. Targeted dead-filter perturbation — re-initialise ONLY the dead
     convolutional filters in a block when dead_fraction > threshold.

Architecture:
  Input : [B, in_channels, H, W]  (in_channels = 3 for RGB or 4 for RGB-D)
  layer1 → 64 ch, stride 1  (after initial 7×7 conv + pool)
  layer2 → 128 ch, stride 2
  layer3 → 256 ch, stride 2
  layer4 → 512 ch, stride 2
  GAP    → [B, 512]                ← this is the visual feature vector

PALR monitors ALL 4 blocks (indices 0–3 correspond to layer1–layer4).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm


# ── Encoder ───────────────────────────────────────────────────────────────────

class PALRResNetEncoder(nn.Module):
    """
    ResNet-18 backbone with PALR gradient-scaling and perturbation support.

    Args:
        in_channels:  3 (RGB) or 4 (RGB-D)
        output_size:  feature dimension after GAP (512 for ResNet-18)
    """

    OUTPUT_DIM = 512   # ResNet-18 layer4 channels

    def __init__(self, in_channels: int = 4):
        super().__init__()
        base = tvm.resnet18(pretrained=False)

        # Replace first conv to accept variable input channels
        base.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Strip the final classifier (avgpool + fc) — we use our own GAP
        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.relu    = base.relu
        self.maxpool = base.maxpool
        self.layer1  = base.layer1   # [B, 64,  H/4,  W/4]
        self.layer2  = base.layer2   # [B, 128, H/8,  W/8]
        self.layer3  = base.layer3   # [B, 256, H/16, W/16]
        self.layer4  = base.layer4   # [B, 512, H/32, W/32]
        self.gap     = nn.AdaptiveAvgPool2d(1)

        # Per-block LR scales (start at 1.0 — updated by PALRTrainer)
        self.register_buffer(
            "lr_scales",
            torch.ones(4, dtype=torch.float32)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [B, 512] visual features."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        return x.flatten(1)   # [B, 512]

    # ── Block accessors (for perturbation) ───────────────────────────────────

    def _get_block(self, k: int) -> nn.Module:
        """Return the nn.Sequential block for index k (0=layer1 ... 3=layer4)."""
        return [self.layer1, self.layer2, self.layer3, self.layer4][k]

    # ── PALR: gradient scaling ────────────────────────────────────────────────

    def scale_gradients(self, lr_scales: np.ndarray):
        """
        Multiply gradients of each block's parameters by lr_scales[k].
        Call this after loss.backward() and before optimizer.step().

        Args:
            lr_scales: float32 array of shape [4] — one scale per block.
        """
        blocks = [self.layer1, self.layer2, self.layer3, self.layer4]
        for k, block in enumerate(blocks):
            scale = float(lr_scales[k])
            if scale == 1.0:
                continue
            for p in block.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)

        # Also scale the initial conv (treated as part of block 0 in deficit)
        scale0 = float(lr_scales[0])
        for p in list(self.conv1.parameters()) + list(self.bn1.parameters()):
            if p.grad is not None:
                p.grad.mul_(scale0)

    # ── PALR: targeted dead-filter perturbation ───────────────────────────────

    def perturb_dead_filters(
        self,
        k: int,
        dead_mask: np.ndarray,
    ):
        """
        Re-initialise dead filters in the k-th ResNet block.

        A filter is 'dead' if dead_mask[j] == True.  We perturb the weight
        of every Conv2d in the block whose OUTPUT channel index is in dead_mask.

        Args:
            k:         block index (0–3)
            dead_mask: boolean array [C_out] — True for dead filters
        """
        if not np.any(dead_mask):
            return

        dead_indices = np.where(dead_mask)[0]
        block = self._get_block(k)

        with torch.no_grad():
            for module in block.modules():
                if not isinstance(module, nn.Conv2d):
                    continue
                weight = module.weight   # [C_out, C_in, kH, kW]
                C_out = weight.shape[0]
                # Only perturb if indices are valid for this conv's output dim
                valid = dead_indices[dead_indices < C_out]
                if len(valid) == 0:
                    continue

                fan_in = float(weight[0].numel())
                std    = np.sqrt(2.0 / fan_in)
                noise  = torch.randn(
                    len(valid), *weight.shape[1:],
                    device=weight.device,
                    dtype=weight.dtype,
                ) * std
                weight.data[valid] = noise

                if module.bias is not None:
                    module.bias.data[valid] = 0.0
