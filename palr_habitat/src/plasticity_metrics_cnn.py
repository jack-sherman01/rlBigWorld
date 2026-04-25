"""
plasticity_metrics_cnn.py
=========================
Block-wise plasticity metrics for a ResNet-18 visual backbone.

Two metrics per ResNet block (layer1–layer4):

  dead_filter_fraction(act):
      Filter j is DEAD if every spatial position (H×W) in every sample (B)
      in the diagnostic batch is ≤ 0 after ReLU.
      dead_fraction = #{dead filters} / C

  effective_rank_gap(act):
      1. Global Average Pool: [B, C, H, W] → [B, C]
      2. Randomized SVD of the centred [B, C] matrix
      3. erank = exp(H(p))  where p_i = sv_i² / Σ sv_j²

These are the same quantities used for CartPole/CW10 but adapted for
4-D feature maps produced by convolutional layers.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional

import torch
import torch.nn as nn


# ── Dead-filter detection ─────────────────────────────────────────────────────

def dead_filter_fraction(act: np.ndarray) -> float:
    """
    Args:
        act: [B, C, H, W]  post-ReLU activations (numpy float32)
    Returns:
        Fraction of filters that are dead (0.0 – 1.0).
    """
    # max over batch dim and both spatial dims → shape [C]
    # A filter is dead iff its max across ALL B*H*W positions is ≤ 0
    max_per_filter = act.max(axis=0).max(axis=-1).max(axis=-1)   # [C]
    return float((max_per_filter <= 0.0).mean())


# ── Effective rank via GAP + SVD ──────────────────────────────────────────────

def effective_rank_gap(act: np.ndarray, n_components: int = 50) -> float:
    """
    Args:
        act:          [B, C, H, W]  post-ReLU activations
        n_components: rank of randomized SVD (capped at min(B, C))
    Returns:
        Effective rank (scalar ≥ 1.0).
    """
    B, C, H, W = act.shape
    gap = act.mean(axis=(2, 3))            # [B, C]  Global Average Pool
    gap = gap - gap.mean(axis=0)           # centre columns (subtract per-filter mean)

    k = min(n_components, B, C)
    # Full SVD is fast enough for B, C ≤ 512; use it for accuracy
    try:
        from sklearn.utils.extmath import randomized_svd
        _, s, _ = randomized_svd(gap, n_components=k, random_state=0)
    except ImportError:
        # fallback: numpy full SVD
        _, s, _ = np.linalg.svd(gap, full_matrices=False)
        s = s[:k]

    s_sq = s ** 2
    total = s_sq.sum()
    if total < 1e-10:
        return 1.0
    p = s_sq / total
    entropy = -(p * np.log(p + 1e-10)).sum()
    return float(np.exp(entropy))


# ── Hook-based activation collector ──────────────────────────────────────────

class BlockActivationCollector:
    """
    Attaches forward hooks to the 4 ResNet block outputs (after the final
    BN+ReLU of each BasicBlock residual branch) and caches [B,C,H,W] tensors.

    Usage:
        collector = BlockActivationCollector(resnet_encoder)
        with torch.no_grad():
            _ = resnet_encoder(obs_batch)
        acts = collector.get()   # dict: block_idx (0-3) → numpy [B,C,H,W]
        collector.remove_hooks()
    """

    def __init__(self, encoder: nn.Module):
        self._acts: Dict[int, np.ndarray] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

        # ResNet block modules: layer1, layer2, layer3, layer4
        # We hook the *last* BasicBlock of each layer (index -1) to capture
        # the block's full output (after residual add + ReLU).
        layers = [encoder.layer1, encoder.layer2,
                  encoder.layer3, encoder.layer4]

        for k, layer in enumerate(layers):
            last_block = layer[-1]   # last BasicBlock in this layer group
            idx = k

            def make_hook(i):
                def hook(module, inp, out):
                    self._acts[i] = out.detach().cpu().float().numpy()
                return hook

            h = last_block.register_forward_hook(make_hook(idx))
            self._hooks.append(h)

    def get(self) -> Dict[int, np.ndarray]:
        return dict(self._acts)

    def clear(self):
        self._acts.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ── Main interface ────────────────────────────────────────────────────────────

def compute_block_metrics(
    encoder: nn.Module,
    obs_rgb: torch.Tensor,
    obs_depth: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Run a diagnostic batch through the ResNet encoder and compute plasticity
    metrics for all 4 blocks.

    Args:
        encoder:   PALRResNetEncoder (has .layer1 – .layer4 attributes)
        obs_rgb:   [B, 3, H, W]  float32 on the same device as encoder
        obs_depth: [B, 1, H, W]  optional depth channel

    Returns:
        dict with keys:
          "block_{k}_dead"   (k=0..3): dead-filter fraction
          "block_{k}_erank"           : effective rank
          "mean_dead", "mean_erank"   : averages over all 4 blocks
    """
    encoder.eval()
    collector = BlockActivationCollector(encoder)

    with torch.no_grad():
        if obs_depth is not None:
            x = torch.cat([obs_rgb, obs_depth], dim=1)
        else:
            x = obs_rgb
        encoder(x)

    acts = collector.get()
    collector.remove_hooks()
    encoder.train()

    metrics: Dict[str, float] = {}
    dead_vals, erank_vals = [], []

    for k in range(4):
        if k not in acts:
            continue
        act = acts[k]
        d  = dead_filter_fraction(act)
        er = effective_rank_gap(act)
        metrics[f"block_{k}_dead"]  = d
        metrics[f"block_{k}_erank"] = er
        dead_vals.append(d)
        erank_vals.append(er)

    metrics["mean_dead"]  = float(np.mean(dead_vals))  if dead_vals  else 0.0
    metrics["mean_erank"] = float(np.mean(erank_vals)) if erank_vals else 1.0
    return metrics
