"""
PALR-SAC Agent for ViT Backbone
=================================
Plasticity-Aware Learning Rates adapted for Vision Transformer (ViT-Small).

Key adaptations from the MLP version (CW10):
  1. Dead-neuron metric: GELU instead of ReLU
     - GELU has no hard zero threshold; we use mean |activation| < threshold
       across batch × tokens as the dead-neuron criterion
  2. Perturbation targets: FFN fc1/fc2 layers + attention QKV/proj
  3. LR scaling: per transformer layer (L0–L5), computed from
     dead-GELU fraction δ(l) and effective rank ρ(l)
  4. Perturb only dead neurons in FFN fc1 weight rows (outgoing weights)
     mirroring the MLP version's targeted perturbation

The PALR formula is identical:
    lr(l) = lr_base × s(l)
    s(l)  = σ (β_rank × [1 - ρ̂(l)] + β_dead × δ(l))
where σ is sigmoid and ρ̂ is normalised effective rank.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from sac_agent import SACAgent, SACL2RegAgent, SACShrinkPerturbAgent
from vit_policy import ViTEncoder, MLP


# ---------------------------------------------------------------------------
# PALR-ViT Agent
# ---------------------------------------------------------------------------
class PALRViTAgent(SACAgent):
    """
    SAC + PALR for ViT backbone.

    Extra hyper-parameters (on top of SACAgent):
        beta_dead:       weight for dead-neuron term in LR scaling
        beta_rank:       weight for effective-rank term in LR scaling
        sigma:           std of Gaussian perturbation noise
        dead_threshold:  GELU dead-neuron criterion threshold
        palr_freq:       measure plasticity + rescale LR every N steps
        perturb_freq:    apply targeted perturbation every N steps
        lr_floor:        minimum LR scale factor (prevents collapse to 0)
        lr_ceil:         maximum LR scale factor
    """

    NAME = "PALR-SAC"

    def __init__(
        self,
        *args,
        beta_dead:      float = 0.0,
        beta_rank:      float = 1.5,
        sigma:          float = 0.05,
        dead_threshold: float = 0.01,
        palr_freq:      int   = 500,
        perturb_freq:   int   = 500,
        lr_floor:       float = 0.05,
        lr_ceil:        float = 5.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.beta_dead      = beta_dead
        self.beta_rank      = beta_rank
        self.sigma          = sigma
        self.dead_threshold = dead_threshold
        self.palr_freq      = palr_freq
        self.perturb_freq   = perturb_freq
        self.lr_floor       = lr_floor
        self.lr_ceil        = lr_ceil
        self._base_lr       = self._lr

        # Per-layer LR scale factors (one per ViT block)
        n_layers = len(self.actor.encoder.blocks)
        self._lr_scales: List[float] = [1.0] * n_layers

        # Plasticity history (for logging)
        self.palr_history: List[Dict] = []

    # ------------------------------------------------------------------
    def _update_step(self, batch: dict) -> Dict[str, float]:
        losses = super()._update_step(batch)

        # PALR: measure + rescale every palr_freq steps
        if self.total_steps % self.palr_freq == 0 and self.total_steps > 0:
            self._update_lr_scales()

        # Targeted perturbation every perturb_freq steps
        if self.total_steps % self.perturb_freq == 0 and self.total_steps > 0:
            self._apply_targeted_perturbation()

        losses["lr_scales"] = float(np.mean(self._lr_scales))
        return losses

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_lr_scales(self):
        """
        Recompute per-layer LR scale from plasticity metrics and update
        the per-parameter-group learning rates.
        """
        encoder = self.actor.encoder
        metrics = encoder.compute_plasticity_metrics(self.dead_threshold)

        n_layers = len(encoder.blocks)

        # Collect dead fractions and effective ranks
        dead  = np.array([metrics.get(f"dead_L{i}", 0.0)  for i in range(n_layers)])
        erank = np.array([metrics.get(f"erank_L{i}", 1.0) for i in range(n_layers)])

        # Normalise effective rank to [0,1]: erank_max ≈ hidden_dim (384 in ViT-S)
        erank_max  = float(encoder.EMBED_DIM * encoder.MLP_RATIO)  # 384×4=1536
        erank_norm = np.clip(erank / erank_max, 0.0, 1.0)

        # PALR scale: sigmoid(β_rank × (1 - ρ̂) + β_dead × δ)
        logit  = self.beta_rank * (1.0 - erank_norm) + self.beta_dead * dead
        scales = 1.0 / (1.0 + np.exp(-logit))  # sigmoid
        scales = np.clip(scales, self.lr_floor, self.lr_ceil)
        self._lr_scales = scales.tolist()

        # Apply to actor encoder parameter groups
        # We group parameters by layer index
        self._apply_per_layer_lr(encoder, scales, self.actor_opt)

        # Log
        entry = {
            "step":   self.total_steps,
            "dead":   dead.tolist(),
            "erank":  erank.tolist(),
            "scales": scales.tolist(),
        }
        self.palr_history.append(entry)

    def _apply_per_layer_lr(
        self,
        encoder: ViTEncoder,
        scales:  np.ndarray,
        opt:     torch.optim.Optimizer,
    ):
        """
        Rebuild actor optimiser parameter groups with per-layer LR.
        Head (actor MLP) uses the mean scale of all layers.
        """
        base_lr = self._base_lr

        # Build layer → params mapping
        layer_params = []
        for i, blk in enumerate(encoder.blocks):
            layer_params.append({
                "params": list(blk.parameters()),
                "lr":     base_lr * float(scales[i]),
                "name":   f"enc_L{i}",
            })

        # Patch embed + positional embedding
        layer_params.append({
            "params": list(encoder.patch_embed.parameters()) +
                      [encoder.cls_token, encoder.pos_embed, *encoder.norm.parameters()],
            "lr":     base_lr,
            "name":   "enc_stem",
        })

        # Actor head (MLP after encoder) — use mean scale
        layer_params.append({
            "params": list(self.actor.head.parameters()) +
                      list(self.actor.mean_layer.parameters()) +
                      list(self.actor.log_std_layer.parameters()),
            "lr":     base_lr * float(scales.mean()),
            "name":   "actor_head",
        })

        # Rebuild optimiser in-place (preserve momentum buffers by reuse)
        # Simplest: update lr in existing groups if structure unchanged,
        # otherwise reinitialise.
        try:
            # Check if group count matches
            if len(opt.param_groups) == len(layer_params):
                for pg, lp in zip(opt.param_groups, layer_params):
                    pg["lr"] = lp["lr"]
            else:
                raise ValueError("group count mismatch")
        except Exception:
            # Reinitialise optimiser (loses momentum — acceptable for PALR)
            self.actor_opt = torch.optim.Adam(layer_params, lr=base_lr)

    # ------------------------------------------------------------------
    def _checkpoint_dict(self) -> dict:
        d = super()._checkpoint_dict()
        d["_lr_scales"]   = self._lr_scales
        d["palr_history"] = self.palr_history
        return d

    def _load_checkpoint_dict(self, d: dict):
        super()._load_checkpoint_dict(d)
        self._lr_scales   = d.get("_lr_scales",   self._lr_scales)
        self.palr_history = d.get("palr_history", [])

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _apply_targeted_perturbation(self):
        """
        Add small Gaussian noise to weights of dead neurons in each FFN layer.
        Only perturbs outgoing rows of fc1 (the neurons that are dead).
        Mirrors the CW10 PALR perturbation logic.
        """
        encoder = self.actor.encoder

        for i, blk in enumerate(encoder.blocks):
            mlp  = blk.mlp
            if mlp._last_activations is None:
                continue

            acts     = mlp._last_activations   # (B, N+1, hidden_dim)
            mean_act = acts.abs().mean(dim=(0, 1))  # (hidden_dim,)
            dead_mask = (mean_act < self.dead_threshold)  # bool (hidden_dim,)

            if dead_mask.sum() == 0:
                continue

            # Perturb fc1 weight rows corresponding to dead neurons
            noise = torch.randn_like(mlp.fc1.weight) * self.sigma
            mlp.fc1.weight[dead_mask] += noise[dead_mask]

            # Also perturb fc2 weight columns (inputs from dead neurons)
            noise2 = torch.randn_like(mlp.fc2.weight) * self.sigma
            mlp.fc2.weight[:, dead_mask] += noise2[:, dead_mask]


# ---------------------------------------------------------------------------
# PALR ablation: No Perturbation (LR scaling only)
# ---------------------------------------------------------------------------
class PALRNoPerturb(PALRViTAgent):
    """PALR with LR scaling but no targeted perturbation."""
    NAME = "PALR-NoPerturb"

    @torch.no_grad()
    def _apply_targeted_perturbation(self):
        pass  # disabled


# ---------------------------------------------------------------------------
# PALR ablation: No LR Scaling (perturbation only)
# ---------------------------------------------------------------------------
class PALRNoScale(PALRViTAgent):
    """PALR with targeted perturbation but no LR scaling."""
    NAME = "PALR-NoScale"

    def _update_lr_scales(self):
        pass  # no-op: LR stays at base_lr


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
AGENT_REGISTRY = {
    0: SACAgent,
    1: SACL2RegAgent,
    2: SACShrinkPerturbAgent,
    3: PALRViTAgent,
    4: PALRNoPerturb,
    5: PALRNoScale,
}

AGENT_NAMES = {
    0: "SAC-FixedLR",
    1: "SAC-L2Reg",
    2: "SAC-ShrinkPerturb",
    3: "PALR-SAC",
    4: "PALR-NoPerturb",
    5: "PALR-NoScale",
}


