"""
ViT-Small Policy Network for SAC
=================================
Vision Transformer (ViT-Small/16, from scratch) as the shared encoder
for SAC actor and critic heads.

Architecture:
  Input:  RGB image (3 × 128 × 128)
  ViT:    patch_size=16 → 8×8=64 patches, embed_dim=384, 6 layers, 6 heads
          (ViT-Small variant — ~21M params total)
  Output: CLS token (384-dim) → actor / critic heads

Plasticity monitoring hooks:
  - Dead GELU neurons in each FFN block (fraction < threshold)
  - Effective rank of each FFN weight matrix (nuclear norm²/Frobenius²)
  These are the same metrics as CW10 PALR, adapted for transformer layers.

Usage:
    encoder  = ViTEncoder(obs_shape=(3, 128, 128))
    actor    = Actor(encoder, action_dim=8)
    critic   = DoubleCritic(encoder_clone, action_dim=8)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional


# ---------------------------------------------------------------------------
# ViT building blocks (from scratch — no pretrained weights)
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Split image into patches and project to embedding dimension."""

    def __init__(
        self,
        img_size:   int = 128,
        patch_size: int = 16,
        in_chans:   int = 3,
        embed_dim:  int = 384,
    ):
        super().__init__()
        assert img_size % patch_size == 0, \
            f"img_size {img_size} not divisible by patch_size {patch_size}"
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, num_patches, embed_dim)
        return self.proj(x).flatten(2).transpose(1, 2)


class MLP(nn.Module):
    """FFN block inside each transformer layer. Uses GELU activation."""

    def __init__(
        self,
        in_features:  int,
        hidden_features: int = None,
        out_features: int = None,
        drop:         float = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        out_features    = out_features    or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        # Hook storage for plasticity monitoring (filled by ViTEncoder)
        self._last_activations: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        # Store post-GELU activations for dead-neuron counting
        self._last_activations = x.detach()
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(
        self,
        dim:       int,
        num_heads: int   = 6,
        qkv_bias:  bool  = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale     = (dim // num_heads) ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj      = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer encoder block: Attention + FFN with LayerNorm."""

    def __init__(
        self,
        dim:       int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop:      float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, num_heads=num_heads, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# ViT Encoder (shared between actor and critic)
# ---------------------------------------------------------------------------

class ViTEncoder(nn.Module):
    """
    ViT-Small from scratch for continual RL.

    patch_size=16; img_size derived from obs_shape (must be square, divisible by 16)
    embed_dim=384, depth=6, num_heads=6, mlp_ratio=4
    Default (128×128): 64 patches → ~21M params; fast mode (32×32): 4 patches → ~11M params

    Output: CLS token representation (B, embed_dim)
    """

    PATCH_SIZE = 16
    EMBED_DIM  = 384
    DEPTH      = 6
    NUM_HEADS  = 6
    MLP_RATIO  = 4.0

    def __init__(self, obs_shape: Tuple[int, int, int] = (3, 128, 128)):
        super().__init__()
        in_chans, img_h, img_w = obs_shape
        assert img_h == img_w and img_h % self.PATCH_SIZE == 0, \
            f"Image size {img_h}×{img_w} must be square and divisible by patch_size={self.PATCH_SIZE}"
        img_size = img_h

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=self.PATCH_SIZE,
            in_chans=in_chans,
            embed_dim=self.EMBED_DIM,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.EMBED_DIM))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, self.EMBED_DIM)
        )

        self.blocks = nn.ModuleList([
            Block(self.EMBED_DIM, self.NUM_HEADS, self.MLP_RATIO)
            for _ in range(self.DEPTH)
        ])
        self.norm = nn.LayerNorm(self.EMBED_DIM)

        self._init_weights()

    def _init_weights(self):
        # ViT weight initialisation: trunc_normal for embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def output_dim(self) -> int:
        return self.EMBED_DIM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) float32 in [0, 1]
        Returns:
            cls: (B, embed_dim) — CLS token representation
        """
        B = x.shape[0]
        x = self.patch_embed(x)                              # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)               # (B, 1, D)
        x   = torch.cat([cls, x], dim=1)                     # (B, N+1, D)
        x   = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]                                       # CLS token

    # ------------------------------------------------------------------
    # Plasticity metrics (for PALR monitoring)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_plasticity_metrics(
        self,
        dead_threshold: float = 0.01,
    ) -> Dict[str, float]:
        """
        Compute per-layer dead-GELU-neuron fraction and effective rank of FFN.

        dead_threshold: neurons with mean |activation| < threshold are dead
        Returns dict: {
            'dead_L0': float, ..., 'dead_L5': float,
            'erank_L0': float, ..., 'erank_L5': float,
        }
        """
        metrics = {}
        for i, blk in enumerate(self.blocks):
            mlp = blk.mlp
            if mlp._last_activations is not None:
                acts = mlp._last_activations   # (B, N+1, hidden_dim)
                # Mean absolute activation per neuron (across batch and tokens)
                mean_act = acts.abs().mean(dim=(0, 1))  # (hidden_dim,)
                dead_frac = float((mean_act < dead_threshold).float().mean())
                metrics[f"dead_L{i}"] = dead_frac
            else:
                metrics[f"dead_L{i}"] = float("nan")

            # Effective rank of FFN fc1 weight (hidden_dim × embed_dim)
            W = mlp.fc1.weight.float()   # (hidden_dim, embed_dim)
            try:
                sv  = torch.linalg.svdvals(W)
                sv  = sv[sv > 1e-10]
                p   = sv / sv.sum()
                erank = float(torch.exp(-(p * torch.log(p + 1e-10)).sum()))
            except Exception:
                erank = float("nan")
            metrics[f"erank_L{i}"] = erank

        return metrics

    def get_ffn_modules(self) -> List[MLP]:
        """Return list of FFN MLP modules (one per transformer layer)."""
        return [blk.mlp for blk in self.blocks]

    def get_all_linear_weights(self) -> Dict[str, torch.Tensor]:
        """Return all linear weight tensors keyed by name (for PALR perturbation)."""
        weights = {}
        for i, blk in enumerate(self.blocks):
            weights[f"L{i}_attn_qkv"]  = blk.attn.qkv.weight
            weights[f"L{i}_attn_proj"] = blk.attn.proj.weight
            weights[f"L{i}_ffn_fc1"]   = blk.mlp.fc1.weight
            weights[f"L{i}_ffn_fc2"]   = blk.mlp.fc2.weight
        return weights


# ---------------------------------------------------------------------------
# Actor and Critic heads (SAC)
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """
    SAC actor: ViT encoder → MLP head → (mean, log_std) for Gaussian policy.
    Action space: tanh-squashed Gaussian in [-1, 1]^action_dim.
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX =  2.0

    def __init__(
        self,
        encoder:    ViTEncoder,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder = encoder
        d = encoder.output_dim
        self.head = nn.Sequential(
            nn.Linear(d, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_layer    = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            action:  (B, action_dim) sampled + tanh-squashed
            log_prob:(B,) log probability of action
        """
        z    = self.encoder(obs)
        h    = self.head(z)
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std  = log_std.exp()

        dist   = torch.distributions.Normal(mean, std)
        x_t    = dist.rsample()                          # reparameterise
        action = torch.tanh(x_t)

        # log prob with tanh correction
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return action, log_prob

    @torch.no_grad()
    def select_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Inference-time action selection."""
        z    = self.encoder(obs)
        h    = self.head(z)
        mean = self.mean_layer(h)
        if deterministic:
            return torch.tanh(mean)
        log_std = self.log_std_layer(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std     = log_std.exp()
        x_t     = mean + std * torch.randn_like(std)
        return torch.tanh(x_t)


class DoubleCritic(nn.Module):
    """
    SAC double Q-critic: two independent (obs, action) → Q value networks.
    Each uses its own ViTEncoder copy (no shared weights with actor).
    """

    def __init__(
        self,
        encoder1:   ViTEncoder,
        encoder2:   ViTEncoder,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        d = encoder1.output_dim

        def _make_q_head():
            return nn.Sequential(
                nn.Linear(d + action_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.q1 = _make_q_head()
        self.q2 = _make_q_head()

    def forward(
        self,
        obs:    torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z1 = self.encoder1(obs)
        z2 = self.encoder2(obs)
        sa1 = torch.cat([z1, action], dim=-1)
        sa2 = torch.cat([z2, action], dim=-1)
        return self.q1(sa1).squeeze(-1), self.q2(sa2).squeeze(-1)

    def q1_value(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        z   = self.encoder1(obs)
        sa  = torch.cat([z, action], dim=-1)
        return self.q1(sa).squeeze(-1)


# ---------------------------------------------------------------------------
# Factory: create all networks for one agent
# ---------------------------------------------------------------------------

def make_networks(
    obs_shape:  Tuple[int, int, int] = (3, 128, 128),
    action_dim: int                  = 8,
    hidden_dim: int                  = 256,
    device:     str                  = "cpu",
) -> dict:
    """
    Create actor + double critic + target critic with separate ViT encoders.

    Returns dict with keys:
        actor, critic, critic_target
    """
    actor_enc    = ViTEncoder(obs_shape).to(device)
    critic_enc1  = ViTEncoder(obs_shape).to(device)
    critic_enc2  = ViTEncoder(obs_shape).to(device)
    target_enc1  = ViTEncoder(obs_shape).to(device)
    target_enc2  = ViTEncoder(obs_shape).to(device)

    actor   = Actor(actor_enc, action_dim, hidden_dim).to(device)
    critic  = DoubleCritic(critic_enc1, critic_enc2, action_dim, hidden_dim).to(device)
    critic_target = DoubleCritic(target_enc1, target_enc2, action_dim, hidden_dim).to(device)

    # Initialise target = critic
    critic_target.load_state_dict(critic.state_dict())
    for p in critic_target.parameters():
        p.requires_grad_(False)

    total_params = sum(p.numel() for p in actor.parameters()) + \
                   sum(p.numel() for p in critic.parameters())
    print(f"  [Networks] actor params: {sum(p.numel() for p in actor.parameters())/1e6:.1f}M")
    print(f"  [Networks] critic params: {sum(p.numel() for p in critic.parameters())/1e6:.1f}M")
    print(f"  [Networks] total: {total_params/1e6:.1f}M")

    return dict(actor=actor, critic=critic, critic_target=critic_target)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    nets = make_networks(obs_shape=(3, 128, 128), action_dim=8, device=device)
    actor          = nets["actor"]
    critic         = nets["critic"]
    critic_target  = nets["critic_target"]

    B   = 4
    obs = torch.rand(B, 3, 128, 128, device=device)
    act = torch.rand(B, 8, device=device) * 2 - 1

    # Forward pass
    action, log_prob = actor(obs)
    q1, q2           = critic(obs, act)
    print(f"  action:   {action.shape}   log_prob: {log_prob.shape}")
    print(f"  q1:       {q1.shape}       q2:       {q2.shape}")

    # Plasticity metrics
    metrics = actor.encoder.compute_plasticity_metrics()
    for k, v in sorted(metrics.items()):
        print(f"  plasticity/{k}: {v:.4f}")

    print("vit_policy smoke test PASSED")
