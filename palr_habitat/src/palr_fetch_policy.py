"""
palr_fetch_policy.py
====================
Actor-Critic policy for Habitat Fetch Rearrangement.

Architecture
------------
  Visual stream:
    RGB [B,3,128,128] + Depth [B,1,128,128]
      → PALRResNetEncoder (ResNet-18, in_channels=4)
      → [B, 512]

  Proprioceptive stream:
    joint positions [B, J] + is_holding [B,1] + gps_compass [B, 4]
      → Linear(J+5, 128) → ReLU
      → [B, 128]

  Fusion:
    cat([visual, proprio]) → [B, 640]
      → GRU(640, hidden_size=512)
      → [B, 512]

  Actor head (continuous Gaussian, arm = 7-DOF):
    Linear(512, action_dim*2) → [B, action_dim], [B, action_dim]

  Critic head:
    Linear(512, 1) → [B, 1]

Registered as "PALRFetchPolicy" for habitat_baselines.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

try:
    from habitat_baselines.common.baseline_registry import baseline_registry
    from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
    HAS_HABITAT = True
except ImportError:
    HAS_HABITAT = False

from palr_resnet_encoder import PALRResNetEncoder


# ── Gaussian policy distribution ──────────────────────────────────────────────

class DiagGaussian(nn.Module):
    """Output layer that predicts mean + log_std for a diagonal Gaussian."""

    LOG_STD_MIN = -5.0
    LOG_STD_MAX =  2.0

    def __init__(self, in_dim: int, action_dim: int):
        super().__init__()
        self.mean_head    = nn.Linear(in_dim, action_dim)
        self.log_std_head = nn.Linear(in_dim, action_dim)

    def forward(self, x: torch.Tensor) -> Normal:
        mean    = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return Normal(mean, log_std.exp())


# ── Main policy network ───────────────────────────────────────────────────────

class PALRFetchNet(nn.Module):
    """
    The neural network for PALR Fetch policy.
    Used by PALRFetchPolicy (habitat_baselines) and also directly in
    the standalone palr_trainer.py.
    """

    def __init__(
        self,
        action_dim:    int,
        joint_dim:     int = 7,    # arm joints; habitat rearrange default
        hidden_size:   int = 512,
        use_palr:      bool = True,
    ):
        super().__init__()
        self.use_palr  = use_palr
        self.hidden_size = hidden_size

        # Visual backbone — RGB-D input (4 channels)
        self.visual_encoder = PALRResNetEncoder(in_channels=4)

        # Proprioceptive encoder
        # inputs: joint (7) + is_holding (1) + start_gps_compass (2) + goal_gps_compass (2) = 12
        proprio_dim = joint_dim + 1 + 2 + 2
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.ReLU(inplace=True),
        )

        # Fusion + GRU
        fused_dim = PALRResNetEncoder.OUTPUT_DIM + 128   # 512 + 128 = 640
        self.gru  = nn.GRU(fused_dim, hidden_size, batch_first=False)

        # Heads
        self.actor  = DiagGaussian(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)

        self._init_heads()

    def _init_heads(self):
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)
        nn.init.orthogonal_(self.actor.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.actor.mean_head.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        obs:            Dict[str, torch.Tensor],
        rnn_hidden:     torch.Tensor,
        masks:          torch.Tensor,
    ) -> Tuple[Normal, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs:        dict with keys "rgb", "depth", "joint",
                        "is_holding", "target_start_gps_compass",
                        "target_goal_gps_compass"
            rnn_hidden: [1, B, hidden_size]
            masks:      [T*B, 1] or [B, 1]  — 0 at episode boundaries

        Returns:
            dist:       action distribution (Normal)
            value:      [B, 1]
            rnn_hidden: [1, B, hidden_size]  (updated)
        """
        # --- visual ---
        rgb   = obs["rgb"].float()    / 255.0  # [B, H, W, 3]
        depth = obs["depth"].float()           # [B, H, W, 1]
        # HWC → CHW
        rgb   = rgb.permute(0, 3, 1, 2)        # [B, 3, H, W]
        depth = depth.permute(0, 3, 1, 2)      # [B, 1, H, W]
        x_vis = torch.cat([rgb, depth], dim=1) # [B, 4, H, W]
        vis_feat = self.visual_encoder(x_vis)  # [B, 512]

        # --- proprio ---
        joint   = obs["joint"].float()
        holding = obs["is_holding"].float()
        gps_s   = obs["target_start_gps_compass"].float()
        gps_g   = obs["target_goal_gps_compass"].float()
        proprio = torch.cat([joint, holding, gps_s, gps_g], dim=-1)
        prop_feat = self.proprio_encoder(proprio)  # [B, 128]

        # --- fuse → GRU ---
        fused = torch.cat([vis_feat, prop_feat], dim=-1)  # [B, 640]
        # GRU expects [T, B, input_size]; with T=1 step (online)
        fused = fused.unsqueeze(0)               # [1, B, 640]
        # Zero hidden state at episode boundaries
        rnn_hidden = rnn_hidden * masks.unsqueeze(0)
        out, rnn_hidden = self.gru(fused, rnn_hidden)   # out: [1, B, 512]
        out = out.squeeze(0)                    # [B, 512]

        dist  = self.actor(out)
        value = self.critic(out)                # [B, 1]
        return dist, value, rnn_hidden

    @property
    def visual_encoder_ref(self) -> PALRResNetEncoder:
        return self.visual_encoder

    def recurrent_hidden_state_size(self) -> int:
        return self.hidden_size

    def num_recurrent_layers(self) -> int:
        return 1


# ── Habitat baseline_registry integration (optional) ─────────────────────────

if HAS_HABITAT:
    from habitat_baselines.rl.ppo.policy import Policy

    @baseline_registry.register_policy
    class PALRFetchPolicy(Policy):
        """
        Thin wrapper so habitat_baselines can instantiate our policy via config.
        The actual PALR gradient scaling logic lives in PALRDDPPOTrainer.
        """

        def __init__(self, observation_space, action_space, **kwargs):
            super().__init__()
            action_dim = action_space.shape[0]
            # Infer joint_dim from observation space
            joint_dim  = observation_space.spaces.get("joint", None)
            joint_dim  = joint_dim.shape[0] if joint_dim is not None else 7
            use_palr   = kwargs.get("use_palr", True)

            self.net = PALRFetchNet(
                action_dim=action_dim,
                joint_dim=joint_dim,
                use_palr=use_palr,
            )

        def act(self, observations, rnn_hidden_states, prev_actions, masks,
                deterministic=False):
            dist, value, rnn_hidden = self.net(observations, rnn_hidden_states, masks)
            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()
            action_log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            return value, action, action_log_prob, rnn_hidden

        def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
            _, value, _ = self.net(observations, rnn_hidden_states, masks)
            return value

        def evaluate_actions(self, observations, rnn_hidden_states, prev_actions,
                             masks, action):
            dist, value, rnn_hidden = self.net(observations, rnn_hidden_states, masks)
            action_log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            dist_entropy    = dist.entropy().sum(-1)
            return value, action_log_prob, dist_entropy, rnn_hidden

        @classmethod
        def from_config(cls, config, observation_space, action_space):
            return cls(
                observation_space,
                action_space,
                use_palr=getattr(config.PALR, "enabled", True),
            )
