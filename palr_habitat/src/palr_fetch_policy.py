"""
palr_fetch_policy.py
====================
Actor-Critic policy for Habitat Fetch Rearrangement.

Architecture (matches the obs dict produced by habitat-lab 0.2.5
`benchmark/rearrange/pick.yaml`):

  Visual stream:
    head_depth [B,1,256,256]
      → PALRResNetEncoder (ResNet-18, in_channels=1)
      → [B, 512]

  Proprioceptive stream:
    joint (7) + is_holding (1) + obj_start_sensor (3)
                              + relative_resting_position (3)        = 14
      → Linear(., 128) → ReLU
      → [B, 128]

  Fusion:
    cat([visual, proprio]) → [B, 640]
      → GRU(640, hidden_size=512)
      → [B, 512]

  Actor head (continuous Gaussian over the task action vector):
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

# Sizes of the proprio sub-vectors expected from the rearrange pick task.
_OBJ_START_DIM           = 3
_RELATIVE_RESTING_DIM    = 3
_HOLDING_DIM             = 1


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

        # Visual backbone — depth-only (1 channel) since habitat rearrange
        # pick task does not expose head_rgb by default.
        self.visual_encoder = PALRResNetEncoder(in_channels=1)

        # Proprioceptive encoder
        # inputs: joint + is_holding + obj_start_sensor + relative_resting_position
        proprio_dim = (
            joint_dim
            + _HOLDING_DIM
            + _OBJ_START_DIM
            + _RELATIVE_RESTING_DIM
        )
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
            obs:        dict with keys "head_depth", "joint", "is_holding",
                        "obj_start_sensor", "relative_resting_position"
            rnn_hidden: [1, B, hidden_size]
            masks:      [T*B, 1] or [B, 1]  — 0 at episode boundaries

        Returns:
            dist:       action distribution (Normal)
            value:      [B, 1]
            rnn_hidden: [1, B, hidden_size]  (updated)
        """
        # --- visual ---
        depth = obs["head_depth"].float()      # [B, H, W, 1]
        depth = depth.permute(0, 3, 1, 2)      # [B, 1, H, W]
        vis_feat = self.visual_encoder(depth)  # [B, 512]

        # --- proprio ---
        joint    = obs["joint"].float()
        holding  = obs["is_holding"].float()
        obj_pos  = obs["obj_start_sensor"].float()
        rest_pos = obs["relative_resting_position"].float()
        proprio  = torch.cat([joint, holding, obj_pos, rest_pos], dim=-1)
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
