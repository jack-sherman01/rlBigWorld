"""
SAC Agent (baseline) for ManiSkill + ViT
==========================================
Standard Soft Actor-Critic with fixed learning rate (no PALR).
Used as SAC-FixedLR baseline.

All other baseline agents (L2Reg, ShrinkAndPerturb) inherit from this
and override the `update` method.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import copy

from vit_policy import make_networks, ViTEncoder


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Simple ring-buffer replay buffer for image observations."""

    def __init__(
        self,
        obs_shape:  Tuple[int, ...],
        action_dim: int,
        capacity:   int   = 100_000,
        device:     str   = "cpu",
    ):
        self.capacity   = capacity
        self.obs_shape  = obs_shape
        self.action_dim = action_dim
        self.device     = device

        self.obs     = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs= np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones   = np.zeros(capacity, dtype=np.float32)

        self.ptr  = 0
        self.size = 0

    def add(
        self,
        obs:      np.ndarray,
        action:   np.ndarray,
        reward:   float,
        next_obs: np.ndarray,
        done:     float,
    ):
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr]    = done
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        idx = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs      = torch.FloatTensor(self.obs[idx]).to(self.device),
            actions  = torch.FloatTensor(self.actions[idx]).to(self.device),
            rewards  = torch.FloatTensor(self.rewards[idx]).to(self.device),
            next_obs = torch.FloatTensor(self.next_obs[idx]).to(self.device),
            dones    = torch.FloatTensor(self.dones[idx]).to(self.device),
        )

    def __len__(self):
        return self.size


# ---------------------------------------------------------------------------
# SAC-FixedLR (baseline)
# ---------------------------------------------------------------------------
class SACAgent:
    """
    Soft Actor-Critic with fixed learning rate.
    Baseline: no plasticity maintenance.
    """

    NAME = "SAC-FixedLR"

    def __init__(
        self,
        obs_shape:      Tuple[int, int, int] = (3, 128, 128),
        action_dim:     int   = 8,
        hidden_dim:     int   = 256,
        lr:             float = 3e-4,
        gamma:          float = 0.99,
        tau:            float = 0.005,
        alpha:          float = 0.2,
        auto_alpha:     bool  = True,
        buffer_capacity:int   = 50_000,
        batch_size:     int   = 256,
        device:         str   = "cpu",
    ):
        self.obs_shape   = obs_shape
        self.action_dim  = action_dim
        self.gamma       = gamma
        self.tau         = tau
        self.batch_size  = batch_size
        self.device      = device
        self._lr         = lr

        # Networks
        nets = make_networks(obs_shape, action_dim, hidden_dim, device)
        self.actor          = nets["actor"]
        self.critic         = nets["critic"]
        self.critic_target  = nets["critic_target"]

        # Optimisers
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Entropy temperature
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -float(action_dim)
            self.log_alpha      = torch.zeros(1, requires_grad=True, device=device)
            self.alpha          = self.log_alpha.exp().item()
            self.alpha_opt      = torch.optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha

        # Replay buffer
        self.buffer = ReplayBuffer(obs_shape, action_dim, buffer_capacity, device)

        # Step counter
        self.total_steps = 0

    # ------------------------------------------------------------------
    def select_action(
        self,
        obs:           np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        obs_t = torch.FloatTensor(obs[None]).to(self.device)
        with torch.no_grad():
            action = self.actor.select_action(obs_t, deterministic)
        return action.cpu().numpy()[0]

    # ------------------------------------------------------------------
    def update(self) -> Optional[Dict[str, float]]:
        """One gradient step. Returns loss dict or None if buffer too small."""
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        return self._update_step(batch)

    def _update_step(self, batch: dict) -> Dict[str, float]:
        obs, actions, rewards, next_obs, dones = (
            batch["obs"], batch["actions"], batch["rewards"],
            batch["next_obs"], batch["dones"],
        )

        # ── Critic update ────────────────────────────────────────────────────
        with torch.no_grad():
            next_actions, next_log_pi = self.actor(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            target_q = rewards + self.gamma * (1.0 - dones) * q_next

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # ── Actor update ─────────────────────────────────────────────────────
        pi, log_pi = self.actor(obs)
        q1_pi, q2_pi = self.critic(obs, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_pi - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # ── Alpha update ─────────────────────────────────────────────────────
        alpha_loss = torch.tensor(0.0)
        if self.auto_alpha:
            alpha_loss = -(
                self.log_alpha * (log_pi.detach() + self.target_entropy)
            ).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()

        # ── Target update ─────────────────────────────────────────────────────
        self._soft_update_target()
        self.total_steps += 1

        return dict(
            critic_loss = float(critic_loss),
            actor_loss  = float(actor_loss),
            alpha_loss  = float(alpha_loss),
            alpha       = self.alpha,
        )

    def _soft_update_target(self):
        for p, tp in zip(self.critic.parameters(),
                         self.critic_target.parameters()):
            tp.data.copy_((1 - self.tau) * tp.data + self.tau * p.data)

    # ------------------------------------------------------------------
    def get_plasticity_metrics(self) -> Dict[str, float]:
        """Return plasticity metrics from actor encoder (for logging)."""
        return self.actor.encoder.compute_plasticity_metrics()


# ---------------------------------------------------------------------------
# SAC-L2Reg
# ---------------------------------------------------------------------------
class SACL2RegAgent(SACAgent):
    """SAC with L2 regularisation (weight decay) on encoder parameters."""

    NAME = "SAC-L2Reg"

    def __init__(self, *args, l2_coef: float = 1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.l2_coef = l2_coef
        # Override optimisers with weight_decay
        lr = self._lr
        self.actor_opt  = torch.optim.Adam(
            self.actor.parameters(),  lr=lr, weight_decay=l2_coef)
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=lr, weight_decay=l2_coef)


# ---------------------------------------------------------------------------
# SAC-ShrinkAndPerturb
# ---------------------------------------------------------------------------
class SACShrinkPerturbAgent(SACAgent):
    """
    SAC with Shrink-and-Perturb applied to actor encoder every
    `perturb_freq` gradient steps.
    """

    NAME = "SAC-ShrinkPerturb"

    def __init__(
        self,
        *args,
        perturb_freq: int   = 1000,
        shrink_factor: float = 0.9,
        perturb_std:   float = 0.01,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.perturb_freq   = perturb_freq
        self.shrink_factor  = shrink_factor
        self.perturb_std    = perturb_std

    def _update_step(self, batch: dict) -> Dict[str, float]:
        losses = super()._update_step(batch)
        if self.total_steps % self.perturb_freq == 0:
            self._shrink_and_perturb()
        return losses

    def _shrink_and_perturb(self):
        with torch.no_grad():
            for p in self.actor.encoder.parameters():
                p.data.mul_(self.shrink_factor)
                p.data.add_(
                    torch.randn_like(p) * self.perturb_std
                )
