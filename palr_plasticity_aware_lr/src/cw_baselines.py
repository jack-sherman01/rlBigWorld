"""
SAC Baseline Agents for CW10  (PyTorch)
=========================================
Three standard continual-RL baselines, each built on top of SACAgent:

1. SACShinkAndPerturbAgent  — shrink-and-perturb every K steps (Ash & Adams 2020)
2. SACPeriodicResetAgent    — full network reset every K episodes
3. SACL2RegAgent            — L2 weight regularisation on critic loss
"""

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from sac_base import SACAgent, ActorNet, CriticNet, DEVICE


class SACShinkAndPerturbAgent(SACAgent):
    """
    Shrink-and-Perturb on both actor and critics.
    Every `perturb_freq` steps: w <- alpha * w + noise(0, sigma).
    """

    def __init__(self, *args, perturb_freq: int = 5_000,
                 shrink_alpha: float = 0.9, sigma: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.perturb_freq  = perturb_freq
        self.shrink_alpha  = shrink_alpha
        self.sigma         = sigma
        self.name          = "SAC-ShrinkAndPerturb"

    def train_step(self):
        loss = super().train_step()
        if self.step_count > 0 and self.step_count % self.perturb_freq == 0:
            self._shrink_and_perturb()
        return loss

    def _shrink_and_perturb(self):
        for net in (self.actor, self.critic1, self.critic2):
            with torch.no_grad():
                for p in net.parameters():
                    noise = torch.randn_like(p) * self.sigma
                    p.mul_(self.shrink_alpha).add_(noise)
        # Sync target critics after perturbation
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())


class SACPeriodicResetAgent(SACAgent):
    """
    Periodic network reset every `reset_freq` episodes.
    Replay buffer is preserved — only weights are reinitialised.
    """

    def __init__(self, *args, reset_freq: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_freq = reset_freq
        self.name       = "SAC-PeriodicReset"

    def on_episode_end(self, episode_reward: float):
        super().on_episode_end(episode_reward)
        if self.episode_count % self.reset_freq == 0:
            self._reset_networks()

    def _reset_networks(self):
        self.actor    = ActorNet(self.obs_dim, self.action_dim,
                                 self.hidden_sizes).to(DEVICE)
        self.critic1  = CriticNet(self.obs_dim, self.action_dim,
                                  self.hidden_sizes).to(DEVICE)
        self.critic2  = CriticNet(self.obs_dim, self.action_dim,
                                  self.hidden_sizes).to(DEVICE)
        self.target_critic1 = CriticNet(self.obs_dim, self.action_dim,
                                        self.hidden_sizes).to(DEVICE)
        self.target_critic2 = CriticNet(self.obs_dim, self.action_dim,
                                        self.hidden_sizes).to(DEVICE)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.target_critic1.eval()
        self.target_critic2.eval()
        # Recreate optimisers with fresh state
        self.actor_optim   = optim.Adam(self.actor.parameters(),   lr=self.lr)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=self.lr)


class SACL2RegAgent(SACAgent):
    """
    L2 regularisation on critic weights.
    Adds l2_coeff * ||W||² to the critic loss to prevent weight-norm explosion.
    """

    def __init__(self, *args, l2_coeff: float = 1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.l2_coeff = l2_coeff
        self.name     = "SAC-L2Reg"

    def _update_critic(self, obs, actions, rewards, next_obs, dones) -> float:
        obs_t      = torch.FloatTensor(obs).to(DEVICE)
        actions_t  = torch.FloatTensor(actions).to(DEVICE)
        rewards_t  = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        next_obs_t = torch.FloatTensor(next_obs).to(DEVICE)
        dones_t    = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor.sample(next_obs_t)
            q1_next = self.target_critic1(next_obs_t, next_action)
            q2_next = self.target_critic2(next_obs_t, next_action)
            q_next  = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            target_q = rewards_t + self.gamma * (1.0 - dones_t) * q_next

        q1 = self.critic1(obs_t, actions_t)
        q2 = self.critic2(obs_t, actions_t)
        td1 = F.mse_loss(q1, target_q)
        td2 = F.mse_loss(q2, target_q)

        l2_c1 = sum(p.pow(2).sum() for p in self.critic1.parameters() if p.dim() >= 2)
        l2_c2 = sum(p.pow(2).sum() for p in self.critic2.parameters() if p.dim() >= 2)

        self.critic1_optim.zero_grad()
        (td1 + self.l2_coeff * l2_c1).backward()
        self._apply_critic1_gradients()

        self.critic2_optim.zero_grad()
        (td2 + self.l2_coeff * l2_c2).backward()
        self._apply_critic2_gradients()

        return float((td1 + td2).item() / 2)
