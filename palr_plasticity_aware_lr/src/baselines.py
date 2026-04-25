"""
Baseline Agents  (PyTorch)
===========================
Three standard approaches for handling plasticity loss in continual RL:

1. ShrinkAndPerturbAgent: Periodically shrink weights toward zero and add noise.
   (Ash & Adams, 2020 -- "Warm-Starting Neural Network Training")

2. PeriodicResetAgent: Fully reset the online network every K episodes.
   Simple but disruptive; serves as an upper bound on adaptation speed.

3. L2RegAgent: Add L2 regularization to keep weights small, preventing
   the weight explosion associated with plasticity loss.
"""

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from dqn_base import DQNAgent, QNet, DEVICE


class ShrinkAndPerturbAgent(DQNAgent):
    """
    Shrink-and-Perturb baseline (Ash & Adams, 2020).
    Every `perturb_freq` steps: w <- alpha * w + noise(0, sigma)
    """

    def __init__(self, *args, perturb_freq=2000, alpha=0.9, sigma=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.perturb_freq = perturb_freq
        self.alpha        = alpha
        self.sigma        = sigma
        self.name         = "ShrinkAndPerturb"

    def train_step(self):
        loss = super().train_step()
        if self.step_count > 0 and self.step_count % self.perturb_freq == 0:
            self._shrink_and_perturb()
        return loss

    def _shrink_and_perturb(self):
        with torch.no_grad():
            for p in self.online_net.parameters():
                noise = torch.randn_like(p) * self.sigma
                p.mul_(self.alpha).add_(noise)
        # Sync target net after perturbation
        self.target_net.load_state_dict(self.online_net.state_dict())


class PeriodicResetAgent(DQNAgent):
    """
    Periodic network reset baseline.
    Every `reset_freq` episodes: reinitialise the online network weights.
    Preserves the replay buffer so experience is not lost.
    """

    def __init__(self, *args, reset_freq=200, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_freq = reset_freq
        self.name       = "PeriodicReset"

    def on_episode_end(self, episode_reward: float):
        super().on_episode_end(episode_reward)
        if self.episode_count % self.reset_freq == 0:
            self._reset_network()

    def _reset_network(self):
        self.online_net = QNet(self.obs_dim, self.n_actions, self.hidden_sizes).to(DEVICE)
        self.target_net = QNet(self.obs_dim, self.n_actions, self.hidden_sizes).to(DEVICE)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        # Reset optimizer state along with the new parameters
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr)


class L2RegAgent(DQNAgent):
    """
    L2 regularisation baseline.
    Adds an L2 penalty on weights to prevent weight-norm explosion,
    a known precursor to plasticity loss.
    """

    def __init__(self, *args, l2_coeff=1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.l2_coeff = l2_coeff
        self.name     = "L2-Regularisation"

    def _update(self, obs, actions, rewards, next_obs, dones) -> float:
        obs_t      = torch.FloatTensor(obs).to(DEVICE)
        actions_t  = torch.LongTensor(actions).to(DEVICE)
        rewards_t  = torch.FloatTensor(rewards).to(DEVICE)
        next_obs_t = torch.FloatTensor(next_obs).to(DEVICE)
        dones_t    = torch.FloatTensor(dones).to(DEVICE)

        with torch.no_grad():
            next_q  = self.target_net(next_obs_t).max(dim=1).values
            targets = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        self.online_net.train()
        q_vals  = self.online_net(obs_t)
        q_pred  = q_vals.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        td_loss = F.mse_loss(q_pred, targets)

        # L2 penalty on weight matrices only (not biases)
        l2_loss = sum(
            p.pow(2).sum()
            for p in self.online_net.parameters()
            if p.dim() >= 2
        )
        loss = td_loss + self.l2_coeff * l2_loss

        self.optimizer.zero_grad()
        loss.backward()
        self._apply_gradients()

        return float(td_loss.item())  # return TD loss for fair comparison
