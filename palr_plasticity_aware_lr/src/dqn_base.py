"""
Base DQN Agent  (PyTorch)
==========================
Shared Q-network architecture and training logic used by all agent variants.
Supports per-layer learning rate scaling for PALR experiments.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from replay_buffer import ReplayBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNet(nn.Module):
    """
    MLP Q-network: Input -> [Linear -> ReLU] * n_hidden -> Linear.

    Architecture (2 hidden layers, default):
        net[0] = Linear(obs_dim, h1)
        net[1] = ReLU
        net[2] = Linear(h1, h2)
        net[3] = ReLU
        net[4] = Linear(h2, n_actions)
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """
    Standard DQN with fixed Adam learning rate.

    Args:
        obs_dim: Observation dimensionality.
        n_actions: Number of discrete actions.
        lr: Adam learning rate.
        gamma: Discount factor.
        buffer_size: Replay buffer capacity.
        batch_size: Training batch size.
        target_update_freq: Steps between target network syncs.
        epsilon_start / epsilon_end / epsilon_decay: Epsilon-greedy schedule.
        hidden_sizes: Tuple of hidden layer widths.
        seed: Random seed.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        target_update_freq: int = 500,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 5_000,
        hidden_sizes=(128, 128),
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.obs_dim            = obs_dim
        self.n_actions          = n_actions
        self.gamma              = gamma
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon_start      = epsilon_start
        self.epsilon_end        = epsilon_end
        self.epsilon_decay      = epsilon_decay
        self.hidden_sizes       = hidden_sizes
        self.lr                 = lr

        self.online_net = QNet(obs_dim, n_actions, hidden_sizes).to(DEVICE)
        self.target_net = QNet(obs_dim, n_actions, hidden_sizes).to(DEVICE)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_size, obs_dim)

        self.step_count    = 0
        self.episode_count = 0
        self.name          = "DQN-FixedLR"

    @property
    def epsilon(self) -> float:
        frac = min(1.0, self.step_count / max(1, self.epsilon_decay))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def act(self, obs: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            q = self.online_net(obs_t)
        return int(q.argmax(dim=1).item())

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
        loss_val = self._update(obs, actions, rewards, next_obs, dones)

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss_val

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
        q_vals = self.online_net(obs_t)
        q_pred = q_vals.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        loss   = F.mse_loss(q_pred, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self._apply_gradients()

        return float(loss.item())

    def _apply_gradients(self):
        """Apply gradients — subclasses override for custom LR scaling."""
        self.optimizer.step()

    def on_episode_end(self, episode_reward: float):
        """Hook called at end of each episode. Override in subclasses."""
        self.episode_count += 1
