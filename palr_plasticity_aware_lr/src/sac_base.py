"""
Soft Actor-Critic Agent  (PyTorch)
====================================
SAC with:
  - 4-layer MLP actor and twin critics (configurable depth via hidden_sizes)
  - Automatic entropy tuning (alpha learned online)
  - Reparameterisation trick + tanh squashing for bounded actions
  - Soft target updates (Polyak averaging, tau=0.005)

Used as the base agent for all CW10 continual-world experiments.
Per-layer LR scaling hooks (for PALR-SAC) are exposed via _update_critic()
and _apply_critic_gradients(), which subclasses can override.

Reference: Haarnoja et al., "Soft Actor-Critic", ICML 2018.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from replay_buffer import ContinuousReplayBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Network building blocks ──────────────────────────────────────────────────

def _build_mlp(in_dim: int, hidden_sizes: tuple) -> nn.Sequential:
    """Build a shared MLP trunk: [Linear → ReLU] × n_layers."""
    layers = []
    for h in hidden_sizes:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())
        in_dim = h
    return nn.Sequential(*layers)


class ActorNet(nn.Module):
    """
    Gaussian policy with tanh squashing.
    Architecture: obs → [Linear→ReLU]×4 → (mean_head, log_std_head)
    """
    LOG_STD_MIN = -20
    LOG_STD_MAX =   2

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple):
        super().__init__()
        self.net          = _build_mlp(obs_dim, hidden_sizes)
        trunk_out         = hidden_sizes[-1]
        self.mean_head    = nn.Linear(trunk_out, action_dim)
        self.log_std_head = nn.Linear(trunk_out, action_dim)

    def forward(self, obs: torch.Tensor):
        x       = self.net(obs)
        mean    = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor):
        """
        Returns:
            action:   tanh-squashed sample,  shape (B, action_dim)
            log_prob: sum log-prob with tanh correction, shape (B, 1)
            mean_act: deterministic action (tanh of mean), shape (B, action_dim)
        """
        mean, log_std = self(obs)
        std  = log_std.exp()
        dist = Normal(mean, std)
        x_t  = dist.rsample()                                    # reparameterised
        action = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t)
        # Tanh correction: log|da/dx_t| = log(1 - tanh²(x_t))
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob  = log_prob.sum(dim=-1, keepdim=True)
        mean_act  = torch.tanh(mean)
        return action, log_prob, mean_act


class CriticNet(nn.Module):
    """
    Q-network: (obs, action) → scalar Q-value.
    Architecture: [obs ‖ action] → [Linear→ReLU]×4 → Linear(1)
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple):
        super().__init__()
        self.net    = _build_mlp(obs_dim + action_dim, hidden_sizes)
        self.q_head = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.q_head(self.net(x))


# ── SAC Agent ────────────────────────────────────────────────────────────────

class SACAgent:
    """
    Standard SAC with automatic entropy tuning.

    Args:
        obs_dim:     Observation dimensionality (39 for MetaWorld).
        action_dim:  Action dimensionality (4 for MetaWorld).
        lr:          Shared learning rate for actor, critic, alpha.
        gamma:       Discount factor.
        tau:         Polyak averaging coefficient for target critics.
        buffer_size: Replay buffer capacity.
        batch_size:  Training batch size (use 512 with 40 GB GPU).
        hidden_sizes: Hidden layer widths — use (256,256,256,256) for 4 layers.
        seed:        Random seed.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 500_000,
        batch_size: int = 512,
        hidden_sizes: tuple = (256, 256, 256, 256),
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.gamma      = gamma
        self.tau        = tau
        self.batch_size = batch_size
        self.hidden_sizes = hidden_sizes
        self.lr         = lr

        # Networks
        self.actor         = ActorNet(obs_dim, action_dim, hidden_sizes).to(DEVICE)
        self.critic1       = CriticNet(obs_dim, action_dim, hidden_sizes).to(DEVICE)
        self.critic2       = CriticNet(obs_dim, action_dim, hidden_sizes).to(DEVICE)
        self.target_critic1 = CriticNet(obs_dim, action_dim, hidden_sizes).to(DEVICE)
        self.target_critic2 = CriticNet(obs_dim, action_dim, hidden_sizes).to(DEVICE)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.target_critic1.eval()
        self.target_critic2.eval()

        # Optimisers
        self.actor_optim  = optim.Adam(self.actor.parameters(),   lr=lr)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=lr)

        # Automatic entropy tuning
        self.target_entropy = -float(action_dim)
        self.log_alpha      = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.alpha_optim    = optim.Adam([self.log_alpha], lr=lr)

        self.buffer     = ContinuousReplayBuffer(buffer_size, obs_dim, action_dim)
        self.step_count = 0
        self.episode_count = 0
        self.name       = "SAC-FixedLR"

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ── Interface ─────────────────────────────────────────────────────────────

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            if deterministic:
                _, _, mean_act = self.actor.sample(obs_t)
                return mean_act.squeeze(0).cpu().numpy()
            action, _, _ = self.actor.sample(obs_t)
            return action.squeeze(0).cpu().numpy()

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
        critic_loss = self._update_critic(obs, actions, rewards, next_obs, dones)
        actor_loss  = self._update_actor(obs)
        self._update_alpha(obs)
        self._soft_update_targets()

        self.step_count += 1
        return float(critic_loss)

    def on_episode_end(self, episode_reward: float):
        self.episode_count += 1

    # ── Internal update steps ─────────────────────────────────────────────────

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
        loss1 = F.mse_loss(q1, target_q)
        loss2 = F.mse_loss(q2, target_q)

        self.critic1_optim.zero_grad()
        loss1.backward()
        self._apply_critic1_gradients()

        self.critic2_optim.zero_grad()
        loss2.backward()
        self._apply_critic2_gradients()

        return float((loss1 + loss2).item() / 2)

    def _apply_critic1_gradients(self):
        """Subclasses override for per-layer LR scaling."""
        self.critic1_optim.step()

    def _apply_critic2_gradients(self):
        """Subclasses override for per-layer LR scaling."""
        self.critic2_optim.step()

    def _update_actor(self, obs) -> float:
        obs_t  = torch.FloatTensor(obs).to(DEVICE)
        action, log_pi, _ = self.actor.sample(obs_t)
        q1 = self.critic1(obs_t, action)
        q2 = self.critic2(obs_t, action)
        q  = torch.min(q1, q2)
        actor_loss = (self.alpha.detach() * log_pi - q).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        return float(actor_loss.item())

    def _update_alpha(self, obs):
        obs_t = torch.FloatTensor(obs).to(DEVICE)
        with torch.no_grad():
            _, log_pi, _ = self.actor.sample(obs_t)
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy)).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

    def _soft_update_targets(self):
        for param, target in zip(self.critic1.parameters(),
                                 self.target_critic1.parameters()):
            target.data.mul_(1.0 - self.tau).add_(self.tau * param.data)
        for param, target in zip(self.critic2.parameters(),
                                 self.target_critic2.parameters()):
            target.data.mul_(1.0 - self.tau).add_(self.tau * param.data)
