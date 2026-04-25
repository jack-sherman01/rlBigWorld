"""
Experience Replay Buffer
========================
Simple ring-buffer implementations for DQN (discrete) and SAC (continuous).
"""

import numpy as np


class ContinuousReplayBuffer:
    """Replay buffer for continuous-action agents (SAC, TD3)."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity   = capacity
        self.obs        = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.actions    = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards    = np.zeros(capacity,               dtype=np.float32)
        self.next_obs   = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.dones      = np.zeros(capacity,               dtype=np.float32)
        self.ptr  = 0
        self.size = 0

    def push(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr]    = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

    def __len__(self):
        return self.size


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions  = np.zeros(capacity, dtype=np.int32)
        self.rewards  = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones    = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def push(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr]    = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

    def __len__(self):
        return self.size
