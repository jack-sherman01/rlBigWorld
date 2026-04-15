"""
Continual CartPole Environment
================================
Wraps CartPole-v1 with periodic physics parameter switches to simulate
a "big world" scenario where the agent must continually adapt.

Task phases cycle through different gravity and pole-mass configurations.
This tests an agent's ability to maintain plasticity across distribution shifts.
"""

import gymnasium as gym
import numpy as np


TASK_CONFIGS = [
    {"gravity": 9.8,  "masspole": 0.1,  "length": 0.5,  "name": "Default"},
    {"gravity": 14.0, "masspole": 0.2,  "length": 0.7,  "name": "Heavy-Slow"},
    {"gravity": 7.0,  "masspole": 0.05, "length": 0.3,  "name": "Light-Fast"},
    {"gravity": 9.8,  "masspole": 0.15, "length": 0.6,  "name": "Default-Heavy"},
]


class ContinualCartPole:
    """
    CartPole with periodic task switches to simulate non-stationarity.

    Args:
        episodes_per_task: Number of episodes before switching to next task.
        seed: Random seed.
    """

    def __init__(self, episodes_per_task: int = 200, seed: int = 42):
        self.env = gym.make("CartPole-v1")
        self.episodes_per_task = episodes_per_task
        self.seed = seed
        self.episode_count = 0
        self.task_idx = 0
        self.task_switch_episodes = []  # record when switches happened
        self._apply_task(self.task_idx)
        np.random.seed(seed)

    def _apply_task(self, idx: int):
        cfg = TASK_CONFIGS[idx % len(TASK_CONFIGS)]
        self.env.unwrapped.gravity   = cfg["gravity"]
        self.env.unwrapped.masspole  = cfg["masspole"]
        self.env.unwrapped.length    = cfg["length"]
        self.env.unwrapped.total_mass = (
            self.env.unwrapped.masscart + cfg["masspole"]
        )
        self.env.unwrapped.polemass_length = cfg["masspole"] * cfg["length"]
        self.current_task_name = cfg["name"]

    def reset(self):
        # Check if it's time to switch tasks
        if (self.episode_count > 0 and
                self.episode_count % self.episodes_per_task == 0):
            self.task_idx += 1
            self._apply_task(self.task_idx)
            self.task_switch_episodes.append(self.episode_count)

        self.episode_count += 1
        obs, _ = self.env.reset(seed=self.seed + self.episode_count)
        return obs.astype(np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs.astype(np.float32), float(reward), done, info

    @property
    def obs_dim(self):
        return self.env.observation_space.shape[0]

    @property
    def n_actions(self):
        return self.env.action_space.n

    @property
    def current_task(self):
        return TASK_CONFIGS[self.task_idx % len(TASK_CONFIGS)]["name"]

    def close(self):
        self.env.close()
