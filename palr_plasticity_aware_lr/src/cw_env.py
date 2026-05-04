"""
ContinualWorld-10 Environment Wrapper
=======================================
Wraps 10 MetaWorld MT10 tasks into a continual RL benchmark:

  - Tasks run sequentially: episodes_per_task episodes each, no oracle.
  - Task switches are silent — the agent is NOT told when the task changes.
  - World state resets between episodes (MetaWorld is episodic by design).
  - Observations: 39-dimensional state vector.
  - Actions:      4-dimensional continuous Box([-1,1]^4).
  - Reward:       dense reward from MetaWorld (reach distance + success bonus).

CW10 task order (10 diverse manipulation tasks):
  1. reach-v3          — move end-effector to goal
  2. push-v3           — push puck to goal position
  3. pick-place-v3     — grasp and place object
  4. door-open-v3      — pull door open
  5. drawer-open-v3    — pull drawer open
  6. drawer-close-v3   — push drawer closed
  7. button-press-topdown-v3 — press button from above
  8. peg-insert-side-v3    — insert peg into side hole
  9. window-open-v3    — slide window open
 10. window-close-v3   — slide window closed

This ordering goes from simple (reach) to progressively harder (peg insert),
creating a natural curriculum that stresses forward transfer and plasticity.
"""

import os
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import metaworld
from metaworld import MT50


# ── Canonical CW10 task order ────────────────────────────────────────────────

CW10_TASKS = [
    "reach-v3",
    "push-v3",
    "pick-place-v3",
    "door-open-v3",
    "drawer-open-v3",
    "drawer-close-v3",
    "button-press-topdown-v3",
    "peg-insert-side-v3",
    "window-open-v3",
    "window-close-v3",
]

# ── Canonical CW20 task order ────────────────────────────────────────────────
# CW10 + 10 additional tasks from MT50 (following Wolczyk et al. 2021 ordering)

CW20_TASKS = CW10_TASKS + [
    "door-close-v3",
    "reach-wall-v3",
    "pick-place-wall-v3",
    "push-wall-v3",
    "button-press-v3",
    "button-press-topdown-wall-v3",
    "button-press-wall-v3",
    "peg-unplug-side-v3",
    "disassemble-v3",
    "hammer-v3",
]


class ContinualWorld:
    """
    CW10 continual RL environment.

    The agent sees a plain 39-dim observation at every step. No task ID
    is provided — the agent must adapt purely from reward signals.

    Args:
        task_names:        Ordered list of task names (default: CW10_TASKS).
        episodes_per_task: Episodes to spend on each task before switching.
        max_steps:         Max steps per episode (MetaWorld default 500).
        seed:              Random seed for task sampling.
    """

    def __init__(
        self,
        task_names: list = None,
        episodes_per_task: int = 100,
        max_steps: int = 500,
        seed: int = 42,
    ):
        self.task_names        = task_names or CW10_TASKS
        self.episodes_per_task = episodes_per_task
        self.max_steps         = max_steps
        self.seed              = seed

        # Build one env per task using MT50 (superset of MT10, supports CW10+CW20)
        mt50 = MT50(seed=seed)
        self._envs: list  = []
        self._tasks: list = []

        for name in self.task_names:
            env_cls = mt50.train_classes[name]
            env     = env_cls()
            # Pick the first task configuration that matches this env name
            task = next(
                t for t in mt50.train_tasks if t.env_name == name
            )
            env.set_task(task)
            self._envs.append(env)
            self._tasks.append(task)

        # Derive obs/action dims from the first env
        obs0, _ = self._envs[0].reset()
        self.obs_dim    = int(obs0.shape[0])
        self.action_dim = int(self._envs[0].action_space.shape[0])

        # State counters
        self.task_idx             = 0
        self.episode_count        = 0
        self._step_in_episode     = 0
        self.task_switch_episodes: list = []

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def current_task(self) -> str:
        return self.task_names[self.task_idx]

    @property
    def n_tasks(self) -> int:
        return len(self.task_names)

    @property
    def total_episodes(self) -> int:
        return self.episodes_per_task * self.n_tasks

    # ── Environment API ───────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """
        Reset for the next episode. Silently advances to the next task when
        `episodes_per_task` episodes have elapsed on the current task.
        """
        # Task boundary check (invisible to agent)
        if (self.episode_count > 0 and
                self.episode_count % self.episodes_per_task == 0):
            new_idx = min(self.task_idx + 1, self.n_tasks - 1)
            if new_idx != self.task_idx:
                self.task_idx = new_idx
                self.task_switch_episodes.append(self.episode_count)

        self.episode_count    += 1
        self._step_in_episode  = 0

        obs, _ = self._envs[self.task_idx].reset()
        return obs.astype(np.float32)

    def step(self, action: np.ndarray):
        """
        Take one step. Returns (obs, reward, done, info).
        done is True after max_steps or on natural termination.
        """
        action = np.clip(action, -1.0, 1.0)
        obs, reward, terminated, truncated, info = \
            self._envs[self.task_idx].step(action)
        self._step_in_episode += 1
        done = bool(terminated or truncated or
                    self._step_in_episode >= self.max_steps)
        info["task_name"] = self.current_task
        info["task_idx"]  = self.task_idx
        return obs.astype(np.float32), float(reward), done, info

    def close(self):
        for env in self._envs:
            env.close()


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing ContinualWorld CW10...")
    env = ContinualWorld(episodes_per_task=3, max_steps=20, seed=42)
    print(f"  obs_dim={env.obs_dim}  action_dim={env.action_dim}")
    print(f"  total_episodes={env.total_episodes}")

    for ep in range(env.total_episodes):
        obs = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            action = np.random.uniform(-1, 1, env.action_dim).astype(np.float32)
            obs, r, done, info = env.step(action)
            ep_r += r
        print(f"  ep {ep:3d} | task={env.current_task:<30s} | reward={ep_r:.2f}")

    print(f"\nTask switches at episodes: {env.task_switch_episodes}")
    env.close()
    print("OK")
