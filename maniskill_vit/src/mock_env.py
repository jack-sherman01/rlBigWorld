"""
Mock Heterogeneous Skill Stream (no ManiSkill / no GPU required)
================================================================
Provides the exact same interface as HeterogeneousSkillStream but uses
simple synthetic physics to generate 128×128 RGB images.

Used for:
  - Pipeline validation (ViT + SAC + PALR training loop)
  - CI / unit tests on machines without GPU/Vulkan

The 4 mock tasks differ in their reward shaping and object dynamics,
creating a genuinely heterogeneous task stream that challenges plasticity.

NOT used for paper results — those use ManiSkill3 on GPU nodes.
"""

import numpy as np
from typing import List, Tuple, Optional

OBS_IMG_SIZE = 128
ACTION_DIM   = 8    # matches real ManiSkill Panda action space
SCENE_RANGE  = 1.0  # scene spans [-0.5, 0.5] in x and y


# ---------------------------------------------------------------------------
# Per-task mock dynamics
# ---------------------------------------------------------------------------
class MockTask:
    """Simple 2D point-mass task with a distinct reward structure."""

    def __init__(self, task_id: str, seed: int):
        self.task_id = task_id
        self.rng     = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> dict:
        """Randomise state, return state dict."""
        self.agent_pos  = self.rng.uniform(-0.2, 0.2, 2)   # end-effector XY
        self.obj_pos    = self.rng.uniform(-0.3, 0.3, 2)   # primary object
        self.target_pos = self.rng.uniform(-0.3, 0.3, 2)   # goal position
        self.obj2_pos   = self.rng.uniform(-0.3, 0.3, 2)   # secondary (StackCube)
        self.obj_angle  = self.rng.uniform(0, 2 * np.pi)   # rotation (TurnFaucet)
        self.target_angle = self.rng.uniform(0, 2 * np.pi)
        self.step_count = 0
        return self._get_state()

    def step(self, action: np.ndarray) -> Tuple[dict, float, bool]:
        """Apply action (first 2 dims = XY velocity) and return (state, reward, done)."""
        # Move agent
        vel          = np.clip(action[:2], -1, 1) * 0.05
        self.agent_pos = np.clip(self.agent_pos + vel, -0.5, 0.5)

        reward = self._compute_reward()
        self.step_count += 1
        done = self.step_count >= 200

        # Object interaction: if agent is close to object, move it slightly
        dist_ao = np.linalg.norm(self.agent_pos - self.obj_pos)
        if dist_ao < 0.06:
            push = (self.obj_pos - self.agent_pos) * 0.1
            self.obj_pos = np.clip(self.obj_pos + push, -0.5, 0.5)

        # TurnFaucet: rotate object angle toward target
        if "Faucet" in self.task_id and dist_ao < 0.1:
            self.obj_angle += action[2] * 0.1

        return self._get_state(), float(reward), done

    def _compute_reward(self) -> float:
        dist_to_obj    = np.linalg.norm(self.agent_pos - self.obj_pos)
        dist_obj_target = np.linalg.norm(self.obj_pos - self.target_pos)

        if "PickCube" in self.task_id:
            # Reward: reach object + move object to target
            r = -dist_to_obj * 0.5 - dist_obj_target
        elif "StackCube" in self.task_id:
            # Reward: place obj on obj2 (target = obj2 position + height)
            stack_target = self.obj2_pos + np.array([0.0, 0.05])
            r = -dist_to_obj * 0.3 - np.linalg.norm(self.obj_pos - stack_target)
        elif "Faucet" in self.task_id:
            # Reward: rotate angle toward target
            angle_diff = abs(self.obj_angle - self.target_angle) % (2 * np.pi)
            angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
            r = -dist_to_obj * 0.3 - angle_diff
        else:  # PushCube
            # Reward: push object without grasping (penalise reaching too close)
            grasp_penalty = max(0, 0.05 - dist_to_obj)
            r = -dist_obj_target - grasp_penalty * 2.0

        return r * 0.1  # scale to reasonable range

    def _get_state(self) -> dict:
        return dict(
            agent_pos    = self.agent_pos.copy(),
            obj_pos      = self.obj_pos.copy(),
            target_pos   = self.target_pos.copy(),
            obj2_pos     = self.obj2_pos.copy(),
            obj_angle    = float(self.obj_angle),
            target_angle = float(self.target_angle),
        )

    @property
    def success(self) -> float:
        dist = np.linalg.norm(self.obj_pos - self.target_pos)
        return float(dist < 0.05)


# ---------------------------------------------------------------------------
# Image rendering from mock state
# ---------------------------------------------------------------------------
TASK_COLOURS = {
    "PickCube":  (200, 60,  60),   # red object
    "StackCube": (60,  200, 60),   # green object
    "TurnFaucet":(60,  60,  200),  # blue object
    "PushCube":  (200, 200, 60),   # yellow object
}

def _draw_circle_fast(img, cx, cy, r, color):
    H, W = img.shape[:2]
    y0, y1 = max(0, cy-r), min(H, cy+r+1)
    x0, x1 = max(0, cx-r), min(W, cx+r+1)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    mask = (xs-cx)**2 + (ys-cy)**2 <= r**2
    img[y0:y1, x0:x1][mask] = color

def _draw_rect_fast(img, cx, cy, hw, color):
    H, W = img.shape[:2]
    img[max(0,cy-hw):min(H,cy+hw+1),
        max(0,cx-hw):min(W,cx+hw+1)] = color

def _w2p(x: float, y: float, size: int = OBS_IMG_SIZE) -> Tuple[int, int]:
    px = int((x / SCENE_RANGE + 0.5) * size)
    py = int((0.5 - y / SCENE_RANGE) * size)
    return np.clip(px, 0, size-1), np.clip(py, 0, size-1)

def state_to_image(state: dict, task_id: str, size: int = OBS_IMG_SIZE) -> np.ndarray:
    img = np.full((size, size, 3), 20, dtype=np.uint8)
    r = max(3, size // 20)   # ensure r≥3 so r-2≥1 at any image size

    # Target (yellow ring)
    tx, ty = _w2p(*state["target_pos"], size)
    _draw_circle_fast(img, tx, ty, r+2, (200, 200, 0))
    _draw_circle_fast(img, tx, ty, max(1, r-1), (20, 20, 20))

    # Secondary object (cyan, StackCube)
    obj2x, obj2y = _w2p(*state["obj2_pos"], size)
    _draw_rect_fast(img, obj2x, obj2y, max(1, r-2), (60, 180, 180))

    # Primary object
    obj_col = TASK_COLOURS.get(task_id.replace("-v1",""), (180, 80, 80))
    ox, oy = _w2p(*state["obj_pos"], size)
    if "Faucet" in task_id:
        # Draw rotated line for faucet handle
        angle = state["obj_angle"]
        ex = int(ox + r * 1.5 * np.cos(angle))
        ey = int(oy - r * 1.5 * np.sin(angle))
        _draw_rect_fast(img, (ox+ex)//2, (oy+ey)//2, max(1, r-2), obj_col)
    else:
        _draw_rect_fast(img, ox, oy, r, obj_col)

    # Agent / end-effector (white)
    ax, ay = _w2p(*state["agent_pos"], size)
    _draw_circle_fast(img, ax, ay, max(1, r-2), (230, 230, 230))

    return img


# ---------------------------------------------------------------------------
# Mock HeterogeneousSkillStream
# ---------------------------------------------------------------------------
MOCK_TASK_SEQUENCE = ["PickCube-v1", "StackCube-v1", "TurnFaucet-v1", "PushCube-v1"]

class MockHeterogeneousSkillStream:
    """
    Drop-in replacement for HeterogeneousSkillStream that requires no GPU.
    Same interface: reset() → obs(CHW float32), step() → (obs, reward, done, info).
    """

    def __init__(
        self,
        task_sequence:     List[str] = None,
        task_episodes:     int       = 100,
        steps_per_episode: int       = 200,
        seed:              int       = 0,
        obs_size:          int       = OBS_IMG_SIZE,
    ):
        self.task_sequence     = task_sequence or MOCK_TASK_SEQUENCE
        self.task_episodes     = task_episodes
        self.steps_per_episode = steps_per_episode
        self.obs_size          = obs_size
        self._base_seed        = seed

        self.n_tasks           = len(self.task_sequence)
        self.action_dim        = ACTION_DIM
        self.obs_shape         = (3, obs_size, obs_size)

        self.episode_count         = 0
        self.task_idx              = 0
        self.task_switch_episodes: List[int] = []
        self._step_in_ep           = 0

        self._task = self._new_task(0)
        print(f"[MockEnv] Initialized with {self.n_tasks} tasks, "
              f"{task_episodes} episodes each, seed={seed}")

    def _new_task(self, task_idx: int) -> MockTask:
        name = self.task_sequence[task_idx].replace("-v1", "")
        return MockTask(name, self._base_seed + task_idx * 1000)

    def _obs_from_state(self, state: dict) -> np.ndarray:
        img = state_to_image(state, self.task_sequence[self.task_idx], self.obs_size)
        return (img.astype(np.float32) / 255.0).transpose(2, 0, 1)

    def reset(self) -> np.ndarray:
        if self.episode_count > 0 and self.episode_count % self.task_episodes == 0:
            self.task_idx = (self.task_idx + 1) % self.n_tasks
            self.task_switch_episodes.append(self.episode_count)
            self._task = self._new_task(self.task_idx)
            print(f"[MockEnv] Task switch → {self.task_sequence[self.task_idx]} "
                  f"at episode {self.episode_count}")

        if self.episode_count % 10 == 0:
            print(f"[MockEnv] Episode {self.episode_count} | "
                  f"task={self.task_sequence[self.task_idx]}")

        self.episode_count += 1
        self._step_in_ep    = 0
        state = self._task.reset()
        return self._obs_from_state(state)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        state, reward, task_done = self._task.step(action)
        self._step_in_ep += 1
        done = task_done or self._step_in_ep >= self.steps_per_episode
        obs  = self._obs_from_state(state)
        info = dict(
            task_idx  = self.task_idx,
            task_name = self.task_sequence[self.task_idx],
            success   = self._task.success,
        )
        return obs, reward, done, info

    @property
    def current_task_name(self) -> str:
        return self.task_sequence[self.task_idx]

    @property
    def current_task_idx(self) -> int:
        return self.task_idx

    def close(self):
        pass
