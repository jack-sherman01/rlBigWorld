"""
palr_trainer.py
===============
DD-PPO training loop for PALR Fetch Rearrangement.

Launch with:
    torchrun --nproc_per_node=4 src/palr_trainer.py \\
        --config configs/ddppo_palr_fetch.yaml \\
        --seed 0 --num_envs 16 --outdir results/palr_seed0

One process per GPU.  Each process:
  - Runs `num_envs` habitat environments (via VectorEnv).
  - Collects rollouts of length `num_steps` (default 64).
  - Performs `ppo_epoch` PPO update rounds per rollout.
  - After `measure_freq` updates, measures CNN plasticity on rank-0
    and all-reduces lr_scales to all ranks.
  - At curriculum phase boundaries, restarts VectorEnv with new task config.
  - Saves checkpoints and TensorBoard logs from rank-0.

PALR hooks:
  (A) After loss.backward():
        policy.net.visual_encoder.scale_gradients(lr_scales)
  (B) Periodically, when dead_fraction > threshold:
        policy.net.visual_encoder.perturb_dead_filters(k, dead_mask)

This file is self-contained and does NOT depend on habitat_baselines.
It uses habitat's gym interface directly for maximum transparency.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml

# habitat gym interface
import habitat.gym   # noqa: F401  (registers habitat gym envs)
import gym

from palr_fetch_policy import PALRFetchNet
from palr_resnet_encoder import PALRResNetEncoder
from fetch_curriculum import FetchCurriculum, make_curriculum_from_config
from plasticity_metrics_cnn import compute_block_metrics, dead_filter_fraction


# ── Replay / rollout storage ──────────────────────────────────────────────────

class RolloutStorage:
    """
    Fixed-length on-device rollout buffer for `num_steps` steps × `num_envs` envs.
    """

    def __init__(self, num_steps, num_envs, obs_space_shapes, action_dim,
                 hidden_size, device):
        self.num_steps = num_steps
        self.num_envs  = num_envs
        self.device    = device

        # Observations (store as float32)
        self.obs = {
            k: torch.zeros(num_steps + 1, num_envs, *shape, device=device)
            for k, shape in obs_space_shapes.items()
        }
        self.actions          = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.rewards          = torch.zeros(num_steps, num_envs, 1,          device=device)
        self.masks            = torch.ones( num_steps + 1, num_envs, 1,      device=device)
        self.value_preds      = torch.zeros(num_steps + 1, num_envs, 1,      device=device)
        self.returns          = torch.zeros(num_steps + 1, num_envs, 1,      device=device)
        self.action_log_probs = torch.zeros(num_steps, num_envs, 1,          device=device)
        self.rnn_hidden       = torch.zeros(num_steps + 1, 1, num_envs, hidden_size, device=device)

        self._step = 0

    def insert(self, obs, action, reward, mask, value, action_log_prob, rnn_hidden):
        t = self._step
        for k in obs:
            self.obs[k][t + 1].copy_(obs[k])
        self.actions[t].copy_(action)
        self.rewards[t].copy_(reward)
        self.masks[t + 1].copy_(mask)
        self.value_preds[t].copy_(value)
        self.action_log_probs[t].copy_(action_log_prob)
        self.rnn_hidden[t + 1].copy_(rnn_hidden)
        self._step = (t + 1) % self.num_steps

    def after_update(self):
        for k in self.obs:
            self.obs[k][0].copy_(self.obs[k][-1])
        self.masks[0].copy_(self.masks[-1])
        self.rnn_hidden[0].copy_(self.rnn_hidden[-1])

    def compute_returns(self, next_value, gamma=0.99, gae_lambda=0.95):
        self.value_preds[-1] = next_value
        gae = 0.0
        for t in reversed(range(self.num_steps)):
            delta = (self.rewards[t]
                     + gamma * self.value_preds[t + 1] * self.masks[t + 1]
                     - self.value_preds[t])
            gae = delta + gamma * gae_lambda * self.masks[t + 1] * gae
            self.returns[t] = gae + self.value_preds[t]

    def get_mini_batches(self, num_mini_batch):
        T, N = self.num_steps, self.num_envs
        assert N % num_mini_batch == 0, "num_envs must be divisible by num_mini_batch"
        envs_per_batch = N // num_mini_batch

        perm = torch.randperm(N)
        for start in range(0, N, envs_per_batch):
            idx = perm[start:start + envs_per_batch]
            obs_b = {k: self.obs[k][:-1, idx].reshape(T * envs_per_batch, *self.obs[k].shape[2:])
                     for k in self.obs}
            actions_b         = self.actions[:, idx].reshape(T * envs_per_batch, -1)
            value_preds_b     = self.value_preds[:-1, idx].reshape(T * envs_per_batch, 1)
            returns_b         = self.returns[:-1, idx].reshape(T * envs_per_batch, 1)
            action_log_probs_b= self.action_log_probs[:, idx].reshape(T * envs_per_batch, 1)
            masks_b           = self.masks[:-1, idx].reshape(T * envs_per_batch, 1)
            rnn_hidden_b      = self.rnn_hidden[0, :, idx]  # [1, envs_per_batch, H]

            advantages = returns_b - value_preds_b
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            yield (obs_b, actions_b, value_preds_b, returns_b,
                   action_log_probs_b, masks_b, rnn_hidden_b, advantages)


# ── VectorEnv wrapper ─────────────────────────────────────────────────────────

# Map task_type strings → bundled habitat hydra configs (shipped inside the
# installed habitat-lab package: habitat/config/benchmark/rearrange/*.yaml).
# habitat-lab >= 0.2.3 uses a hydra-based config system, so we cannot point
# `habitat.get_config` at an arbitrary local YAML — it must be a path that
# the bundled hydra search path can resolve.
_TASK_TO_CONFIG = {
    "RearrangePickTask-v0":       "benchmark/rearrange/pick.yaml",
    "RearrangePlaceTask-v0":      "benchmark/rearrange/place.yaml",
    "RearrangeOpenFridgeTask-v0": "benchmark/rearrange/open_fridge.yaml",
}


def _make_single_env(task_type: str, dataset_path: str,
                     seed: int, rank: int, env_idx: int):
    """Construct a single habitat.Env for habitat.VectorEnv.

    This is a top-level function (not a closure) so it can be pickled
    by habitat.VectorEnv's worker processes.
    """
    import habitat
    from habitat.config.read_write import read_write

    config_path = _TASK_TO_CONFIG.get(task_type)
    if config_path is None:
        raise ValueError(
            f"Unsupported task_type {task_type!r}; "
            f"add it to _TASK_TO_CONFIG."
        )

    cfg = habitat.get_config(config_path)
    with read_write(cfg):
        cfg.habitat.environment.max_episode_steps = 200
        cfg.habitat.dataset.data_path = dataset_path
        cfg.habitat.dataset.split = "train"
        cfg.habitat.simulator.habitat_sim_v0.gpu_device_id = rank
        cfg.habitat.simulator.seed = seed + rank * 1000 + env_idx
    # habitat.VectorEnv expects a GymHabitatEnv (it queries
    # `original_action_space` etc.), so we go through habitat.gym instead of
    # constructing habitat.Env directly.
    from habitat.gym import make_gym_from_config
    return make_gym_from_config(cfg)


def make_env_fn(task_type: str, dataset_path: str, seed: int, rank: int, env_idx: int):
    """Returns a callable that creates one habitat env (legacy helper)."""
    def _init():
        return _make_single_env(task_type, dataset_path, seed, rank, env_idx)
    return _init


# ── PALR logic ────────────────────────────────────────────────────────────────

class PALRState:
    """Tracks per-block LR scales and plasticity history."""

    N_BLOCKS = 4
    BASELINE_ERANK = [48.0, 96.0, 192.0, 384.0]   # ~75% of channel count per block
    BASELINE_DEAD  = 0.0

    def __init__(self, cfg_palr: dict):
        self.enabled          = cfg_palr.get("enabled", True)
        self.beta             = cfg_palr.get("beta", 3.0)
        self.rank_beta        = cfg_palr.get("rank_beta", 1.5)
        self.max_lr_scale     = cfg_palr.get("max_lr_scale", 5.0)
        self.perturb_threshold= cfg_palr.get("perturb_threshold", 0.10)
        self.perturb_sigma    = cfg_palr.get("perturb_sigma", 0.3)
        self.measure_freq     = cfg_palr.get("measure_freq", 200)
        self.diag_batch_size  = cfg_palr.get("diag_batch_size", 256)
        baseline_erank        = cfg_palr.get("baseline_erank", self.BASELINE_ERANK)

        self.lr_scales        = np.ones(self.N_BLOCKS, dtype=np.float32)
        self.baseline_erank   = np.array(baseline_erank, dtype=np.float32)
        self.history: list    = []

    def update(
        self,
        metrics: Dict[str, float],
        encoder: PALRResNetEncoder,
        obs_sample: Dict[str, torch.Tensor],
        update_idx: int,
    ):
        """Compute new lr_scales and optionally perturb, given block metrics."""
        if not self.enabled:
            return

        new_scales = np.ones(self.N_BLOCKS, dtype=np.float32)
        entry = {"update": update_idx}

        for k in range(self.N_BLOCKS):
            dead_k  = metrics.get(f"block_{k}_dead",  0.0)
            erank_k = metrics.get(f"block_{k}_erank", 1.0)

            dead_deficit = max(0.0, dead_k - self.BASELINE_DEAD)
            rank_deficit = max(
                0.0,
                (self.baseline_erank[k] - erank_k) / max(self.baseline_erank[k], 1.0)
            )
            combined = dead_deficit + self.rank_beta * rank_deficit
            # Normalise combined to [0, 1] then linearly map to
            # [1, max_lr_scale].  Saturates only when fully degraded
            # (dead=1.0 AND rank_deficit=1.0 simultaneously).
            combined_norm = combined / (1.0 + self.rank_beta)
            new_scales[k] = float(
                1.0 + (self.max_lr_scale - 1.0)
                * float(np.clip(combined_norm, 0.0, 1.0))
            )

            entry[f"block_{k}_dead"]     = dead_k
            entry[f"block_{k}_erank"]    = erank_k
            entry[f"block_{k}_lr_scale"] = new_scales[k]

            # Targeted perturbation
            if dead_k > self.perturb_threshold:
                # Reconstruct dead mask for this block from metrics
                # (full dead_mask is computed in measure_plasticity)
                pass  # handled separately in measure_plasticity

        self.lr_scales = new_scales
        self.history.append(entry)

    def on_task_switch(self, encoder: PALRResNetEncoder,
                       obs_sample: Dict[str, torch.Tensor]):
        """Proactively boost LR and perturb at task switch."""
        if not self.enabled:
            return
        for k in range(self.N_BLOCKS):
            self.lr_scales[k] = max(self.lr_scales[k], 2.0)
        # Perturbation at task switch happens in measure_plasticity call
        # that the trainer makes immediately after switch


# ── Main trainer ──────────────────────────────────────────────────────────────

class PALRDDPPOTrainer:

    def __init__(self, args):
        self.args = args
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_main = (self.rank == 0)
        self.device = torch.device(f"cuda:{self.rank}")

        # Parse config
        with open(args.config) as f:
            self.cfg = yaml.safe_load(f)

        self.palr_cfg   = self.cfg.get("PALR", {})
        self.ppo_cfg    = self.cfg["RL"]["PPO"]
        self.num_steps  = self.cfg["RL"]["ROLLOUT_STORAGE"]["num_steps"]
        self.num_envs   = args.num_envs
        self.total_steps= args.steps
        self.seed       = args.seed
        self.outdir     = args.outdir

        self.success_keys = self.cfg.get(
            "SUCCESS_MEASURE_KEYS",
            [
                "rearrange_pick_success",
                "rearrange_place_success",
                "rearrange_open_fridge_success",
                "success",  # fallback
            ],
        )

        os.makedirs(self.outdir, exist_ok=True)
        os.makedirs(f"{self.outdir}/checkpoints", exist_ok=True)

    # ── Setup ──────────────────────────────────────────────────────────────────

    def setup_distributed(self):
        dist.init_process_group("nccl")
        torch.cuda.set_device(self.rank)

    def build_policy(self, action_dim: int, joint_dim: int) -> PALRFetchNet:
        policy = PALRFetchNet(
            action_dim  = action_dim,
            joint_dim   = joint_dim,
            use_palr    = self.palr_cfg.get("enabled", True),
        ).to(self.device)
        policy = nn.parallel.DistributedDataParallel(
            policy, device_ids=[self.rank]
        )
        return policy

    def build_optimizer(self, policy) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            policy.parameters(),
            lr   = self.ppo_cfg["lr"],
            eps  = self.ppo_cfg["eps"],
        )

    def _extract_success(self, info: dict) -> float:
        """Return the first matching success measurement from habitat info."""
        for key in self.success_keys:
            if key in info:
                return float(info[key])
        return 0.0

    # ── PPO update ─────────────────────────────────────────────────────────────

    def ppo_update(
        self,
        policy:      nn.Module,
        optimizer:   torch.optim.Optimizer,
        rollouts:    RolloutStorage,
        palr_state:  PALRState,
        clip_param:  float,
        ppo_epochs:  int,
        num_mini_batch: int,
        value_coef:  float,
        entropy_coef: float,
        max_grad_norm: float,
    ) -> Dict[str, float]:
        logs = defaultdict(list)

        for _ in range(ppo_epochs):
            for batch in rollouts.get_mini_batches(num_mini_batch):
                (obs_b, actions_b, value_preds_b, returns_b,
                 old_log_probs_b, masks_b, rnn_hidden_b, advantages) = batch

                dist, value, _ = policy.module.forward(obs_b, rnn_hidden_b, masks_b)
                new_log_probs = dist.log_prob(actions_b).sum(-1, keepdim=True)
                entropy       = dist.entropy().sum(-1).mean()

                ratio = torch.exp(new_log_probs - old_log_probs_b)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (returns_b - value).pow(2).mean()

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()

                # ── PALR gradient scaling ──────────────────────────────────
                if palr_state.enabled:
                    policy.module.visual_encoder.scale_gradients(palr_state.lr_scales)

                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

                logs["policy_loss"].append(policy_loss.item())
                logs["value_loss"].append(value_loss.item())
                logs["entropy"].append(entropy.item())

        return {k: float(np.mean(v)) for k, v in logs.items()}

    # ── Plasticity measurement ─────────────────────────────────────────────────

    @torch.no_grad()
    def measure_plasticity(
        self,
        policy:     nn.Module,
        rollouts:   RolloutStorage,
        palr_state: PALRState,
        update_idx: int,
    ):
        if not palr_state.enabled:
            return {}

        encoder: PALRResNetEncoder = policy.module.visual_encoder
        B = min(palr_state.diag_batch_size, self.num_steps * self.num_envs)

        # Sample observations from rollout buffer
        all_rgb   = rollouts.obs["rgb"][:-1].reshape(-1, *rollouts.obs["rgb"].shape[2:])
        all_depth = rollouts.obs["depth"][:-1].reshape(-1, *rollouts.obs["depth"].shape[2:])
        idx = torch.randperm(all_rgb.shape[0])[:B]
        rgb_b   = all_rgb[idx].float() / 255.0     # [B, H, W, 3]
        depth_b = all_depth[idx].float()            # [B, H, W, 1]
        # HWC → CHW
        rgb_b   = rgb_b.permute(0, 3, 1, 2).to(self.device)
        depth_b = depth_b.permute(0, 3, 1, 2).to(self.device)

        metrics = compute_block_metrics(encoder, rgb_b, depth_b)

        # Update lr_scales
        palr_state.update(metrics, encoder, {}, update_idx)

        # All-reduce lr_scales across GPUs (so all ranks use the same scales)
        scales_tensor = torch.tensor(palr_state.lr_scales, device=self.device)
        dist.all_reduce(scales_tensor, op=dist.ReduceOp.AVG)
        palr_state.lr_scales = scales_tensor.cpu().numpy()

        # Targeted perturbation per block
        for k in range(PALRState.N_BLOCKS):
            dead_k = metrics.get(f"block_{k}_dead", 0.0)
            if dead_k > palr_state.perturb_threshold:
                # Recompute dead mask for this block
                from plasticity_metrics_cnn import BlockActivationCollector
                collector = BlockActivationCollector(encoder)
                x = torch.cat([rgb_b, depth_b], dim=1)
                encoder.eval()
                encoder(x)
                encoder.train()
                acts = collector.get()
                collector.remove_hooks()
                if k in acts:
                    act_k     = acts[k]                        # [B, C, H, W]
                    max_per_f = act_k.max(axis=0).max(-1).max(-1)  # [C]
                    dead_mask = max_per_f <= 0.0
                    encoder.perturb_dead_filters(k, dead_mask)

        return metrics

    # ── Main training loop ─────────────────────────────────────────────────────

    def train(self):
        self.setup_distributed()
        torch.manual_seed(self.seed + self.rank)

        curriculum = make_curriculum_from_config(self.cfg["CURRICULUM"])
        palr_state = PALRState(self.palr_cfg)

        # Import TensorBoard only on rank 0
        writer = None
        if self.is_main:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(f"{self.outdir}/tb")

        # ── Build environments (rank-local) ───────────────────────────────────
        phase = curriculum.current_phase

        def make_envs(task_type, dataset_path):
            """Build a habitat.VectorEnv with `self.num_envs` parallel envs.

            Uses habitat.VectorEnv directly (rather than
            habitat_baselines.utils.env_utils.construct_envs) because the
            latter's signature has changed across habitat-baselines versions
            and is not needed for our use case.
            """
            import habitat
            # ── DEBUG: instantiate one env in-process so worker exceptions
            #          aren't swallowed by VectorEnv's pipe and we can see
            #          the real traceback.  Remove once stable.
            if os.environ.get("PALR_DEBUG_SINGLE_ENV", "0") == "1":
                probe = _make_single_env(task_type, dataset_path,
                                         self.seed, self.rank, 0)
                print("[PALR-DEBUG] single env OK; "
                      f"obs_space={probe.observation_space} "
                      f"action_space={probe.action_space}",
                      flush=True)
                obs = probe.reset()
                print(f"[PALR-DEBUG] reset OK; "
                      f"obs keys={list(obs.keys()) if hasattr(obs, 'keys') else type(obs)}",
                      flush=True)
                probe.close()
            env_fn_args = [
                (task_type, dataset_path, self.seed, self.rank, i)
                for i in range(self.num_envs)
            ]
            return habitat.VectorEnv(
                make_env_fn=_make_single_env,
                env_fn_args=env_fn_args,
            )

        envs = make_envs(phase.task_type, phase.dataset_path)

        # Determine spaces
        obs_space  = envs.observation_spaces[0]
        act_space  = envs.action_spaces[0]
        action_dim = act_space.shape[0]
        joint_dim  = obs_space["joint"].shape[0]

        # ── Build policy and optimizer ────────────────────────────────────────
        policy    = self.build_policy(action_dim, joint_dim)
        optimizer = self.build_optimizer(policy)

        hidden_size = policy.module.hidden_size
        obs_shapes  = {k: obs_space[k].shape for k in obs_space.spaces}

        rollouts = RolloutStorage(
            self.num_steps, self.num_envs,
            obs_shapes, action_dim, hidden_size, self.device,
        )

        # ── Collect initial observations ──────────────────────────────────────
        obs_list = envs.reset()
        for k in obs_shapes:
            t = torch.from_numpy(np.stack([o[k] for o in obs_list])).to(self.device)
            rollouts.obs[k][0].copy_(t)

        rnn_hidden = torch.zeros(1, self.num_envs, hidden_size, device=self.device)
        masks      = torch.ones(self.num_envs, 1, device=self.device)

        # ── Stats ──────────────────────────────────────────────────────────────
        episode_rewards: List[float] = []
        episode_successes: List[float] = []
        update_idx  = 0
        total_env_steps = 0
        t_start = time.time()

        # Per-env accumulators for episode returns (reset at done boundaries)
        running_ep_returns = np.zeros(self.num_envs, dtype=np.float32)

        while total_env_steps < self.total_steps:

            # ── Curriculum switch check ────────────────────────────────────────
            if curriculum.should_switch():
                new_phase = curriculum.advance()
                if new_phase is not None:
                    envs.close()
                    envs = make_envs(new_phase.task_type, new_phase.dataset_path)
                    obs_list = envs.reset()
                    for k in obs_shapes:
                        t = torch.from_numpy(
                            np.stack([o[k] for o in obs_list])
                        ).to(self.device)
                        rollouts.obs[k][0].copy_(t)
                    rnn_hidden = torch.zeros(1, self.num_envs, hidden_size, device=self.device)
                    masks      = torch.ones(self.num_envs, 1, device=self.device)

                    # Reset per-env return accumulators when environments are restarted
                    running_ep_returns[:] = 0.0

                    # PALR proactive boost at task switch
                    palr_state.lr_scales = np.clip(palr_state.lr_scales, 2.0, palr_state.max_lr_scale)

            # ── Rollout collection ─────────────────────────────────────────────
            policy.eval()
            with torch.no_grad():
                for step in range(self.num_steps):
                    obs_b = {k: rollouts.obs[k][step] for k in obs_shapes}
                    rnn_h = rollouts.rnn_hidden[step]
                    mask  = rollouts.masks[step]

                    dist, value, rnn_hidden = policy.module.forward(obs_b, rnn_h, mask)
                    action = dist.sample()
                    action_log_prob = dist.log_prob(action).sum(-1, keepdim=True)

                    # Step environments
                    actions_np = action.cpu().numpy()
                    obs_list, rewards, dones, infos = envs.step(actions_np)

                    reward_t = torch.tensor(rewards, dtype=torch.float32,
                                            device=self.device).unsqueeze(1)
                    mask_t   = torch.tensor(
                        [[0.0] if d else [1.0] for d in dones],
                        dtype=torch.float32, device=self.device
                    )

                    for k in obs_shapes:
                        t = torch.from_numpy(
                            np.stack([o[k] for o in obs_list])
                        ).to(self.device)
                        rollouts.obs[k][step + 1].copy_(t)

                    rollouts.insert(
                        obs   = {k: rollouts.obs[k][step + 1] for k in obs_shapes},
                        action= action,
                        reward= reward_t,
                        mask  = mask_t,
                        value = value,
                        action_log_prob = action_log_prob,
                        rnn_hidden      = rnn_hidden,
                    )

                    # Accumulate per-env episode returns
                    running_ep_returns += np.asarray(rewards, dtype=np.float32).reshape(self.num_envs)

                    for i, (done, info) in enumerate(zip(dones, infos)):
                        if done:
                            episode_rewards.append(float(running_ep_returns[i]))
                            episode_successes.append(self._extract_success(info))
                            running_ep_returns[i] = 0.0

                    total_env_steps += self.num_envs
                    curriculum.step(self.num_envs)

                # Bootstrap value
                obs_last = {k: rollouts.obs[k][-1] for k in obs_shapes}
                _, next_value, _ = policy.module.forward(
                    obs_last, rollouts.rnn_hidden[-1], rollouts.masks[-1]
                )

            rollouts.compute_returns(next_value.detach())

            # ── PALR: measure plasticity ────────────────────────────────────────
            policy.train()
            plast_metrics = {}
            if (update_idx % palr_state.measure_freq == 0
                    and total_env_steps > self.num_steps * self.num_envs):
                plast_metrics = self.measure_plasticity(
                    policy, rollouts, palr_state, update_idx
                )

            # ── PPO update ─────────────────────────────────────────────────────
            ppo_logs = self.ppo_update(
                policy, optimizer, rollouts, palr_state,
                clip_param    = self.ppo_cfg["clip_param"],
                ppo_epochs    = self.ppo_cfg["ppo_epoch"],
                num_mini_batch= self.ppo_cfg["num_mini_batch"],
                value_coef    = self.ppo_cfg["value_loss_coef"],
                entropy_coef  = self.ppo_cfg["entropy_coef"],
                max_grad_norm = self.ppo_cfg["max_grad_norm"],
            )
            rollouts.after_update()
            update_idx += 1

            # ── Logging ────────────────────────────────────────────────────────
            log_interval = self.cfg.get("LOG_INTERVAL", 10)
            if self.is_main and update_idx % log_interval == 0:
                fps = total_env_steps / (time.time() - t_start)
                mean_r = np.mean(episode_rewards[-100:]) if episode_rewards else 0.0
                mean_s = np.mean(episode_successes[-100:]) if episode_successes else 0.0
                print(
                    f"  update={update_idx:6d}  "
                    f"steps={total_env_steps:10,d}  "
                    f"fps={fps:.0f}  "
                    f"task={curriculum.current_phase.label:<15s}  "
                    f"r={mean_r:.2f}  "
                    f"suc={mean_s:.2f}  "
                    f"lr_scales={palr_state.lr_scales.round(2)}"
                )
                if writer:
                    writer.add_scalar("train/mean_reward", mean_r, total_env_steps)
                    writer.add_scalar("train/success_rate", mean_s, total_env_steps)
                    writer.add_scalar("train/fps", fps, total_env_steps)
                    for k, v in ppo_logs.items():
                        writer.add_scalar(f"ppo/{k}", v, total_env_steps)
                    for k in range(4):
                        if f"block_{k}_dead" in plast_metrics:
                            writer.add_scalar(f"palr/block{k}_dead",
                                              plast_metrics[f"block_{k}_dead"], total_env_steps)
                            writer.add_scalar(f"palr/block{k}_erank",
                                              plast_metrics[f"block_{k}_erank"], total_env_steps)
                            writer.add_scalar(f"palr/block{k}_lr_scale",
                                              float(palr_state.lr_scales[k]), total_env_steps)

            # ── Checkpoint ──────────────────────────────────────────────────────
            ckpt_interval = self.cfg.get("CHECKPOINT_INTERVAL", 500)
            if self.is_main and update_idx % ckpt_interval == 0:
                ckpt = {
                    "update":       update_idx,
                    "total_steps":  total_env_steps,
                    "policy":       policy.module.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "curriculum":   curriculum.state_dict(),
                    "palr_history": palr_state.history,
                    "lr_scales":    palr_state.lr_scales.tolist(),
                }
                path = f"{self.outdir}/checkpoints/ckpt_{update_idx:06d}.pt"
                torch.save(ckpt, path)
                print(f"  [checkpoint saved → {path}]")

        # ── Cleanup ────────────────────────────────────────────────────────────
        envs.close()
        if writer:
            writer.close()

        # Save final plasticity history
        if self.is_main:
            hist_path = f"{self.outdir}/palr_plasticity_history.json"
            with open(hist_path, "w") as f:
                json.dump(palr_state.history, f, indent=2)
            sw_path = f"{self.outdir}/curriculum_switch_log.json"
            with open(sw_path, "w") as f:
                json.dump(curriculum.switch_log, f, indent=2)
            print(f"\nTraining complete.  Steps: {total_env_steps:,d}")
            print(f"  Plasticity history → {hist_path}")
            print(f"  Switch log         → {sw_path}")

        dist.destroy_process_group()


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",   required=True)
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--num_envs", type=int, default=16,
                   help="Environments per GPU")
    p.add_argument("--steps",    type=int, default=200_000_000,
                   help="Total env steps (across all phases)")
    p.add_argument("--outdir",   default="results/run")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = PALRDDPPOTrainer(args)
    trainer.train()
