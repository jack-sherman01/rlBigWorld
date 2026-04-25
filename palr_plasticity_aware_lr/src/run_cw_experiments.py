"""
CW10 Experiment Runner
=======================
Runs PALR-SAC and all baseline agents on ContinualWorld-10 (10 MT10 tasks
sequentially, no task oracle) and saves structured results.

Key design choices
------------------
  - Batch size 512 (exploits 40 GB GPU for more accurate NTK rank estimation).
  - 4-layer MLP critic (256×4) for per-layer LR visualisation.
  - Per-seed checkpoint files — safe to interrupt and resume.
  - Parallel-friendly: --seed_offset + --ckpt_suffix flags.
  - Learning-speed tracking: records episodes-to-threshold for EACH task,
    enabling the task-1 vs task-10 plasticity-loss comparison.

Usage
-----
    # Full run (100 ep/task, 5 seeds)
    bash run.sh palr_plasticity_aware_lr/src/run_cw_experiments.py

    # Fast debug run (20 ep/task, 1 seed)
    bash run.sh palr_plasticity_aware_lr/src/run_cw_experiments.py --fast

    # Single seed (for parallel launch)
    bash run.sh palr_plasticity_aware_lr/src/run_cw_experiments.py \\
        --episodes_per_task 100 --seeds 1 --seed_offset 2 --ckpt_suffix _seed2

Results saved to:  results/cw_checkpoint_seed<N>.json
"""

import sys, os, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "egl")

from cw_env import ContinualWorld, CW10_TASKS
from sac_base import SACAgent
from cw_baselines import SACShinkAndPerturbAgent, SACPeriodicResetAgent, SACL2RegAgent
from palr_sac_agent import PALRSACAgent
from plasticity_metrics import compute_all_metrics
from palr_sac_agent import CW_HIDDEN_INDICES

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Shared hyperparameters ────────────────────────────────────────────────────

def make_common(obs_dim: int, action_dim: int, batch_size: int) -> dict:
    return dict(
        obs_dim      = obs_dim,
        action_dim   = action_dim,
        hidden_sizes = (256, 256, 256, 256),   # 4-layer MLP
        buffer_size  = 500_000,
        batch_size   = batch_size,
        gamma        = 0.99,
        tau          = 0.005,
        lr           = 3e-4,
    )


def make_agents(obs_dim: int, action_dim: int, seed: int,
                batch_size: int = 512) -> list:
    common = make_common(obs_dim, action_dim, batch_size)
    return [
        SACAgent(seed=seed, **common),
        SACShinkAndPerturbAgent(seed=seed, perturb_freq=5_000,
                                shrink_alpha=0.9, sigma=0.01, **common),
        SACPeriodicResetAgent(seed=seed, reset_freq=20, **common),
        SACL2RegAgent(seed=seed, l2_coeff=1e-4, **common),
        # PALR full
        PALRSACAgent(base_lr=3e-4, seed=seed, beta=3.0, rank_beta=1.5,
                     measure_freq=500, perturb_threshold=0.10,
                     perturb_sigma=0.3, **common),
        # Ablation: LR scaling only
        PALRSACAgent(base_lr=3e-4, seed=seed, beta=3.0, rank_beta=1.5,
                     measure_freq=500, perturb_threshold=0.10,
                     no_perturb=True, **common),
        # Ablation: perturbation only
        PALRSACAgent(base_lr=3e-4, seed=seed, beta=3.0, rank_beta=1.5,
                     measure_freq=500, perturb_threshold=0.10,
                     perturb_sigma=0.3, no_scale=True, **common),
    ]


# ── Training loop ─────────────────────────────────────────────────────────────

def train_agent_cw(
    agent,
    env: ContinualWorld,
    n_episodes: int,
    measure_plasticity_every: int = 5,
    verbose: bool = True,
    verbose_every: int = 20,
    reward_threshold: float = None,
) -> dict:
    """
    Train one agent on ContinualWorld for n_episodes.

    Returns a dict containing:
      episode_rewards, task_ids, task_switch_episodes, plasticity_log,
      palr_plasticity_history (PALR only), learning_speed (episodes-to-threshold
      per task — the key metric for task-1 vs task-N comparison).
    """
    episode_rewards = []
    task_ids        = []
    plasticity_log  = []
    last_task_idx   = 0

    # Learning-speed tracking: for each task, record the episode (within-task)
    # at which the agent first crosses `reward_threshold`.
    # Shape: list of (task_idx, episodes_to_threshold or None)
    task_learning_speeds: dict = {i: None for i in range(env.n_tasks)}
    task_episode_counts:  dict = {i: 0    for i in range(env.n_tasks)}
    task_reward_sums:     dict = {i: []   for i in range(env.n_tasks)}

    for ep in range(n_episodes):
        obs  = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action  = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            agent.push(obs, action, reward, next_obs, done)
            agent.train_step()
            obs = next_obs
            ep_reward += reward

        agent.on_episode_end(ep_reward)
        episode_rewards.append(ep_reward)
        task_ids.append(env.task_idx)

        # Track per-task reward for learning-speed computation
        tid = env.task_idx
        task_episode_counts[tid] += 1
        task_reward_sums[tid].append(ep_reward)

        # Learning-speed: episodes to cross threshold within each task
        if reward_threshold is not None and task_learning_speeds[tid] is None:
            if ep_reward >= reward_threshold:
                task_learning_speeds[tid] = task_episode_counts[tid]

        # Detect task switch and notify PALR
        if env.task_idx != last_task_idx:
            if hasattr(agent, "reset_plasticity_baseline"):
                agent.reset_plasticity_baseline()
            last_task_idx = env.task_idx

        # Periodic plasticity logging
        if ep % measure_plasticity_every == 0 and len(agent.buffer) >= agent.batch_size:
            obs_b, act_b, _, _, _ = agent.buffer.sample(
                min(512, len(agent.buffer))
            )
            # Pass actions as extra_input for SAC critics
            m = compute_all_metrics(agent.critic1, obs_b, CW_HIDDEN_INDICES,
                                    extra_input=act_b)
            m["episode"]   = ep
            m["task_idx"]  = env.task_idx
            m["ep_reward"] = ep_reward
            plasticity_log.append(m)

        if verbose and (ep + 1) % verbose_every == 0:
            recent = np.mean(episode_rewards[-verbose_every:])
            print(
                f"  [{agent.name}] Ep {ep+1:4d}/{n_episodes} | "
                f"Task: {env.current_task:<30s} | "
                f"Avg reward (last {verbose_every}): {recent:.2f}"
            )

    result = {
        "agent_name":           agent.name,
        "episode_rewards":      episode_rewards,
        "task_ids":             task_ids,
        "task_switch_episodes": env.task_switch_episodes,
        "plasticity_log":       plasticity_log,
        "learning_speed":       task_learning_speeds,
        "task_mean_rewards":    {
            i: float(np.mean(v)) if v else 0.0
            for i, v in task_reward_sums.items()
        },
    }
    if hasattr(agent, "plasticity_history"):
        result["palr_plasticity_history"] = agent.plasticity_history
    return result


# ── Serialisation ─────────────────────────────────────────────────────────────

def to_serialisable(obj):
    if isinstance(obj, np.ndarray):       return obj.tolist()
    if isinstance(obj, (np.integer,)):    return int(obj)
    if isinstance(obj, (np.floating,)):   return float(obj)
    if isinstance(obj, dict):             return {k: to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):             return [to_serialisable(i) for i in obj]
    return obj


def save_checkpoint(all_results, ckpt_suffix=""):
    fname     = f"cw_checkpoint{ckpt_suffix}.json"
    ckpt_path = os.path.join(RESULTS_DIR, fname)
    with open(ckpt_path, "w") as f:
        json.dump(to_serialisable(all_results), f)
    print(f"  [checkpoint → {ckpt_path}]")


# ── Multi-seed runner ─────────────────────────────────────────────────────────

def run_all(episodes_per_task: int, n_seeds: int, seed_offset_start: int = 0,
            ckpt_suffix: str = "", batch_size: int = 512,
            reward_threshold: float = None):

    all_results = {}

    # Resume from existing checkpoint
    ckpt_path = os.path.join(RESULTS_DIR, f"cw_checkpoint{ckpt_suffix}.json")
    if seed_offset_start > 0 and os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            all_results = json.load(f)
        print(f"Loaded checkpoint: {sum(len(v) for v in all_results.values())} runs")

    for seed_offset in range(seed_offset_start, seed_offset_start + n_seeds):
        seed = 42 + seed_offset * 13
        print(f"\n{'='*70}")
        print(f"SEED {seed_offset+1}/{seed_offset_start+n_seeds}  (seed={seed})")
        print(f"{'='*70}")

        # Probe env for dims
        probe = ContinualWorld(episodes_per_task=1, max_steps=1, seed=seed)
        obs_dim, action_dim = probe.obs_dim, probe.action_dim
        probe.close()
        print(f"CW10: obs_dim={obs_dim}  action_dim={action_dim}")

        n_episodes = episodes_per_task * len(CW10_TASKS)
        agents = make_agents(obs_dim, action_dim, seed, batch_size)

        for agent in agents:
            print(f"\n--- {agent.name} ---")
            env = ContinualWorld(episodes_per_task=episodes_per_task, seed=seed)
            result = train_agent_cw(
                agent, env, n_episodes,
                verbose=True,
                verbose_every=max(1, n_episodes // 20),
                reward_threshold=reward_threshold,
            )
            env.close()

            if agent.name not in all_results:
                all_results[agent.name] = []
            all_results[agent.name].append(result)

        save_checkpoint(all_results, ckpt_suffix)

    return all_results


# ── Summary + table ───────────────────────────────────────────────────────────

def save_results(all_results, episodes_per_task: int):
    raw_path = os.path.join(RESULTS_DIR, "cw_raw_results.json")
    with open(raw_path, "w") as f:
        json.dump(to_serialisable(all_results), f, indent=2)
    print(f"\nSaved raw results → {raw_path}")

    summary = {}
    for name, runs in all_results.items():
        all_rewards = np.array([r["episode_rewards"] for r in runs])
        n_tasks     = len(CW10_TASKS)

        # Per-task mean reward (across all seeds)
        task_means = {}
        for tid in range(n_tasks):
            task_eps = [
                ep for ep, t in enumerate(runs[0]["task_ids"]) if t == tid
            ]
            if task_eps:
                task_means[f"task_{tid+1}"] = float(
                    all_rewards[:, task_eps].mean()
                )

        # Learning speed: mean episodes-to-threshold per task
        speed_by_task = {}
        for tid in range(n_tasks):
            speeds = [
                r["learning_speed"].get(str(tid)) or r["learning_speed"].get(tid)
                for r in runs
            ]
            valid = [s for s in speeds if s is not None]
            speed_by_task[f"task_{tid+1}"] = float(np.mean(valid)) if valid else None

        # Task-1 vs task-10 learning speed gap (the key plasticity metric)
        s1  = speed_by_task.get("task_1")
        s10 = speed_by_task.get("task_10")
        plasticity_gap = (s10 - s1) if (s1 is not None and s10 is not None) else None

        summary[name] = {
            "n_runs":              len(runs),
            "overall_mean_reward": float(all_rewards.mean()),
            "overall_std_reward":  float(all_rewards.std()),
            "final_task_mean":     float(all_rewards[:, -episodes_per_task:].mean()),
            "task_means":          task_means,
            "learning_speed":      speed_by_task,
            "plasticity_gap":      plasticity_gap,
        }

    summary_path = os.path.join(RESULTS_DIR, "cw_summary_stats.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary → {summary_path}")
    return summary


def print_table(summary):
    print("\n" + "="*80)
    print("CW10 RESULTS SUMMARY")
    print("="*80)
    print(f"{'Agent':<30} {'N':>3} {'Overall':>10} {'T10 Mean':>10} {'Plast.Gap':>12}")
    print("-"*80)
    for name, s in sorted(summary.items(),
                           key=lambda x: x[1]["overall_mean_reward"], reverse=True):
        gap = f"{s['plasticity_gap']:.1f}ep" if s["plasticity_gap"] is not None else "  n/a"
        print(
            f"{name:<30} "
            f"{s['n_runs']:>3} "
            f"{s['overall_mean_reward']:>8.2f}±{s['overall_std_reward']:<5.2f}"
            f"{s['final_task_mean']:>10.2f}  "
            f"{gap:>10}"
        )
    print("="*80)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes_per_task", type=int, default=100)
    parser.add_argument("--seeds",             type=int, default=5)
    parser.add_argument("--batch_size",        type=int, default=512)
    parser.add_argument("--reward_threshold",  type=float, default=None,
                        help="Reward threshold for learning-speed tracking")
    parser.add_argument("--seed_offset",       type=int, default=0)
    parser.add_argument("--ckpt_suffix",       type=str, default="")
    parser.add_argument("--fast", action="store_true",
                        help="Quick debug: 20 ep/task, 1 seed, batch 64")
    args = parser.parse_args()

    if args.fast:
        ep_per_task  = 20
        n_seeds      = 1
        batch_size   = 64
    else:
        ep_per_task  = args.episodes_per_task
        n_seeds      = args.seeds
        batch_size   = args.batch_size

    n_total = ep_per_task * len(CW10_TASKS)
    print(f"CW10 Experiment")
    print(f"  episodes_per_task={ep_per_task}  total={n_total}  "
          f"seeds={n_seeds}  batch={batch_size}")
    print(f"  Results → {RESULTS_DIR}\n")

    all_results = run_all(
        ep_per_task, n_seeds,
        seed_offset_start=args.seed_offset,
        ckpt_suffix=args.ckpt_suffix,
        batch_size=batch_size,
        reward_threshold=args.reward_threshold,
    )
    summary = save_results(all_results, ep_per_task)
    print_table(summary)
    print("\nDone. Run plot_cw_results.py to generate figures.")
