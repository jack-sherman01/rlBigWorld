"""
JBW Experiment Runner
=====================
Runs PALR and baseline agents on ContinualJBW and saves results.

Features
--------
  - 4 phases of `phase_episodes` episodes each (reward sign flips, no oracle)
  - Same agent set as CartPole experiments
  - Per-seed checkpoint files (jbw_checkpoint_seed<N>.json) — safe to interrupt
  - Video recording every `video_every` episodes for PALR agent only
    Videos saved to: plots/jbw_videos/seed<N>/ep<E>_<agent>.mp4
    Frames = upscaled 11×11 vision observations (what the agent sees)
  - Parallel-friendly: --seed_offset + --ckpt_suffix flags

Usage
-----
    # Full run (200 episodes, 3 seeds)
    python run_jbw_experiments.py

    # Fast debug run (80 episodes, 1 seed)
    python run_jbw_experiments.py --fast

    # Single seed (for parallel launch)
    python run_jbw_experiments.py --episodes 400 --seeds 1 --seed_offset 2 --ckpt_suffix _seed2

Results saved to:  results/jbw_checkpoint_seed<N>.json
Plots  saved to:   plots/   (via plot_jbw_results.py)
Videos saved to:   plots/jbw_videos/
"""

import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from jbw_env import ContinualJBW
from dqn_base import DQNAgent
from baselines import ShrinkAndPerturbAgent, PeriodicResetAgent, L2RegAgent
from palr_agent import PALRAgent
from plasticity_metrics import compute_all_metrics, HIDDEN_LAYER_INDICES

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "plots")
VIDEOS_DIR  = os.path.join(PLOTS_DIR, "jbw_videos")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR,  exist_ok=True)


# ---------------------------------------------------------------------------
# Video saving helper
# ---------------------------------------------------------------------------
def save_video(frames, path, fps=10, upscale=20):
    """
    Save a list of (H, W, 3) uint8 frames as an mp4 video.
    Upscales each frame by `upscale` factor for visibility.
    Falls back to GIF if ffmpeg is unavailable.
    """
    try:
        import imageio
        import numpy as np

        if not frames:
            return

        # Upscale each frame (11×11 → 220×220 by default)
        big_frames = []
        for f in frames:
            h, w = f.shape[:2]
            big = np.repeat(np.repeat(f, upscale, axis=0), upscale, axis=1)
            big_frames.append(big)

        # Try mp4 first, fall back to gif
        try:
            imageio.mimwrite(path, big_frames, fps=fps, macro_block_size=None)
        except Exception:
            gif_path = path.replace(".mp4", ".gif")
            imageio.mimwrite(gif_path, big_frames, fps=fps)
            path = gif_path

        print(f"    [video saved → {os.path.basename(path)}]")
    except Exception as e:
        print(f"    [video save failed: {e}]")


# ---------------------------------------------------------------------------
# Shared hyperparameters
# ---------------------------------------------------------------------------
def make_common(obs_dim: int, n_actions: int) -> dict:
    return dict(
        obs_dim      = obs_dim,
        n_actions    = n_actions,
        hidden_sizes = (128, 128),
        buffer_size  = 50_000,
        batch_size   = 64,
        gamma        = 0.99,
        target_update_freq = 500,
        epsilon_start = 1.0,
        epsilon_end   = 0.05,
        epsilon_decay = 5_000,
    )


def make_agents(obs_dim: int, n_actions: int, seed: int) -> list:
    common = make_common(obs_dim, n_actions)
    return [
        DQNAgent(lr=1e-3, seed=seed, **common),
        ShrinkAndPerturbAgent(lr=1e-3, seed=seed, perturb_freq=2000,
                              alpha=0.9, sigma=0.01, **common),
        PeriodicResetAgent(lr=1e-3, seed=seed, reset_freq=200, **common),
        L2RegAgent(lr=1e-3, seed=seed, l2_coeff=1e-4, **common),
        # PALR full
        PALRAgent(base_lr=1e-3, seed=seed, beta=3.0, rank_beta=1.5,
                  measure_freq=100, perturb_threshold=0.10,
                  perturb_sigma=0.3, **common),
        # Ablation: LR scaling only
        PALRAgent(base_lr=1e-3, seed=seed, beta=3.0, rank_beta=1.5,
                  measure_freq=100, perturb_threshold=0.10,
                  no_perturb=True, **common),
        # Ablation: perturbation only
        PALRAgent(base_lr=1e-3, seed=seed, beta=3.0, rank_beta=1.5,
                  measure_freq=100, perturb_threshold=0.10,
                  perturb_sigma=0.3, no_scale=True, **common),
    ]


# ---------------------------------------------------------------------------
# Training loop with video recording
# ---------------------------------------------------------------------------
def train_agent_jbw(
    agent,
    env: ContinualJBW,
    n_episodes: int,
    measure_plasticity_every: int = 5,
    verbose: bool = True,
    verbose_every: int = 40,
    video_every: int = 50,
    video_dir: str = None,
    seed_tag: str = "",
) -> dict:
    """
    Train one agent on ContinualJBW for n_episodes.

    video_every: record a full episode as video every this many episodes.
    video_dir:   directory to save videos (None = skip).
    """
    episode_rewards = []
    task_ids        = []
    plasticity_log  = []
    last_phase      = 0

    for ep in range(n_episodes):
        obs  = env.reset()
        done = False
        ep_reward    = 0.0
        video_frames = []
        record_video = (video_dir is not None) and ((ep + 1) % video_every == 0)

        while not done:
            action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            agent.push(obs, action, reward, next_obs, done)
            agent.train_step()
            obs = next_obs
            ep_reward += reward

            # Capture vision frame for video
            if record_video:
                frame = env.get_vision_frame()
                if frame is not None:
                    video_frames.append(frame)

        agent.on_episode_end(ep_reward)
        episode_rewards.append(ep_reward)
        task_ids.append(env.current_phase)

        # Notify PALR of phase change
        if env.current_phase != last_phase:
            if hasattr(agent, "reset_plasticity_baseline"):
                agent.reset_plasticity_baseline()
            last_phase = env.current_phase

        # Save video for this episode
        if record_video and video_frames and video_dir:
            safe_name = agent.name.replace(" ", "_").replace("(", "").replace(")", "")
            vid_path  = os.path.join(
                video_dir,
                f"ep{ep+1:04d}_{safe_name}{seed_tag}.mp4"
            )
            save_video(video_frames, vid_path)

        # Periodic plasticity logging
        if ep % measure_plasticity_every == 0 and len(agent.buffer) >= agent.batch_size:
            obs_batch, _, _, _, _ = agent.buffer.sample(
                min(256, len(agent.buffer))
            )
            m = compute_all_metrics(agent.online_net, obs_batch, HIDDEN_LAYER_INDICES)
            m["episode"]   = ep
            m["phase"]     = env.current_phase
            m["ep_reward"] = ep_reward
            plasticity_log.append(m)

        if verbose and (ep + 1) % verbose_every == 0:
            recent = np.mean(episode_rewards[-verbose_every:])
            print(
                f"  [{agent.name}] Ep {ep+1:4d}/{n_episodes} | "
                f"Phase: {env.current_task:<22s} | "
                f"Avg reward (last {verbose_every}): {recent:.2f}"
            )

    result = {
        "agent_name":           agent.name,
        "episode_rewards":      episode_rewards,
        "task_ids":             task_ids,
        "task_switch_episodes": env.task_switch_episodes,
        "plasticity_log":       plasticity_log,
    }
    if hasattr(agent, "plasticity_history"):
        result["palr_plasticity_history"] = agent.plasticity_history
    return result


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------
def to_serialisable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serialisable(i) for i in obj]
    return obj


# ---------------------------------------------------------------------------
# Checkpoint save
# ---------------------------------------------------------------------------
def save_checkpoint(all_results, ckpt_suffix=""):
    fname     = f"jbw_checkpoint{ckpt_suffix}.json"
    ckpt_path = os.path.join(RESULTS_DIR, fname)
    with open(ckpt_path, "w") as f:
        json.dump(to_serialisable(all_results), f)
    print(f"  [checkpoint saved → {ckpt_path}]")


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------
def run_all(n_episodes: int, n_seeds: int, phase_episodes: int,
            steps_per_episode: int, seed_offset_start: int = 0,
            ckpt_suffix: str = "", video_every: int = 50):
    all_results = {}

    # Load existing checkpoint if resuming
    fname     = f"jbw_checkpoint{ckpt_suffix}.json"
    ckpt_path = os.path.join(RESULTS_DIR, fname)
    if seed_offset_start > 0 and os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            all_results = json.load(f)
        print(f"Loaded checkpoint: {sum(len(v) for v in all_results.values())} runs")

    for seed_offset in range(seed_offset_start, seed_offset_start + n_seeds):
        seed     = 42 + seed_offset * 13
        seed_tag = f"_seed{seed_offset}"
        print(f"\n{'='*65}")
        print(f"SEED {seed_offset+1}/{seed_offset_start + n_seeds}  (seed={seed})")
        print(f"{'='*65}")

        # Probe env to get obs_dim / n_actions
        probe_env = ContinualJBW(phase_episodes=phase_episodes,
                                 steps_per_episode=steps_per_episode,
                                 seed=seed)
        obs_dim   = probe_env.obs_dim
        n_actions = probe_env.n_actions
        probe_env.close()
        print(f"JBW: obs_dim={obs_dim}, n_actions={n_actions}")

        # Per-seed video directory
        vid_dir = os.path.join(VIDEOS_DIR, f"seed{seed_offset}")
        os.makedirs(vid_dir, exist_ok=True)

        agents = make_agents(obs_dim, n_actions, seed)

        for agent in agents:
            print(f"\n--- {agent.name} ---")
            env = ContinualJBW(phase_episodes=phase_episodes,
                               steps_per_episode=steps_per_episode,
                               seed=seed)
            result = train_agent_jbw(
                agent, env,
                n_episodes=n_episodes,
                verbose=True,
                verbose_every=max(1, n_episodes // 10),
                video_every=video_every,
                video_dir=vid_dir,
                seed_tag=seed_tag,
            )
            env.close()

            if agent.name not in all_results:
                all_results[agent.name] = []
            all_results[agent.name].append(result)

        save_checkpoint(all_results, ckpt_suffix)

    return all_results


# ---------------------------------------------------------------------------
# Final results save + summary
# ---------------------------------------------------------------------------
def save_results(all_results, n_episodes, phase_episodes):
    raw_path = os.path.join(RESULTS_DIR, "jbw_raw_results.json")
    with open(raw_path, "w") as f:
        json.dump(to_serialisable(all_results), f, indent=2)
    print(f"\nSaved raw results → {raw_path}")

    summary = {}
    for name, runs in all_results.items():
        all_rewards = np.array([r["episode_rewards"] for r in runs])
        switch_eps  = runs[0]["task_switch_episodes"]

        phase_rewards = {}
        phase_starts = [0] + switch_eps
        phase_ends   = switch_eps + [n_episodes]
        for i, (s, e) in enumerate(zip(phase_starts, phase_ends)):
            phase_rewards[f"phase_{i+1}"] = {
                "mean": float(all_rewards[:, s:e].mean()),
                "std":  float(all_rewards[:, s:e].std()),
            }

        max_r     = float(all_rewards.max())
        threshold = 0.5 * max_r if max_r > 0 else 1.0
        recovery_speeds = []
        for sw in switch_eps:
            for run in runs:
                found = False
                for j in range(sw, min(sw + phase_episodes, n_episodes)):
                    if run["episode_rewards"][j] >= threshold:
                        recovery_speeds.append(j - sw)
                        found = True
                        break
                if not found:
                    recovery_speeds.append(phase_episodes)

        summary[name] = {
            "n_runs":               int(len(runs)),
            "overall_mean_reward":  float(all_rewards.mean()),
            "overall_std_reward":   float(all_rewards.std()),
            "final_20ep_mean":      float(all_rewards[:, -20:].mean()),
            "final_20ep_std":       float(all_rewards[:, -20:].std()),
            "phase_rewards":        phase_rewards,
            "mean_recovery_speed":  float(np.mean(recovery_speeds)) if recovery_speeds else float(phase_episodes),
            "std_recovery_speed":   float(np.std(recovery_speeds))  if recovery_speeds else 0.0,
        }

    summary_path = os.path.join(RESULTS_DIR, "jbw_summary_stats.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary stats  → {summary_path}")
    return summary


def print_table(summary):
    print("\n" + "="*75)
    print("JBW RESULTS SUMMARY")
    print("="*75)
    header = f"{'Agent':<28} {'N':>4} {'Overall Mean':>14} {'Final-20ep':>12} {'Recovery':>12}"
    print(header)
    print("-"*75)
    for name, s in sorted(summary.items(),
                           key=lambda x: x[1]["overall_mean_reward"],
                           reverse=True):
        print(
            f"{name:<28} "
            f"{s['n_runs']:>4} "
            f"{s['overall_mean_reward']:>10.3f}±{s['overall_std_reward']:<5.3f} "
            f"{s['final_20ep_mean']:>8.3f}±{s['final_20ep_std']:<5.3f} "
            f"{s['mean_recovery_speed']:>8.1f}ep"
        )
    print("="*75)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",       type=int, default=200)
    parser.add_argument("--seeds",          type=int, default=5)
    parser.add_argument("--phase_episodes", type=int, default=50)
    parser.add_argument("--steps_per_ep",   type=int, default=500)
    parser.add_argument("--video_every",    type=int, default=50,
                        help="Record a video episode every N episodes")
    parser.add_argument("--fast", action="store_true",
                        help="Quick debug: 80 episodes, 1 seed, 20 ep/phase")
    parser.add_argument("--seed_offset",    type=int, default=0)
    parser.add_argument("--ckpt_suffix",    type=str, default="")
    args = parser.parse_args()

    if args.fast:
        n_ep     = 80
        n_seeds  = 1
        phase_ep = 20
        steps_ep = 200
    else:
        n_ep     = args.episodes
        n_seeds  = args.seeds
        phase_ep = args.phase_episodes
        steps_ep = args.steps_per_ep

    print(f"JBW Experiment")
    print(f"  episodes={n_ep}, seeds={n_seeds}, "
          f"phase_episodes={phase_ep}, steps_per_ep={steps_ep}")
    print(f"  video every {args.video_every} episodes → {VIDEOS_DIR}")
    print(f"  Results → {RESULTS_DIR}\n")

    all_results = run_all(
        n_ep, n_seeds, phase_ep, steps_ep,
        seed_offset_start=args.seed_offset,
        ckpt_suffix=args.ckpt_suffix,
        video_every=args.video_every,
    )
    summary = save_results(all_results, n_ep, phase_ep)
    print_table(summary)
    print("\nDone. Run plot_jbw_results.py to generate figures.")
