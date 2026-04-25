"""
Training Loop
=============
Runs a single agent on the Continual CartPole environment and returns
a structured results dictionary. Used by the main experiment runner.
"""

import sys
import os
import numpy as np

# Make src importable
sys.path.insert(0, os.path.dirname(__file__))

from continual_cartpole import ContinualCartPole
from plasticity_metrics import compute_all_metrics, HIDDEN_LAYER_INDICES


def train_agent(
    agent,
    n_episodes: int = 1200,
    episodes_per_task: int = 200,
    train_every: int = 1,
    measure_plasticity_every: int = 10,   # episodes
    seed: int = 42,
    verbose: bool = True,
    verbose_every: int = 100,
) -> dict:
    """
    Run one agent on Continual CartPole for n_episodes.

    Returns a dict with:
      - episode_rewards: list of per-episode returns
      - task_ids: list of task index at each episode
      - task_switch_episodes: list of episode numbers where task switched
      - plasticity_log: list of dicts with plasticity metrics
      - agent_name: str
    """
    env = ContinualCartPole(
        episodes_per_task=episodes_per_task,
        seed=seed,
    )

    episode_rewards   = []
    task_ids          = []
    plasticity_log    = []
    last_task_idx     = 0

    for ep in range(n_episodes):
        obs  = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)
            agent.push(obs, action, reward, next_obs, done)

            for _ in range(train_every):
                agent.train_step()

            obs = next_obs
            ep_reward += reward

        agent.on_episode_end(ep_reward)
        episode_rewards.append(ep_reward)
        task_ids.append(env.task_idx % 4)

        # Detect task switch and notify PALR agent to reset baseline
        if env.task_idx != last_task_idx:
            if hasattr(agent, "reset_plasticity_baseline"):
                agent.reset_plasticity_baseline()
            last_task_idx = env.task_idx

        # Periodically measure plasticity for logging
        if ep % measure_plasticity_every == 0 and len(agent.buffer) >= agent.batch_size:
            obs_batch, _, _, _, _ = agent.buffer.sample(
                min(256, len(agent.buffer))
            )
            m = compute_all_metrics(
                agent.online_net, obs_batch, HIDDEN_LAYER_INDICES
            )
            m["episode"] = ep
            m["task_id"] = task_ids[-1]
            m["ep_reward"] = ep_reward
            plasticity_log.append(m)

        if verbose and (ep + 1) % verbose_every == 0:
            recent = np.mean(episode_rewards[-verbose_every:])
            print(
                f"  [{agent.name}] Ep {ep+1:4d}/{n_episodes} | "
                f"Task: {env.current_task:<15s} | "
                f"Avg reward (last {verbose_every}): {recent:.1f}"
            )

    env.close()

    result = {
        "agent_name":           agent.name,
        "episode_rewards":      episode_rewards,
        "task_ids":             task_ids,
        "task_switch_episodes": env.task_switch_episodes,
        "plasticity_log":       plasticity_log,
    }

    # Export per-measurement-step plasticity history from PALR agents.
    # This includes lr_scale_l0/l1, erank_l0/l1, redeath_rate_l0/l1 — used
    # for the LR-scale control-loop figure and the re-death rate figure.
    if hasattr(agent, "plasticity_history"):
        result["palr_plasticity_history"] = agent.plasticity_history

    return result
