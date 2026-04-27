"""
PALR-SAC: Plasticity-Aware Learning Rates for SAC  (PyTorch)  [OUR METHOD]
============================================================================
Extends SACAgent with PALR's two mechanisms applied to the critic networks:

  1. Per-layer LR scaling — gradient scaling on critic hidden layers
     proportional to the plasticity deficit (dead-fraction + rank-deficit).

  2. Targeted neuron-level perturbation — re-initialise ONLY the dead
     neurons in critic1 and critic2 when the dead fraction exceeds threshold.

PALR is applied to the CRITIC (not the actor) because:
  - Critic represents the value landscape; plasticity loss here prevents
    the agent from re-evaluating task-relevant features.
  - The actor's policy gradient benefits from a healthy critic's Q-values;
    fixing the critic indirectly improves actor learning.
  - Per-layer LR visualisation of the 4-layer critic gives the multi-level
    LR dynamic figure requested for the paper.

Ablation variants (same as CartPole experiments):
  - PALR-SAC-NoScale:   perturbation only, no LR scaling
  - PALR-SAC-NoPerturb: LR scaling only, no perturbation

The 4-layer critic (hidden_sizes=(256,256,256,256)) lets us visualise
LR scales for layers 0-3 — deeper layers typically accumulate more dead
neurons and receive larger LR boosts.
"""

import numpy as np
import torch

from sac_base import SACAgent, DEVICE
from plasticity_metrics import (
    compute_all_metrics,
    collect_layer_activations,
)

# Monitor all 4 hidden ReLU layers (indices 0-3) of the critic trunk
CW_HIDDEN_INDICES = [0, 1, 2, 3]


class PALRSACAgent(SACAgent):
    """
    PALR applied to SAC's twin critics.

    Args:
        base_lr:           Base Adam LR (shared for actor, critic, alpha).
        beta:              Dead-fraction sensitivity for LR scaling.
        rank_beta:         Rank-deficit sensitivity for LR scaling.
        measure_freq:      Critic update steps between plasticity measurements.
        perturb_threshold: Dead-fraction threshold above which perturbation fires.
        perturb_sigma:     Scale on He-init std for perturbation noise.
        no_scale:          Disable LR scaling (ablation).
        no_perturb:        Disable targeted perturbation (ablation).
    """

    def __init__(
        self,
        *args,
        base_lr: float = 3e-4,
        beta: float = 3.0,
        rank_beta: float = 1.5,
        measure_freq: int = 500,
        perturb_threshold: float = 0.10,
        perturb_sigma: float = 0.3,
        no_scale: bool = False,
        no_perturb: bool = False,
        **kwargs,
    ):
        kwargs.pop("lr", None)   # base_lr takes precedence over any lr in **kwargs
        super().__init__(*args, lr=base_lr, **kwargs)
        self.base_lr           = base_lr
        self.beta              = beta
        self.rank_beta         = rank_beta
        self.measure_freq      = measure_freq
        self.perturb_threshold = perturb_threshold
        self.perturb_sigma     = perturb_sigma
        self.no_scale          = no_scale
        self.no_perturb        = no_perturb

        n_hidden = len(CW_HIDDEN_INDICES)
        self.lr_scales = np.ones(n_hidden, dtype=np.float32)

        # Ideal baselines: 0 dead, conservative target rank.
        self.baseline_dead  = {i: 0.0  for i in range(n_hidden)}
        self.baseline_erank = {i: 64.0 for i in range(n_hidden)}  # larger for 256-unit layers

        self.plasticity_history: list = []
        self.revived_indices: dict    = {i: np.array([], dtype=int)
                                         for i in range(n_hidden)}

        if no_scale and no_perturb:
            self.name = "PALR-SAC-NoAdapt"
        elif no_scale:
            self.name = "PALR-SAC-NoScale"
        elif no_perturb:
            self.name = "PALR-SAC-NoPerturb"
        else:
            self.name = "PALR-SAC (ours)"

    # ── Override train_step to inject measurement ─────────────────────────────

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None
        if self.step_count > 0 and self.step_count % self.measure_freq == 0:
            self._update_plasticity_state()
        return super().train_step()

    # ── Plasticity measurement ────────────────────────────────────────────────

    def _update_plasticity_state(self):
        if len(self.buffer) < self.batch_size:
            return

        obs_b, act_b, _, _, _ = self.buffer.sample(min(512, len(self.buffer)))

        # Use critic1 as the reference network for metrics
        metrics = compute_all_metrics(
            self.critic1, obs_b, CW_HIDDEN_INDICES, extra_input=act_b
        )
        raw_acts = collect_layer_activations(
            self.critic1, obs_b, CW_HIDDEN_INDICES, extra_input=act_b
        )

        new_scales    = np.ones(len(CW_HIDDEN_INDICES), dtype=np.float32)
        redeath_rates = {}

        for k, layer_idx in enumerate(CW_HIDDEN_INDICES):
            dead_k  = metrics.get(f"layer_{layer_idx}_dead", 0.0)
            erank_k = metrics.get(f"layer_{layer_idx}_erank", 1.0)

            # Re-death rate
            revived = self.revived_indices[k]
            if len(revived) > 0:
                act_k = raw_acts[layer_idx]
                still_dead = np.all(act_k[:, revived] <= 0, axis=0)
                redeath_rates[k] = float(still_dead.mean())
            else:
                redeath_rates[k] = 0.0

            dead_deficit = max(0.0, dead_k - self.baseline_dead[k])
            rank_deficit = max(
                0.0,
                (self.baseline_erank[k] - erank_k) / max(self.baseline_erank[k], 1.0)
            )

            if not self.no_scale:
                combined = dead_deficit + self.rank_beta * rank_deficit
                # Normalise combined to [0, 1] then linearly map to
                # [1, max_lr_scale].  Saturates only when fully degraded.
                combined_norm = combined / (1.0 + self.rank_beta)
                max_lr_scale = 5.0  # same hard-coded ceiling as before
                new_scales[k] = float(
                    1.0 + (max_lr_scale - 1.0)
                    * float(np.clip(combined_norm, 0.0, 1.0))
                )

            if not self.no_perturb and dead_k > self.perturb_threshold:
                self._targeted_perturbation(k, obs_b, act_b)

        self.lr_scales = new_scales

        # Build history entry with all 4 layers
        entry = {
            "step":       self.step_count,
            "mean_dead":  metrics.get("mean_dead",  0.0),
            "mean_erank": metrics.get("mean_erank", 1.0),
        }
        for k, li in enumerate(CW_HIDDEN_INDICES):
            entry[f"dead_l{k}"]         = metrics.get(f"layer_{li}_dead",  0.0)
            entry[f"erank_l{k}"]        = metrics.get(f"layer_{li}_erank", 1.0)
            entry[f"lr_scale_l{k}"]     = float(self.lr_scales[k])
            entry[f"redeath_rate_l{k}"] = redeath_rates.get(k, 0.0)
        self.plasticity_history.append(entry)

    # ── Targeted perturbation ────────────────────────────────────────────────

    def _targeted_perturbation(self, k: int, obs_b: np.ndarray, act_b: np.ndarray):
        """
        Re-initialise dead neurons in the k-th hidden Linear of BOTH critics.
        The k-th hidden Linear is at net[2*k] in the Sequential trunk.
        PyTorch weight shape: (out_features=n_units, in_features).
        Dead neurons correspond to rows (out_features dimension).
        """
        layer_idx = CW_HIDDEN_INDICES[k]
        acts = collect_layer_activations(
            self.critic1, obs_b, [layer_idx], extra_input=act_b
        )
        act_k     = acts[layer_idx]
        dead_mask = np.all(act_k <= 0, axis=0)
        if not np.any(dead_mask):
            return

        dead_indices = np.where(dead_mask)[0]

        for critic in (self.critic1, self.critic2):
            linear = critic.net[2 * k]
            fan_in = linear.weight.data.shape[1]
            std    = np.sqrt(2.0 / fan_in) * self.perturb_sigma
            with torch.no_grad():
                noise = torch.randn(len(dead_indices), fan_in,
                                    device=DEVICE,
                                    dtype=linear.weight.dtype) * std
                linear.weight.data[dead_indices, :] = noise
                linear.bias.data[dead_indices]       = 0.0

        self.revived_indices[k] = dead_indices

    # ── Per-layer gradient scaling on critics ─────────────────────────────────

    def _scale_critic_grads(self, critic):
        """Scale gradients of each hidden Linear in the critic trunk."""
        for k in range(len(CW_HIDDEN_INDICES)):
            scale  = float(self.lr_scales[k])
            linear = critic.net[2 * k]
            if linear.weight.grad is not None:
                linear.weight.grad.mul_(scale)
            if linear.bias.grad is not None:
                linear.bias.grad.mul_(scale)

    def _apply_critic1_gradients(self):
        if not self.no_scale:
            self._scale_critic_grads(self.critic1)
        self.critic1_optim.step()

    def _apply_critic2_gradients(self):
        if not self.no_scale:
            self._scale_critic_grads(self.critic2)
        self.critic2_optim.step()

    # ── Task-switch hook ──────────────────────────────────────────────────────

    def reset_plasticity_baseline(self):
        """
        Called at each detected task switch. Proactively boosts LR and
        perturbs dead neurons before the first gradient step on the new task.
        """
        if len(self.buffer) < self.batch_size:
            return

        obs_b, act_b, _, _, _ = self.buffer.sample(min(512, len(self.buffer)))

        for k in range(len(CW_HIDDEN_INDICES)):
            if not self.no_scale:
                self.lr_scales[k] = max(self.lr_scales[k], 2.0)
            if not self.no_perturb:
                self._targeted_perturbation(k, obs_b, act_b)
