"""
PALR: Plasticity-Aware Learning Rate Agent  (PyTorch)  [OUR METHOD]
=====================================================================
The core idea:

  1. Every `measure_freq` training steps, compute per-layer plasticity metrics
     (dead neuron fraction, effective rank of activations) on a diagnostic batch.

  2. Each layer gets an adaptive LR scale factor computed as:
        scale_i = 1 + beta * plasticity_deficit_i
     where plasticity_deficit_i measures how much plasticity has degraded
     relative to a running baseline (set at the start of each task phase).

  3. If mean dead neuron fraction exceeds `perturb_threshold`, apply a small
     targeted perturbation ONLY to dead-neuron units in that layer, re-
     initialising just those weights from He initialisation. This is more
     surgical than full shrink-and-perturb: healthy neurons are untouched.

  4. Per-layer LR scaling is applied by multiplying each hidden layer's
     gradients before the optimizer step -- a lightweight approximation of
     bi-level LR meta-learning at the cost of a single forward pass.

Ablation variants:
  - PALR-NoScale:   only perturbation, no LR scaling
  - PALR-NoPerturb: only LR scaling, no perturbation

Connection to the research idea:
  The plasticity deficit IS the gradient signal for the outer loop -- we avoid
  full bi-level optimisation by making the outer step a simple closed-form rule
  grounded in explicit plasticity measurements.
"""

import numpy as np
import torch

from dqn_base import DQNAgent, DEVICE
from plasticity_metrics import (
    compute_all_metrics,
    collect_layer_activations,
    HIDDEN_LAYER_INDICES,
)


class PALRAgent(DQNAgent):
    """
    Plasticity-Aware Learning Rate agent.

    Args:
        base_lr: Base Adam learning rate.
        beta: Sensitivity of LR scaling to plasticity deficit.
        rank_beta: Weight for the rank-deficit component.
        measure_freq: Steps between plasticity measurements.
        perturb_threshold: Dead neuron fraction above which targeted perturbation
                           is triggered.
        perturb_sigma: Scale factor on He-init std for perturbation noise.
        no_scale: If True, disable LR scaling (ablation).
        no_perturb: If True, disable targeted perturbation (ablation).
    """

    def __init__(
        self,
        *args,
        base_lr: float = 1e-3,
        beta: float = 3.0,
        rank_beta: float = 1.5,
        measure_freq: int = 200,
        perturb_threshold: float = 0.10,
        perturb_sigma: float = 0.3,
        no_scale: bool = False,
        no_perturb: bool = False,
        **kwargs,
    ):
        super().__init__(*args, lr=base_lr, **kwargs)
        self.base_lr           = base_lr
        self.beta              = beta
        self.rank_beta         = rank_beta
        self.measure_freq      = measure_freq
        self.perturb_threshold = perturb_threshold
        self.perturb_sigma     = perturb_sigma
        self.no_scale          = no_scale
        self.no_perturb        = no_perturb

        n_hidden = len(HIDDEN_LAYER_INDICES)

        # Per-layer LR scale factors (start at 1.0)
        self.lr_scales = np.ones(n_hidden, dtype=np.float32)

        # Ideal baselines: 0 dead, conservative target rank.
        # Never reset to current (degraded) state — always compare against ideal.
        self.baseline_dead  = {i: 0.0  for i in range(n_hidden)}
        self.baseline_erank = {i: 32.0 for i in range(n_hidden)}

        self.plasticity_history: list = []

        # Re-death tracking: indices of neurons revived at the last perturbation.
        self.revived_indices: dict = {i: np.array([], dtype=int)
                                      for i in range(n_hidden)}

        if no_scale and no_perturb:
            self.name = "PALR-NoAdapt(ablation)"
        elif no_scale:
            self.name = "PALR-NoScale"
        elif no_perturb:
            self.name = "PALR-NoPerturb"
        else:
            self.name = "PALR (ours)"

    # ------------------------------------------------------------------
    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        if self.step_count > 0 and self.step_count % self.measure_freq == 0:
            self._update_plasticity_state()

        return super().train_step()

    # ------------------------------------------------------------------
    def _update_plasticity_state(self):
        """Measure plasticity metrics and update per-layer LR scales."""
        if len(self.buffer) < self.batch_size:
            return

        obs_batch, _, _, _, _ = self.buffer.sample(min(256, len(self.buffer)))
        metrics = compute_all_metrics(self.online_net, obs_batch, HIDDEN_LAYER_INDICES)

        new_scales     = np.ones(len(HIDDEN_LAYER_INDICES), dtype=np.float32)
        redeath_rates  = {}

        # Collect activations once for re-death checking
        raw_acts = collect_layer_activations(
            self.online_net, obs_batch, HIDDEN_LAYER_INDICES
        )

        for k, layer_idx in enumerate(HIDDEN_LAYER_INDICES):
            dead_k  = metrics.get(f"layer_{layer_idx}_dead", 0.0)
            erank_k = metrics.get(f"layer_{layer_idx}_erank", 1.0)

            # Re-death rate: fraction of previously-revived neurons that are
            # dead again at this measurement step.
            revived = self.revived_indices[k]
            if len(revived) > 0:
                act_k = raw_acts[layer_idx]                       # (batch, n_units)
                still_dead = np.all(act_k[:, revived] <= 0, axis=0)
                redeath_rates[k] = float(still_dead.mean())
            else:
                redeath_rates[k] = 0.0

            # Dead-neuron deficit (vs ideal baseline of 0)
            dead_deficit = max(0.0, dead_k - self.baseline_dead[k])

            # Effective-rank deficit, normalised to [0, 1]
            rank_deficit = max(
                0.0,
                (self.baseline_erank[k] - erank_k) / max(self.baseline_erank[k], 1.0)
            )

            if not self.no_scale:
                combined = dead_deficit + self.rank_beta * rank_deficit
                # Normalise combined to [0, 1] (max possible = 1 + rank_beta),
                # then linearly map to [1, max_lr_scale].  This preserves
                # per-layer differentiation: scale only saturates at max
                # when dead_deficit = 1.0 AND rank_deficit = 1.0 (fully dead
                # and fully rank-collapsed simultaneously).
                # Note: self.beta is NOT applied here -- using it as an extra
                # multiplier on already-normalised [0,1] value re-introduces
                # premature saturation (the original bug).
                combined_norm = combined / (1.0 + self.rank_beta)
                max_lr_scale = 5.0  # same hard-coded ceiling as before
                new_scales[k] = float(
                    1.0 + (max_lr_scale - 1.0)
                    * float(np.clip(combined_norm, 0.0, 1.0))
                )

            if not self.no_perturb and dead_k > self.perturb_threshold:
                self._targeted_perturbation(k, obs_batch)

        self.lr_scales = new_scales

        self.plasticity_history.append({
            "step":            self.step_count,
            "dead_l0":         metrics.get(f"layer_{HIDDEN_LAYER_INDICES[0]}_dead",  0.0),
            "dead_l1":         metrics.get(f"layer_{HIDDEN_LAYER_INDICES[1]}_dead",  0.0),
            "erank_l0":        metrics.get(f"layer_{HIDDEN_LAYER_INDICES[0]}_erank", 1.0),
            "erank_l1":        metrics.get(f"layer_{HIDDEN_LAYER_INDICES[1]}_erank", 1.0),
            "lr_scale_l0":     float(self.lr_scales[0]),
            "lr_scale_l1":     float(self.lr_scales[1]),
            "redeath_rate_l0": redeath_rates.get(0, 0.0),
            "redeath_rate_l1": redeath_rates.get(1, 0.0),
            "mean_dead":       metrics.get("mean_dead",  0.0),
            "mean_erank":      metrics.get("mean_erank", 1.0),
        })

    # ------------------------------------------------------------------
    def _targeted_perturbation(self, k: int, obs_batch: np.ndarray):
        """
        Re-initialise ONLY the input weights and biases of dead neurons in
        hidden layer k. Healthy neurons are untouched.

        In the QNet architecture:
            net[2*k]     = Linear (k-th hidden linear layer)
            net[2*k + 1] = ReLU

        PyTorch weight layout: weight.shape = (out_features, in_features).
        Dead neurons correspond to rows in out_features dimension.

        Records revived neuron indices for re-death rate computation at
        the next _update_plasticity_state call.
        """
        layer_idx = HIDDEN_LAYER_INDICES[k]
        acts = collect_layer_activations(self.online_net, obs_batch, [layer_idx])
        act_k = acts[layer_idx]                   # (batch, n_units)
        dead_mask = np.all(act_k <= 0, axis=0)   # (n_units,)
        if not np.any(dead_mask):
            return

        dead_indices = np.where(dead_mask)[0]

        # Access the k-th hidden Linear layer: net[0], net[2], ... for 2-layer net
        linear = self.online_net.net[2 * k]       # nn.Linear
        fan_in = linear.weight.data.shape[1]
        std    = np.sqrt(2.0 / fan_in) * self.perturb_sigma  # He init, scaled

        with torch.no_grad():
            noise = torch.randn(len(dead_indices), fan_in,
                                device=DEVICE, dtype=linear.weight.dtype) * std
            linear.weight.data[dead_indices, :] = noise
            linear.bias.data[dead_indices] = 0.0   # reset bias so neuron escapes ≤0

        self.revived_indices[k] = dead_indices

    # ------------------------------------------------------------------
    def _apply_gradients(self):
        """
        Apply gradients with per-layer LR scaling.
        Scale the gradients for each hidden Linear layer according to lr_scales,
        then call the base optimizer step.
        """
        for k in range(len(HIDDEN_LAYER_INDICES)):
            scale  = float(self.lr_scales[k])
            linear = self.online_net.net[2 * k]
            if linear.weight.grad is not None:
                linear.weight.grad.mul_(scale)
            if linear.bias.grad is not None:
                linear.bias.grad.mul_(scale)

        self.optimizer.step()

    # ------------------------------------------------------------------
    def reset_plasticity_baseline(self):
        """
        Called at every detected task switch. Two proactive actions:

        1. Keep ideal baselines (0 dead, target rank) — do NOT reset to the
           current (degraded) state. Resetting to the degraded state would
           neutralise the LR boost precisely when we need it most.

        2. Proactively boost LR and apply targeted perturbation for any
           already-dead neurons so the agent immediately adapts to the new
           task dynamics.
        """
        if len(self.buffer) < self.batch_size:
            return

        obs_batch, _, _, _, _ = self.buffer.sample(min(256, len(self.buffer)))

        for k in range(len(HIDDEN_LAYER_INDICES)):
            if not self.no_scale:
                # Proactive LR boost: at least 2.0 at each switch.
                self.lr_scales[k] = max(self.lr_scales[k], 2.0)

            if not self.no_perturb:
                # Proactive perturbation: revive dead neurons immediately.
                self._targeted_perturbation(k, obs_batch)
