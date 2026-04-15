"""
PALR: Plasticity-Aware Learning Rate Agent  [OUR METHOD]
=========================================================
The core idea:

  1. Every `measure_freq` training steps, compute per-layer plasticity metrics
     (dead neuron fraction, effective rank of activations) on a diagnostic batch.

  2. Each layer gets an adaptive LR scale factor computed as:
        scale_i = 1 + beta * plasticity_deficit_i
     where plasticity_deficit_i measures how much plasticity has degraded
     relative to a running baseline (set at the start of each task phase).

  3. If mean dead neuron fraction exceeds `perturb_threshold`, apply a small
     targeted perturbation ONLY to dead-neuron units in that layer, re-
     initialising just those weights from their original initialisation
     distribution. This is more surgical than full shrink-and-perturb.

  4. Per-layer LR scaling is achieved by splitting the optimizer and applying
     separate gradient updates per layer group. In practice, we use gradient
     re-scaling (multiply grads before applying) as a lightweight approximation.

Ablation variants:
  - PALR-NoScale: only perturbation, no LR scaling
  - PALR-NoPerturb: only LR scaling, no perturbation

Connection to the research idea:
  This is a computationally cheap approximation of the bi-level meta-objective
  that would explicitly optimise step sizes to maximise NTK effective rank.
  The plasticity deficit IS the gradient signal for the outer loop -- we avoid
  full bi-level optimisation by making the outer step a simple closed-form rule.
"""

import numpy as np
import tensorflow as tf
from dqn_base import DQNAgent, build_qnet
from plasticity_metrics import compute_all_metrics, HIDDEN_LAYER_INDICES


class PALRAgent(DQNAgent):
    """
    Plasticity-Aware Learning Rate agent.

    Args:
        base_lr: Base Adam learning rate.
        beta: Sensitivity of LR scaling to plasticity deficit. Higher => more
              aggressive adaptation.
        measure_freq: Steps between plasticity measurements.
        perturb_threshold: Dead neuron fraction above which targeted perturbation
                           is triggered.
        perturb_sigma: Std dev of perturbation noise for dead neurons.
        dead_target: Target dead neuron fraction (ideally kept near this level).
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
        dead_target: float = 0.05,
        no_scale: bool = False,
        no_perturb: bool = False,
        **kwargs,
    ):
        super().__init__(*args, lr=base_lr, **kwargs)
        self.base_lr   = base_lr
        self.beta      = beta
        self.rank_beta = rank_beta
        self.measure_freq = measure_freq
        self.perturb_threshold = perturb_threshold
        self.perturb_sigma = perturb_sigma
        self.dead_target = dead_target
        self.no_scale  = no_scale
        self.no_perturb = no_perturb

        # Per-layer LR scale factors (start at 1.0)
        n_hidden = len(HIDDEN_LAYER_INDICES)
        self.lr_scales = np.ones(n_hidden, dtype=np.float32)

        # Ideal baselines: 0 dead neurons, max effective rank for 64-unit layer
        # Never reset to current state — always compare against the ideal.
        self.baseline_dead  = {i: 0.0  for i in range(n_hidden)}
        self.baseline_erank = {i: 32.0 for i in range(n_hidden)}  # conservative ideal
        self.plasticity_history = []  # (step, dead_layer0, dead_layer1, erank_l0, erank_l1, lr_scale_l0, lr_scale_l1)

        # Set name based on ablation
        if no_scale and no_perturb:
            self.name = "PALR-NoAdapt(ablation)"
        elif no_scale:
            self.name = "PALR-NoScale"
        elif no_perturb:
            self.name = "PALR-NoPerturb"
        else:
            self.name = "PALR (ours)"

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        # Periodically measure plasticity and update LR scales
        if self.step_count > 0 and self.step_count % self.measure_freq == 0:
            self._update_plasticity_state()

        loss = super().train_step()
        return loss

    def _update_plasticity_state(self):
        """Measure plasticity metrics and update per-layer LR scales."""
        if len(self.buffer) < self.batch_size:
            return

        # Diagnostic batch from replay buffer (not used for training)
        obs_batch, _, _, _, _ = self.buffer.sample(
            min(256, len(self.buffer))
        )
        metrics = compute_all_metrics(
            self.online_net, obs_batch, HIDDEN_LAYER_INDICES
        )

        new_scales = np.ones(len(HIDDEN_LAYER_INDICES), dtype=np.float32)
        for k, layer_idx in enumerate(HIDDEN_LAYER_INDICES):
            dead_k  = metrics.get(f"layer_{layer_idx}_dead", 0.0)
            erank_k = metrics.get(f"layer_{layer_idx}_erank", 1.0)

            # Dead-neuron deficit (vs ideal baseline of 0)
            dead_deficit = max(0.0, dead_k - self.baseline_dead[k])

            # Effective-rank deficit: how far below the ideal rank we are
            # Normalise by ideal so both signals are on [0, 1] scale
            rank_deficit = max(
                0.0,
                (self.baseline_erank[k] - erank_k) / max(self.baseline_erank[k], 1.0)
            )

            # Combined plasticity deficit drives LR boost
            if not self.no_scale:
                combined = dead_deficit + self.rank_beta * rank_deficit
                new_scales[k] = 1.0 + self.beta * combined
                new_scales[k] = float(np.clip(new_scales[k], 1.0, 5.0))

            # Targeted perturbation for heavily dead layers
            if (not self.no_perturb and
                    dead_k > self.perturb_threshold):
                self._targeted_perturbation(layer_idx, obs_batch)

        self.lr_scales = new_scales

        # Log
        self.plasticity_history.append({
            "step": self.step_count,
            "dead_l0": metrics.get(f"layer_{HIDDEN_LAYER_INDICES[0]}_dead", 0.0),
            "dead_l1": metrics.get(f"layer_{HIDDEN_LAYER_INDICES[1]}_dead", 0.0),
            "erank_l0": metrics.get(f"layer_{HIDDEN_LAYER_INDICES[0]}_erank", 1.0),
            "erank_l1": metrics.get(f"layer_{HIDDEN_LAYER_INDICES[1]}_erank", 1.0),
            "lr_scale_l0": float(self.lr_scales[0]),
            "lr_scale_l1": float(self.lr_scales[1]),
            "mean_dead": metrics.get("mean_dead", 0.0),
            "mean_erank": metrics.get("mean_erank", 1.0),
        })

    def _targeted_perturbation(self, layer_idx: int, obs_batch: np.ndarray):
        """
        Re-initialise ONLY the outgoing weights of dead neurons in a layer.
        This is more surgical than full shrink-and-perturb: healthy neurons
        are untouched, preserving accumulated knowledge.
        """
        from plasticity_metrics import collect_layer_activations
        acts = collect_layer_activations(
            self.online_net, obs_batch, [layer_idx]
        )[layer_idx]
        dead_mask = np.all(acts <= 0, axis=0)  # shape: (n_units,)
        if not np.any(dead_mask):
            return

        # Identify the weight matrix for this layer
        # Layer layout: Input -> Dense(128, relu)[L1] -> Dense(128, relu)[L2] -> Dense(2)[out]
        # online_net.layers[layer_idx] corresponds to the Dense layer at position layer_idx
        keras_layer = self.online_net.layers[layer_idx]
        weights = keras_layer.get_weights()  # [W, b]
        W, b = weights[0], weights[1]  # W: (in_features, out_features)

        # Re-initialise the input weights AND biases of dead neurons.
        # Biases must be reset too: a strongly-negative bias keeps a neuron
        # dead even after its input weights are re-initialised.
        fan_in = W.shape[0]
        std = np.sqrt(2.0 / fan_in)  # He init
        dead_indices = np.where(dead_mask)[0]
        noise = np.random.normal(0, self.perturb_sigma * std,
                                 size=(fan_in, len(dead_indices))).astype(W.dtype)
        W[:, dead_indices] = noise
        b[dead_indices] = 0.0  # reset bias so neuron is no longer trapped at ≤0
        keras_layer.set_weights([W, b])

    def _apply_gradients(self, grads):
        """
        Apply gradients with per-layer LR scaling.
        We scale gradients for each Dense layer according to self.lr_scales.
        """
        scaled_grads = []
        layer_var_map = self._get_layer_var_map()

        for grad, var in zip(grads, self.online_net.trainable_variables):
            if grad is None:
                scaled_grads.append(grad)
                continue
            scale = layer_var_map.get(var.name, 1.0)
            scaled_grads.append(grad * scale)

        self.optimizer.apply_gradients(
            zip(scaled_grads, self.online_net.trainable_variables)
        )

    def _get_layer_var_map(self):
        """Map variable names to their LR scale factors."""
        var_map = {}
        for k, layer_idx in enumerate(HIDDEN_LAYER_INDICES):
            keras_layer = self.online_net.layers[layer_idx]
            for var in keras_layer.trainable_variables:
                var_map[var.name] = float(self.lr_scales[k])
        return var_map

    def reset_plasticity_baseline(self):
        """
        Called at every task switch. Two actions:

        1. Keep the ideal baselines (0 dead neurons, target erank) — do NOT
           reset to the current (degraded) state.  Setting the baseline to the
           current dead fraction would neutralise the LR boost precisely when
           we need it most (right after a task switch).

        2. Proactively boost LR and apply targeted perturbation for any already-
           dead neurons so the agent immediately adapts to the new task dynamics.
        """
        if len(self.buffer) < self.batch_size:
            return

        obs_batch, _, _, _, _ = self.buffer.sample(
            min(256, len(self.buffer))
        )

        # Proactive: measure current state and act immediately at the switch
        for k, layer_idx in enumerate(HIDDEN_LAYER_INDICES):
            # Proactive LR boost: set scales to at least 2.0 at every switch
            # so the agent has headroom to re-learn the new task quickly.
            # The next _update_plasticity_state will fine-tune from there.
            if not self.no_scale:
                self.lr_scales[k] = max(self.lr_scales[k], 2.0)

            # Proactive perturbation: revive any accumulated dead neurons NOW,
            # rather than waiting for the next measure_freq boundary.
            if not self.no_perturb:
                self._targeted_perturbation(layer_idx, obs_batch)
