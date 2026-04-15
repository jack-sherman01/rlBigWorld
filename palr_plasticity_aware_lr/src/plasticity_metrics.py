"""
Plasticity Metrics
==================
Lightweight measures of neural network plasticity adapted for online RL.

Metrics implemented:
  - dead_neuron_fraction: fraction of ReLU units with zero activation on a batch
  - effective_rank: numerical rank of the activation matrix (approximates NTK rank)
  - weight_norm: L2 norm of all weights (proxy for weight magnitude explosion)
  - gradient_norm: L2 norm of parameter gradients (proxy for gradient flow health)

All metrics are computed per-layer for fine-grained diagnosis.
"""

import numpy as np
import tensorflow as tf

# Layer indices of the ReLU hidden layers in our 2-hidden-layer network
# Input -> Dense(relu)[1] -> Dense(relu)[2] -> Dense(linear)[3]
HIDDEN_LAYER_INDICES = [1, 2]


def dead_neuron_fraction(activations: np.ndarray) -> float:
    """
    Fraction of neurons that are dead (output <= 0 for all samples in batch).

    Args:
        activations: shape (batch, n_units) -- post-ReLU activations.

    Returns:
        Fraction in [0, 1]. Higher means more dead neurons (worse plasticity).
    """
    if activations.ndim == 1:
        activations = activations[np.newaxis, :]
    dead = np.all(activations <= 0, axis=0)
    return float(dead.mean())


def effective_rank(activations: np.ndarray, eps: float = 1e-6) -> float:
    """
    Effective rank of the activation matrix via entropy of normalised singular values.
    Roy & Vetterli (2007): erank(A) = exp(H(sigma/||sigma||_1)).

    Higher effective rank => more diverse feature directions => better plasticity.

    Args:
        activations: shape (batch, n_units).
        eps: small constant for numerical stability.

    Returns:
        Effective rank in [1, n_units].
    """
    if activations.ndim == 1 or activations.shape[0] < 2:
        return 1.0
    # Normalise rows (zero-mean per sample)
    A = activations - activations.mean(axis=0, keepdims=True)
    try:
        sv = np.linalg.svd(A, compute_uv=False)
    except np.linalg.LinAlgError:
        return 1.0
    sv = sv[sv > eps]
    if len(sv) == 0:
        return 1.0
    p = sv / sv.sum()
    entropy = -np.sum(p * np.log(p + eps))
    return float(np.exp(entropy))


def weight_norm(model: tf.keras.Model) -> float:
    """Mean L2 norm of all trainable weight tensors."""
    norms = [
        float(tf.norm(w).numpy())
        for w in model.trainable_variables
        if len(w.shape) >= 2  # exclude bias vectors
    ]
    return float(np.mean(norms)) if norms else 0.0


# Module-level probe cache: model_id -> probe Keras model.
# Probes share weights with the parent model so they always reflect current
# weights without needing to be recreated after each gradient update.
_probe_cache: dict = {}


def _get_probe(model: tf.keras.Model, hidden_layer_indices: tuple):
    """Return (and cache) a probe model that outputs the requested layers."""
    key = (id(model), hidden_layer_indices)
    if key not in _probe_cache:
        outputs = [model.layers[i].output for i in hidden_layer_indices]
        _probe_cache[key] = tf.keras.Model(inputs=model.inputs, outputs=outputs)
    return _probe_cache[key]


def collect_layer_activations(
    model: tf.keras.Model,
    inputs: np.ndarray,
    hidden_layer_indices: list
) -> dict:
    """
    Run a forward pass and return activations at specified hidden layers.

    Args:
        model: Keras model.
        inputs: batch of observations, shape (batch, obs_dim).
        hidden_layer_indices: list of layer indices to probe.

    Returns:
        dict mapping layer_index -> activation array (batch, n_units).
    """
    probe = _get_probe(model, tuple(hidden_layer_indices))
    acts = probe(inputs, training=False)
    if not isinstance(acts, (list, tuple)):
        acts = [acts]
    return {
        idx: a.numpy()
        for idx, a in zip(hidden_layer_indices, acts)
    }


def compute_all_metrics(
    model: tf.keras.Model,
    sample_batch: np.ndarray,
    hidden_layer_indices: list
) -> dict:
    """
    Compute dead neuron fraction and effective rank for each hidden layer.

    Returns dict with keys:
      'layer_{i}_dead', 'layer_{i}_erank', 'mean_dead', 'mean_erank', 'weight_norm'
    """
    acts = collect_layer_activations(model, sample_batch, hidden_layer_indices)
    metrics = {}
    dead_vals, erank_vals = [], []
    for i, act in acts.items():
        dn = dead_neuron_fraction(act)
        er = effective_rank(act)
        metrics[f"layer_{i}_dead"]  = dn
        metrics[f"layer_{i}_erank"] = er
        dead_vals.append(dn)
        erank_vals.append(er)
    metrics["mean_dead"]    = float(np.mean(dead_vals)) if dead_vals else 0.0
    metrics["mean_erank"]   = float(np.mean(erank_vals)) if erank_vals else 1.0
    metrics["weight_norm"]  = weight_norm(model)
    return metrics
