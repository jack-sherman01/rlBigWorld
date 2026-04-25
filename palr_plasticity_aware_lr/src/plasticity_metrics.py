"""
Plasticity Metrics  (PyTorch)
==============================
Lightweight measures of neural network plasticity adapted for online RL.

Metrics implemented:
  - dead_neuron_fraction: fraction of ReLU units with zero activation on a batch
  - effective_rank: entropy-based rank of the activation matrix (NTK-rank proxy)
  - weight_norm: mean L2 norm of weight matrices
  - gradient_norm: L2 norm of parameter gradients (proxy for gradient flow health)

All metrics are computed per-layer for fine-grained diagnosis.
"""

import numpy as np
import torch
import torch.nn as nn

# Indices into the list of ReLU hidden layers in our 2-hidden-layer network.
# 0 = first ReLU, 1 = second ReLU.
HIDDEN_LAYER_INDICES = [0, 1]


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


def weight_norm(model: nn.Module) -> float:
    """Mean L2 norm of all weight matrices (excludes bias vectors)."""
    norms = [
        p.data.norm(2).item()
        for p in model.parameters()
        if p.dim() >= 2
    ]
    return float(np.mean(norms)) if norms else 0.0


def collect_layer_activations(
    model: nn.Module,
    inputs: np.ndarray,
    hidden_layer_indices: list,
    extra_input: np.ndarray = None,
) -> dict:
    """
    Run a forward pass and return post-ReLU activations at specified hidden layers.

    Uses forward hooks so no model surgery is required.

    Args:
        model: nn.Module with a .net Sequential containing Linear/ReLU layers.
        inputs: batch of observations, shape (batch, obs_dim).
        hidden_layer_indices: which ReLU positions to probe (0-indexed over ReLU layers).
        extra_input: optional secondary input (e.g. actions for SAC critics),
                     shape (batch, action_dim). When provided, model is called as
                     model(inputs, extra_input).

    Returns:
        dict mapping relu_index -> activation array (batch, n_units).
    """
    acts: dict = {}
    hooks = []

    # Enumerate only ReLU layers and register hooks for requested indices.
    relu_counter = 0
    for layer in model.net:
        if isinstance(layer, nn.ReLU):
            if relu_counter in hidden_layer_indices:
                def _make_hook(idx):
                    def _hook(module, inp, output):
                        acts[idx] = output.detach().cpu().numpy()
                    return _hook
                hooks.append(layer.register_forward_hook(_make_hook(relu_counter)))
            relu_counter += 1

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(inputs).to(device)
        if extra_input is not None:
            a = torch.FloatTensor(extra_input).to(device)
            model(x, a)
        else:
            model(x)
    model.train()

    for h in hooks:
        h.remove()

    return acts


def compute_all_metrics(
    model: nn.Module,
    sample_batch: np.ndarray,
    hidden_layer_indices: list,
    extra_input: np.ndarray = None,
) -> dict:
    """
    Compute dead neuron fraction and effective rank for each hidden layer.

    Returns dict with keys:
      'layer_{i}_dead', 'layer_{i}_erank', 'mean_dead', 'mean_erank', 'weight_norm'

    extra_input: optional secondary input for critic models (SAC actions).
    """
    acts = collect_layer_activations(model, sample_batch, hidden_layer_indices,
                                     extra_input=extra_input)
    metrics = {}
    dead_vals, erank_vals = [], []
    for i, act in acts.items():
        dn = dead_neuron_fraction(act)
        er = effective_rank(act)
        metrics[f"layer_{i}_dead"]  = dn
        metrics[f"layer_{i}_erank"] = er
        dead_vals.append(dn)
        erank_vals.append(er)
    metrics["mean_dead"]   = float(np.mean(dead_vals))  if dead_vals  else 0.0
    metrics["mean_erank"]  = float(np.mean(erank_vals)) if erank_vals else 1.0
    metrics["weight_norm"] = weight_norm(model)
    return metrics
