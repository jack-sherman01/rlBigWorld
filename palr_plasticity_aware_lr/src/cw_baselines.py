"""
SAC Baseline Agents for CW10  (PyTorch)
=========================================
Five standard continual-RL baselines, each built on top of SACAgent:

1. SACShinkAndPerturbAgent  — shrink-and-perturb every K steps (Ash & Adams 2020)
2. SACPeriodicResetAgent    — full network reset every K episodes
3. SACL2RegAgent            — L2 weight regularisation on critic loss
4. CBPSACAgent              — Continual Backprop (Dohare et al. 2024): continuous,
                              utility-based replacement of low-utility hidden units.
                              Task-agnostic (uses no task-switch oracle).
5. SACReDo                  — ReDo-style dormant-neuron recycling (Sokar et al. 2023):
                              identifies dormant neurons by low mean activation and
                              re-initialises them on a fixed schedule.
                              Task-agnostic (uses no task-switch oracle).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sac_base import SACAgent, ActorNet, CriticNet, DEVICE


class SACShinkAndPerturbAgent(SACAgent):
    """
    Shrink-and-Perturb on both actor and critics.
    Every `perturb_freq` steps: w <- alpha * w + noise(0, sigma).
    """

    def __init__(self, *args, perturb_freq: int = 5_000,
                 shrink_alpha: float = 0.9, sigma: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.perturb_freq  = perturb_freq
        self.shrink_alpha  = shrink_alpha
        self.sigma         = sigma
        self.name          = "SAC-ShrinkAndPerturb"

    def train_step(self):
        loss = super().train_step()
        if self.step_count > 0 and self.step_count % self.perturb_freq == 0:
            self._shrink_and_perturb()
        return loss

    def _shrink_and_perturb(self):
        for net in (self.actor, self.critic1, self.critic2):
            with torch.no_grad():
                for p in net.parameters():
                    noise = torch.randn_like(p) * self.sigma
                    p.mul_(self.shrink_alpha).add_(noise)
        # Sync target critics after perturbation
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())


class SACPeriodicResetAgent(SACAgent):
    """
    Periodic network reset every `reset_freq` episodes.
    Replay buffer is preserved — only weights are reinitialised.
    """

    def __init__(self, *args, reset_freq: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_freq = reset_freq
        self.name       = "SAC-PeriodicReset"

    def on_episode_end(self, episode_reward: float):
        super().on_episode_end(episode_reward)
        if self.episode_count % self.reset_freq == 0:
            self._reset_networks()

    def _reset_networks(self):
        self.actor    = ActorNet(self.obs_dim, self.action_dim,
                                 self.hidden_sizes).to(DEVICE)
        self.critic1  = CriticNet(self.obs_dim, self.action_dim,
                                  self.hidden_sizes).to(DEVICE)
        self.critic2  = CriticNet(self.obs_dim, self.action_dim,
                                  self.hidden_sizes).to(DEVICE)
        self.target_critic1 = CriticNet(self.obs_dim, self.action_dim,
                                        self.hidden_sizes).to(DEVICE)
        self.target_critic2 = CriticNet(self.obs_dim, self.action_dim,
                                        self.hidden_sizes).to(DEVICE)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.target_critic1.eval()
        self.target_critic2.eval()
        # Recreate optimisers with fresh state
        self.actor_optim   = optim.Adam(self.actor.parameters(),   lr=self.lr)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=self.lr)


class SACL2RegAgent(SACAgent):
    """
    L2 regularisation on critic weights.
    Adds l2_coeff * ||W||² to the critic loss to prevent weight-norm explosion.
    """

    def __init__(self, *args, l2_coeff: float = 1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.l2_coeff = l2_coeff
        self.name     = "SAC-L2Reg"

    def _update_critic(self, obs, actions, rewards, next_obs, dones) -> float:
        obs_t      = torch.FloatTensor(obs).to(DEVICE)
        actions_t  = torch.FloatTensor(actions).to(DEVICE)
        rewards_t  = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        next_obs_t = torch.FloatTensor(next_obs).to(DEVICE)
        dones_t    = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor.sample(next_obs_t)
            q1_next = self.target_critic1(next_obs_t, next_action)
            q2_next = self.target_critic2(next_obs_t, next_action)
            q_next  = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            target_q = rewards_t + self.gamma * (1.0 - dones_t) * q_next

        q1 = self.critic1(obs_t, actions_t)
        q2 = self.critic2(obs_t, actions_t)
        td1 = F.mse_loss(q1, target_q)
        td2 = F.mse_loss(q2, target_q)

        l2_c1 = sum(p.pow(2).sum() for p in self.critic1.parameters() if p.dim() >= 2)
        l2_c2 = sum(p.pow(2).sum() for p in self.critic2.parameters() if p.dim() >= 2)

        self.critic1_optim.zero_grad()
        (td1 + self.l2_coeff * l2_c1).backward()
        self._apply_critic1_gradients()

        self.critic2_optim.zero_grad()
        (td2 + self.l2_coeff * l2_c2).backward()
        self._apply_critic2_gradients()

        return float((td1 + td2).item() / 2)


class CBPSACAgent(SACAgent):
    """
    Continual Backprop (CBP) baseline — Dohare et al., "Loss of plasticity in
    deep continual learning", Nature 2024.

    Generate-and-test: at every step a tiny fraction of mature hidden units with
    the lowest *contribution utility* is re-initialised. For each replaced unit,
    the incoming weights are re-sampled from the initialisation distribution, the
    outgoing weights are zeroed (so the fresh feature does not disturb the current
    function), and its utility/age are reset. Applied here to BOTH critics — the
    same networks PALR intervenes on — for a fair comparison.

    Unlike PALR, CBP is fully task-agnostic: it uses no task-switch signal, which
    is why we deliberately do NOT expose `reset_plasticity_baseline`.

    Hyperparameters follow the CBP paper defaults:
        replacement_rate : per-step fraction of mature units replaced (1e-4)
        decay_rate       : running-average decay for contribution utility (0.99)
        maturity_thresh  : min age (in CBP steps) before a unit may be replaced (100)
    """

    def __init__(self, *args,
                 replacement_rate: float = 1e-4,
                 decay_rate: float = 0.99,
                 maturity_thresh: int = 100,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.replacement_rate = replacement_rate
        self.decay_rate       = decay_rate
        self.maturity_thresh  = maturity_thresh
        self.name             = "CBP-SAC"

        self.n_hidden = len(self.hidden_sizes)
        # Per-critic, per-hidden-layer generate-and-test state.
        self._cbp_state = {
            cid: [
                {
                    "util":  torch.zeros(self.hidden_sizes[k], device=DEVICE),
                    "age":   torch.zeros(self.hidden_sizes[k], device=DEVICE),
                    "accum": 0.0,
                }
                for k in range(self.n_hidden)
            ]
            for cid in (1, 2)
        }

    # ── CBP hook: run generate-and-test after every gradient update ────────────

    def train_step(self):
        loss = super().train_step()
        if loss is not None:                       # an update actually happened
            self._cbp_step()
        return loss

    def _cbp_step(self):
        bs = min(256, len(self.buffer))
        obs, actions, _, _, _ = self.buffer.sample(bs)
        obs_t = torch.FloatTensor(obs).to(DEVICE)
        act_t = torch.FloatTensor(actions).to(DEVICE)
        self._cbp_update(self.critic1, 1, self.critic1_optim, obs_t, act_t)
        self._cbp_update(self.critic2, 2, self.critic2_optim, obs_t, act_t)

    @staticmethod
    def _forward_hidden(critic, obs_t, act_t):
        """Forward pass returning the post-ReLU activation of each hidden layer."""
        x = torch.cat([obs_t, act_t], dim=-1)
        acts = []
        for layer in critic.net:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                acts.append(x)
        return acts

    def _cbp_update(self, critic, cid, optimiser, obs_t, act_t):
        with torch.no_grad():
            acts = self._forward_hidden(critic, obs_t, act_t)
            for k in range(self.n_hidden):
                st = self._cbp_state[cid][k]
                st["age"] += 1.0

                # Contribution utility: mean |activation| × sum |outgoing weights|
                mean_act = acts[k].abs().mean(dim=0)                 # (n_units,)
                out_w = (critic.net[2 * (k + 1)].weight if k < self.n_hidden - 1
                         else critic.q_head.weight)                  # (next, n_units)
                out_sum  = out_w.abs().sum(dim=0)                    # (n_units,)
                contrib  = mean_act * out_sum
                st["util"] = self.decay_rate * st["util"] \
                             + (1.0 - self.decay_rate) * contrib

                eligible = st["age"] > self.maturity_thresh
                n_elig   = int(eligible.sum().item())
                if n_elig == 0:
                    continue

                st["accum"] += self.replacement_rate * n_elig
                n_replace = int(st["accum"])
                if n_replace < 1:
                    continue
                st["accum"] -= n_replace

                util_masked = st["util"].clone()
                util_masked[~eligible] = float("inf")
                idx = torch.argsort(util_masked)[:n_replace]

                self._reinit_units(critic, optimiser, k, idx)
                st["util"][idx] = 0.0
                st["age"][idx]  = 0.0

    def _reinit_units(self, critic, optimiser, k, idx):
        """Re-init incoming weights, zero outgoing weights, reset Adam state."""
        in_lin = critic.net[2 * k]
        fan_in = in_lin.weight.shape[1]
        w = torch.empty(len(idx), fan_in, device=DEVICE, dtype=in_lin.weight.dtype)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        in_lin.weight.data[idx, :] = w
        in_lin.bias.data[idx]      = 0.0

        if k < self.n_hidden - 1:
            out_lin = critic.net[2 * (k + 1)]
        else:
            out_lin = critic.q_head
        out_lin.weight.data[:, idx] = 0.0

        # Clear stale Adam moments for the replaced rows / columns.
        self._reset_adam(optimiser, in_lin.weight, rows=idx)
        self._reset_adam(optimiser, in_lin.bias,   rows=idx)
        self._reset_adam(optimiser, out_lin.weight, cols=idx)

    @staticmethod
    def _reset_adam(optimiser, param, rows=None, cols=None):
        st = optimiser.state.get(param)
        if not st:
            return
        for key in ("exp_avg", "exp_avg_sq"):
            if key in st:
                if rows is not None:
                    st[key][rows] = 0.0
                if cols is not None:
                    st[key][:, cols] = 0.0


class SACReDo(SACAgent):
    """
    ReDo-style dormant-neuron recycling baseline — Sokar et al., "The Dormant
    Neuron Phenomenon in Deep Reinforcement Learning", ICML 2023.

    Every `recycle_freq` gradient steps, neurons whose mean activation score
    falls below `dormancy_threshold` are re-initialised: incoming weights are
    re-sampled from the Kaiming distribution, outgoing weights are zeroed, and
    Adam moments are cleared. Applied to both critics for a fair comparison with
    PALR (which intervenes on the same networks).

    The dormancy score for unit i is:
        score_i = mean_batch |h_i| / (mean_batch mean_units |h_j|)
    A unit is dormant when score_i < dormancy_threshold (default 0.025, as in
    the ReDo paper).

    Task-agnostic: no task-switch oracle is used.
    """

    def __init__(self, *args,
                 recycle_freq: int = 1000,
                 dormancy_threshold: float = 0.025,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.recycle_freq         = recycle_freq
        self.dormancy_threshold   = dormancy_threshold
        self.name                 = "ReDo-SAC"
        self._redo_step_count     = 0

    def train_step(self):
        loss = super().train_step()
        if loss is not None:
            self._redo_step_count += 1
            if self._redo_step_count % self.recycle_freq == 0:
                self._recycle_dormant()
        return loss

    @staticmethod
    def _get_hidden_activations(critic, obs_t, act_t):
        """Return post-ReLU activations for each hidden layer."""
        x = torch.cat([obs_t, act_t], dim=-1)
        acts = []
        for layer in critic.net:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                acts.append(x.detach())
        return acts

    def _recycle_dormant(self):
        bs = min(256, len(self.buffer))
        obs, actions, _, _, _ = self.buffer.sample(bs)
        obs_t = torch.FloatTensor(obs).to(DEVICE)
        act_t = torch.FloatTensor(actions).to(DEVICE)
        with torch.no_grad():
            for critic, optimiser in [
                (self.critic1, self.critic1_optim),
                (self.critic2, self.critic2_optim),
            ]:
                acts = self._get_hidden_activations(critic, obs_t, act_t)
                n_hidden = len(acts)
                for k, h in enumerate(acts):
                    # Dormancy score per unit
                    mean_act  = h.abs().mean(dim=0)           # (n_units,)
                    layer_mean = mean_act.mean().clamp(min=1e-8)
                    scores    = mean_act / layer_mean          # normalised
                    dormant   = scores < self.dormancy_threshold
                    idx       = dormant.nonzero(as_tuple=False).squeeze(1)
                    if idx.numel() == 0:
                        continue
                    self._reinit_units(critic, optimiser, k, idx, n_hidden)

    def _reinit_units(self, critic, optimiser, k, idx, n_hidden):
        in_lin  = critic.net[2 * k]
        fan_in  = in_lin.weight.shape[1]
        w = torch.empty(len(idx), fan_in, device=DEVICE,
                        dtype=in_lin.weight.dtype)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        in_lin.weight.data[idx, :] = w
        in_lin.bias.data[idx]      = 0.0

        out_lin = (critic.net[2 * (k + 1)] if k < n_hidden - 1
                   else critic.q_head)
        out_lin.weight.data[:, idx] = 0.0

        CBPSACAgent._reset_adam(optimiser, in_lin.weight,  rows=idx)
        CBPSACAgent._reset_adam(optimiser, in_lin.bias,    rows=idx)
        CBPSACAgent._reset_adam(optimiser, out_lin.weight, cols=idx)
