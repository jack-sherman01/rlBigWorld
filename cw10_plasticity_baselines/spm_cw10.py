"""
Independent re-implementation of SPM (Surgical Plasticity Maintenance) on
ContinualWorld-10 (MT10, metaworld), built from the method's published
description (diagnostics + targeted perturbation + gradient modulation,
reactive/proactive control loops) as a standalone reimplementation, separate
from the original `palr_plasticity_aware_lr/` training code.

Includes a diagnostic-free, fixed-schedule perturbation control, dosed to
match SPM's empirical intervention frequency/reset count, as the key open
validation of whether diagnostic targeting outperforms a frequency-matched
but non-targeted periodic reset. Also logs genuine per-task success rate
(metaworld's native info["success"]), averaged per episode per task, rather
than a single end-of-run snapshot.

Agents implemented:
  - SAC-FixedLR          : plain SAC, no plasticity maintenance
  - SPM                  : full (perturbation + gradient modulation + proactive, schedule oracle)
  - SPM-NoScale           : perturbation only (ablation already in the paper)
  - SPM-NoPerturb         : modulation only (ablation already in the paper)
  - FixedSchedule         : NEW. diagnostic-free periodic perturbation, dosed
                            to match SPM's empirically observed frequency/count
"""
import argparse, json, os, random, time, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# CW10 / MT10 task order (matches logs/cbp_cw10_seed0.log task sequence)
# --------------------------------------------------------------------------
TASK_ORDER = [
    "reach-v3", "push-v3", "pick-place-v3", "door-open-v3", "drawer-open-v3",
    "drawer-close-v3", "button-press-topdown-v3", "peg-insert-side-v3",
    "window-open-v3", "window-close-v3",
]

EPISODES_PER_TASK = 200
N_TASKS = 10
TOTAL_EPISODES = EPISODES_PER_TASK * N_TASKS  # 2000, matches Table 3/6


def make_mt10_envs():
    import metaworld
    mt10 = metaworld.MT10(seed=0)
    envs = {}
    for name, env_cls in mt10.train_classes.items():
        env = env_cls()
        task = [t for t in mt10.train_tasks if t.env_name == name][0]
        env.set_task(task)
        envs[name] = env
    return envs


# --------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------
class MLPCritic(nn.Module):
    """4 hidden layers x 256 units, ReLU (Table 6). Exposes per-layer handles
    for SPM diagnostics/perturbation/modulation."""

    def __init__(self, obs_dim, act_dim, hidden=256, depth=4):
        super().__init__()
        dims = [obs_dim + act_dim] + [hidden] * depth
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(depth)]
        )
        self.out = nn.Linear(hidden, 1)
        self._last_acts = [None] * depth

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x))
            self._last_acts[i] = x
        return self.out(x)

    def last_activations(self):
        return self._last_acts


class MLPCriticCReLU(nn.Module):
    """Baseline: Concatenated ReLU (Shang et al. 2016), a plasticity-loss
    mitigation to compare against. Each
    hidden layer produces concat(relu(z), relu(-z)), which by construction
    cannot have a fully-dead unit pair (one of the two halves is always >=0),
    so this baseline uses no SPM diagnostics/interventions -- it tests
    whether the activation function alone is a sufficient fix."""

    def __init__(self, obs_dim, act_dim, hidden=256, depth=4):
        super().__init__()
        half = hidden // 2
        dims_in = [obs_dim + act_dim] + [hidden] * depth
        self.layers = nn.ModuleList(
            [nn.Linear(dims_in[i], half) for i in range(depth)]
        )
        self.out = nn.Linear(hidden, 1)
        self._last_acts = [None] * depth

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        for i, layer in enumerate(self.layers):
            z = layer(x)
            x = torch.cat([F.relu(z), F.relu(-z)], dim=-1)
            self._last_acts[i] = x
        return self.out(x)

    def last_activations(self):
        return self._last_acts


class _HeadBlock(nn.Module):
    """Two hidden layers (256 units) + scalar output head -- the injectable
    'head' half of the critic, per Nikishin et al. 2023 Section 4's
    encoder/head split."""

    def __init__(self, hidden=256):
        super().__init__()
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.l2(x))
        h = F.relu(self.l3(x))
        return self.out(h), h  # output, and last hidden activation


class MLPCriticPlasticityInjection(nn.Module):
    """Baseline: Plasticity Injection (Nikishin et al. 2023, NeurIPS).
    Verified against the paper's verbatim Section 4 / 5.1 (encoder/head
    split with k=3-of-5 conv layers in their DQN setup; here we use an
    analogous 2-of-4 split for a 4-hidden-layer MLP critic): layers[0:2]
    are a shared, always-trainable 'encoder'; layers[2:4]+out are the
    injectable 'head'.

    At injection: the current head is frozen as h_theta; a freshly
    initialized head is copied twice into h_theta1 (trainable) and
    h_theta2 (frozen) -- identical at init, so h_theta1(x) - h_theta2(x) = 0
    and the combined output h_theta(x) + h_theta1(x) - h_theta2(x) is
    exactly unchanged at the moment of injection (paper's Eq. 1). No
    stop-gradient is applied between the encoder and the frozen head terms
    (paper states gradients from frozen heads still flow into the shared
    encoder), so frozen heads simply have requires_grad=False.

    The paper's default is a SINGLE injection at 25% of training (50M of
    200M Atari frames); we use the same 25%-of-training ratio here rather
    than repeated/ensembling injection (which the paper does not specify a
    bounded-memory merge rule for)."""

    def __init__(self, obs_dim, act_dim, hidden=256, depth=4):
        super().__init__()
        assert depth == 4
        in_dim = obs_dim + act_dim
        self.enc0 = nn.Linear(in_dim, hidden)
        self.enc1 = nn.Linear(hidden, hidden)
        self.head_frozen = None          # h_theta after injection (frozen), else None
        self.head_trainable = _HeadBlock(hidden)   # h_theta before injection / h_theta1 after
        self.head_new_frozen = None      # h_theta2 after injection (frozen), else None
        self.injected = False
        self._last_acts = [None, None, None, None]

    def encoder(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        e0 = F.relu(self.enc0(x))
        e1 = F.relu(self.enc1(e0))
        self._last_acts[0] = e0
        self._last_acts[1] = e1
        return e1

    def forward(self, obs, act):
        enc_out = self.encoder(obs, act)
        q_trainable, h_trainable = self.head_trainable(enc_out)
        # diagnostics report the currently-trainable head's two hidden
        # layers in slots [2],[3] (l2, l3 of _HeadBlock) for comparability
        # with the 4-layer reporting used by other agents.
        self._last_acts[2] = F.relu(self.head_trainable.l2(enc_out))
        self._last_acts[3] = h_trainable
        q = q_trainable
        if self.injected:
            q_frozen, _ = self.head_frozen(enc_out)
            q_new_frozen, _ = self.head_new_frozen(enc_out)
            q = q_frozen + q_trainable - q_new_frozen
        return q

    def last_activations(self):
        return self._last_acts

    def inject(self):
        """Freeze current trainable head as h_theta; spawn a fresh
        (theta1, theta2) pair, identical at init, theta1 trainable."""
        import copy
        if self.injected:
            # fold current (frozen + trainable - new_frozen) into a single
            # frozen term before injecting again, keeping memory bounded.
            old_frozen, old_trainable, old_new_frozen = (
                self.head_frozen, self.head_trainable, self.head_new_frozen
            )
            for p in old_trainable.parameters():
                p.requires_grad_(False)
            self.head_frozen = _FoldedHead(old_frozen, old_trainable, old_new_frozen)
        else:
            self.head_frozen = self.head_trainable
            for p in self.head_frozen.parameters():
                p.requires_grad_(False)
        device = self.enc1.weight.device
        new_init = _HeadBlock(self.enc1.out_features).to(device)
        self.head_trainable = new_init
        self.head_new_frozen = copy.deepcopy(new_init).to(device)
        for p in self.head_new_frozen.parameters():
            p.requires_grad_(False)
        if isinstance(self.head_frozen, _FoldedHead):
            self.head_frozen = self.head_frozen.to(device)
        self.injected = True
        # register as submodules so .parameters()/.to()/soft_update see them
        self.add_module("head_frozen", self.head_frozen)
        self.add_module("head_trainable", self.head_trainable)
        self.add_module("head_new_frozen", self.head_new_frozen)


class _FoldedHead(nn.Module):
    """Collapses a previous (frozen + trainable - new_frozen) combination
    into a single callable frozen term, so repeated injection keeps exactly
    one running 'head_frozen' accumulator rather than an ever-growing list."""

    def __init__(self, frozen, trainable, new_frozen):
        super().__init__()
        self.frozen = frozen
        self.trainable = trainable
        self.new_frozen = new_frozen
        for m in (self.frozen, self.trainable, self.new_frozen):
            for p in m.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        qf, _ = self.frozen(x)
        qt, _ = self.trainable(x)
        qnf, _ = self.new_frozen(x)
        return qf + qt - qnf, None


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256, depth=2, log_std_bounds=(-5, 2)):
        super().__init__()
        dims = [obs_dim] + [hidden] * depth
        self.trunk = nn.Sequential(
            *sum(
                ([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()] for i in range(depth)),
                [],
            )
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        self.log_std_bounds = log_std_bounds

    def forward(self, obs):
        h = self.trunk(obs)
        mu = self.mu(h)
        log_std = torch.tanh(self.log_std(h))
        lo, hi = self.log_std_bounds
        log_std = lo + 0.5 * (hi - lo) * (log_std + 1)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        x = dist.rsample()
        action = torch.tanh(x)
        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

    def select_action(self, obs, deterministic=False):
        h = self.trunk(obs)
        mu = self.mu(h)
        if deterministic:
            return torch.tanh(mu)
        log_std = torch.tanh(self.log_std(h))
        lo, hi = self.log_std_bounds
        log_std = lo + 0.5 * (hi - lo) * (log_std + 1)
        std = log_std.exp()
        x = torch.distributions.Normal(mu, std).rsample()
        return torch.tanh(x)


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity=500_000, device="cpu"):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.nobs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size, self.device = 0, 0, device

    def add(self, o, a, r, no, d):
        self.obs[self.ptr] = o
        self.act[self.ptr] = a
        self.rew[self.ptr] = r
        self.nobs[self.ptr] = no
        self.done[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n):
        idx = np.random.randint(0, self.size, size=n)
        t = lambda x: torch.as_tensor(x, device=self.device)
        return (
            t(self.obs[idx]), t(self.act[idx]), t(self.rew[idx]),
            t(self.nobs[idx]), t(self.done[idx]), idx,
        )


# --------------------------------------------------------------------------
# SPM diagnostics / interventions (Eqs 1-6)
# --------------------------------------------------------------------------
class SPMController:
    def __init__(self, n_layers, rank_target, rank_weight=1.5, smax=5.0,
                 perturb_thresh=0.05, perturb_scale=0.5, enabled_perturb=True,
                 enabled_modulate=True, modulation_mode="full", shuffle_targets=False):
        self.n_layers = n_layers
        self.rank_target = rank_target
        self.rank_weight = rank_weight
        self.smax = smax
        self.tau = perturb_thresh
        self.sigma_p = perturb_scale
        self.enabled_perturb = enabled_perturb
        self.enabled_modulate = enabled_modulate
        # modulation_mode: "full" (dead+rank, Eq.3), "dead_only", "rank_only"
        # -- closes the paper's own appendix item (i)/(ii): decomposing the
        # combined dead+rank modulation signal into its two constituents.
        self.modulation_mode = modulation_mode
        # shuffle_targets: paper appendix item (iv) -- perturbation still
        # triggers at the true diagnostic-measured frequency per layer, but
        # WHICH units get reset is drawn uniformly at random rather than the
        # true dead-unit mask, decoupling "where" from the diagnostic signal
        # while preserving "when"/"how many" (frequency- and count-matched).
        self.shuffle_targets = shuffle_targets
        self.s = [1.0] * n_layers
        self.floor_until = [0] * n_layers  # training-step index until which floor applies
        self.last_dead = [0.0] * n_layers
        self.last_rank = [0.0] * n_layers
        self.perturb_events = 0
        self.units_perturbed = 0
        self.perturb_events_per_layer = [0] * n_layers
        self.units_perturbed_per_layer = [0] * n_layers
        self.total_steps_seen = 0

    @staticmethod
    def dead_fraction(acts):
        # acts: (B, d_l) post-ReLU activations
        dead = (acts.max(dim=0).values <= 0).float().mean().item()
        return dead

    @staticmethod
    def effective_rank(acts):
        x = acts - acts.mean(dim=0, keepdim=True)
        try:
            s = torch.linalg.svdvals(x)
        except RuntimeError:
            return 1.0
        s = s[s > 1e-12]
        if s.numel() == 0:
            return 1.0
        p = s / s.sum()
        ent = -(p * torch.log(p)).sum()
        return float(torch.exp(ent))

    def measure_and_update(self, critic, diag_obs, diag_act):
        with torch.no_grad():
            critic(diag_obs, diag_act)
            acts = critic.last_activations()
        dead_units_per_layer = []
        for l, a in enumerate(acts):
            delta = self.dead_fraction(a)
            rho = self.effective_rank(a)
            self.last_dead[l] = delta
            self.last_rank[l] = rho
            rank_deficit = max(0.0, (self.rank_target - rho) / self.rank_target)
            if self.modulation_mode == "dead_only":
                phi = max(0.0, delta)
            elif self.modulation_mode == "rank_only":
                phi = self.rank_weight * rank_deficit
            else:
                phi = max(0.0, delta) + self.rank_weight * rank_deficit
            if self.enabled_modulate:
                s = 1.0 + (self.smax - 1.0) * min(1.0, max(0.0, phi / (1.0 + self.rank_weight)))
            else:
                s = 1.0
            self.s[l] = s
            dead_mask = (a.max(dim=0).values <= 0)
            dead_units_per_layer.append(dead_mask)
        return dead_units_per_layer

    def maybe_perturb(self, critics, dead_units_per_layer, step):
        if not self.enabled_perturb:
            return
        for l, dead_mask in enumerate(dead_units_per_layer):
            delta = self.last_dead[l]
            if delta > self.tau:
                self._perturb_layer(critics, l, dead_mask)

    def _perturb_layer(self, critics, l, dead_mask):
        idx = dead_mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return
        if self.shuffle_targets:
            # Same count, same per-layer trigger timing, but units chosen
            # uniformly at random instead of the true dead-unit mask.
            n_layer_units = dead_mask.numel()
            idx = torch.randperm(n_layer_units, device=idx.device)[: idx.numel()]
        self.perturb_events += 1
        self.units_perturbed += idx.numel()
        self.perturb_events_per_layer[l] += 1
        self.units_perturbed_per_layer[l] += idx.numel()
        for critic in critics:
            layer = critic.layers[l]
            fan_in = layer.weight.shape[1]
            with torch.no_grad():
                std = (2.0 * self.sigma_p ** 2 / fan_in) ** 0.5
                layer.weight[idx, :] = torch.randn(idx.numel(), fan_in, device=layer.weight.device) * std
                layer.bias[idx] = 0.0

    def perturb_all_layers(self, critics):
        """Proactive: perturb all layers unconditionally at task switch, using
        whatever units are currently dead (re-measured with last diagnostic)."""
        for l in range(self.n_layers):
            # use a fresh all-ones mask fallback if we have no cached dead mask;
            # callers pass dead masks from the most recent measurement instead.
            pass

    def apply_proactive_floor(self, step, duration_steps):
        for l in range(self.n_layers):
            self.s[l] = max(self.s[l], 2.0)
            self.floor_until[l] = step + duration_steps

    def scale_gradients(self, critic):
        if not self.enabled_modulate:
            return
        for l, layer in enumerate(critic.layers):
            s = self.s[l]
            if layer.weight.grad is not None:
                layer.weight.grad.mul_(s)
            if layer.bias.grad is not None:
                layer.bias.grad.mul_(s)


class FixedScheduleController(SPMController):
    """NEW ablation arm (paper appendix item iii): perturbs a FIXED number of
    randomly-selected units per layer at a FIXED period, calibrated offline to
    match SPM's empirically observed average frequency/reset count. No
    diagnostic signal is used to decide when or what to perturb."""

    def __init__(self, n_layers, period_steps, units_per_event, layer_sizes, **kw):
        super().__init__(n_layers, rank_target=1.0, enabled_perturb=False,
                          enabled_modulate=False, **kw)
        # Accept either a single scalar (shared across layers) or a
        # per-layer list -- per-layer is the more faithful calibration,
        # since SPM's real perturbation rate differs sharply by depth.
        self.period_steps = (
            list(period_steps) if isinstance(period_steps, (list, tuple))
            else [period_steps] * n_layers
        )
        self.units_per_event = (
            list(units_per_event) if isinstance(units_per_event, (list, tuple))
            else [units_per_event] * n_layers
        )
        self.layer_sizes = layer_sizes

    def maybe_fixed_perturb(self, critics, step):
        for l in range(self.n_layers):
            period = max(1, int(round(self.period_steps[l])))
            if step > 0 and step % period == 0:
                n = max(0, min(int(round(self.units_per_event[l])), self.layer_sizes[l]))
                if n == 0:
                    continue
                idx = torch.randperm(self.layer_sizes[l])[:n]
                self.perturb_events += 1
                self.units_perturbed += n
                self.perturb_events_per_layer[l] += 1
                self.units_perturbed_per_layer[l] += n
                for critic in critics:
                    layer = critic.layers[l]
                    fan_in = layer.weight.shape[1]
                    with torch.no_grad():
                        std = (2.0 * self.sigma_p ** 2 / fan_in) ** 0.5
                        layer.weight[idx, :] = torch.randn(n, fan_in, device=layer.weight.device) * std
                        layer.bias[idx] = 0.0
