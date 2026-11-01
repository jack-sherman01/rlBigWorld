import argparse, json, os, random, time
import numpy as np
import torch
import torch.nn.functional as F

from spm_cw10 import (
    MLPCritic, MLPCriticCReLU, MLPCriticPlasticityInjection, GaussianActor,
    ReplayBuffer, SPMController, FixedScheduleController, make_mt10_envs,
    TASK_ORDER, EPISODES_PER_TASK, N_TASKS,
)

OBS_DIM, ACT_DIM = 39, 4
HIDDEN = 256
DEPTH = 4
GAMMA = 0.99
TAU_SOFT = 0.005
LR = 3e-4
BUFFER_CAP = 500_000
BATCH = 2048
MEASURE_INTERVAL = 500
DIAG_BATCH = 256
RANK_TARGET = HIDDEN // 4  # 64, d_l/4 for 256-unit layers
RANK_WEIGHT = 1.5
SMAX = 5.0
PERTURB_TAU = 0.05
PERTURB_SIGMA = 0.5
PROACTIVE_FLOOR_STEPS = MEASURE_INTERVAL  # "for T_measure steps"


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def build_agent(device, critic_cls=MLPCritic):
    actor = GaussianActor(OBS_DIM, ACT_DIM).to(device)
    critic1 = critic_cls(OBS_DIM, ACT_DIM, HIDDEN, DEPTH).to(device)
    critic2 = critic_cls(OBS_DIM, ACT_DIM, HIDDEN, DEPTH).to(device)
    critic1_t = critic_cls(OBS_DIM, ACT_DIM, HIDDEN, DEPTH).to(device)
    critic2_t = critic_cls(OBS_DIM, ACT_DIM, HIDDEN, DEPTH).to(device)
    critic1_t.load_state_dict(critic1.state_dict())
    critic2_t.load_state_dict(critic2.state_dict())
    return actor, critic1, critic2, critic1_t, critic2_t


def soft_update(net, target):
    # Skip frozen (requires_grad=False) parameters -- relevant for
    # Plasticity Injection's frozen head terms, which should stay exactly
    # at their own init/frozen value rather than being Polyak-blended
    # between online and target copies.
    for p, tp in zip(net.parameters(), target.parameters()):
        if p.requires_grad:
            tp.data.copy_((1 - TAU_SOFT) * tp.data + TAU_SOFT * p.data)


def sac_update(batch, actor, critic1, critic2, critic1_t, critic2_t,
                actor_opt, critic1_opt, critic2_opt, log_alpha, alpha_opt,
                target_entropy, device, spm=None):
    obs, act, rew, nobs, done, idx = batch
    rew = rew.unsqueeze(-1); done = done.unsqueeze(-1)
    alpha = log_alpha.exp().item()

    with torch.no_grad():
        nact, nlogp = actor(nobs)
        q1_t = critic1_t(nobs, nact)
        q2_t = critic2_t(nobs, nact)
        q_t = torch.min(q1_t, q2_t) - alpha * nlogp
        target_q = rew + GAMMA * (1 - done) * q_t

    q1 = critic1(obs, act)
    q2 = critic2(obs, act)
    closs1 = F.mse_loss(q1, target_q)
    closs2 = F.mse_loss(q2, target_q)

    critic1_opt.zero_grad(); closs1.backward()
    if spm is not None:
        spm.scale_gradients(critic1)
    torch.nn.utils.clip_grad_norm_(critic1.parameters(), 1.0)
    critic1_opt.step()

    critic2_opt.zero_grad(); closs2.backward()
    if spm is not None:
        spm.scale_gradients(critic2)
    torch.nn.utils.clip_grad_norm_(critic2.parameters(), 1.0)
    critic2_opt.step()

    pi, logp = actor(obs)
    q1_pi = critic1(obs, pi); q2_pi = critic2(obs, pi)
    q_pi = torch.min(q1_pi, q2_pi)
    aloss = (alpha * logp - q_pi).mean()
    actor_opt.zero_grad(); aloss.backward(); actor_opt.step()

    alpha_loss = -(log_alpha * (logp.detach() + target_entropy)).mean()
    alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()

    soft_update(critic1, critic1_t)
    soft_update(critic2, critic2_t)


# ---------------------------------------------------------------------------
# C-CHAIN (Tang, Obando-Ceron, Castro, Courville, Berseth, ICML 2025)
# "Mitigating Plasticity Loss in Continual RL by Reducing Churn"
# Verified against the paper's verbatim Eq. 3 (churn), Eq. 8 (churn-reduction
# loss L_f^cr = 0.5 * E[C_f(x_bar, theta, Delta_theta)^2]) and Algorithm 1
# (regular update on B_train, THEN a separate regularization step reducing
# churn on a disjoint reference batch B_ref ~ same buffer). The paper
# specifies DQN/PPO, not SAC; applying it to our SAC critic is our own
# adaptation (documented), following the same pattern already used for the
# paper's own CBP-SAC/ReDo-SAC baselines. No coefficient-schedule formula
# was verifiable from the paper text, so the churn-reduction step uses the
# same learning rate as the main critic optimizer (a deliberate, documented
# simplification).
# ---------------------------------------------------------------------------
def cchain_churn_reduction_step(critic, critic_opt, ref_obs, ref_act, pre_update_q):
    q_new = critic(ref_obs, ref_act)
    churn = q_new - pre_update_q
    loss_cr = 0.5 * (churn ** 2).mean()
    critic_opt.zero_grad()
    loss_cr.backward()
    critic_opt.step()
    return float(loss_cr)


def run(agent_type, seed, device, max_episodes=EPISODES_PER_TASK * N_TASKS,
        fixed_schedule_period=None, fixed_schedule_units=None, outdir="results"):
    set_seed(seed)
    envs = make_mt10_envs()
    if agent_type == "crelu":
        critic_cls = MLPCriticCReLU
    elif agent_type == "plasticity_injection":
        critic_cls = MLPCriticPlasticityInjection
    else:
        critic_cls = MLPCritic
    actor, critic1, critic2, critic1_t, critic2_t = build_agent(device, critic_cls)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=LR)
    critic1_opt = torch.optim.Adam(critic1.parameters(), lr=LR)
    critic2_opt = torch.optim.Adam(critic2.parameters(), lr=LR)
    target_entropy = -float(ACT_DIM)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_opt = torch.optim.Adam([log_alpha], lr=LR)

    buf = ReplayBuffer(OBS_DIM, ACT_DIM, BUFFER_CAP, device=device)

    use_spm_perturb = agent_type in ("spm", "spm_noscale", "spm_deadonly", "spm_rankonly", "spm_shuffled")
    use_spm_modulate = agent_type in ("spm", "spm_noperturb", "spm_deadonly", "spm_rankonly")
    use_fixed_schedule = agent_type == "fixedschedule"
    use_cchain = agent_type == "cchain"
    use_plasticity_injection = agent_type == "plasticity_injection"
    injection_done = False
    injection_step = int(0.25 * max_episodes * 500)  # paper's 50M/200M=25% ratio

    modulation_mode = {"spm_deadonly": "dead_only", "spm_rankonly": "rank_only"}.get(agent_type, "full")
    shuffle_targets = agent_type == "spm_shuffled"

    if use_fixed_schedule:
        spm = FixedScheduleController(
            DEPTH, period_steps=fixed_schedule_period,
            units_per_event=fixed_schedule_units,
            layer_sizes=[HIDDEN] * DEPTH, perturb_scale=PERTURB_SIGMA,
        )
    else:
        spm = SPMController(
            DEPTH, rank_target=RANK_TARGET, rank_weight=RANK_WEIGHT, smax=SMAX,
            perturb_thresh=PERTURB_TAU, perturb_scale=PERTURB_SIGMA,
            enabled_perturb=use_spm_perturb, enabled_modulate=use_spm_modulate,
            modulation_mode=modulation_mode, shuffle_targets=shuffle_targets,
        )
    use_proactive = agent_type in ("spm",)  # schedule-oracle proactive, full SPM only

    log = {
        "agent": agent_type, "seed": seed,
        "episode_rewards": [], "episode_success": [], "episode_task": [],
        "dead_frac_per_layer": [], "eff_rank_per_layer": [],
        "perturb_events_per_layer": None, "units_perturbed_per_layer": None,
    }

    step = 0
    t0 = time.time()
    for ep in range(max_episodes):
        task_idx = ep // EPISODES_PER_TASK
        task_name = TASK_ORDER[task_idx]
        env = envs[task_name]

        if ep > 0 and ep % EPISODES_PER_TASK == 0 and use_proactive:
            with torch.no_grad():
                if buf.size >= DIAG_BATCH:
                    o, a, _, _, _, _ = buf.sample(DIAG_BATCH)
                    dead_masks = spm.measure_and_update(critic1, o, a)
                    for l, dm in enumerate(dead_masks):
                        spm._perturb_layer([critic1, critic2], l, dm)
            spm.apply_proactive_floor(step, PROACTIVE_FLOOR_STEPS)

        obs, _ = env.reset(seed=seed * 10000 + ep)
        ep_rew, ep_success = 0.0, 0.0
        done = False
        t = 0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            if buf.size < 1000:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = actor.select_action(obs_t, deterministic=False).cpu().numpy()[0]
            nobs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buf.add(obs, action, rew, nobs, float(terminated))
            obs = nobs
            ep_rew += rew
            ep_success = max(ep_success, float(info.get("success", 0.0)))
            step += 1; t += 1

            if buf.size >= BATCH:
                if use_cchain and buf.size >= BATCH + DIAG_BATCH:
                    ref_o, ref_a, _, _, _, _ = buf.sample(DIAG_BATCH)
                    with torch.no_grad():
                        pre_q1 = critic1(ref_o, ref_a)
                        pre_q2 = critic2(ref_o, ref_a)
                batch = buf.sample(BATCH)
                spm_arg = spm if use_spm_modulate else None
                sac_update(batch, actor, critic1, critic2, critic1_t, critic2_t,
                           actor_opt, critic1_opt, critic2_opt, log_alpha, alpha_opt,
                           target_entropy, device, spm=spm_arg)
                if use_cchain and buf.size >= BATCH + DIAG_BATCH:
                    cchain_churn_reduction_step(critic1, critic1_opt, ref_o, ref_a, pre_q1)
                    cchain_churn_reduction_step(critic2, critic2_opt, ref_o, ref_a, pre_q2)

            if use_plasticity_injection and (not injection_done) and step >= injection_step:
                for c in (critic1, critic2, critic1_t, critic2_t):
                    c.inject()
                injection_done = True

            if (not use_fixed_schedule) and step % MEASURE_INTERVAL == 0 and buf.size >= DIAG_BATCH:
                o, a, _, _, _, _ = buf.sample(DIAG_BATCH)
                dead_masks = spm.measure_and_update(critic1, o, a)
                spm.maybe_perturb([critic1, critic2], dead_masks, step)

            if use_fixed_schedule:
                spm.maybe_fixed_perturb([critic1, critic2], step)

        log["episode_rewards"].append(ep_rew)
        log["episode_success"].append(ep_success)
        log["episode_task"].append(task_name)
        if ep % 20 == 0:
            with torch.no_grad():
                if buf.size >= DIAG_BATCH:
                    # Read-only diagnostic snapshot for comparison purposes;
                    # for FixedSchedule this does NOT drive any intervention
                    # (that agent's perturbation timing is purely periodic).
                    o, a, _, _, _, _ = buf.sample(DIAG_BATCH)
                    critic1(o, a)
                    acts = critic1.last_activations()
                    dead = [SPMController.dead_fraction(x) for x in acts]
                    rank = [SPMController.effective_rank(x) for x in acts]
                    log["dead_frac_per_layer"].append(dead)
                    log["eff_rank_per_layer"].append(rank)
            elapsed = time.time() - t0
            print(f"[{agent_type} seed{seed}] ep {ep}/{max_episodes} task={task_name} "
                  f"rew={ep_rew:.1f} succ={ep_success:.0f} steps={step} elapsed={elapsed:.0f}s",
                  flush=True)

    log["perturb_events_per_layer"] = spm.perturb_events_per_layer
    log["units_perturbed_per_layer"] = spm.units_perturbed_per_layer
    log["total_steps"] = step

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{agent_type}_seed{seed}.json")
    with open(outpath, "w") as f:
        json.dump(log, f)
    print(f"DONE {agent_type} seed{seed} -> {outpath} ({time.time()-t0:.0f}s)", flush=True)
    return outpath


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True,
                    choices=["fixedlr", "spm", "spm_noscale", "spm_noperturb", "fixedschedule",
                             "spm_deadonly", "spm_rankonly", "spm_shuffled", "crelu",
                             "cchain", "plasticity_injection"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=EPISODES_PER_TASK * N_TASKS)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fixed_schedule_period", type=str, default=None,
                    help="scalar int, or comma-separated per-layer e.g. 450,600,600,480")
    p.add_argument("--fixed_schedule_units", type=str, default=None,
                    help="scalar int, or comma-separated per-layer e.g. 8,20,22,30")
    p.add_argument("--outdir", default="results")
    args = p.parse_args()

    def parse_maybe_list(s):
        if s is None:
            return None
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        vals = [float(p) for p in parts]
        return vals[0] if len(vals) == 1 else vals

    run(args.agent, args.seed, args.device, max_episodes=args.episodes,
        fixed_schedule_period=parse_maybe_list(args.fixed_schedule_period),
        fixed_schedule_units=parse_maybe_list(args.fixed_schedule_units), outdir=args.outdir)
