# CW10 Plasticity-Maintenance Baselines

Independent, from-scratch implementation of several plasticity-maintenance
methods for continual RL, evaluated on ContinualWorld-10 (MT10, metaworld)
with a SAC agent. Built as a standalone reimplementation, not derived from
or dependent on `palr_plasticity_aware_lr/`.

## Agents implemented

- `fixedlr` — plain SAC, no plasticity intervention (baseline)
- `spm` — full diagnostic-driven plasticity maintenance: dead-neuron
  detection + targeted perturbation + gradient modulation, with a
  proactive (schedule-triggered) and reactive (periodic) control loop
- `spm_noscale` / `spm_noperturb` — ablations isolating perturbation-only
  and modulation-only contributions
- `spm_deadonly` / `spm_rankonly` — decomposing the combined dead-fraction +
  effective-rank diagnostic signal into its two constituents
- `spm_shuffled` — frequency/count-matched perturbation with targets chosen
  uniformly at random rather than from the true dead-unit mask (tests
  whether diagnostic targeting matters beyond timing/dose)
- `fixedschedule` — diagnostic-free periodic perturbation, dosed (per-layer
  period and unit count) to match an SPM run's own empirically observed
  intervention frequency
- `crelu` — Concatenated ReLU (Shang et al. 2016) critic activation
- `plasticity_injection` — Plasticity Injection (Nikishin et al. 2023,
  NeurIPS), encoder/head split with a single injection at 25% of training
- `cchain` — C-CHAIN (Tang, Obando-Ceron, Castro, Courville, Berseth 2025,
  ICML), churn-reduction regularization applied as a separate gradient step
  on a disjoint reference batch after each critic update

The last three were independently reimplemented against each paper's
published algorithm (verified against primary-source equations before
coding, not from memory) and adapted to a SAC critic / CW10, since none of
the three original papers evaluates on SAC or on MetaWorld/ContinualWorld.

## Setup

```
conda create -n cw10_baselines python=3.10 pip -y
conda activate cw10_baselines
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Usage

```
python train_cw10.py --agent spm --seed 0 --outdir results
python train_cw10.py --agent fixedschedule --seed 0 \
    --fixed_schedule_period "66667,28571,18182,5714" \
    --fixed_schedule_units "17,30,32,32" --outdir results
```

`--fixed_schedule_period`/`--fixed_schedule_units` accept either a single
scalar (shared across layers) or a comma-separated per-layer list. For a
faithful fixed-schedule comparison, calibrate these from a completed `spm`
run's `perturb_events_per_layer`/`units_perturbed_per_layer`/`total_steps`
(see the per-run JSON log) rather than guessing values.

Full CW10 protocol: 10 sequential MT10 tasks, 200 episodes/task (2,000
total), schedule-oracle task switches, matching the standard
ContinualWorld-10 continual-RL setup.

Each run writes a JSON log with per-episode reward, per-episode success
(metaworld's native `info["success"]`), periodic per-layer dead-fraction and
effective-rank snapshots, and perturbation-event counts.
