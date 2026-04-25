"""
fetch_curriculum.py
===================
Fetch Continual Curriculum: 4 tasks cycling endlessly.

  Phase 0 — Apple Pick     (RearrangePickTask-v0,  object: apple)
  Phase 1 — Bowl Pick      (RearrangePickTask-v0,  object: bowl)
  Phase 2 — Open Fridge    (RearrangeOpenFridgeTask-v0)
  Phase 3 — Place in Sink  (RearrangePlaceTask-v0, receptacle: sink)
  Phase 4 → back to Phase 0

At each task switch the trainer calls:
  curriculum.advance()         → returns new (task_name, dataset_path)
  curriculum.should_switch()   → True when step count crosses phase boundary
  curriculum.on_task_switch()  → called to update PALR baselines

The curriculum is step-based (total env steps), not episode-based,
so it is stable under varying episode lengths typical of rearrangement.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class CurriculumPhase:
    label:        str
    task_type:    str    # habitat task registration name
    dataset_path: str    # path to .json.gz episode dataset


class FetchCurriculum:
    """
    Tracks which task phase is active and signals when to switch.

    Args:
        phases:            List of CurriculumPhase (in order).
        steps_per_phase:   Env steps to spend on each phase.
        cyclic:            If True, loop back to phase 0 after the last.
    """

    def __init__(
        self,
        phases:          List[CurriculumPhase],
        steps_per_phase: int,
        cyclic:          bool = True,
    ):
        self.phases          = phases
        self.steps_per_phase = steps_per_phase
        self.cyclic          = cyclic

        self._phase_idx:       int = 0
        self._total_steps:     int = 0
        self._switch_log:      List[dict] = []  # for logging / plotting

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def current_phase(self) -> CurriculumPhase:
        return self.phases[self._phase_idx]

    @property
    def phase_idx(self) -> int:
        return self._phase_idx

    @property
    def n_phases(self) -> int:
        return len(self.phases)

    # ── Step counting ──────────────────────────────────────────────────────────

    def step(self, n: int = 1) -> bool:
        """
        Advance step counter by n.
        Returns True if a task switch is needed (caller should call advance()).
        """
        self._total_steps += n
        boundary = (self._phase_idx + 1) * self.steps_per_phase
        return self._total_steps >= boundary

    def should_switch(self) -> bool:
        boundary = (self._phase_idx + 1) * self.steps_per_phase
        return self._total_steps >= boundary

    # ── Task switching ─────────────────────────────────────────────────────────

    def advance(self) -> Optional[CurriculumPhase]:
        """
        Advance to the next phase.
        Returns the new CurriculumPhase, or None if training is complete
        (non-cyclic mode and last phase finished).
        """
        next_idx = self._phase_idx + 1
        if next_idx >= len(self.phases):
            if self.cyclic:
                next_idx = 0
            else:
                return None

        old_label = self.current_phase.label
        self._phase_idx = next_idx
        new_phase = self.current_phase

        self._switch_log.append({
            "step":      self._total_steps,
            "from":      old_label,
            "to":        new_phase.label,
            "phase_idx": self._phase_idx,
        })

        print(
            f"[Curriculum] step={self._total_steps:,d}  "
            f"task switch: {old_label} → {new_phase.label}"
        )
        return new_phase

    # ── Serialisation ──────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "phase_idx":   self._phase_idx,
            "total_steps": self._total_steps,
            "switch_log":  self._switch_log,
        }

    def load_state_dict(self, d: dict):
        self._phase_idx   = d["phase_idx"]
        self._total_steps = d["total_steps"]
        self._switch_log  = d.get("switch_log", [])

    @property
    def switch_log(self) -> List[dict]:
        return list(self._switch_log)


# ── Factory: build from YAML config dict ──────────────────────────────────────

def make_curriculum_from_config(cfg: dict) -> FetchCurriculum:
    """
    Build a FetchCurriculum from the CURRICULUM section of the YAML config.

    Expected format:
        CURRICULUM:
          steps_per_phase: 50000000
          phases:
            - task:    "RearrangePickTask-v0"
              label:   "apple_pick"
              dataset: "data/datasets/rearrange/apple_pick_train.json.gz"
            - ...
    """
    steps = cfg["steps_per_phase"]
    phases = []
    for p in cfg["phases"]:
        phases.append(CurriculumPhase(
            label      = p["label"],
            task_type  = p["task"],
            dataset_path = p["dataset"],
        ))
    return FetchCurriculum(phases=phases, steps_per_phase=steps, cyclic=True)
