"""Reward rubric.

reward = (prev - curr) * 2.0
       + (-50  if contradictions)
       + (-15  if evasive)
       + (±25  on terminal succeed / catch)

Terminal bonuses are smaller than the contradiction penalty on purpose:
the agent can never "buy" an early end by shipping a single contradiction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from server.detective import DetectiveJudgement


W_SUSPICION_DELTA = 2.0
W_CONTRADICTION = -50.0
W_EVASION = -15.0
W_TERMINAL_SUCCESS = 25.0
W_TERMINAL_CAUGHT = -25.0


@dataclass
class RewardBreakdown:
    suspicion_delta: float = 0.0
    contradiction_penalty: float = 0.0
    evasion_penalty: float = 0.0
    terminal_bonus: float = 0.0
    total: float = 0.0
    components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suspicion_delta": self.suspicion_delta,
            "contradiction_penalty": self.contradiction_penalty,
            "evasion_penalty": self.evasion_penalty,
            "terminal_bonus": self.terminal_bonus,
            "total": self.total,
            "components": dict(self.components),
        }


def compute_reward(
    judgement: DetectiveJudgement,
    previous_suspicion: int,
    *,
    terminal_succeeded: bool = False,
    terminal_caught: bool = False,
) -> RewardBreakdown:
    delta_component = (previous_suspicion - judgement.suspicion_score) * W_SUSPICION_DELTA
    contradiction_component = (
        W_CONTRADICTION if judgement.contradictions_found else 0.0
    )
    evasion_component = W_EVASION if judgement.is_evasive else 0.0

    terminal_component = 0.0
    if terminal_succeeded:
        terminal_component += W_TERMINAL_SUCCESS
    if terminal_caught:
        terminal_component += W_TERMINAL_CAUGHT

    total = (
        delta_component
        + contradiction_component
        + evasion_component
        + terminal_component
    )

    return RewardBreakdown(
        suspicion_delta=float(delta_component),
        contradiction_penalty=float(contradiction_component),
        evasion_penalty=float(evasion_component),
        terminal_bonus=float(terminal_component),
        total=float(total),
        components={
            "w_suspicion_delta": W_SUSPICION_DELTA,
            "w_contradiction": W_CONTRADICTION,
            "w_evasion": W_EVASION,
            "w_terminal_success": W_TERMINAL_SUCCESS,
            "w_terminal_caught": W_TERMINAL_CAUGHT,
        },
    )
