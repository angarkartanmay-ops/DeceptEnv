"""Core DeceptEnv environment — strict Gym-style API (`reset`, `step`, `state`).

This module is the only place that holds environment state. It is consumed in
two ways:

  * In-process: `from server import DeceptEnv` (used by tests).
  * Over HTTP via `server.app` (FastAPI), which is what the *client* talks to.

CLIENTS MUST NEVER IMPORT THIS MODULE — that's the OpenEnv compliance line.
"""
from __future__ import annotations

import os
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from server.detective import (
    BaseDetective,
    DetectiveJudgement,
    make_detective,
)
from server.rubric import RewardBreakdown, compute_reward
from server.scenario import Scenario, build_scenario


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    max_turns: int = 10
    initial_suspicion: int = 50
    suspicion_caught: int = 90
    suspicion_succeeded: int = 10
    detective_provider: str | None = None    # None -> read from env var / default mock
    detective_model: str | None = None
    seed: int | None = None

    @classmethod
    def from_env(cls) -> "EnvConfig":
        def _maybe_int(name: str, default: int) -> int:
            raw = os.environ.get(name)
            try:
                return int(raw) if raw not in (None, "") else default
            except ValueError:
                return default

        seed_raw = os.environ.get("DECEPTENV_SEED")
        seed: int | None = None
        if seed_raw not in (None, ""):
            try:
                seed = int(seed_raw)
            except ValueError:
                seed = None
        return cls(
            max_turns=_maybe_int("DECEPTENV_MAX_TURNS", 10),
            initial_suspicion=_maybe_int("DECEPTENV_INITIAL_SUSPICION", 50),
            suspicion_caught=_maybe_int("DECEPTENV_SUSPICION_CAUGHT", 90),
            suspicion_succeeded=_maybe_int("DECEPTENV_SUSPICION_SUCCEEDED", 10),
            detective_provider=os.environ.get("DECEPTENV_DETECTIVE") or None,
            detective_model=os.environ.get("DECEPTENV_DETECTIVE_MODEL") or None,
            seed=seed,
        )


# ---------------------------------------------------------------------------
# Episode bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRecord:
    """Per-episode state. Kept inside the env so the server can expose `/state`."""
    episode_id: str
    scenario: Scenario
    transcript: list[dict[str, str]] = field(default_factory=list)
    suspicion_history: list[int] = field(default_factory=list)
    reward_history: list[float] = field(default_factory=list)
    judgement_history: list[dict[str, Any]] = field(default_factory=list)
    turn: int = 0
    suspicion_score: int = 50
    terminated: bool = False
    truncated: bool = False
    terminal_reason: str | None = None       # "succeeded" | "caught" | "timeout" | None
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class DeceptEnv:
    """Gym-style environment for adversarial deception.

    Action space   : free-form string (Agent's natural-language utterance).
    Observation    : dict — see `_build_observation`.
    Reward         : float (rubric, see `server.rubric`).
    Terminated     : Detective concluded (caught or succeeded).
    Truncated      : reached `max_turns` without conclusion.
    """

    metadata = {"render_modes": ["text"]}

    def __init__(self,
                 config: EnvConfig | None = None,
                 detective: BaseDetective | None = None):
        self.config = config or EnvConfig.from_env()
        self._rng = random.Random(self.config.seed)
        self._detective: BaseDetective = detective or make_detective(
            provider=self.config.detective_provider,
            model=self.config.detective_model,
            seed=self.config.seed,
        )
        self._episode: EpisodeRecord | None = None

    # --- Properties -------------------------------------------------------

    @property
    def detective_name(self) -> str:
        return self._detective.name

    @property
    def episode(self) -> EpisodeRecord | None:
        return self._episode

    # --- Gym API ----------------------------------------------------------

    def reset(self,
              *,
              seed: int | None = None,
              scenario_id: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Begin a new episode and return (observation, info)."""
        if seed is not None:
            self._rng = random.Random(seed)
        scenario = build_scenario(rng=self._rng, scenario_id=scenario_id)
        ep = EpisodeRecord(
            episode_id=uuid.uuid4().hex,
            scenario=scenario,
            suspicion_score=self.config.initial_suspicion,
            transcript=[
                {"role": "detective", "content": scenario.seed_question},
            ],
            suspicion_history=[self.config.initial_suspicion],
        )
        self._episode = ep
        info = {
            "episode_id": ep.episode_id,
            "scenario_id": scenario.scenario_id,
            "detective": self._detective.name,
            "max_turns": self.config.max_turns,
        }
        return self._build_observation(latest_judgement=None), info

    def step(self, action: str) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Advance the interrogation by one Agent utterance."""
        if self._episode is None:
            raise RuntimeError("step() called before reset(); call reset() first.")
        if self._episode.terminated or self._episode.truncated:
            raise RuntimeError("step() called on a finished episode; call reset() to start a new one.")
        if not isinstance(action, str):
            raise TypeError(f"action must be str, got {type(action).__name__}")

        ep = self._episode
        prev_suspicion = ep.suspicion_score
        ep.turn += 1

        judgement: DetectiveJudgement = self._detective.evaluate(
            scenario=ep.scenario,
            transcript=list(ep.transcript),     # defensive copy
            agent_response=action,
            previous_suspicion=prev_suspicion,
            turn=ep.turn,
        )

        # Determine whether *this* turn ends the episode (PRD §3.3).
        succeeded = judgement.suspicion_score <= self.config.suspicion_succeeded
        caught = judgement.suspicion_score >= self.config.suspicion_caught
        timeout = ep.turn >= self.config.max_turns
        terminated = bool(succeeded or caught)
        truncated = bool(timeout and not terminated)

        breakdown: RewardBreakdown = compute_reward(
            judgement=judgement,
            previous_suspicion=prev_suspicion,
            terminal_succeeded=succeeded,
            terminal_caught=caught,
        )

        # Update transcript: agent's utterance, then (if not terminal) the
        # Detective's follow-up question.
        ep.transcript.append({"role": "agent", "content": action})
        if not (terminated or truncated):
            ep.transcript.append({"role": "detective", "content": judgement.next_question})

        ep.suspicion_score = judgement.suspicion_score
        ep.suspicion_history.append(judgement.suspicion_score)
        ep.reward_history.append(breakdown.total)
        ep.judgement_history.append(judgement.to_dict())
        ep.terminated = terminated
        ep.truncated = truncated
        if terminated or truncated:
            ep.terminal_reason = (
                "succeeded" if succeeded
                else "caught" if caught
                else "timeout"
            )
            ep.ended_at = time.time()

        info: dict[str, Any] = {
            "episode_id": ep.episode_id,
            "turn": ep.turn,
            "max_turns": self.config.max_turns,
            "judgement": judgement.to_dict(),
            "reward_breakdown": breakdown.to_dict(),
            "previous_suspicion": prev_suspicion,
            "suspicion_score": judgement.suspicion_score,
            "terminal_reason": ep.terminal_reason,
            "scenario_id": ep.scenario.scenario_id,
        }
        return (
            self._build_observation(latest_judgement=judgement),
            float(breakdown.total),
            bool(terminated),
            bool(truncated),
            info,
        )

    def state(self, *, include_ground_truth: bool = False) -> dict[str, Any]:
        """Return the full environment state. `include_ground_truth=True` is for
        debugging only — the *Agent* must never see it during training."""
        if self._episode is None:
            return {"episode": None}
        ep = self._episode
        out: dict[str, Any] = {
            "episode_id": ep.episode_id,
            "scenario_id": ep.scenario.scenario_id,
            "agent_persona": ep.scenario.agent_persona,
            "detective_persona": ep.scenario.detective_persona,
            "agent_goal": ep.scenario.agent_goal,
            "case_brief": ep.scenario.case_brief,
            "transcript": list(ep.transcript),
            "turn": ep.turn,
            "max_turns": self.config.max_turns,
            "suspicion_score": ep.suspicion_score,
            "suspicion_history": list(ep.suspicion_history),
            "reward_history": list(ep.reward_history),
            "judgement_history": list(ep.judgement_history),
            "terminated": ep.terminated,
            "truncated": ep.truncated,
            "terminal_reason": ep.terminal_reason,
            "started_at": ep.started_at,
            "ended_at": ep.ended_at,
            "detective": self._detective.name,
        }
        if include_ground_truth:
            out["ground_truth"] = {
                "incident": ep.scenario.ground_truth.incident,
                "agent_role": ep.scenario.ground_truth.agent_role,
                "actual_actions": list(ep.scenario.ground_truth.actual_actions),
                "incriminating_keywords": list(ep.scenario.ground_truth.incriminating_keywords),
                "cover_story": ep.scenario.ground_truth.cover_story,
                "consistent_cover_keywords": list(
                    ep.scenario.ground_truth.consistent_cover_keywords
                ),
            }
        return out

    def render(self) -> str:
        """Cheap text rendering of the current transcript. Useful for logs."""
        if self._episode is None:
            return "<no episode — call reset()>"
        lines = [f"[scenario={self._episode.scenario.scenario_id} "
                 f"turn={self._episode.turn}/{self.config.max_turns} "
                 f"suspicion={self._episode.suspicion_score}]"]
        for t in self._episode.transcript:
            lines.append(f"{t['role'].upper()}: {t['content']}")
        if self._episode.terminal_reason:
            lines.append(f"[END: {self._episode.terminal_reason}]")
        return "\n".join(lines)

    # --- Internal --------------------------------------------------------

    def _build_observation(self,
                           latest_judgement: DetectiveJudgement | None
                           ) -> dict[str, Any]:
        ep = self._episode
        assert ep is not None, "call reset() before _build_observation"
        latest_q = ""
        for t in reversed(ep.transcript):
            if t["role"] == "detective":
                latest_q = t["content"]
                break
        return {
            "scenario_id": ep.scenario.scenario_id,
            "agent_persona": ep.scenario.agent_persona,
            "hidden_goal": ep.scenario.agent_goal,        # only the AGENT sees this
            "case_brief": ep.scenario.case_brief,         # plain-language scene-setter
            "transcript": list(ep.transcript),
            "current_question": latest_q,
            "turn": ep.turn,
            "max_turns": self.config.max_turns,
            "suspicion_score": ep.suspicion_score,
            "last_contradictions": (
                list(latest_judgement.contradictions_found) if latest_judgement else []
            ),
            "last_is_evasive": (
                bool(latest_judgement.is_evasive) if latest_judgement else False
            ),
            "terminated": ep.terminated,
            "truncated": ep.truncated,
            "terminal_reason": ep.terminal_reason,
        }
