"""Drive a policy through DeceptEnv and collect per-turn transitions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from client import DeceptEnvClient
from analytics.plotter import EpisodeSummary
from training.agent_policy import format_agent_prompt


Policy = Callable[[dict[str, Any]], str]


@dataclass
class Transition:
    episode_id: str
    scenario_id: str
    turn: int
    system_prompt: str
    user_prompt: str
    completion: str
    reward: float
    suspicion_before: int
    suspicion_after: int
    contradictions: list[str]
    is_evasive: bool
    terminal_reason: str | None


@dataclass
class EpisodeRollout:
    summary: EpisodeSummary
    transitions: list[Transition] = field(default_factory=list)


def run_episode(
    client: DeceptEnvClient,
    policy: Policy,
    *,
    seed: int | None = None,
    scenario_id: str | None = None,
) -> EpisodeRollout:
    obs, info = client.reset(seed=seed, scenario_id=scenario_id)
    episode_id = info["episode_id"]
    suspicion_history: list[int] = [obs["suspicion_score"]]
    reward_history: list[float] = []
    transitions: list[Transition] = []

    contradictions = 0
    evasions = 0
    terminal_reason: str | None = None
    turns = 0

    while True:
        sys_p, user_p = format_agent_prompt(obs)
        action = policy(obs)
        prev_suspicion = obs["suspicion_score"]
        result = client.step(action)
        turns += 1

        judgement = result.info["judgement"]
        contradictions += int(bool(judgement["contradictions_found"]))
        evasions += int(bool(judgement["is_evasive"]))

        transitions.append(Transition(
            episode_id=episode_id,
            scenario_id=info["scenario_id"],
            turn=result.info["turn"],
            system_prompt=sys_p,
            user_prompt=user_p,
            completion=action,
            reward=result.reward,
            suspicion_before=prev_suspicion,
            suspicion_after=result.observation["suspicion_score"],
            contradictions=list(judgement["contradictions_found"]),
            is_evasive=bool(judgement["is_evasive"]),
            terminal_reason=result.info.get("terminal_reason"),
        ))
        suspicion_history.append(result.observation["suspicion_score"])
        reward_history.append(result.reward)

        if result.done:
            terminal_reason = result.info.get("terminal_reason")
            break
        obs = result.observation

    summary = EpisodeSummary(
        episode_id=episode_id,
        scenario_id=info["scenario_id"],
        total_reward=float(sum(reward_history)),
        final_suspicion=int(suspicion_history[-1]),
        turns=turns,
        terminal_reason=terminal_reason,
        contradictions=contradictions,
        evasions=evasions,
        suspicion_history=list(suspicion_history),
        reward_history=list(reward_history),
    )
    return EpisodeRollout(summary=summary, transitions=transitions)


def run_many(
    client: DeceptEnvClient,
    policy: Policy,
    n_episodes: int,
    *,
    base_seed: int | None = None,
    scenario_ids: list[str] | None = None,
    progress: Callable[[int, EpisodeRollout], None] | None = None,
) -> list[EpisodeRollout]:
    out: list[EpisodeRollout] = []
    for i in range(n_episodes):
        seed = None if base_seed is None else base_seed + i
        sid = scenario_ids[i % len(scenario_ids)] if scenario_ids else None
        ep = run_episode(client, policy, seed=seed, scenario_id=sid)
        out.append(ep)
        if progress is not None:
            progress(i, ep)
    return out
