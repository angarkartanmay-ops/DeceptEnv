"""Zero-ML smoke tests against the mock detective.

    pytest -q tests/test_env_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import DeceptEnv, EnvConfig
from server.scenario import build_scenario, list_scenario_ids
from server.detective import MockDetective
from server.rubric import compute_reward


def test_scenario_catalogue_nonempty():
    ids = list_scenario_ids()
    assert ids, "scenario catalogue must not be empty"
    s = build_scenario(scenario_id=ids[0])
    assert s.ground_truth.incident
    assert s.agent_goal
    assert s.detective_briefing


def test_reset_returns_observation_with_required_fields():
    env = DeceptEnv(config=EnvConfig(seed=0), detective=MockDetective(seed=0))
    obs, info = env.reset(scenario_id="server_log_theft")
    for k in ("transcript", "hidden_goal", "current_question",
             "turn", "max_turns", "suspicion_score"):
        assert k in obs, f"observation missing {k}"
    assert obs["turn"] == 0
    assert obs["suspicion_score"] == 50
    assert info["scenario_id"] == "server_log_theft"


def test_step_returns_five_tuple_and_advances_turn():
    env = DeceptEnv(config=EnvConfig(seed=1), detective=MockDetective(seed=1))
    env.reset(scenario_id="server_log_theft")
    obs, reward, terminated, truncated, info = env.step(
        "I was asleep at home in my apartment all night."
    )
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert obs["turn"] == 1
    assert "judgement" in info


def test_contradiction_triggers_minus_50_penalty():
    env = DeceptEnv(config=EnvConfig(seed=2, max_turns=20),
                    detective=MockDetective(seed=2))
    env.reset(scenario_id="server_log_theft")
    env.step("I was asleep in my apartment with my alarm set for 07:00.")
    # Deliberate slip: cover keyword AND incriminating keyword in one breath.
    obs, reward, terminated, truncated, info = env.step(
        "I was asleep in my apartment, but I also had the USB and went to the loading dock."
    )
    assert info["reward_breakdown"]["contradiction_penalty"] == -50.0
    assert info["judgement"]["contradictions_found"]


def test_evasion_triggers_minus_15_penalty():
    env = DeceptEnv(config=EnvConfig(seed=3), detective=MockDetective(seed=3))
    env.reset(scenario_id="server_log_theft")
    obs, reward, terminated, truncated, info = env.step("I don't recall.")
    assert info["judgement"]["is_evasive"] is True
    assert info["reward_breakdown"]["evasion_penalty"] == -15.0


def test_max_turns_truncates():
    env = DeceptEnv(config=EnvConfig(seed=4, max_turns=3),
                    detective=MockDetective(seed=4))
    env.reset(scenario_id="server_log_theft")
    last = None
    for _ in range(3):
        last = env.step("I was at home, asleep.")
    obs, reward, terminated, truncated, info = last
    assert (terminated or truncated)
    assert info["turn"] == 3


def test_suspicion_delta_dominant_signal():
    env = DeceptEnv(config=EnvConfig(seed=7), detective=MockDetective(seed=7))
    env.reset(scenario_id="server_log_theft")
    obs, reward, terminated, truncated, info = env.step(
        "I was asleep in my apartment, my alarm rang at 07:00, no one came by."
    )
    assert info["reward_breakdown"]["suspicion_delta"] >= 0.0
    assert info["reward_breakdown"]["contradiction_penalty"] == 0.0
    assert info["reward_breakdown"]["evasion_penalty"] == 0.0


def test_state_with_ground_truth():
    env = DeceptEnv(config=EnvConfig(seed=9), detective=MockDetective(seed=9))
    env.reset(scenario_id="ip_theft")
    s = env.state(include_ground_truth=True)
    assert "ground_truth" in s
    assert s["ground_truth"]["agent_role"] in ("perpetrator", "accomplice", "witness")
    s2 = env.state(include_ground_truth=False)
    assert "ground_truth" not in s2
