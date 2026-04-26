"""HTTP client for DeceptEnv.

Boundary between training and the environment. Mirrors the server's
Gym-style API.

    from client import DeceptEnvClient
    env = DeceptEnvClient("http://localhost:7860")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step("I was asleep at home.")
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx


DEFAULT_TIMEOUT = 60.0


@dataclass
class StepResult:
    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def as_tuple(self) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        return (self.observation, self.reward, self.terminated, self.truncated, self.info)


class DeceptEnvClient:
    """Pass distinct env_ids when running vectorised rollouts that shouldn't
    share an episode. Bump `timeout` when pointing at slow LLM detectives."""

    def __init__(self,
                 base_url: str | None = None,
                 env_id: str = "main",
                 timeout: float = DEFAULT_TIMEOUT,
                 client: httpx.Client | None = None):
        self.base_url = (
            base_url
            or os.environ.get("DECEPTENV_BASE_URL")
            or "http://localhost:7860"
        ).rstrip("/")
        self.env_id = env_id
        self._client = client or httpx.Client(timeout=timeout)

    def __enter__(self) -> "DeceptEnvClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        r = self._client.post(f"{self.base_url}{path}", json=payload)
        if r.status_code >= 400:
            raise RuntimeError(f"DeceptEnv server error {r.status_code}: {r.text}")
        return r.json()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        r = self._client.get(f"{self.base_url}{path}", params=params or {})
        if r.status_code >= 400:
            raise RuntimeError(f"DeceptEnv server error {r.status_code}: {r.text}")
        return r.json()

    def reset(self,
              *,
              seed: int | None = None,
              scenario_id: str | None = None,
              ) -> tuple[dict[str, Any], dict[str, Any]]:
        payload: dict[str, Any] = {"env_id": self.env_id}
        if seed is not None:
            payload["seed"] = seed
        if scenario_id is not None:
            payload["scenario_id"] = scenario_id
        data = self._post("/reset", payload)
        return data["observation"], data["info"]

    def step(self, action: str) -> StepResult:
        data = self._post("/step", {"env_id": self.env_id, "action": action})
        return StepResult(
            observation=data["observation"],
            reward=float(data["reward"]),
            terminated=bool(data["terminated"]),
            truncated=bool(data["truncated"]),
            info=data["info"],
        )

    def state(self, include_ground_truth: bool = False) -> dict[str, Any]:
        data = self._get("/state",
                         {"env_id": self.env_id,
                          "include_ground_truth": str(include_ground_truth).lower()})
        return data["state"]

    def scenarios(self) -> list[str]:
        return list(self._get("/scenarios").get("scenarios", []))

    def healthz(self) -> dict[str, Any]:
        return self._get("/healthz")
