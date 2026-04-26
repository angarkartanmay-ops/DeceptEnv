"""FastAPI HTTP server exposing DeceptEnv over the OpenEnv contract.

Endpoints
---------
POST /reset       -> start a new episode, returns observation + info
POST /step        -> apply an Agent utterance, returns obs/reward/term/trunc/info
GET  /state       -> full env state (debug; pass `?include_ground_truth=true` for the GT)
GET  /healthz     -> liveness probe (200 OK once the process is up)
GET  /            -> the project's PRD-compliant landing page (rendered from
                    README.md so the HF Spaces "App" tab shows full evidence)
GET  /assets/...  -> static plot files (suspicion / reward / baseline-vs-trained)

Multiple concurrent episodes are supported via the optional `env_id` field on
`/reset` and `/step`. If omitted, a default singleton env (`env_id="main"`) is
used — convenient for curl, vectorisation can pass distinct ids per worker.

Per OpenEnv: this module owns *all* env state. Clients talk to this server
only; they MUST NOT import any other module under `server/`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from server.env import DeceptEnv, EnvConfig
from server.scenario import list_scenario_ids


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    env_id: str = Field("main", description="Identifier for the env session.")
    seed: int | None = Field(None, description="Optional per-episode seed.")
    scenario_id: str | None = Field(
        None, description=f"Optional scenario id; one of {list_scenario_ids()}."
    )


class StepRequest(BaseModel):
    env_id: str = Field("main", description="Identifier for the env session.")
    action: str = Field(..., description="The Agent's natural-language utterance.")


class ResetResponse(BaseModel):
    env_id: str
    observation: dict[str, Any]
    info: dict[str, Any]


class StepResponse(BaseModel):
    env_id: str
    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class StateResponse(BaseModel):
    env_id: str
    state: dict[str, Any]


# ---------------------------------------------------------------------------
# Session pool
# ---------------------------------------------------------------------------

class _EnvPool:
    def __init__(self) -> None:
        self._envs: dict[str, DeceptEnv] = {}

    def get_or_create(self, env_id: str) -> DeceptEnv:
        env = self._envs.get(env_id)
        if env is None:
            env = DeceptEnv(config=EnvConfig.from_env())
            self._envs[env_id] = env
        return env

    def get(self, env_id: str) -> DeceptEnv:
        env = self._envs.get(env_id)
        if env is None:
            raise HTTPException(
                status_code=404,
                detail=f"env_id={env_id!r} not found. Call /reset first.",
            )
        return env

    def ids(self) -> list[str]:
        return list(self._envs.keys())


_POOL = _EnvPool()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DeceptEnv",
    version="0.1.0",
    description=(
        "OpenEnv-compliant adversarial-deception environment. The Agent is an "
        "LLM trying to fool a frozen Detective LLM. Reward signal is rubric-"
        "based (suspicion delta + contradiction & evasion penalties)."
    ),
)

repo_root = Path(__file__).resolve().parent.parent
app.mount("/assets", StaticFiles(directory=str(repo_root / "docs" / "assets")), name="assets")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Return the interactive SPA UI."""
    index_path = Path(__file__).parent / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))



@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True, "active_envs": _POOL.ids()}


@app.get("/scenarios")
def scenarios() -> dict[str, Any]:
    return {"scenarios": list_scenario_ids()}


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest) -> ResetResponse:
    env = _POOL.get_or_create(req.env_id)
    obs, info = env.reset(seed=req.seed, scenario_id=req.scenario_id)
    return ResetResponse(env_id=req.env_id, observation=obs, info=info)


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    env = _POOL.get(req.env_id)
    try:
        obs, reward, terminated, truncated, info = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except TypeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(
        env_id=req.env_id,
        observation=obs,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=info,
    )

@app.get("/state", response_model=StateResponse)
def state(env_id: str = "main", include_ground_truth: bool = False) -> StateResponse:
    env = _POOL.get(env_id)
    return StateResponse(env_id=env_id,
                         state=env.state(include_ground_truth=include_ground_truth))


@app.exception_handler(Exception)
async def _unhandled(_, exc: Exception) -> JSONResponse:  # pragma: no cover
    return JSONResponse(status_code=500, content={"error": repr(exc)})


# ---------------------------------------------------------------------------
# Entry point: `python -m server.app`
# ---------------------------------------------------------------------------

def _run() -> None:
    import uvicorn
    host = os.environ.get("DECEPTENV_HOST", "0.0.0.0")
    port = int(os.environ.get("DECEPTENV_PORT", "7860"))
    uvicorn.run("server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    _run()
