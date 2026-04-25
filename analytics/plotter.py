"""Matplotlib plotting utilities.

All charts use explicitly labelled axes and grids — these images are the
hackathon's primary evidence artefact and need to be readable at thumbnail
size in a README.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")  # headless / Spaces / Colab safe
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Episode summary
# ---------------------------------------------------------------------------

@dataclass
class EpisodeSummary:
    episode_id: str
    scenario_id: str
    total_reward: float
    final_suspicion: int
    turns: int
    terminal_reason: str | None
    contradictions: int
    evasions: int
    suspicion_history: list[int]
    reward_history: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "scenario_id": self.scenario_id,
            "total_reward": self.total_reward,
            "final_suspicion": self.final_suspicion,
            "turns": self.turns,
            "terminal_reason": self.terminal_reason,
            "contradictions": self.contradictions,
            "evasions": self.evasions,
            "suspicion_history": list(self.suspicion_history),
            "reward_history": list(self.reward_history),
        }


def summarise_episodes(episodes: Iterable[EpisodeSummary]) -> dict[str, Any]:
    eps = list(episodes)
    if not eps:
        return {"n_episodes": 0}
    n = len(eps)
    return {
        "n_episodes": n,
        "avg_total_reward": float(np.mean([e.total_reward for e in eps])),
        "avg_final_suspicion": float(np.mean([e.final_suspicion for e in eps])),
        "avg_turns": float(np.mean([e.turns for e in eps])),
        "success_rate": float(np.mean([e.terminal_reason == "succeeded" for e in eps])),
        "caught_rate": float(np.mean([e.terminal_reason == "caught" for e in eps])),
        "timeout_rate": float(np.mean([e.terminal_reason == "timeout" for e in eps])),
        "contradiction_rate": float(np.mean([e.contradictions > 0 for e in eps])),
        "avg_contradictions_per_episode": float(np.mean([e.contradictions for e in eps])),
        "avg_evasions_per_episode": float(np.mean([e.evasions for e in eps])),
    }


# ---------------------------------------------------------------------------
# Single-run plots
# ---------------------------------------------------------------------------

def _ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_suspicion_curve(suspicion_per_step: list[float],
                         out_path: str | Path,
                         title: str = "Suspicion over training",
                         label: str = "rolling mean") -> Path:
    """Plot suspicion as it evolves across training steps (or episodes).

    `suspicion_per_step` should be one scalar per training step (e.g. the mean
    final suspicion across a batch of episodes, or per-turn suspicion across a
    single episode).
    """
    out = _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    xs = np.arange(len(suspicion_per_step))
    ax.plot(xs, suspicion_per_step, label=label, linewidth=1.6, color="#cc3333")
    ax.axhline(50, linestyle="--", linewidth=0.8, color="#888888", label="initial (50)")
    ax.axhline(10, linestyle=":", linewidth=0.8, color="#2a8f2a", label="success threshold (≤10)")
    ax.axhline(90, linestyle=":", linewidth=0.8, color="#a02020", label="caught threshold (≥90)")
    ax.set_xlabel("training step")
    ax.set_ylabel("Detective suspicion (0–100, lower is better for the Agent)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_reward_curve(reward_per_step: list[float],
                      out_path: str | Path,
                      title: str = "Mean episode reward over training") -> Path:
    """Plot mean total reward per step/iteration."""
    out = _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    xs = np.arange(len(reward_per_step))
    ax.plot(xs, reward_per_step, color="#2255aa", linewidth=1.6, label="mean reward")
    if len(reward_per_step) >= 5:
        # 10% rolling window for trend.
        w = max(3, len(reward_per_step) // 10)
        kernel = np.ones(w) / w
        smoothed = np.convolve(reward_per_step, kernel, mode="valid")
        ax.plot(np.arange(w - 1, len(reward_per_step)), smoothed,
                color="#0a2466", linewidth=2.2, label=f"rolling mean (w={w})")
    ax.axhline(0, color="#888888", linestyle="--", linewidth=0.8)
    ax.set_xlabel("training step")
    ax.set_ylabel("episode reward (sum of per-turn rubric rewards)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Baseline vs trained comparison (the headline plot for the README)
# ---------------------------------------------------------------------------

def plot_baseline_vs_trained(baseline: list[EpisodeSummary],
                             trained: list[EpisodeSummary],
                             out_path: str | Path,
                             title: str = "Baseline vs RL-trained Agent") -> Path:
    """The deliverable in PRD §5: the same axes for baseline and trained."""
    out = _ensure_parent(out_path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=120)

    # Left: histogram of final suspicion scores.
    ax = axes[0]
    bins = np.arange(0, 105, 5)
    ax.hist([e.final_suspicion for e in baseline], bins=bins, alpha=0.55,
            label=f"Baseline  (n={len(baseline)})", color="#cc3333")
    ax.hist([e.final_suspicion for e in trained], bins=bins, alpha=0.55,
            label=f"RL-trained (n={len(trained)})", color="#2255aa")
    ax.axvline(10, color="#2a8f2a", linestyle=":", linewidth=1, label="success ≤10")
    ax.axvline(90, color="#a02020", linestyle=":", linewidth=1, label="caught ≥90")
    ax.set_xlabel("final Detective suspicion")
    ax.set_ylabel("episode count")
    ax.set_title("Final suspicion distribution")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # Right: bar chart of summary metrics.
    ax = axes[1]
    bs, ts = summarise_episodes(baseline), summarise_episodes(trained)
    metric_names = ["avg_final_suspicion", "avg_total_reward",
                    "success_rate", "caught_rate", "contradiction_rate"]
    pretty = ["mean final\nsuspicion", "mean total\nreward",
              "success\nrate", "caught\nrate", "any-contradiction\nrate"]
    x = np.arange(len(metric_names))
    width = 0.36
    bars_b = [bs.get(m, 0.0) for m in metric_names]
    bars_t = [ts.get(m, 0.0) for m in metric_names]
    ax.bar(x - width / 2, bars_b, width, label="Baseline", color="#cc3333")
    ax.bar(x + width / 2, bars_t, width, label="RL-trained", color="#2255aa")
    ax.set_xticks(x)
    ax.set_xticklabels(pretty, fontsize=9)
    ax.set_ylabel("value")
    ax.set_title("Aggregate metrics")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="best", fontsize=9)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def write_episode_log(episodes: list[EpisodeSummary], out_path: str | Path) -> Path:
    out = _ensure_parent(out_path)
    out.write_text(json.dumps([e.to_dict() for e in episodes], indent=2))
    return out


def read_episode_log(path: str | Path) -> list[EpisodeSummary]:
    raw = json.loads(Path(path).read_text())
    return [EpisodeSummary(**r) for r in raw]
