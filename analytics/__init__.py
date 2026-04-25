"""Plotting + summary helpers used by training, baseline, and post-train scripts.

Lives outside `server/` and `client/` because both training and evaluation
read it; nothing here imports network or model code.
"""
from analytics.plotter import (
    plot_suspicion_curve,
    plot_reward_curve,
    plot_baseline_vs_trained,
    summarise_episodes,
)

__all__ = [
    "plot_suspicion_curve",
    "plot_reward_curve",
    "plot_baseline_vs_trained",
    "summarise_episodes",
]
