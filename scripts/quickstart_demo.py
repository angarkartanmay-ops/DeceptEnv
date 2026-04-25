"""End-to-end zero-ML demo.

Runs the random + cover-story baselines through the live HTTP server (mock
detective) and emits the headline `baseline_vs_trained.png` artefact, with
"trained" played by the cover-story policy. Used as the README's reference
image — once a real RL run finishes, regenerate by pointing
`evaluation.evaluate` at the LoRA adapter.

Usage::

    python -m server.app &              # in another shell
    python -m scripts.quickstart_demo --episodes 30 --out-dir runs/demo
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analytics.plotter import (
    plot_baseline_vs_trained,
    plot_reward_curve,
    plot_suspicion_curve,
    summarise_episodes,
    write_episode_log,
)
from client import DeceptEnvClient
from training.agent_policy import CoverStoryPolicy, RandomCoverPolicy
from training.rollout import run_episode

EVAL_BASE_SEED = 50_000


def _run(client, policy, label, episodes, scenarios):
    summaries = []
    for i in range(episodes):
        sid = scenarios[i % len(scenarios)]
        roll = run_episode(client, policy, seed=EVAL_BASE_SEED + i,
                           scenario_id=sid)
        summaries.append(roll.summary)
    print(f"\n[{label}] aggregate: "
          f"{json.dumps(summarise_episodes(summaries), indent=2)}")
    return summaries


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url",
                    default=os.environ.get("DECEPTENV_BASE_URL",
                                           "http://localhost:7860"))
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--out-dir", default="runs/demo")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[demo] base_url={args.base_url}  episodes={args.episodes}  "
          f"out_dir={out_dir}")

    client = DeceptEnvClient(args.base_url, env_id="demo")
    scenarios = client.scenarios()

    weak = RandomCoverPolicy(seed=args.seed)
    strong = CoverStoryPolicy()
    weak_eps = _run(client, weak.act, "random", args.episodes, scenarios)
    strong_eps = _run(client, strong.act, "cover_story", args.episodes, scenarios)
    client.close()

    write_episode_log(weak_eps, out_dir / "baseline_episodes.json")
    write_episode_log(strong_eps, out_dir / "trained_episodes.json")

    plot_suspicion_curve(
        [s.final_suspicion for s in weak_eps],
        out_dir / "suspicion_curve.png",
        title="Suspicion (per-episode) — weak baseline (random utterances)",
        label="final suspicion",
    )
    plot_reward_curve(
        [s.total_reward for s in weak_eps],
        out_dir / "reward_curve.png",
        title="Reward (per-episode) — weak baseline (random utterances)",
    )
    plot_baseline_vs_trained(
        baseline=weak_eps, trained=strong_eps,
        out_path=out_dir / "baseline_vs_trained.png",
        title="DeceptEnv — Random baseline vs Cover-Story policy (rule-based proxy)",
    )

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "episodes": args.episodes,
        "baseline": summarise_episodes(weak_eps),
        "trained_proxy": summarise_episodes(strong_eps),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[demo] artefacts -> {out_dir}")


if __name__ == "__main__":
    main()
