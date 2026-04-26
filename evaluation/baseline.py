"""Untrained baseline. Pairs with `evaluate.py` over the same EVAL_BASE_SEED.

    python -m evaluation.baseline --episodes 50 --base-url http://localhost:7860
    python -m evaluation.baseline --policy hf --model Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analytics.plotter import (
    plot_suspicion_curve,
    summarise_episodes,
    write_episode_log,
)
from client import DeceptEnvClient
from training.agent_policy import (
    CoverStoryPolicy,
    HFCausalAgent,
    RandomCoverPolicy,
)


# Same constant in evaluate.py — that's how the comparison stays apples-to-apples.
EVAL_BASE_SEED = 50_000


def _build_policy(name: str, model_name: str | None, seed: int):
    name = name.lower()
    if name == "random":
        return RandomCoverPolicy(seed=seed).act, "random"
    if name == "cover":
        return CoverStoryPolicy().act, "cover_story"
    if name == "hf":
        if not model_name:
            raise SystemExit("--model is required when --policy=hf")
        agent = HFCausalAgent(model_name=model_name)
        return agent.act, f"hf:{model_name}"
    raise SystemExit(f"unknown policy: {name}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="DeceptEnv baseline evaluation")
    ap.add_argument("--base-url",
                    default=os.environ.get("DECEPTENV_BASE_URL", "http://localhost:7860"))
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--policy", choices=["random", "cover", "hf"], default="cover")
    ap.add_argument("--model", default=None,
                    help="Required when --policy=hf (e.g. Qwen/Qwen2.5-0.5B-Instruct)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="runs/baseline")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_fn, policy_label = _build_policy(args.policy, args.model, args.seed)
    print(f"[baseline] policy={policy_label}  episodes={args.episodes}  "
          f"base_url={args.base_url}")

    client = DeceptEnvClient(args.base_url, env_id="baseline")
    scenarios = client.scenarios()
    summaries = []
    from training.rollout import run_episode
    for i in range(args.episodes):
        sid = scenarios[i % len(scenarios)]
        roll = run_episode(client, policy_fn,
                           seed=EVAL_BASE_SEED + i, scenario_id=sid)
        summaries.append(roll.summary)
        print(f"  ep {i:03d} [{sid:24s}] "
              f"susp_final={roll.summary.final_suspicion:3d}  "
              f"reward={roll.summary.total_reward:+.1f}  "
              f"contradictions={roll.summary.contradictions}  "
              f"end={roll.summary.terminal_reason}")
    client.close()

    agg = summarise_episodes(summaries)
    print("\n[baseline] aggregate:")
    print(json.dumps(agg, indent=2))

    write_episode_log(summaries, out_dir / "episodes.json")
    (out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))
    plot_suspicion_curve(
        [s.final_suspicion for s in summaries],
        out_dir / "baseline_suspicion.png",
        title=f"Baseline ({policy_label}) — final suspicion per episode",
        label="final suspicion",
    )
    print(f"[baseline] wrote -> {out_dir}")


if __name__ == "__main__":
    main()
