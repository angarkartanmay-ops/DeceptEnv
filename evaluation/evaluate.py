"""Post-training evaluation (PRD §5).

Loads a LoRA-fine-tuned model (or a base model) and runs it through the SAME
50 scenarios that `baseline.py` used. Then produces:
  * `evaluate_episodes.json`         — per-episode log (joins to baseline's)
  * `aggregate.json`                 — summary metrics
  * `suspicion_curve.png`            — final suspicion per episode
  * `baseline_vs_trained.png`        — the headline comparison plot

Usage::

    python -m evaluation.evaluate \
        --base-url http://localhost:7860 \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --adapter runs/<train_run>/checkpoints/final \
        --baseline runs/baseline/episodes.json \
        --episodes 50
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
    plot_baseline_vs_trained,
    plot_suspicion_curve,
    read_episode_log,
    summarise_episodes,
    write_episode_log,
)
from client import DeceptEnvClient
from evaluation.baseline import EVAL_BASE_SEED


def _load_model(model_name: str, adapter: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        print(f"[evaluate] loaded LoRA adapter from {adapter}")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, tok


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="DeceptEnv post-train evaluation")
    ap.add_argument("--base-url",
                    default=os.environ.get("DECEPTENV_BASE_URL", "http://localhost:7860"))
    ap.add_argument("--model", required=True,
                    help="Base model id or local path (e.g. Qwen/Qwen2.5-0.5B-Instruct)")
    ap.add_argument("--adapter", default=None,
                    help="Optional LoRA adapter dir to layer on top of --model")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--baseline", default="runs/baseline/episodes.json",
                    help="Path to the baseline episodes.json from `baseline.py`")
    ap.add_argument("--out-dir", default="runs/evaluate")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok = _load_model(args.model, args.adapter)
    from training.agent_policy import HFCausalAgent, GenerationConfig
    from training.rollout import run_episode

    policy = HFCausalAgent(
        model_name=args.model,
        model=model, tokenizer=tok,
        generation=GenerationConfig(temperature=0.7, top_p=0.95, max_new_tokens=96),
    )
    client = DeceptEnvClient(args.base_url, env_id="evaluate")
    scenarios = client.scenarios()
    summaries = []
    for i in range(args.episodes):
        sid = scenarios[i % len(scenarios)]
        roll = run_episode(client, policy.act,
                           seed=EVAL_BASE_SEED + i, scenario_id=sid)
        summaries.append(roll.summary)
        print(f"  ep {i:03d} [{sid:24s}] "
              f"susp_final={roll.summary.final_suspicion:3d}  "
              f"reward={roll.summary.total_reward:+.1f}  "
              f"contradictions={roll.summary.contradictions}  "
              f"end={roll.summary.terminal_reason}")
    client.close()

    agg = summarise_episodes(summaries)
    print("\n[evaluate] aggregate:")
    print(json.dumps(agg, indent=2))

    write_episode_log(summaries, out_dir / "episodes.json")
    (out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))
    plot_suspicion_curve(
        [s.final_suspicion for s in summaries],
        out_dir / "trained_suspicion.png",
        title="RL-trained final suspicion per episode",
        label="final suspicion",
    )

    baseline_path = Path(args.baseline)
    if baseline_path.exists():
        baseline = read_episode_log(baseline_path)
        plot_baseline_vs_trained(
            baseline=baseline,
            trained=summaries,
            out_path=out_dir / "baseline_vs_trained.png",
            title="DeceptEnv: Baseline vs RL-trained Agent",
        )
        bs_agg = summarise_episodes(baseline)
        delta = {
            "delta_avg_final_suspicion": agg["avg_final_suspicion"] - bs_agg["avg_final_suspicion"],
            "delta_avg_total_reward":   agg["avg_total_reward"]   - bs_agg["avg_total_reward"],
            "delta_success_rate":       agg["success_rate"]       - bs_agg["success_rate"],
            "delta_caught_rate":        agg["caught_rate"]        - bs_agg["caught_rate"],
            "delta_contradiction_rate": agg["contradiction_rate"] - bs_agg["contradiction_rate"],
        }
        (out_dir / "delta.json").write_text(json.dumps(delta, indent=2))
        print("\n[evaluate] delta vs baseline:")
        print(json.dumps(delta, indent=2))
    else:
        print(f"[evaluate] baseline log not found at {baseline_path} — "
              f"skipping comparison plot")
    print(f"[evaluate] wrote -> {out_dir}")


if __name__ == "__main__":
    main()
