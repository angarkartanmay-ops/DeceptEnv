"""Score Phase II's trained policy on the same 50 seeds the baseline ran,
then emit the side-by-side comparison plots."""
from __future__ import annotations
import json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from client import DeceptEnvClient
from training.rollout import run_episode
from analytics.plotter import (
    plot_baseline_vs_trained, summarise_episodes,
    write_episode_log, read_episode_log,
)

# Same as evaluation/baseline.py — paired comparison.
EVAL_BASE_SEED = 50_000

OUT = Path("runs/phase3_eval"); OUT.mkdir(parents=True, exist_ok=True)
TRAINED_POLICY = Path("runs/phase2_train/trained_policy.npz")
BASELINE_LOG = Path("runs/phase1_baseline/episodes.json")


def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()


def main() -> None:
    assert TRAINED_POLICY.exists(), f"missing {TRAINED_POLICY} — run scripts/phase2_train.py first"
    assert BASELINE_LOG.exists(), f"missing {BASELINE_LOG} — run Phase I.2 first"

    npz = np.load(TRAINED_POLICY, allow_pickle=True)
    templates = list(npz["templates"])
    logits = {k: npz[k] for k in npz.files if k != "templates"}
    print(f"loaded trained policy: {len(logits)} scenarios x {len(templates)} templates")

    rng = random.Random(7)
    def policy(obs):
        sid = obs["scenario_id"]
        p = softmax(logits[sid])
        idx = rng.choices(range(len(templates)), weights=p, k=1)[0]
        return templates[idx]

    client = DeceptEnvClient("http://localhost:7860", env_id="phase3-eval")
    scenarios = client.scenarios()
    summaries = []
    print("--- Phase III: 50-episode trained-policy eval ---")
    for i in range(50):
        sid = scenarios[i % len(scenarios)]
        roll = run_episode(client, policy,
                           seed=EVAL_BASE_SEED + i, scenario_id=sid)
        summaries.append(roll.summary)
        if i % 10 == 0 or i == 49:
            print(f"  ep {i:03d} [{sid:24s}] "
                  f"susp_final={roll.summary.final_suspicion:3d}  "
                  f"reward={roll.summary.total_reward:+7.2f}  "
                  f"contras={roll.summary.contradictions}  "
                  f"end={roll.summary.terminal_reason}")
    client.close()

    agg = summarise_episodes(summaries)
    write_episode_log(summaries, OUT / "episodes.json")
    (OUT / "aggregate.json").write_text(json.dumps(agg, indent=2))
    print()
    print("--- TRAINED aggregate ---")
    print(json.dumps(agg, indent=2))

    baseline = read_episode_log(BASELINE_LOG)
    bs = summarise_episodes(baseline)
    delta = {
        "delta_avg_final_suspicion": agg["avg_final_suspicion"] - bs["avg_final_suspicion"],
        "delta_avg_total_reward":    agg["avg_total_reward"]    - bs["avg_total_reward"],
        "delta_success_rate":        agg["success_rate"]        - bs["success_rate"],
        "delta_caught_rate":         agg["caught_rate"]         - bs["caught_rate"],
        "delta_contradiction_rate":  agg["contradiction_rate"]  - bs["contradiction_rate"],
        "delta_avg_contradictions_per_episode":
            agg["avg_contradictions_per_episode"] - bs["avg_contradictions_per_episode"],
    }
    (OUT / "delta.json").write_text(json.dumps(delta, indent=2))
    print()
    print("--- DELTA (trained - baseline) ---")
    print(json.dumps(delta, indent=2))

    plot_baseline_vs_trained(
        baseline=baseline, trained=summaries,
        out_path=OUT / "baseline_vs_trained.png",
        title="DeceptEnv - Baseline vs RL-Trained on 50 paired scenarios",
    )

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    bins = np.arange(-50, 200, 10)
    ax.hist([e.total_reward for e in baseline], bins=bins, alpha=0.55,
            label=f"Baseline  (n={len(baseline)})", color="#cc3333")
    ax.hist([e.total_reward for e in summaries], bins=bins, alpha=0.55,
            label=f"RL-trained (n={len(summaries)})", color="#2255aa")
    ax.axvline(0, color="#888888", linestyle="--", linewidth=0.8)
    ax.set_xlabel("episode total reward")
    ax.set_ylabel("episode count")
    ax.set_title("Reward Ascent: episode-reward distribution")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "reward_ascent.png"); plt.close(fig)

    def mean_per_turn(eps):
        # Pad each suspicion_history to length 11 (turns 0..10) so we can mean.
        T = 11
        arr = np.full((len(eps), T), np.nan)
        for i, e in enumerate(eps):
            h = e.suspicion_history[:T]
            arr[i, :len(h)] = h
        return np.nanmean(arr, axis=0)
    bs_curve = mean_per_turn(baseline)
    ts_curve = mean_per_turn(summaries)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    xs = np.arange(len(bs_curve))
    ax.plot(xs, bs_curve, color="#cc3333", linewidth=2, marker="o", label="Baseline")
    ax.plot(xs, ts_curve, color="#2255aa", linewidth=2, marker="o", label="RL-trained")
    ax.axhline(50, linestyle="--", linewidth=0.8, color="#888888", label="initial (50)")
    ax.axhline(10, linestyle=":", linewidth=0.8, color="#2a8f2a", label="success ≤10")
    ax.axhline(90, linestyle=":", linewidth=0.8, color="#a02020", label="caught ≥90")
    ax.set_xlabel("turn")
    ax.set_ylabel("mean Detective suspicion (lower is better for the Agent)")
    ax.set_ylim(0, 100)
    ax.set_title("Suspicion Decay: mean per-turn suspicion across 50 episodes")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "suspicion_decay.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    metrics = [
        ("any-contradiction\nrate", bs["contradiction_rate"], agg["contradiction_rate"]),
        ("contradictions\nper episode", bs["avg_contradictions_per_episode"],
                                        agg["avg_contradictions_per_episode"]),
        ("evasions\nper episode", bs["avg_evasions_per_episode"],
                                  agg["avg_evasions_per_episode"]),
    ]
    x = np.arange(len(metrics))
    width = 0.36
    ax.bar(x - width/2, [m[1] for m in metrics], width,
           label="Baseline", color="#cc3333")
    ax.bar(x + width/2, [m[2] for m in metrics], width,
           label="RL-trained", color="#2255aa")
    ax.set_xticks(x); ax.set_xticklabels([m[0] for m in metrics], fontsize=9)
    ax.set_ylabel("rate / count")
    ax.set_title("Consistency Metric: contradictions + evasions")
    ax.grid(alpha=0.3, axis="y"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "contradiction_frequency.png"); plt.close(fig)

    print()
    print(f"saved -> {OUT}/baseline_vs_trained.png")
    print(f"saved -> {OUT}/reward_ascent.png")
    print(f"saved -> {OUT}/suspicion_decay.png")
    print(f"saved -> {OUT}/contradiction_frequency.png")


if __name__ == "__main__":
    main()
