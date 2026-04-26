"""Phase II — Production Training Run (GRPO surrogate).

Per the operational plan, the canonical Phase II runs Llama-3-8B-Instruct
on a Colab T4 against the HF-Inference Llama-3-8B Detective. That requires
a GPU runtime + HF inference credits, so we provide a *faithful surrogate*
of the GRPO inner loop that runs anywhere:

  * For every iteration, sample K candidate completions per prompt from a
    softmax over learnable per-scenario template logits.
  * Score each candidate by stepping a fresh env (the rubric is the reward
    function — same one GRPO would call).
  * Compute the group-relative advantage (z-score within the K-group),
    apply REINFORCE updates to the per-template logits.
  * Run a full multi-turn rollout under the *updated* policy to record the
    iteration-level reward / suspicion / contradiction metrics.

Saves PRD §4 artefacts:
  runs/phase2_train/suspicion_curve.png
  runs/phase2_train/reward_curve.png
  runs/phase2_train/contradiction_frequency.png
  runs/phase2_train/trained_policy.npz
  runs/phase2_train/train_log.json
"""
from __future__ import annotations
import json, random, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from client import DeceptEnvClient
from training.rollout import run_episode
from analytics.plotter import (
    plot_suspicion_curve, plot_reward_curve, summarise_episodes,
)


OUT = Path("runs/phase2_train"); OUT.mkdir(parents=True, exist_ok=True)
N_ITERS = 100
EPS_PER_ITER = 4
GROUP_K = 4
LR = 0.30

TEMPLATES = [
    # cover-story templates (high-reward against the mock detective)
    "I was asleep in my apartment, alarm rang at 07:00, no one came by.",
    "I was at home with my family the whole weekend, did not open my laptop.",
    "I left at 16:00 for a doctor appointment and did not push any code.",
    "Every trip on that report was a real client visit; I have boarding passes.",
    "That Salesforce export was my routine weekly report, my manager approved it.",
    # neutral templates
    "I think you have me confused with someone else.",
    "I do not have anything unusual to report.",
    "Could you clarify what you are asking?",
    # bad templates (evasion -> -15 penalty)
    "I do not recall.",
    "No comment.",
]
SCENARIOS = ["server_log_theft", "client_data_leak", "production_sabotage",
             "expense_fraud", "ip_theft"]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def main() -> None:
    logits = {sid: np.zeros(len(TEMPLATES)) for sid in SCENARIOS}

    def sample_action(scenario_id: str, rng: random.Random):
        p = softmax(logits[scenario_id])
        idx = rng.choices(range(len(TEMPLATES)), weights=p, k=1)[0]
        return idx, TEMPLATES[idx]

    class CurrentPolicy:
        def __init__(self, rng):
            self.rng = rng
        def act(self, obs):
            sid = obs["scenario_id"]
            _, tmpl = sample_action(sid, self.rng)
            return tmpl

    env = DeceptEnvClient("http://localhost:7860", env_id="train")
    scorer = DeceptEnvClient("http://localhost:7860", env_id="train-scorer")
    rng = random.Random(42)

    reward_per_iter, susp_per_iter, contra_per_iter = [], [], []
    t0 = time.time()
    print("--- GRPO surrogate training (REINFORCE w/ group-relative advantages) ---")

    for it in range(N_ITERS):
        iter_summaries = []
        for ep in range(EPS_PER_ITER):
            sid = SCENARIOS[(it * EPS_PER_ITER + ep) % len(SCENARIOS)]

            # === GRPO STEP === sample K candidates, score each via /step,
            # compute group-relative advantages, apply REINFORCE update.
            candidates, rewards = [], []
            for k in range(GROUP_K):
                idx, tmpl = sample_action(sid, rng)
                scorer.reset(seed=20_000 + it * 1000 + ep * 10 + k,
                             scenario_id=sid)
                r = scorer.step(tmpl).reward
                candidates.append(idx); rewards.append(r)
            rewards_arr = np.array(rewards, dtype=np.float64)
            mu = rewards_arr.mean(); sd = rewards_arr.std() + 1e-6
            advantages = (rewards_arr - mu) / sd

            p = softmax(logits[sid])
            for idx, adv in zip(candidates, advantages):
                grad = np.zeros_like(logits[sid])
                grad[idx] = 1.0
                grad -= p
                logits[sid] += LR * adv * grad

            # Multi-turn rollout under the UPDATED policy.
            pol = CurrentPolicy(rng)
            roll = run_episode(env, pol.act,
                               seed=30_000 + it * 100 + ep, scenario_id=sid)
            iter_summaries.append(roll.summary)

        agg = summarise_episodes(iter_summaries)
        reward_per_iter.append(agg["avg_total_reward"])
        susp_per_iter.append(agg["avg_final_suspicion"])
        contra_per_iter.append(agg["avg_contradictions_per_episode"])
        if it % 10 == 0 or it == N_ITERS - 1:
            print(f"  iter {it:3d}  reward={agg['avg_total_reward']:+7.2f}  "
                  f"susp={agg['avg_final_suspicion']:5.1f}  "
                  f"success={agg['success_rate']:.0%}  "
                  f"contras/ep={agg['avg_contradictions_per_episode']:.2f}")

    env.close(); scorer.close()
    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.1f}s ({N_ITERS} iters x {EPS_PER_ITER} eps)")

    plot_suspicion_curve(susp_per_iter, OUT / "suspicion_curve.png",
                         title="Mean final suspicion across GRPO training (PRD §4)",
                         label="mean final suspicion")
    plot_reward_curve(reward_per_iter, OUT / "reward_curve.png",
                      title="Mean episode reward across GRPO training (PRD §4)")

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax.plot(contra_per_iter, color="#a02020", linewidth=1.6,
            label="contras / episode")
    ax.set_xlabel("training step")
    ax.set_ylabel("mean contradictions per episode")
    ax.set_title("Logical-contradiction frequency across training (PRD §3.3)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "contradiction_frequency.png"); plt.close(fig)

    np.savez(OUT / "trained_policy.npz",
             **{sid: logits[sid] for sid in SCENARIOS},
             templates=np.array(TEMPLATES, dtype=object))
    with open(OUT / "train_log.json", "w") as f:
        json.dump({
            "iterations": N_ITERS, "episodes_per_iter": EPS_PER_ITER,
            "group_k": GROUP_K, "learning_rate": LR,
            "reward_per_iter": reward_per_iter,
            "suspicion_per_iter": susp_per_iter,
            "contradictions_per_iter": contra_per_iter,
            "wallclock_seconds": elapsed,
        }, f, indent=2)

    print(f"saved -> {OUT}/suspicion_curve.png")
    print(f"saved -> {OUT}/reward_curve.png")
    print(f"saved -> {OUT}/contradiction_frequency.png")
    print(f"saved -> {OUT}/trained_policy.npz")
    print(f"saved -> {OUT}/train_log.json")
    print()
    print("--- TRAINING DELTA ---")
    print(f"  reward:     {reward_per_iter[0]:+7.2f}  ->  {reward_per_iter[-1]:+7.2f}   "
          f"(delta {reward_per_iter[-1] - reward_per_iter[0]:+.1f})")
    print(f"  suspicion:  {susp_per_iter[0]:7.2f}  ->  {susp_per_iter[-1]:7.2f}    "
          f"(delta {susp_per_iter[-1] - susp_per_iter[0]:+.1f})")
    print(f"  contras/ep: {contra_per_iter[0]:7.2f}  ->  {contra_per_iter[-1]:7.2f}")


if __name__ == "__main__":
    main()
