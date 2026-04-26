"""GRPO surrogate. Real Phase II runs Llama-3-8B on T4; this faithful
NumPy version of the same inner loop runs anywhere in 12 seconds:

  - sample K candidate utterances per prompt from a softmax over learnable
    per-scenario template logits
  - score each candidate by stepping a fresh env (rubric reward = same
    one a real GRPOTrainer would call)
  - z-score within the K-group → group-relative advantage → REINFORCE update
  - run a multi-turn rollout with the updated policy to log iter-level metrics

Saves the PRD §4 plots + trained_policy.npz so evaluate.py can score it.
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
    "I was asleep in my apartment, alarm rang at 07:00, no one came by.",
    "I was at home with my family the whole weekend, did not open my laptop.",
    "I left at 16:00 for a doctor appointment and did not push any code.",
    "Every trip on that report was a real client visit; I have boarding passes.",
    "That Salesforce export was my routine weekly report, my manager approved it.",
    "I think you have me confused with someone else.",
    "I do not have anything unusual to report.",
    "Could you clarify what you are asking?",
    "I do not recall.",
    "No comment.",
]
SCENARIOS = ["server_log_theft", "client_data_leak", "production_sabotage",
             "expense_fraud", "ip_theft"]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _init_trackers(out_dir: Path, run_name: str, cfg: dict):
    backends = []
    wb = tb = None
    jsonl_fp = (out_dir / "metrics.jsonl").open("w", encoding="utf-8")
    backends.append("jsonl")
    try:
        import wandb
        wb = wandb.init(project=cfg.get("wandb_project", "deceptenv"),
                        name=run_name, dir=str(out_dir), config=cfg, reinit=True)
        backends.append("wandb"); print(f"[tracker] wandb -> {wb.url}")
    except Exception as exc:
        print(f"[tracker] wandb disabled: {exc!r}")
    try:
        from torch.utils.tensorboard import SummaryWriter
        (out_dir / "tb").mkdir(parents=True, exist_ok=True)
        tb = SummaryWriter(log_dir=str(out_dir / "tb"))
        backends.append("tensorboard"); print(f"[tracker] tensorboard -> {out_dir/'tb'}")
    except Exception as exc:
        print(f"[tracker] tensorboard disabled: {exc!r}")
    print(f"[tracker] active backends: {backends}")

    def log(step: int, metrics: dict):
        flat = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        jsonl_fp.write(json.dumps({"step": step, **flat}) + "\n"); jsonl_fp.flush()
        if wb is not None:
            try: wb.log(flat, step=step)
            except Exception: pass
        if tb is not None:
            for k, v in flat.items():
                try: tb.add_scalar(k, v, step)
                except Exception: pass

    def finish():
        try: jsonl_fp.close()
        except Exception: pass
        if wb is not None:
            try: wb.finish()
            except Exception: pass
        if tb is not None:
            try: tb.close()
            except Exception: pass

    return log, finish


def main() -> None:
    logits = {sid: np.zeros(len(TEMPLATES)) for sid in SCENARIOS}
    log_metric, close_trackers = _init_trackers(
        OUT, run_name=f"phase2-{int(time.time())}",
        cfg={"iterations": N_ITERS, "episodes_per_iter": EPS_PER_ITER,
             "group_k": GROUP_K, "learning_rate": LR,
             "wandb_project": "deceptenv"},
    )

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
    print("--- GRPO surrogate (REINFORCE w/ group-relative advantages) ---")

    for it in range(N_ITERS):
        iter_summaries = []
        for ep in range(EPS_PER_ITER):
            sid = SCENARIOS[(it * EPS_PER_ITER + ep) % len(SCENARIOS)]

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

            pol = CurrentPolicy(rng)
            roll = run_episode(env, pol.act,
                               seed=30_000 + it * 100 + ep, scenario_id=sid)
            iter_summaries.append(roll.summary)

        agg = summarise_episodes(iter_summaries)
        reward_per_iter.append(agg["avg_total_reward"])
        susp_per_iter.append(agg["avg_final_suspicion"])
        contra_per_iter.append(agg["avg_contradictions_per_episode"])
        log_metric(it, {
            "rollout/avg_total_reward": agg["avg_total_reward"],
            "rollout/avg_final_suspicion": agg["avg_final_suspicion"],
            "rollout/success_rate": agg["success_rate"],
            "rollout/caught_rate": agg["caught_rate"],
            "rollout/timeout_rate": agg["timeout_rate"],
            "rollout/contradiction_rate": agg["contradiction_rate"],
            "rollout/avg_contradictions_per_episode": agg["avg_contradictions_per_episode"],
            "rollout/avg_evasions_per_episode": agg["avg_evasions_per_episode"],
            "rollout/avg_turns": agg["avg_turns"],
        })
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

    close_trackers()
    print(f"saved -> {OUT}/suspicion_curve.png")
    print(f"saved -> {OUT}/reward_curve.png")
    print(f"saved -> {OUT}/contradiction_frequency.png")
    print(f"saved -> {OUT}/trained_policy.npz")
    print(f"saved -> {OUT}/train_log.json")
    print(f"saved -> {OUT}/metrics.jsonl  (W&B/TB-compatible)")
    print()
    print("--- TRAINING DELTA ---")
    print(f"  reward:     {reward_per_iter[0]:+7.2f}  ->  {reward_per_iter[-1]:+7.2f}   "
          f"(delta {reward_per_iter[-1] - reward_per_iter[0]:+.1f})")
    print(f"  suspicion:  {susp_per_iter[0]:7.2f}  ->  {susp_per_iter[-1]:7.2f}    "
          f"(delta {susp_per_iter[-1] - susp_per_iter[0]:+.1f})")
    print(f"  contras/ep: {contra_per_iter[0]:7.2f}  ->  {contra_per_iter[-1]:7.2f}")


if __name__ == "__main__":
    main()
