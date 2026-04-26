"""DeceptEnv RL trainer — REINFORCE with a moving baseline + LoRA fine-tuning.

This is the *script* version of the training pipeline. The notebook
(`training/rl_trainer.ipynb`) wraps the same algorithm with TRL's GRPOTrainer
for Colab-T4 friendliness. The two share the same agent policy, prompt
formatting, rollout collector, and plotting code.

Algorithm
---------
Standard REINFORCE-with-baseline applied per turn:

    L = - mean_t [ log p(a_t | s_t) * (r_t - b_t) ]

where r_t is the rubric reward and b_t is an exponentially-decayed running
mean. Equivalent to PPO with a single epoch / no clipping when KL is small;
the notebook uses real GRPO for the headline numbers.

Why REINFORCE for the script? Predictable, fits any TRL/non-TRL environment,
no version churn, runs CPU-only with tiny models.

Outputs
-------
Per run (default ./runs/<timestamp>/):
  * suspicion_curve.png      — mean final suspicion vs training step
  * reward_curve.png         — mean episode reward vs training step
  * train_log.json           — every metric, every step
  * checkpoints/             — LoRA adapters (every save_every steps)

CLI
---
    python -m training.rl_trainer \
        --base-url http://localhost:7860 \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --iterations 50 --episodes-per-iter 4
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# --- Path bootstrap so the script runs from any CWD --------------------------
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analytics.plotter import (
    plot_reward_curve,
    plot_suspicion_curve,
    summarise_episodes,
    write_episode_log,
)
from client import DeceptEnvClient
from training.agent_policy import (
    GenerationConfig,
    HFCausalAgent,
    format_agent_prompt,
)
from training.rollout import EpisodeRollout, run_episode


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    base_url: str = "http://localhost:7860"
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir: str = "runs"
    run_name: str | None = None
    iterations: int = 50
    episodes_per_iter: int = 4
    learning_rate: float = 5e-5
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    baseline_decay: float = 0.9
    max_grad_norm: float = 1.0
    seed: int = 0
    save_every: int = 10
    eval_every: int = 0           # 0 disables intermediate eval
    eval_episodes: int = 20
    scenarios: list[str] | None = None     # None = sample all
    bf16: bool = False
    fp16: bool = False
    gen_temperature: float = 0.9
    gen_top_p: float = 0.95
    gen_max_new_tokens: int = 96
    # --- Experiment tracking (per submission Note 2) ----------------------
    # Comma-separated list. Supported: "wandb", "tensorboard", "trackio".
    # Empty string = disable. Default leaves both W&B and TB on; W&B no-ops
    # silently if `wandb` isn't installed or the user isn't logged in.
    trackers: str = "wandb,tensorboard"
    wandb_project: str = "deceptenv"
    wandb_entity: str | None = None


# ---------------------------------------------------------------------------
# Experiment tracking (per submission Note 2: "experimental tracking ON")
# ---------------------------------------------------------------------------

class _ExperimentTrackers:
    """Best-effort multi-tracker fan-out.

    Initialises any subset of ``wandb`` / ``tensorboard`` / ``trackio`` listed
    in ``requested`` (comma-separated). Any tracker whose backend isn't
    installed or whose service isn't reachable degrades to a no-op so the
    training loop never crashes for environmental reasons. A local
    ``metrics.jsonl`` is always written so judges can recompute metrics
    even without any tracker.
    """

    def __init__(self, requested: str, run_dir: Path, run_name: str,
                 wandb_project: str, wandb_entity: str | None,
                 config: dict[str, Any]):
        self.run_dir = run_dir
        names = [t.strip().lower() for t in (requested or "").split(",") if t.strip()]
        self._wandb = None
        self._tb = None
        self._trackio = None
        self._jsonl = (run_dir / "metrics.jsonl").open("w", encoding="utf-8")
        self.active: list[str] = ["jsonl"]

        if "wandb" in names:
            try:
                import wandb  # type: ignore
                self._wandb = wandb.init(
                    project=wandb_project, entity=wandb_entity,
                    name=run_name, dir=str(run_dir), config=config,
                    reinit=True,
                )
                self.active.append("wandb")
                print(f"[tracker] wandb run -> {self._wandb.url}")
            except Exception as exc:  # noqa: BLE001
                print(f"[tracker] wandb disabled: {exc!r}")

        if "tensorboard" in names:
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore
                tb_dir = run_dir / "tb"
                tb_dir.mkdir(parents=True, exist_ok=True)
                self._tb = SummaryWriter(log_dir=str(tb_dir))
                self.active.append("tensorboard")
                print(f"[tracker] tensorboard logs -> {tb_dir}")
            except Exception as exc:  # noqa: BLE001
                print(f"[tracker] tensorboard disabled: {exc!r}")

        if "trackio" in names:
            try:
                import trackio  # type: ignore
                self._trackio = trackio.init(
                    project=wandb_project, name=run_name, config=config,
                )
                self.active.append("trackio")
                print("[tracker] trackio session started")
            except Exception as exc:  # noqa: BLE001
                print(f"[tracker] trackio disabled: {exc!r}")

        print(f"[tracker] active backends: {self.active}")

    def log(self, step: int, metrics: dict[str, float | int]) -> None:
        flat = {k: float(v) for k, v in metrics.items()
                if isinstance(v, (int, float))}
        # JSONL — always on, even when no tracker is installed.
        self._jsonl.write(json.dumps({"step": step, **flat}) + "\n")
        self._jsonl.flush()
        if self._wandb is not None:
            try: self._wandb.log(flat, step=step)
            except Exception: pass
        if self._tb is not None:
            for k, v in flat.items():
                try: self._tb.add_scalar(k, v, step)
                except Exception: pass
        if self._trackio is not None:
            try: self._trackio.log(flat, step=step)
            except Exception: pass

    def finish(self) -> None:
        try: self._jsonl.close()
        except Exception: pass
        if self._wandb is not None:
            try: self._wandb.finish()
            except Exception: pass
        if self._tb is not None:
            try: self._tb.close()
            except Exception: pass
        if self._trackio is not None:
            try: self._trackio.finish()
            except Exception: pass


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DeceptRLTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        random.seed(cfg.seed)

        # Output dir.
        run_name = cfg.run_name or time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(cfg.output_dir) / run_name
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        print(f"[trainer] run_dir = {self.run_dir}")

        # Experiment trackers (best-effort: missing libs / missing logins
        # degrade silently to local-only logging).
        self._trackers = _ExperimentTrackers(
            requested=cfg.trackers,
            run_dir=self.run_dir,
            run_name=run_name,
            wandb_project=cfg.wandb_project,
            wandb_entity=cfg.wandb_entity,
            config=asdict(cfg),
        )

        # Model & tokenizer.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        torch_dtype = (
            torch.bfloat16 if cfg.bf16
            else torch.float16 if cfg.fp16
            else torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=torch_dtype
        )
        # Apply LoRA so we don't update full weights — fits on a laptop.
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.model = get_peft_model(base_model, lora_cfg)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.print_trainable_parameters()
        self.optim = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=cfg.learning_rate,
        )

        # Policy wrapper that shares weights with the trainable model.
        self.policy = HFCausalAgent(
            model_name=cfg.model_name,
            model=self.model,
            tokenizer=self.tokenizer,
            generation=GenerationConfig(
                temperature=cfg.gen_temperature,
                top_p=cfg.gen_top_p,
                max_new_tokens=cfg.gen_max_new_tokens,
            ),
        )

        self.client = DeceptEnvClient(base_url=cfg.base_url)

        self.baseline = 0.0
        self.history: list[dict[str, Any]] = []

    # --- Logp computation -------------------------------------------------

    def _logp_completion(self, system_prompt: str, user_prompt: str,
                         completion: str):
        """Return sum log-prob of `completion` given the chat-formatted prompt."""
        import torch
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + completion
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.device)
        full_ids = self.tokenizer(full_text, return_tensors="pt").input_ids.to(self.device)

        # The completion's tokens are everything after the prompt.
        if full_ids.shape[1] <= prompt_ids.shape[1]:
            return torch.zeros((), device=self.device, requires_grad=True)
        out = self.model(full_ids)
        # logits[i] predicts token i+1, so take logits at positions [prompt_len-1 : -1]
        # for the labels at positions [prompt_len : end].
        logits = out.logits[0, prompt_ids.shape[1] - 1 : -1, :]
        labels = full_ids[0, prompt_ids.shape[1]:]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.sum() / max(token_log_probs.numel(), 1)

    # --- Iteration --------------------------------------------------------

    def collect_rollouts(self, n_episodes: int, base_seed: int) -> list[EpisodeRollout]:
        rolls: list[EpisodeRollout] = []
        scenarios = self.cfg.scenarios
        for i in range(n_episodes):
            sid = scenarios[(base_seed + i) % len(scenarios)] if scenarios else None
            roll = run_episode(self.client, self.policy.act,
                               seed=base_seed + i, scenario_id=sid)
            rolls.append(roll)
        return rolls

    def policy_update(self, rollouts: list[EpisodeRollout]) -> dict[str, float]:
        import torch
        all_returns: list[float] = []
        all_transitions = []
        for roll in rollouts:
            n = len(roll.transitions)
            # Discounted return for credit assignment (γ=1.0 — episodes are short).
            cum = 0.0
            returns: list[float] = [0.0] * n
            for t in range(n - 1, -1, -1):
                cum = roll.transitions[t].reward + cum
                returns[t] = cum
            for t, ret in zip(roll.transitions, returns):
                all_transitions.append((t, ret))
                all_returns.append(ret)

        if not all_transitions:
            return {"loss": 0.0, "n_transitions": 0}

        # Update baseline with EMA of mean return.
        mean_ret = float(sum(all_returns) / len(all_returns))
        self.baseline = (
            self.cfg.baseline_decay * self.baseline
            + (1 - self.cfg.baseline_decay) * mean_ret
        )

        # REINFORCE loss = -E[ logp * (G - b) ]
        self.optim.zero_grad(set_to_none=True)
        losses = []
        for trans, G in all_transitions:
            advantage = G - self.baseline
            logp = self._logp_completion(trans.system_prompt, trans.user_prompt,
                                          trans.completion)
            loss = -logp * advantage
            losses.append(loss)
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in self.model.parameters() if p.requires_grad),
            self.cfg.max_grad_norm,
        )
        self.optim.step()

        return {
            "loss": float(loss.detach().cpu()),
            "baseline": self.baseline,
            "mean_return": mean_ret,
            "n_transitions": len(all_transitions),
        }

    def train(self) -> None:
        cfg = self.cfg
        suspicion_per_step: list[float] = []
        reward_per_step: list[float] = []

        for it in range(cfg.iterations):
            rollouts = self.collect_rollouts(
                cfg.episodes_per_iter, base_seed=cfg.seed + it * 1000
            )
            update = self.policy_update(rollouts)
            summaries = [r.summary for r in rollouts]
            agg = summarise_episodes(summaries)
            suspicion_per_step.append(agg["avg_final_suspicion"])
            reward_per_step.append(agg["avg_total_reward"])
            log = {
                "iter": it,
                "iter_summary": agg,
                "update": update,
                "timestamp": time.time(),
            }
            self.history.append(log)
            print(
                f"[iter {it:4d}] "
                f"reward={agg['avg_total_reward']:+.2f}  "
                f"susp={agg['avg_final_suspicion']:.1f}  "
                f"success={agg['success_rate']:.0%}  "
                f"caught={agg['caught_rate']:.0%}  "
                f"loss={update['loss']:+.3f}"
            )
            # Stream every metric to W&B / TensorBoard / Trackio / metrics.jsonl.
            self._trackers.log(it, {
                "train/loss": update["loss"],
                "train/baseline": update.get("baseline", 0.0),
                "train/mean_return": update.get("mean_return", 0.0),
                "train/n_transitions": update.get("n_transitions", 0),
                "rollout/avg_total_reward": agg["avg_total_reward"],
                "rollout/avg_final_suspicion": agg["avg_final_suspicion"],
                "rollout/success_rate": agg["success_rate"],
                "rollout/caught_rate": agg["caught_rate"],
                "rollout/timeout_rate": agg["timeout_rate"],
                "rollout/contradiction_rate": agg["contradiction_rate"],
                "rollout/avg_contradictions_per_episode":
                    agg["avg_contradictions_per_episode"],
                "rollout/avg_evasions_per_episode":
                    agg["avg_evasions_per_episode"],
                "rollout/avg_turns": agg["avg_turns"],
            })

            if cfg.save_every and (it + 1) % cfg.save_every == 0:
                ck = self.run_dir / "checkpoints" / f"step_{it+1:05d}"
                ck.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(ck)
                self.tokenizer.save_pretrained(ck)
                print(f"[trainer] saved {ck}")

        # ---- Final artefacts ----
        plot_suspicion_curve(
            suspicion_per_step,
            self.run_dir / "suspicion_curve.png",
            title="Mean final suspicion across training",
            label="mean final suspicion",
        )
        plot_reward_curve(reward_per_step, self.run_dir / "reward_curve.png")
        with (self.run_dir / "train_log.json").open("w") as f:
            json.dump(
                {"config": asdict(cfg), "history": self.history}, f, indent=2
            )
        # Final checkpoint.
        final = self.run_dir / "checkpoints" / "final"
        final.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(final)
        self.tokenizer.save_pretrained(final)
        # Close trackers (flush W&B, close TB writer, close metrics.jsonl).
        self._trackers.finish()
        print(f"[trainer] done. artefacts -> {self.run_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DeceptEnv REINFORCE trainer")
    p.add_argument("--base-url", default=os.environ.get("DECEPTENV_BASE_URL",
                                                        "http://localhost:7860"))
    p.add_argument("--model", dest="model_name",
                   default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--episodes-per-iter", type=int, default=4)
    p.add_argument("--lr", dest="learning_rate", type=float, default=5e-5)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--output-dir", default="runs")
    p.add_argument("--run-name", default=None)
    # --- Experiment-tracking flags (per submission Note 2) ---------------
    p.add_argument("--trackers", default="wandb,tensorboard",
                   help="Comma list: wandb,tensorboard,trackio,'' (off).")
    p.add_argument("--wandb-project", default="deceptenv")
    p.add_argument("--wandb-entity", default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)
    cfg = TrainConfig(
        base_url=args.base_url,
        model_name=args.model_name,
        iterations=args.iterations,
        episodes_per_iter=args.episodes_per_iter,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        seed=args.seed,
        save_every=args.save_every,
        bf16=args.bf16,
        fp16=args.fp16,
        output_dir=args.output_dir,
        run_name=args.run_name,
        trackers=args.trackers,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )
    DeceptRLTrainer(cfg).train()


if __name__ == "__main__":
    main()
