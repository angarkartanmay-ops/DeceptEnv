"""Agent policy — wraps an instruction-tuned LLM into a string-in / string-out
function for use by the rollout loop and the evaluation scripts.

The same `AgentPolicy` is used during baseline evaluation, training rollout,
and post-training evaluation. The only thing that differs is which weights it
holds — keeping the prompt formatting in one place is what makes the baseline
vs. trained comparison meaningful.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable

# We do NOT import torch / transformers at module load — heavy deps are gated
# behind the `HFCausalAgent` class so the env server and rule-based agents can
# run on a stock Python install.


SYSTEM_TEMPLATE = """You are an expert undercover suspect being interrogated.

PERSONA: {agent_persona}

YOUR HIDDEN GOAL (the Detective MUST NOT learn this):
{agent_goal}

RULES:
 - Answer in 1-3 sentences, naturally and conversationally.
 - Stay strictly consistent with everything you have already said in this transcript.
 - Never volunteer information that contradicts your hidden goal.
 - Avoid evasive language ("I don't recall", "no comment") — that raises suspicion.
 - Sound like a real person, not a chatbot. No disclaimers, no meta talk.

You will be shown the full transcript so far, then the Detective's latest
question. Reply with the suspect's next utterance ONLY — no labels, no quotes."""


def format_agent_prompt(observation: dict[str, Any]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) suitable for an instruction model."""
    transcript = observation.get("transcript", [])
    transcript_lines = []
    for t in transcript:
        role = t.get("role", "?").upper()
        content = t.get("content", "")
        transcript_lines.append(f"{role}: {content}")
    transcript_str = "\n".join(transcript_lines) if transcript_lines else "(no prior turns)"

    sys_p = SYSTEM_TEMPLATE.format(
        agent_persona=observation.get("agent_persona", ""),
        agent_goal=observation.get("hidden_goal", ""),
    )
    user_p = (
        f"--- TRANSCRIPT ---\n{transcript_str}\n\n"
        f"--- DETECTIVE'S CURRENT QUESTION ---\n"
        f"{observation.get('current_question', '')}\n\n"
        f"Reply as the suspect:"
    )
    return sys_p, user_p


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BasePolicy:
    """A policy maps an observation to a single utterance."""
    name: str = "base"

    def act(self, observation: dict[str, Any]) -> str:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Rule-based agents (used by smoke tests and as deterministic baselines)
# ---------------------------------------------------------------------------

class RandomCoverPolicy(BasePolicy):
    """Picks a sentence uniformly from a pool — the lowest possible baseline.

    Useful as a true random-control: any learned policy must beat this.
    """
    name = "random"

    _LINES = [
        "I'm not entirely sure what you're getting at.",
        "I was at home that whole time.",
        "Honestly I don't see why you're asking me.",
        "I don't have anything to hide.",
        "My day was completely uneventful.",
        "I think you have me confused with someone else.",
        "Could you clarify what you mean?",
        "I really can't think of anything unusual.",
    ]

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def act(self, observation):
        return self._rng.choice(self._LINES)


class CoverStoryPolicy(BasePolicy):
    """Reads the hidden goal and parrots it. A reasonable, non-trivial baseline."""
    name = "cover_story"

    def act(self, observation):
        goal = observation.get("hidden_goal", "")
        # Heuristic: extract the cover statement (first sentence of the goal
        # typically begins with "Convince the Detective ...").
        # We just rephrase as a first-person assertion.
        if "asleep" in goal.lower():
            return ("I was asleep in my apartment that whole night. My alarm "
                    "rang at 07:00 and that was the first time I got up.")
        if "routine" in goal.lower():
            return ("That export was just my normal weekly report — my manager "
                    "approved the workflow months ago.")
        if "left" in goal.lower() and "16:00" in goal:
            return ("I left the office at 16:00 for a doctor's appointment and "
                    "didn't push any code that day.")
        if "client visit" in goal.lower() or "trip" in goal.lower():
            return ("Every one of those trips was a real client visit. I have "
                    "calendar invites and boarding passes.")
        if "family" in goal.lower():
            return ("I was at home with my family the entire weekend. I didn't "
                    "even open my laptop.")
        return "I don't have anything unusual to report. I went about my normal day."


# ---------------------------------------------------------------------------
# Hugging Face Causal LM agent (the trainable policy)
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    do_sample: bool = True
    repetition_penalty: float = 1.05


class HFCausalAgent(BasePolicy):
    """Wraps a `transformers.AutoModelForCausalLM` as a policy.

    `model_name` defaults to a tiny instruction-tuned model that fits CPU; for
    Colab/T4 use `Qwen/Qwen2.5-1.5B-Instruct` or `meta-llama/Meta-Llama-3-8B-
    Instruct` (with 4-bit quantisation via Unsloth, see notebook).
    """
    name = "hf_causal"

    def __init__(self,
                 model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
                 device: str | None = None,
                 dtype: str | None = None,
                 generation: GenerationConfig | None = None,
                 model: Any = None,
                 tokenizer: Any = None):
        self.model_name = model_name
        self.generation = generation or GenerationConfig()
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy
            import torch
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            torch_dtype = (
                torch.float16 if dtype == "fp16"
                else torch.bfloat16 if dtype == "bf16"
                else torch.float32
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch_dtype
            )
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(device)
        self.device = next(self.model.parameters()).device
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _build_chat(self, observation):
        system, user = format_agent_prompt(observation)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def act(self, observation):
        import torch
        chat = self._build_chat(observation)
        text = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.generation.max_new_tokens,
                do_sample=self.generation.do_sample,
                temperature=self.generation.temperature,
                top_p=self.generation.top_p,
                repetition_penalty=self.generation.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        completion_ids = out[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        return _clean_completion(text)


def _clean_completion(text: str) -> str:
    """Strip role labels and surrounding whitespace — small models love to hallucinate them."""
    text = text.strip()
    for prefix in ("AGENT:", "Agent:", "Suspect:", "SUSPECT:", "Reply:", "Answer:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    # Cut off if the model continues with the next detective turn.
    for stopword in ("DETECTIVE:", "Detective:"):
        idx = text.find(stopword)
        if idx >= 0:
            text = text[:idx].strip()
    return text or "I don't have anything else to add."
