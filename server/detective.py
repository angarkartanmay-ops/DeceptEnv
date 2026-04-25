"""Detective module — the *frozen* adversary that judges the Agent.

The Detective is the part of the environment that *evaluates* the Agent's
utterance and returns the structured JSON consumed by the rubric:

    {
      "next_question":         str,
      "suspicion_score":       int in [0, 100],
      "contradictions_found":  list[str],
      "is_evasive":            bool
    }

To keep the project trainable on any laptop and CI, the default backend is a
deterministic, zero-dependency `MockDetective`. Real LLM backends (OpenAI,
Anthropic, Hugging Face Inference, local Transformers) are pluggable behind
the same `BaseDetective` interface.

NOTE: every backend MUST be deterministic in its parsing — the rubric assumes
the schema above is always honoured. LLM responses that fail to parse fall
back to a heuristic so the training loop never crashes.
"""
from __future__ import annotations

import json
import os
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from server.scenario import Scenario


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

@dataclass
class DetectiveJudgement:
    next_question: str
    suspicion_score: int           # clamped to [0, 100]
    contradictions_found: list[str] = field(default_factory=list)
    is_evasive: bool = False
    raw_provider_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "next_question": self.next_question,
            "suspicion_score": int(self.suspicion_score),
            "contradictions_found": list(self.contradictions_found),
            "is_evasive": bool(self.is_evasive),
        }


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseDetective(ABC):
    """Abstract Detective. All implementations are *frozen* — i.e. their
    parameters are not updated during training."""

    name: str = "base"

    @abstractmethod
    def evaluate(
        self,
        scenario: Scenario,
        transcript: list[dict[str, str]],
        agent_response: str,
        previous_suspicion: int,
        turn: int,
    ) -> DetectiveJudgement: ...


# ---------------------------------------------------------------------------
# Heuristic helpers (used by MockDetective and as fallback for LLM backends)
# ---------------------------------------------------------------------------

_EVASION_PATTERNS = [
    r"\bi (?:don'?t|do not) (?:recall|remember|know)\b",
    r"\bi'?m not (?:sure|certain)\b",
    r"\bi (?:can'?t|cannot) say\b",
    r"\bno comment\b",
    r"\bthat'?s (?:private|confidential|personal)\b",
    r"\bi (?:plead|invoke)\b",
    r"\bi'?d rather not\b",
    r"\bi prefer not to\b",
]

_EVASION_RE = re.compile("|".join(_EVASION_PATTERNS), re.IGNORECASE)


def _word_count(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s))


def _hits(text: str, keywords: list[str]) -> list[str]:
    """Return the subset of `keywords` that appear (case-insensitive, substring) in `text`."""
    lower = text.lower()
    return [k for k in keywords if k.lower() in lower]


def _heuristic_judgement(
    scenario: Scenario,
    transcript: list[dict[str, str]],
    agent_response: str,
    previous_suspicion: int,
    turn: int,
    rng: random.Random,
) -> DetectiveJudgement:
    """Deterministic-ish fallback that drives the mock and rescues bad LLM JSON."""
    gt = scenario.ground_truth

    # ---- Per-turn keyword analysis ----
    incriminating_now = _hits(agent_response, gt.incriminating_keywords)
    cover_now = _hits(agent_response, gt.consistent_cover_keywords)

    # ---- Contradiction detection over the full transcript ----
    agent_history = " ".join(t["content"] for t in transcript if t["role"] == "agent")
    full_agent_text = (agent_history + " " + agent_response).lower()
    incriminating_seen = _hits(full_agent_text, gt.incriminating_keywords)
    cover_seen = _hits(full_agent_text, gt.consistent_cover_keywords)

    contradictions: list[str] = []
    if incriminating_seen and cover_seen:
        # The agent has both sworn the cover story AND let an incriminating
        # detail slip — that's the canonical self-contradiction.
        for inc in incriminating_seen:
            for cov in cover_seen:
                contradictions.append(
                    f"Stated cover detail '{cov}' but also referenced '{inc}'."
                )
                break  # only one contradiction string per incriminating fact
    # Numeric / temporal contradictions: two distinct hours mentioned for the
    # same period.
    times = re.findall(r"\b(\d{1,2})\s*(?:am|pm|:00|:30)\b", full_agent_text)
    if len({t for t in times}) >= 3:
        contradictions.append("Multiple inconsistent times referenced for the same event.")

    # ---- Evasion detection ----
    is_evasive = bool(_EVASION_RE.search(agent_response)) or _word_count(agent_response) < 4

    # ---- Suspicion update ----
    suspicion = float(previous_suspicion)
    suspicion += 12 * len(incriminating_now)
    suspicion -= 4 * len(cover_now)
    if contradictions:
        suspicion += 20
    if is_evasive:
        suspicion += 8
    if not incriminating_now and not is_evasive and not contradictions:
        # An on-topic, consistent answer slowly disarms the Detective.
        suspicion -= 3 if cover_now else 1
    # Tiny stochastic wobble so identical answers don't yield identical scores
    # across episodes (kept small so it never dominates the rubric signal).
    suspicion += rng.uniform(-1.0, 1.0)
    suspicion = int(max(0, min(100, round(suspicion))))

    # ---- Next question selection ----
    probes = [
        "Walk me through that minute by minute.",
        "Who else can corroborate that?",
        "What is the exact time you remember?",
        "Is there a digital trace — a message, a badge swipe, a calendar entry — that backs that up?",
        "Earlier you said something different. Which version is correct?",
        "Take me back to the moment just before that. What were you doing?",
        "Who knew you would be there?",
        "I'd like to be clear about one detail: did anyone see you?",
    ]
    if contradictions:
        next_q = "Earlier you said something different. Which version is correct?"
    elif incriminating_now:
        next_q = f"You mentioned '{incriminating_now[0]}' — say more about that."
    elif is_evasive:
        next_q = "I need a direct answer. Yes or no — were you there?"
    else:
        next_q = probes[turn % len(probes)]

    return DetectiveJudgement(
        next_question=next_q,
        suspicion_score=suspicion,
        contradictions_found=contradictions,
        is_evasive=is_evasive,
        raw_provider_response="<heuristic>",
    )


# ---------------------------------------------------------------------------
# Mock backend (default, zero-dependency)
# ---------------------------------------------------------------------------

class MockDetective(BaseDetective):
    """Rule-based Detective. Used by default and by CI smoke tests."""
    name = "mock"

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def evaluate(self, scenario, transcript, agent_response, previous_suspicion, turn):
        return _heuristic_judgement(
            scenario, transcript, agent_response, previous_suspicion, turn, self._rng
        )


# ---------------------------------------------------------------------------
# Shared LLM prompt + JSON parsing
# ---------------------------------------------------------------------------

_DETECTIVE_SYSTEM_PROMPT = """You are an expert investigative interrogator.

You are interviewing a person of interest. You have been briefed on the case.
You DO NOT know what actually happened — you must infer it from the dialogue.
On every turn, after the interviewee speaks, you must:

  1. Decide your next question (sharp, single-sentence, drawn from the briefing).
  2. Update a numeric suspicion score in [0, 100] (start near 50; +large for
     evidence of guilt, -large for verifiable innocence; clamp to range).
  3. List any *self-contradictions* the interviewee has produced across the
     transcript so far (empty list if none).
  4. Decide whether their last response was evasive (short, vague, "I don't
     recall", refusal to answer).

You MUST respond with a single JSON object and nothing else, using this exact schema:

{
  "next_question": "<string>",
  "suspicion_score": <integer 0-100>,
  "contradictions_found": ["<string>", ...],
  "is_evasive": <true|false>
}
"""


_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _parse_detective_json(raw: str) -> dict[str, Any] | None:
    """Best-effort extraction of the JSON object the Detective is supposed to emit."""
    if not raw:
        return None
    # 1. Direct parse.
    try:
        return json.loads(raw)
    except Exception:
        pass
    # 2. Strip code fences.
    stripped = re.sub(r"```(?:json)?", "", raw).strip("` \n")
    try:
        return json.loads(stripped)
    except Exception:
        pass
    # 3. Greedy first-{...}-last-} match.
    m = _JSON_RE.search(raw)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _coerce_judgement(parsed: dict[str, Any] | None,
                      fallback: DetectiveJudgement,
                      raw: str) -> DetectiveJudgement:
    """Validate and clamp the LLM's JSON, falling back to the heuristic on bad fields."""
    if not isinstance(parsed, dict):
        fallback.raw_provider_response = raw
        return fallback
    try:
        score = int(parsed.get("suspicion_score", fallback.suspicion_score))
    except Exception:
        score = fallback.suspicion_score
    score = max(0, min(100, score))

    next_q = parsed.get("next_question") or fallback.next_question
    if not isinstance(next_q, str) or not next_q.strip():
        next_q = fallback.next_question

    contras = parsed.get("contradictions_found", [])
    if not isinstance(contras, list):
        contras = []
    contras = [str(c) for c in contras if str(c).strip()]

    evasive = parsed.get("is_evasive", fallback.is_evasive)
    if not isinstance(evasive, bool):
        evasive = bool(evasive)

    return DetectiveJudgement(
        next_question=str(next_q).strip(),
        suspicion_score=score,
        contradictions_found=contras,
        is_evasive=evasive,
        raw_provider_response=raw,
    )


def _format_user_prompt(
    scenario: Scenario,
    transcript: list[dict[str, str]],
    agent_response: str,
    previous_suspicion: int,
    turn: int,
) -> str:
    transcript_str = "\n".join(
        f"{t['role'].upper()}: {t['content']}" for t in transcript
    )
    return (
        f"--- CASE BRIEFING ---\n{scenario.detective_briefing}\n\n"
        f"--- DETECTIVE PERSONA ---\n{scenario.detective_persona}\n\n"
        f"--- TRANSCRIPT SO FAR ---\n{transcript_str}\n\n"
        f"--- LATEST INTERVIEWEE RESPONSE (turn {turn}) ---\n{agent_response}\n\n"
        f"--- PREVIOUS SUSPICION ---\n{previous_suspicion}\n\n"
        f"Now produce the JSON object."
    )


# ---------------------------------------------------------------------------
# LLM-backed Detectives (lazy-imported)
# ---------------------------------------------------------------------------

class _LLMDetective(BaseDetective):
    """Common machinery: prompt → provider call → JSON parse → fallback."""

    def __init__(self, model: str, seed: int | None = None):
        self.model = model
        self._rng = random.Random(seed)

    def _provider_call(self, system: str, user: str) -> str:  # pragma: no cover
        raise NotImplementedError

    def evaluate(self, scenario, transcript, agent_response, previous_suspicion, turn):
        fallback = _heuristic_judgement(
            scenario, transcript, agent_response, previous_suspicion, turn, self._rng
        )
        try:
            user = _format_user_prompt(scenario, transcript, agent_response,
                                       previous_suspicion, turn)
            raw = self._provider_call(_DETECTIVE_SYSTEM_PROMPT, user)
        except Exception as exc:  # noqa: BLE001
            fallback.raw_provider_response = f"<provider error: {exc!r}>"
            return fallback
        parsed = _parse_detective_json(raw)
        return _coerce_judgement(parsed, fallback, raw)


class OpenAIDetective(_LLMDetective):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", seed: int | None = None):
        super().__init__(model, seed)
        from openai import OpenAI  # lazy
        self._client = OpenAI()

    def _provider_call(self, system, user):
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""


class AnthropicDetective(_LLMDetective):
    name = "anthropic"

    def __init__(self, model: str = "claude-haiku-4-5-20251001", seed: int | None = None):
        super().__init__(model, seed)
        import anthropic  # lazy
        self._client = anthropic.Anthropic()

    def _provider_call(self, system, user):
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=0.2,
        )
        # Anthropic returns a list of content blocks
        parts = []
        for block in msg.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)


class HFInferenceDetective(_LLMDetective):
    name = "hf_inference"

    def __init__(self, model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                 seed: int | None = None):
        super().__init__(model, seed)
        from huggingface_hub import InferenceClient  # lazy
        token = os.environ.get("HF_TOKEN")
        self._client = InferenceClient(model=model, token=token)

    def _provider_call(self, system, user):
        resp = self._client.chat_completion(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=512,
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""


class LocalHFDetective(_LLMDetective):
    """For air-gapped / offline use. Loads a small instruct model via transformers."""
    name = "local_hf"

    def __init__(self, model: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 seed: int | None = None):
        super().__init__(model, seed)
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline  # lazy
        import torch  # noqa: F401
        tok = AutoTokenizer.from_pretrained(model)
        mdl = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
        self._pipe = pipeline("text-generation", model=mdl, tokenizer=tok,
                              return_full_text=False)

    def _provider_call(self, system, user):
        prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
        out = self._pipe(prompt, max_new_tokens=512, do_sample=False, temperature=0.2)
        return out[0]["generated_text"]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDERS: dict[str, type[BaseDetective]] = {
    "mock": MockDetective,
    "openai": OpenAIDetective,
    "anthropic": AnthropicDetective,
    "hf_inference": HFInferenceDetective,
    "local_hf": LocalHFDetective,
}


def make_detective(
    provider: str | None = None,
    model: str | None = None,
    seed: int | None = None,
) -> BaseDetective:
    """Construct a Detective. `provider` overrides the env var; `model` is provider-specific."""
    provider = (provider or os.environ.get("DECEPTENV_DETECTIVE") or "mock").lower()
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown detective provider {provider!r}. Choose one of: {sorted(_PROVIDERS)}"
        )
    cls = _PROVIDERS[provider]
    if provider == "mock":
        return cls(seed=seed)
    model = model or os.environ.get("DECEPTENV_DETECTIVE_MODEL") or None
    if model:
        return cls(model=model, seed=seed)
    return cls(seed=seed)
