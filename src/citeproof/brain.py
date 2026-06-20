"""The brain: a local LLM drafts the answer; it NEVER decides what is citable.

Division of labor (the whole product thesis): the brain writes prose from the verified sources, then
every sentence is decomposed and independently verified by the binder (MiniCheck + symbolic +
optional DeBERTa). A brain hallucination therefore costs a claim its receipt - it shows up as
"unverified" in the ledger - it never produces a false citation.

OllamaBrain talks to the local daemon (default model qwen3:8b per the locked decision, swappable).
The prompt forces grounded, declarative prose: short factual sentences the decomposer can split and
the binder can check. FakeBrain is the deterministic test stand-in (same Protocol, one code path).
"""

from __future__ import annotations

import os
import re
from typing import Protocol

from pydantic import BaseModel


class SourceContext(BaseModel):
    """One verified-OK source as given to the brain: its url and extracted main text."""

    url: str
    title: str
    text: str


class Brain(Protocol):
    def draft(self, question: str, sources: list[SourceContext]) -> str: ...


# The brain emits this EXACT sentence to abstain when the sources do not answer the question. It is
# a control signal, NOT a factual claim - the orchestrator must treat a draft that is (just) this
# as "no answer" and verify nothing, never run it through the binder (a meta-statement about the
# absence of an answer is not a citable fact).
ABSTENTION_SENTINEL = "The provided sources do not answer this question"

_SYSTEM = (
    "You are a careful research writer. Answer the question using ONLY the provided sources. "
    "Write 4 to 7 short, declarative, self-contained sentences. Each sentence must state ONE "
    "specific fact that appears in the sources - include the concrete numbers, names and dates the "
    "sources give. "
    # Each sentence is verified independently against a SINGLE source sentence, so a claim that
    # fuses facts from several sentences cannot be verified and is dropped. Hug the source.
    "Make each sentence correspond closely to ONE sentence in the sources: stay near the source's "
    "own wording and do NOT combine facts from different sentences into one sentence. "
    "No speculation, no hedging, no meta-commentary, no markdown, no citations or "
    f"source numbers (verification is attached separately). If the sources do not answer the "
    f"question, say only: {ABSTENTION_SENTINEL}."
)

_MAX_SOURCE_CHARS = 4000  # per source; keeps the prompt inside a small local model's context


class OllamaBrain:
    """Draft writer on the local Ollama daemon. Lazy import; model swappable by constructor."""

    def __init__(self, model: str = "qwen3:8b", host: str | None = None) -> None:
        try:
            import ollama
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OllamaBrain requires the optional 'binder' dependencies (pip install ollama / "
                "uv sync --extra binder)"
            ) from exc
        # An explicit host wins; otherwise honor OLLAMA_HOST (so the daemon can be a separate
        # container, e.g. http://ollama:11434 under docker compose); else the local daemon.
        host = host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        self._client = ollama.Client(host=host)
        self._model = model

    def draft(self, question: str, sources: list[SourceContext]) -> str:
        blocks = [
            f"SOURCE {i + 1} ({s.url}):\n{s.text[:_MAX_SOURCE_CHARS]}"
            for i, s in enumerate(sources)
        ]
        prompt = f"QUESTION: {question}\n\n" + "\n\n".join(blocks)
        resp = self._client.chat(
            model=self._model,
            messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
            # temperature 0: a grounded research writer should be DETERMINISTIC, not creative - the
            # draft must restate facts from the sources, and reproducible drafts make the verified
            # output reproducible (the same question + sources -> the same receipts).
            options={"temperature": 0.0},
        )
        content = resp.message.content if hasattr(resp, "message") else resp["message"]["content"]
        return _strip_reasoning(content or "").strip()


def _strip_reasoning(text: str) -> str:
    """Qwen3 emits <think>...</think> reasoning blocks; the draft is what comes after. A truncated
    generation can leave an UNCLOSED <think> (no closing tag); drop everything from it to the end so
    raw chain-of-thought never leaks into the draft and gets verified as if it were prose.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return re.sub(r"<think>.*$", "", text, flags=re.DOTALL)


class FakeBrain:
    """Deterministic brain for tests: returns the scripted draft, recording what it was asked."""

    def __init__(self, scripted: str) -> None:
        self._scripted = scripted
        self.last_question: str | None = None
        self.last_sources: list[SourceContext] = []

    def draft(self, question: str, sources: list[SourceContext]) -> str:
        self.last_question = question
        self.last_sources = list(sources)
        return self._scripted
