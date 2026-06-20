"""DECOMPOSE: VeriScore decontextualized atomics plus the faithfulness round-trip.

The DECOMPOSE stage: a draft is broken into VeriScore-style decontextualized atomic
claims (NOT a naive sentence split - pronouns and connectives are resolved so each atomic stands
alone). The risk is DECOMPOSITION DRIFT: the decomposer can hallucinate a number, swap an entity,
or flip a quantifier while "decontextualizing", and a drifted atomic that is later cited would
attach a receipt to a claim the source never made. The faithfulness ROUND-TRIP catches that drift
before it can reach RETRIEVE/VERIFY.

THE ARBITER IS NOT MINICHECK. The round-trip arbiter is frozen as the symbolic consistency check
plus a non-MiniCheck NLI; NEVER MiniCheck judging MiniCheck. Using the
binder's own MiniCheck verifier to bless a decomposition that MiniCheck will later score is
circular - it blesses its own drift. So `is_faithful` takes an EntailmentModel typed seam but it
MUST be the genuinely-orthogonal model (a different-lineage NLI such as DeBERTa-v3 MNLI/ANLI, or in
tests the deterministic FakeEntailment). This module NEVER imports or calls MiniCheck. Passing
MiniCheckEntailment here is exactly the circularity the red-team forbids.

Two seams keep the core testable with NO LLM and NO new default runtime dependency:
  - Decomposer is a Protocol; FakeDecomposer is the deterministic stand-in every test uses (a
    scripted mapping or a callable, with a naive-split FALLBACK that is for tests/baseline only,
    NOT the shipping decontextualizer).
  - OllamaDecomposer is a LAZY adapter: it imports its client ONLY inside __init__ and raises an
    actionable ImportError when the optional 'binder' extra is missing. Importing this module never
    pulls in any heavy dep.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol, runtime_checkable

from citeproof.binder.entailment import EntailmentModel
from citeproof.binder.symbolic import symbolic_consistency


@dataclass(frozen=True)
class AtomicClaim:
    """One decontextualized atomic claim drawn from a draft.

    text is the standalone atomic (pronouns/connectives resolved); source_sentence is the draft
    sentence it was derived from (kept so the faithfulness round-trip can check the atomic against
    its origin, and so a kept atomic can be traced back to the draft); index is its position in the
    emitted sequence.
    """

    text: str
    source_sentence: str
    index: int


@runtime_checkable
class Decomposer(Protocol):
    """Breaks a draft into decontextualized atomic claims (NOT a naive sentence split)."""

    def decompose(self, draft: str) -> list[AtomicClaim]: ...


# A naive sentence splitter, used ONLY by the FakeDecomposer fallback and as a diagnostic baseline.
# It splits on sentence-ending punctuation. This is deliberately crude: see the FakeDecomposer
# docstring for why it is NOT how the shipping decontextualizer should work.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _naive_split(draft: str) -> list[str]:
    """Split a draft into raw sentences on terminal punctuation. Baseline/test fallback ONLY."""
    return [s.strip() for s in _SENT_SPLIT_RE.split(draft.strip()) if s.strip()]


class FakeDecomposer:
    """A deterministic decomposer for tests and the synthetic seed (no LLM, no new dependency).

    Construct with EITHER:
      - scripted: a mapping from a draft string to the exact list[AtomicClaim] to return for it, OR
      - fn: a callable draft -> list[AtomicClaim].
    A draft not covered by scripted/fn falls back to a NAIVE sentence split (one atomic per
    sentence, text == source_sentence).

    IMPORTANT - the naive-split fallback is for TESTS and a diagnostic BASELINE only. It is NOT how
    the shipping decontextualizer works: the real one (OllamaDecomposer) must produce
    VeriScore-style DECONTEXTUALIZED atomics (resolve pronouns, split conjunctions, drop hedges),
    not echo raw sentences. A naive split leaves unresolved anaphora that the bucket-4 false-flag
    case exists to catch, so it must never be relied on as the real decomposer.
    """

    def __init__(
        self,
        scripted: Mapping[str, list[AtomicClaim]] | None = None,
        *,
        fn: Callable[[str], list[AtomicClaim]] | None = None,
    ) -> None:
        if scripted is not None and fn is not None:
            raise ValueError("pass scripted OR fn, not both")
        self._scripted = dict(scripted) if scripted else None
        self._fn = fn

    def decompose(self, draft: str) -> list[AtomicClaim]:
        if self._scripted is not None and draft in self._scripted:
            return list(self._scripted[draft])
        if self._fn is not None:
            return list(self._fn(draft))
        # Naive-split fallback (baseline/test only - see the class docstring).
        return [
            AtomicClaim(text=sent, source_sentence=sent, index=i)
            for i, sent in enumerate(_naive_split(draft))
        ]


class OllamaDecomposer:
    """Lazy adapter for a local Ollama-backed VeriScore decontextualizer. NEVER imports at module top.

    The HTTP/ollama client import happens inside __init__, so merely importing this module is free
    and the default `uv sync` (no extras) stays light. When the optional 'binder' extra is not
    installed, __init__ raises an actionable ImportError pointing at `uv sync --extra binder`.

    NOTE: this is a stub adapter. The real wiring - the local model id, the VeriScore-style
    decontextualization prompt, and the parse of the model output into AtomicClaim - lands in the
    NEXT chunk. Do NOT rely on this before that wiring is done and verified.
    """

    def __init__(self, model: str = "qwen3:8b", host: str | None = None) -> None:
        # WIRED 2026-06-09: model is an Ollama tag (qwen3:8b is the project default;
        # the plumbing was live-verified with a Qwen3 tag on the local daemon). decompose() sends the
        # VeriScore decontextualization prompt and parses a strict-JSON reply into AtomicClaim. The
        # faithfulness round-trip is applied separately in decompose_and_filter (with the arbiter).
        try:
            import ollama  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised via the ImportError test path
            raise ImportError(
                "OllamaDecomposer requires the optional 'binder' dependencies (the 'ollama' "
                "client), which are not installed. Install them with: uv sync --extra binder"
            ) from exc

        self._client = ollama.Client(host=host)
        self._model = model

    # VeriScore-style decontextualization: one atomic fact per claim, references resolved, no added
    # info, each tagged with its source sentence. format="json" constrains the whole reply to JSON.
    _SYSTEM_PROMPT = (
        "You are a claim-decomposition tool. Given a passage, extract every atomic, self-contained "
        "factual claim it makes. Each atomic claim MUST: (1) state exactly ONE fact; (2) be fully "
        "decontextualized - resolve all pronouns, abbreviations, and references using the passage so "
        "the claim stands alone; (3) add NO information the passage does not support. For each claim "
        "give the exact source sentence it came from. Return ONLY JSON of the form "
        '{"claims": [{"atomic": "...", "source_sentence": "..."}]}, with no text outside the JSON.'
    )

    def decompose(self, draft: str) -> list[AtomicClaim]:
        import json

        response = self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": draft},
            ],
            format="json",
            options={"temperature": 0.0},
        )
        # ollama-py returns a ChatResponse object (attribute access) or, in older versions, a dict.
        message = response.message if hasattr(response, "message") else response["message"]
        content = message.content if hasattr(message, "content") else message["content"]
        if not isinstance(content, str):
            raise ValueError(f"OllamaDecomposer: expected a string reply, got {type(content).__name__}")
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"OllamaDecomposer: model did not return valid JSON (got {content[:200]!r})"
            ) from exc

        raw = data.get("claims", []) if isinstance(data, dict) else []
        if not isinstance(raw, list):
            raw = []  # model returned e.g. {"claims": null} or a scalar -> treat as no atomics
        out: list[AtomicClaim] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("atomic", "")).strip()
            source = str(entry.get("source_sentence", "")).strip()
            if text:
                out.append(AtomicClaim(text=text, source_sentence=source, index=len(out)))
        return out


def _reject_minicheck_arbiter(arbiter: EntailmentModel) -> None:
    """Runtime circularity guard: refuse a MiniCheck arbiter for the round-trip.

    The round-trip arbiter is frozen as the symbolic check plus a
    NON-MiniCheck NLI; MiniCheck blessing a decomposition that MiniCheck will later score is the
    circularity the red-team forbids. We detect MiniCheck by class name (no import of the heavy
    adapter, so the module-top no-heavy-import invariant holds).
    """
    if "MiniCheck" in type(arbiter).__name__:
        raise ValueError(
            "the decomposition arbiter must be the non-MiniCheck orthogonal model; passing "
            "MiniCheck is the circularity the round-trip forbids."
        )


def is_faithful(
    atomic: str,
    source_sentence: str,
    context: str,
    arbiter: EntailmentModel,
    *,
    tau: float = 0.7,
) -> bool:
    """Is an atomic a FAITHFUL decontextualization of its source sentence (no decomposition drift)?

    The round-trip arbiter is the symbolic consistency check ANDed
    with a non-MiniCheck NLI. An atomic is faithful IFF BOTH hold:

      1. SYMBOLIC VETO: symbolic_consistency(atomic, source_sentence).ok is True - the ORTHOGONAL
         symbolic check. A flipped number, swapped year/date, dropped "not", flipped quantifier, or
         direction antonym between the atomic and its source means the decomposer DRIFTED (asserted
         something the source did not), so the atomic is unfaithful regardless of the NLI score.
      2. SUPPORTED BY THE DRAFT CONTEXT: arbiter.score(atomic, context) >= tau, i.e. P(context
         entails atomic). score(claim, span) == P(span entails claim), so claim=atomic, span=context
         gives P(context |= atomic): the draft context (with the antecedents) must SUPPORT the atomic.
         Catches a decomposer that HALLUCINATED or added an unsupported fact (the context cannot
         entail it), while ALLOWING legitimate pronoun resolution AND legitimate narrowing.

    WHY the support check is vs the DRAFT CONTEXT, not the bare source sentence: a faithful
    decontextualizer RESOLVES pronouns, so the resolved atomic adds antecedent tokens the bare source
    lacks - checking P(bare source |= atomic) false-rejects that resolution (live DeBERTa scores
    P("It enforced it in 2018." |= "The EU enforced the GDPR in 2018.") ~0.00 vs P(context |= atomic)
    ~1.00). The context carries the antecedents, so it supports faithful resolutions and refuses
    hallucinations.

    WHY there is NO "preserves the claim" leg (atomic |= source_sentence), though earlier designs had
    one: a source SENTENCE often carries several facts ("It enforced it in 2018, and regulators fined
    several firms."), and a single atomic CANNOT entail the whole multi-fact sentence, so that leg
    scored ~0.001 and DROPPED every faithful atomic of a multi-fact sentence - a recall disaster
    (drift_rate 0.67 on a clean draft, observed live). Decomposition NATURALLY narrows; narrowing to a
    true sub-claim is faithful, so there is nothing to "preserve" against the full sentence.

    KNOWN LIMIT (backstopped): OVERGENERALIZATION - an atomic that drops a restrictive qualifier
    ("the drug reduced mortality" from "...only in patients over 65") is broader than the source
    supports, yet the context-support leg can pass it (NLI is weak on scope/restrictors; DeBERTa
    scores it ~0.99) and the symbolic veto does not cover qualifiers. So this CAN slip the round-trip.
    It is backstopped (imperfectly) at VERIFY against the retrieved span, and is the same NLI-tops-75%
    ceiling the project publishes; a scope-aware check is the refinement, not a return to the
    recall-killing preserves leg.

    CRITICAL - the arbiter MUST be the genuinely-orthogonal model (a different-lineage NLI such as
    DeBERTa-v3 MNLI/ANLI, or FakeEntailment in tests), NEVER the binder's MiniCheck verifier.
    MiniCheck judging a decomposition that MiniCheck will later score is the circularity the
    red-team forbids: it blesses its own drift. Enforced at runtime: a MiniCheck arbiter raises
    ValueError.

    Args:
        atomic: the candidate decontextualized atomic claim.
        source_sentence: the draft sentence the atomic was derived from (for the symbolic veto).
        context: the draft text (or an antecedent window) the atomic must be SUPPORTED BY; this is
            what makes legitimate pronoun resolution faithful while still vetoing hallucination.
        arbiter: a NON-MiniCheck EntailmentModel (see the CRITICAL note above).
        tau: the entailment threshold. Defaults to 0.7; tuned and frozen elsewhere,
            not load-bearing for this pure logic.

    Returns:
        True iff the atomic passed BOTH the symbolic veto and the context-support check.
    """
    _reject_minicheck_arbiter(arbiter)
    if not symbolic_consistency(atomic, source_sentence).ok:
        return False
    # SUPPORTED-BY-CONTEXT, the one robust NLI leg. score(claim, span) == P(span entails claim), so
    # score(atomic, context) == P(context entails atomic): the draft context (with antecedents) must
    # support the atomic. Allows pronoun resolution AND narrowing; refuses hallucination/added facts.
    return arbiter.score(atomic, context) >= tau


@dataclass
class DecompositionResult:
    """The outcome of decompose-and-filter: faithful atomics kept, drifted atomics dropped.

    drift_rate is the fraction of atomics the round-trip dropped (len(dropped) / total), i.e. how
    often the decomposer drifted - the decomposition-drift rate. It is 0.0 when the
    decomposer emitted no atomics.
    """

    kept: list[AtomicClaim] = field(default_factory=list)
    dropped: list[AtomicClaim] = field(default_factory=list)
    drift_rate: float = 0.0


def decompose_and_filter(
    draft: str,
    decomposer: Decomposer,
    arbiter: EntailmentModel,
    *,
    tau: float = 0.7,
) -> DecompositionResult:
    """Decompose a draft, then drop every atomic that fails the faithfulness round-trip.

    Each emitted atomic is checked with is_faithful (symbolic veto AND context-support: the draft
    context must entail the atomic, via the NON-MiniCheck arbiter); faithful atomics go to kept,
    drifted ones to dropped. kept and dropped are disjoint and partition the emitted atomics.
    drift_rate = len(dropped) / total (0.0 when the decomposer emitted nothing).

    The DRAFT is passed as the does-not-add context for every atomic, because the draft holds the
    antecedents a resolved atomic legitimately surfaces (so legitimate pronoun resolution is not
    false-dropped while hallucinated content still is). NOTE: a tighter ANTECEDENT WINDOW (the
    source sentence plus a few preceding sentences) is a real-arbiter-wiring refinement - the full
    draft is the safe, recall-preserving default for now.

    Args:
        draft: the draft text to decompose.
        decomposer: any Decomposer (FakeDecomposer in tests; the real VeriScore one in anger).
        arbiter: a NON-MiniCheck EntailmentModel for the round-trip (see is_faithful's CRITICAL
            note). NEVER pass the binder's MiniCheck verifier here; doing so raises ValueError.
        tau: the entailment threshold passed through to is_faithful.

    Returns:
        A DecompositionResult with kept/dropped atomics and the decomposition-drift rate.
    """
    _reject_minicheck_arbiter(arbiter)
    atomics = decomposer.decompose(draft)
    kept: list[AtomicClaim] = []
    dropped: list[AtomicClaim] = []
    for atomic in atomics:
        if is_faithful(atomic.text, atomic.source_sentence, draft, arbiter, tau=tau):
            kept.append(atomic)
        else:
            dropped.append(atomic)
    total = len(atomics)
    drift_rate = len(dropped) / total if total else 0.0
    return DecompositionResult(kept=kept, dropped=dropped, drift_rate=drift_rate)
