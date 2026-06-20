"""Data model for the M0 eval: the labeled pair, the binder output, the Binder protocol.

These are the frozen units the harness scores. A ClaimSourcePair is one human-labeled
(claim, source) item in one of the five pre-registered buckets; a BinderOutput is what a
binder emits for it (a citation with a receipt, or an abstention). The Binder protocol is
the single seam every binder (baselines now, the real MiniCheck binder later) plugs into.

The verdict field reuses veriscrape's Verdict enum directly; the cite-gate (cite only when
verdict is Verdict.OK) is enforced by the harness, not by these models, because a deliberately
broken binder must be CONSTRUCTIBLE so the coverage check can prove it gets caught.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, Self

from pydantic import BaseModel, model_validator
from veriscrape import Verdict


class Bucket(str, Enum):
    """The five pre-registered binder-eval buckets."""

    CLEAN_ENTAILED = "clean_entailed"
    CLEAN_NOT_ENTAILED = "clean_not_entailed"
    NEAR_MISS_PARAPHRASE = "near_miss_paraphrase"
    DECONTEXT_FAILURE = "decontext_failure"
    ANSWERABLE_HARD = "answerable_hard"


class Fold(str, Enum):
    """Dev fold (threshold tuning) vs the write-once frozen test fold."""

    DEV = "dev"
    TEST = "test"


class ClaimSourcePair(BaseModel):
    """One human-labeled (claim, source) item; the atomic unit the harness scores.

    Labels are frozen before any model runs. `entailed` is the primary human label
    (LLM-AggreFact standard: fully supported by THIS source alone); `annotator2_entailed`
    holds the second independent annotator's label (buckets 3 and 4) for Cohen's kappa.
    `answerable` is the recall-denominator predicate: a gold-entailing span exists in at
    least one verdict==OK snapshot, independent of the binder's retrieval outcome.
    """

    id: str
    bucket: Bucket
    fold: Fold
    claim: str
    source_url: str
    source_text: str
    verdict: Verdict
    gold_span: str | None
    entailed: bool
    answerable: bool
    annotator2_entailed: bool | None = None
    notes: str | None = None


class BinderOutput(BaseModel):
    """What a binder emits for one pair: a citation with a receipt, or an abstention.

    Invariants (enforced at construction):
      - cited == (not abstained): exactly one of the two is true.
      - a CITED output MUST carry a non-empty cited_span AND a source_url (no receipt,
        no citation). An abstention carries neither.
    """

    pair_id: str
    cited: bool
    abstained: bool
    cited_span: str | None = None
    # The cited span's start offset in source_text (the verified occurrence the binder chose). A
    # receipt re-anchors the span text in the artifact's VISIBLE text, which repeats can make
    # ambiguous; this offset lets the re-anchor pick the occurrence NEAREST the verified one instead
    # of blindly taking the first (which could highlight the wrong line). None on an abstention.
    cited_span_start: int | None = None
    source_url: str | None = None
    entailment_prob: float | None = None
    symbolic_ok: bool | None = None
    # The optional second-signal (DeBERTa) probability, when a second signal is configured; None
    # otherwise. Recorded on both the cite and the second-signal-veto path so the pre-registered
    # "P(2nd wrong | MiniCheck wrong)" independence diagnostic is computable from binder outputs.
    second_signal_prob: float | None = None

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        if self.cited == self.abstained:
            raise ValueError("exactly one of cited / abstained must be true (cited == not abstained)")
        if self.cited:
            if self.cited_span is None or not self.cited_span.strip():
                raise ValueError("a cited output must carry a non-empty cited_span")
            if self.source_url is None or not self.source_url.strip():
                raise ValueError("a cited output must carry a source_url")
        return self


class Binder(Protocol):
    """The single seam every binder plugs into: bind one pair, return one output."""

    def bind(self, pair: ClaimSourcePair) -> BinderOutput: ...
