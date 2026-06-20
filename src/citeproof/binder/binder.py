"""EntailmentBinder: the VERIFY + ATTACH-OR-FLAG core, behind a strict cite-gate.

This is the (claim, source) unit the eval harness scores. It implements the locked binder posture:
abstention-first; cite ONLY when a verbatim span anchors AND entailment clears the
threshold AND the orthogonal symbolic check finds no contradiction; else drop and flag. The two
signals are ANDed exactly as pre-registered: entailment and the
symbolic check are SEPARATE gates, and a high entailment score can NEVER override a symbolic
contradiction - that is the whole point of the orthogonal gate.

Order of checks (the cite-gate is non-negotiably FIRST):
  1. CITE-GATE: if verdict is not Verdict.OK -> abstain. UNVERIFIED is excluded exactly like
     BLOCKED. No candidate is even considered for a non-OK page.
  2. Split the source into candidate spans (positional offsets preserved).
  3. For each candidate: entailment score e, symbolic ok s, and anchorability in the source.
  4. eligible = candidates with e >= tau_mc AND s True AND anchorable (two separate gates).
  5. eligible empty -> abstain (carry the best candidate's diagnostics). Else pick the highest-e
     eligible candidate.
  6. OPTIONAL third gate (Section 7.2/7.3): if a different-lineage NLI (DeBERTa) is configured, it
     must also score the claim against the FULL source at >= tau_db, else abstain. This catches a
     near-miss where the primary is overconfident but the orthogonal NLI is not (it scored ~0.01-0.09
     on the live over-attributions MiniCheck waved through at ~0.80). Default off.

Deliberately OUT of scope for this chunk (they operate above the (claim, source) unit, at the
draft/corpus level): DECOMPOSE, RETRIEVE, and the ALCE leave-one-out PRUNE.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from veriscrape import Verdict

from citeproof.binder.entailment import EntailmentModel
from citeproof.binder.spans import Span, candidate_spans, find_anchor
from citeproof.binder.symbolic import symbolic_consistency
from citeproof.eval.models import BinderOutput, ClaimSourcePair

# Cheap lexical pre-filter: a long real page yields ~2000 candidate sentences, and scoring EVERY one
# with the (slow) entailment model per claim is the dominant cost of a live query. The supporting
# sentence almost always shares the claim's distinctive content words, so we rank candidates by
# IDF-weighted content-word overlap and run the expensive model only on the top _PREFILTER_K. A
# sentence with no shared content word was never going to entail (after the citation-aware sentence
# split, candidates are sentence-sized, so the old "huge span happens to contain the support" case is
# gone). This cuts precision NOTHING (the winner is in the kept set) and recall only in the rare case
# a true support shares zero distinctive words - guarded by a generous K and validated against a full
# scan. Disabled (k <= 0) for exact backward-compatibility; a no-op when candidates <= k (short M0
# sources), so the frozen thresholds and unit tests are untouched.
_PREFILTER_K = 48
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "the a an and or but if then else of to in on at by for with from into over under as is are was "
    "were be been being it its this that these those they them their he she his her him we us our you "
    "your i me my not no nor so than too very can could should would may might must will shall do does "
    "did has have had having which who whom whose what when where why how all any both each few more "
    "most other some such only own same s t will just don there here out up down off above below "
    "between through during before after about against among also another because been being get got "
    "make made many much new now one two three use used using like include including".split()
)


# Non-prose sections (navigation, reference lists, infobox tables) are NOT citable - they mention
# the topic, so an NLI scores them spuriously high (a Wikipedia "See also" list out-scored the real
# birth-date sentence 0.984 vs 0.980 for "X was born on ..."), and highlighting them is a
# mis-attribution. Drop them before scoring. Targeted so it never drops real prose.
_BOILERPLATE_FIRST = frozenset([
    "see also", "references", "external links", "further reading", "notes", "citations",
    "bibliography", "footnotes", "sources", "related articles", "general references", "works cited",
])


def _is_citable_prose(text: str) -> bool:
    """False for navigation / reference / table chunks that an NLI mis-scores as support."""
    t = text.strip()
    if not t:
        return False
    first_line = t.split("\n", 1)[0].strip().strip(":").lower()
    if first_line in _BOILERPLATE_FIRST:
        return False  # "See also", "References", "External links", ...
    if t.startswith("^") or t.lstrip("-* ").startswith("^"):
        return False  # a footnote/reference entry
    if t.count("|") >= 4:
        return False  # an infobox / wikitable, not a sentence
    return True


def _content_words(text: str) -> set[str]:
    """Distinctive content words (lowercased, >2 chars, stopwords removed) for cheap lexical ranking.
    Dropping stopwords is what stops a candidate from ranking high on a bare subject overlap like
    'They are ...' - the exact pattern that mis-ranked an unrelated sentence before."""
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 2 and w not in _STOPWORDS}


def _prefilter_candidates(claim: str, candidates: list[Span], k: int) -> list[Span]:
    """Keep the top-k candidates by IDF-weighted shared content words with the claim (rarer words
    count more, so 'sunscreen' outweighs 'water'). No-op when k <= 0 or candidates <= k."""
    if k <= 0 or len(candidates) <= k:
        return candidates
    claim_words = _content_words(claim)
    if not claim_words:
        return candidates[:k]
    cand_words = [_content_words(c.text) for c in candidates]
    df: Counter[str] = Counter()
    for cw in cand_words:
        df.update(claim_words & cw)
    n = len(candidates)
    weight = {w: math.log(1 + n / (1 + df[w])) for w in claim_words}
    order = sorted(
        range(len(candidates)),
        key=lambda i: sum(weight[w] for w in (claim_words & cand_words[i])),
        reverse=True,
    )
    return [candidates[i] for i in order[:k]]


@dataclass(frozen=True)
class _Scored:
    """A candidate span with its two-signal scores and anchor, for ranking and diagnostics."""

    span: Span
    entailment: float
    symbolic_ok: bool
    anchor_start: int
    anchor_end: int


class EntailmentBinder:
    """Binds a claim to a verified-OK source span, or abstains. Implements the Binder protocol.

    Args:
        entailment: any EntailmentModel (FakeEntailment in tests; MiniCheckEntailment in anger).
        tau_mc: the MiniCheck entailment threshold. A candidate is entailment-eligible only when
            its score is >= tau_mc. Tuned on the DEV fold and FROZEN before the test fold is scored;
            the default here is a placeholder, not the frozen value.
        second_signal: an OPTIONAL different-lineage NLI used as a third gate (a different-lineage
            NLI such as DeBERTa-v3 MNLI/ANLI).
            When set, a citation the primary would attach is VETOED unless this model scores the
            claim against the FULL source at >= tau_db. It scores the whole source (not the winning
            span) so it catches a contradiction in a DIFFERENT sentence than the supporting span
            (e.g. the source supports the claim in one clause and refutes it in another). It must be
            a DIFFERENT lineage from `entailment` (a DeBERTa NLI vs MiniCheck), per the registration.
            Default None -> the gate is inert and behavior is identical to the two-signal binder.
        tau_db: the second-signal threshold. Tuned on the DEV fold and FROZEN like tau_mc; the
            default is a placeholder.
    """

    def __init__(
        self,
        entailment: EntailmentModel,
        tau_mc: float = 0.7,
        *,
        second_signal: EntailmentModel | None = None,
        tau_db: float = 0.5,
        prefilter_k: int = _PREFILTER_K,
    ) -> None:
        if not (0.0 <= tau_mc <= 1.0):
            raise ValueError(f"tau_mc must be in [0, 1], got {tau_mc}")
        if not (0.0 <= tau_db <= 1.0):
            raise ValueError(f"tau_db must be in [0, 1], got {tau_db}")
        self.entailment = entailment
        self.tau_mc = tau_mc
        self.second_signal = second_signal
        self.tau_db = tau_db
        # Cap on candidates scored by the entailment model per bind (the speed lever). <= 0 disables.
        self.prefilter_k = prefilter_k

    def bind(self, pair: ClaimSourcePair) -> BinderOutput:
        # 1. CITE-GATE FIRST (non-negotiable): only verdict == OK pages are ever citable. UNVERIFIED
        #    is excluded exactly like BLOCKED. We do not even score candidates for a non-OK page.
        if pair.verdict is not Verdict.OK:
            return self._abstain(pair)

        # 2. Candidate spans, with char offsets preserved for anchoring. Drop non-prose chunks
        #    (nav lists, references, infobox tables) - an NLI mis-scores them as support.
        candidates = [c for c in candidate_spans(pair.source_text) if _is_citable_prose(c.text)]
        if not candidates:
            return self._abstain(pair)

        # 2b. Cheap lexical pre-filter so the expensive entailment model scores only the candidates
        #     that could plausibly support the claim (the speed lever; see _prefilter_candidates).
        candidates = _prefilter_candidates(pair.claim, candidates, self.prefilter_k)

        # 3. Score every candidate on BOTH signals and locate it in the source.
        scored: list[_Scored] = []
        for cand in candidates:
            e = self.entailment.score(pair.claim, cand.text)
            s = symbolic_consistency(pair.claim, cand.text).ok
            anchor = find_anchor(cand.text, pair.source_text, near=cand.start)
            if anchor is None:
                # Genuinely unlocatable: skip (it can never be a receipt). Repetition is NOT this
                # case - find_anchor relocates repeats by offset and only returns None on absence.
                continue
            scored.append(
                _Scored(
                    span=cand,
                    entailment=e,
                    symbolic_ok=s,
                    anchor_start=anchor.start,
                    anchor_end=anchor.end,
                )
            )

        if not scored:
            return self._abstain(pair)

        # 4. Two SEPARATE gates ANDed: entailment >= tau_mc AND symbolic ok (anchorability already
        #    enforced above by skipping unlocatable candidates). A symbolic contradiction is fatal
        #    no matter how high the entailment score - the orthogonal gate is the whole point.
        eligible = [c for c in scored if c.entailment >= self.tau_mc and c.symbolic_ok]

        if not eligible:
            # 5a. Abstain, carrying the BEST candidate's diagnostics (highest entailment) so a
            #     reviewer can see why nothing attached.
            best = max(scored, key=lambda c: c.entailment)
            return self._abstain(
                pair,
                entailment_prob=best.entailment,
                symbolic_ok=best.symbolic_ok,
            )

        # 5b. Attach the highest-entailment eligible candidate; its anchor offset is recorded.
        winner = max(eligible, key=lambda c: c.entailment)

        # 5c. OPTIONAL THIRD GATE: a different-lineage NLI vetoes a
        #     citation the primary would attach. It scores the claim against the FULL source so it
        #     catches a contradiction in a sentence OTHER than the supporting span (the primary scores
        #     per-span and can miss "supports here, refutes there"). Below tau_db -> abstain; a
        #     non-finite score is treated as a veto (fail safe: never cite on a broken signal).
        #     LIMIT: the second model truncates to its max input length (DeBERTa-v3 ~512 tokens), so on
        #     a long real page only the head is scored and a far-apart refutation can be truncated
        #     away; chunk-and-take-min over an over-length source is the M1 refinement (M0 sources are
        #     short). The score is recorded on the output either way for the independence diagnostic.
        second_prob: float | None = None
        if self.second_signal is not None:
            second_prob = self.second_signal.score(pair.claim, pair.source_text)
            if not math.isfinite(second_prob) or second_prob < self.tau_db:
                return self._abstain(
                    pair,
                    entailment_prob=winner.entailment,
                    symbolic_ok=True,
                    second_signal_prob=second_prob,
                )

        # A citation needs a receipt URL: BinderOutput requires a non-empty source_url on a cited
        # output, so without one we ABSTAIN (degrade gracefully) rather than raise from bind() and
        # crash the harness. Unreachable for veriscrape-fetched pairs (always have a URL); a guard
        # against malformed/hand-authored input.
        if not (pair.source_url and pair.source_url.strip()):
            return self._abstain(
                pair, entailment_prob=winner.entailment, symbolic_ok=True, second_signal_prob=second_prob
            )

        return BinderOutput(
            pair_id=pair.id,
            cited=True,
            abstained=False,
            cited_span=winner.span.text,
            cited_span_start=winner.span.start,
            source_url=pair.source_url,
            entailment_prob=winner.entailment,
            symbolic_ok=True,
            second_signal_prob=second_prob,
        )

    @staticmethod
    def _abstain(
        pair: ClaimSourcePair,
        *,
        entailment_prob: float | None = None,
        symbolic_ok: bool | None = None,
        second_signal_prob: float | None = None,
    ) -> BinderOutput:
        return BinderOutput(
            pair_id=pair.id,
            cited=False,
            abstained=True,
            entailment_prob=entailment_prob,
            symbolic_ok=symbolic_ok,
            second_signal_prob=second_signal_prob,
        )
