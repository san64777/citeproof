"""The research orchestrator: question -> verified sources -> draft -> per-claim receipts -> ledger.

The M2 vertical slice, composing only already-tested parts:

  search/BYO urls -> acquire (SSRF guard + verdict + EMPTY_SHELL-only render)
    -> cite-gate: non-OK sources EXCLUDED, listed honestly with their verdicts (the demo moment)
    -> snapshot the verified bytes + hash-checked extract
    -> the brain drafts from the OK sources only
    -> the draft splits into claims (a seam: sentence-split by default, the M0 decomposer drop-in)
    -> EVERY claim runs through the binder against the best-overlap sources; a cite must then
       re-anchor in that source's artifact, or it is dropped to unverified
    -> the LEDGER: N cited / M unverified / K excluded - the product's honesty surface.

A claim with no receipt stays in the report VISIBLY UNVERIFIED. Nothing is hidden, nothing is
padded: the numbers in the ledger are the product.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Callable, Protocol
from urllib.parse import urlparse

from pydantic import BaseModel
from veriscrape import Verdict

from citeproof.anchor import anchor_quote, proportional_near
from citeproof.binder.binder import EntailmentBinder
from citeproof.binder.spans import candidate_spans
from citeproof.brain import ABSTENTION_SENTINEL, Brain, SourceContext
from citeproof.eval.models import Bucket, ClaimSourcePair, Fold
from citeproof.extract import extract_text, read_artifact
from citeproof.receipt import render_receipt_html
from citeproof.search import SearchProvider, SearchResult, run_search
from citeproof.snapshot import snapshot_raw
from citeproof.spine import Receipt, _visible_text, acquire


class SourceReport(BaseModel):
    url: str
    title: str = ""
    verdict: str
    status: str  # ok | excluded | no_content | fetch_error
    reason: str | None = None


class ClaimReport(BaseModel):
    claim: str
    status: str  # cited | unverified
    receipt_id: str | None = None
    url: str | None = None
    reason: str | None = None


class Ledger(BaseModel):
    cited: int
    unverified: int
    excluded: int


class ResearchReport(BaseModel):
    question: str
    draft: str
    claims: list[ClaimReport]
    sources: list[SourceReport]
    ledger: Ledger


class _OkSource(BaseModel):
    url: str
    title: str
    extracted: str
    visible: str
    artifact_html: str  # the integrity-checked artifact bytes (read back via the recorded digest)
    artifact_path: str
    artifact_sha256: str
    tactic: str | None  # how the content was obtained ("rendered" when escalated), for disclosure


def sentence_claims(draft: str) -> list[str]:
    """Default claim splitter: the brain is prompted to write self-contained declarative sentences,
    so sentences ARE the claims. The M0 Ollama decomposer (decontextualization + drift filter) drops
    into this seam for production runs that need pronoun resolution.
    """
    return [s.text.strip() for s in candidate_spans(draft) if len(s.text.strip()) >= 25]


_WORD = re.compile(r"[a-z0-9]+")


def _overlap(a: str, b: str) -> float:
    wa, wb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    return len(wa & wb) / len(wa) if wa else 0.0


def _host(url: str) -> str:
    """The host of a url, for human-readable progress messages."""
    return urlparse(url).netloc or url


def focus_source(question: str, text: str, budget: int) -> str:
    """Give the brain the most QUESTION-RELEVANT slice of a long source, not just its head.

    The brain's prompt budget is small, so a long document (an RFC, a long article) must be
    windowed. Head-truncation is wrong: the answer is often DEEP past the head (an RFC's status-code
    definitions are in section 10, not the title block), so the brain would see only boilerplate and
    abstain. Instead we score paragraphs by lexical overlap with the QUESTION and keep the top ones
    in document order up to the budget. Short sources pass through unchanged. This affects only what
    the brain SEES (draft recall); the binder still verifies each claim against the FULL source, so
    precision is untouched.
    """
    if len(text) <= budget:
        return text
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    if not paras:
        return text[:budget]
    ranked = sorted(range(len(paras)), key=lambda i: _overlap(question, paras[i]), reverse=True)
    chosen: set[int] = set()
    total = 0
    for i in ranked:
        if total + len(paras[i]) + 1 > budget and chosen:
            continue
        chosen.add(i)
        total += len(paras[i]) + 1
        if total >= budget:
            break
    return "\n".join(paras[i] for i in sorted(chosen))[:budget]


class ReceiptStore(Protocol):
    def put(self, html: str) -> str: ...


class MemoryReceiptStore:
    """In-memory receipt store for the v0 app: uuid keys (server-generated - never path-derived)."""

    def __init__(self, cap: int = 500) -> None:
        self._cap = cap
        self._store: dict[str, str] = {}

    def put(self, html: str) -> str:
        if len(self._store) >= self._cap:
            self._store.pop(next(iter(self._store)))
        rid = uuid.uuid4().hex
        self._store[rid] = html
        return rid

    def get(self, rid: str) -> str | None:
        return self._store.get(rid)


def run_research(
    question: str,
    *,
    binder: EntailmentBinder,
    brain: Brain,
    provider: SearchProvider,
    store: MemoryReceiptStore,
    out_dir: Path,
    urls: list[str] | None = None,
    k_sources: int = 6,
    decompose: Callable[[str], list[str]] = sentence_claims,
    timeout: float = 20.0,
    brain_source_chars: int = 4000,
    on_progress: Callable[[str], None] | None = None,
) -> ResearchReport:
    def emit(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    # 1. candidate sources: explicit URLs win; otherwise search proposes. run_search derives entity
    #    query variants from the question and merges results, so a conversational question still
    #    surfaces the authoritative page (not just articles tangential to it).
    candidates: list[SearchResult] = [SearchResult(title=u, url=u) for u in (urls or [])]
    if not candidates:
        emit("Searching for sources...")
        candidates = run_search(provider, question, k=k_sources)

    pool = candidates[: max(k_sources, len(urls or []))]
    sources: list[SourceReport] = []
    ok: list[_OkSource] = []
    for si, cand in enumerate(pool, 1):
        emit(f"Fetching and verifying source {si}/{len(pool)}: {_host(cand.url)}")
        try:
            record = acquire(cand.url, timeout=timeout)
        except Exception as exc:
            sources.append(SourceReport(url=cand.url, title=cand.title, verdict="FETCH_ERROR",
                                        status="fetch_error", reason=f"{type(exc).__name__}"))
            continue
        verdict = record.verdict.value
        if not record.ok:
            sources.append(SourceReport(url=cand.url, title=cand.title, verdict=verdict,
                                        status="excluded",
                                        reason=f"not verified-OK (verdict {verdict})"))
            continue
        body = record.text or ""
        artifact = snapshot_raw(body, cand.url, out_dir)
        extracted = extract_text(artifact)
        if not extracted:
            sources.append(SourceReport(url=cand.url, title=cand.title, verdict=verdict,
                                        status="no_content", reason="no extractable main content"))
            continue
        # Read the artifact back through the recorded digest, so the bytes the receipt renders are
        # the SAME bytes the one-artifact invariant pins - not a separate in-memory copy.
        verified_html = read_artifact(artifact)
        sources.append(SourceReport(url=cand.url, title=cand.title, verdict=verdict, status="ok"))
        ok.append(_OkSource(url=cand.url, title=cand.title, extracted=extracted,
                            visible=_visible_text(verified_html), artifact_html=verified_html,
                            artifact_path=artifact.path, artifact_sha256=artifact.sha256,
                            tactic=record.tactic))

    excluded_count = sum(1 for s in sources if s.status != "ok")

    # 2. draft from the verified sources only. Each source is windowed to the brain's budget around
    #    the QUESTION-RELEVANT passages (focus_source), so the answer to a long document is not lost
    #    to head-truncation. The binder still verifies against the FULL extracted text below.
    if not ok:
        return ResearchReport(question=question, draft="", claims=[], sources=sources,
                              ledger=Ledger(cited=0, unverified=0, excluded=excluded_count))
    emit(f"Drafting an answer from {len(ok)} verified source{'s' if len(ok) != 1 else ''}...")
    draft = brain.draft(question, [
        SourceContext(url=s.url, title=s.title, text=focus_source(question, s.extracted, brain_source_chars))
        for s in ok
    ])

    # The brain abstains with an EXACT sentinel when the sources do not answer the question. That is
    # a control signal, not a claim - return it as the draft with ZERO claims, never verify it (a
    # meta-statement about the absence of an answer must never become a citation).
    if draft.strip().rstrip(".").strip() == ABSTENTION_SENTINEL:
        return ResearchReport(question=question, draft=draft, claims=[], sources=sources,
                              ledger=Ledger(cited=0, unverified=0, excluded=excluded_count))

    # 3. verify every claim; attach a receipt or leave it visibly unverified.
    decomposed = decompose(draft)
    claims: list[ClaimReport] = []
    for ci, claim in enumerate(decomposed):
        emit(f"Verifying claim {ci + 1}/{len(decomposed)} against the sources...")
        ranked = sorted(ok, key=lambda s: _overlap(claim, s.extracted), reverse=True)
        report = ClaimReport(claim=claim, status="unverified",
                             reason="no source passed the verification gates for this claim")
        verified_but_unanchored = False  # the binder backed the claim, but no receipt could be anchored
        for src in ranked[:3]:
            pair = ClaimSourcePair(
                id=f"q{ci}", bucket=Bucket.CLEAN_ENTAILED, fold=Fold.TEST, claim=claim,
                source_url=src.url, source_text=src.extracted, verdict=Verdict.OK,
                gold_span=None, entailed=False, answerable=False,
            )
            out = binder.bind(pair)
            if not out.cited or not out.cited_span:
                continue
            # Disambiguate a repeated span: map the binder's verified offset (into extracted text)
            # to an approximate offset in the visible text, so a repeat highlights the RIGHT line.
            near = proportional_near(out.cited_span_start, len(src.extracted), len(src.visible))
            anchor = anchor_quote(out.cited_span, src.visible, near=near)
            if anchor is None:
                # The binder verified a span, but it could not be re-located in the snapshot the
                # receipt renders. We never mis-highlight, so try the next source; if all fail, say
                # so honestly (this is a recall loss in ANCHORING, not a failure to verify).
                verified_but_unanchored = True
                continue
            receipt = Receipt(
                claim=claim, url=src.url, verdict="OK", tactic=src.tactic,
                artifact_path=src.artifact_path, artifact_sha256=src.artifact_sha256,
                quote=out.cited_span, anchor_exact=anchor.exact, anchor_prefix=anchor.prefix,
                anchor_suffix=anchor.suffix, anchor_strategy=anchor.strategy,
                entailment_prob=out.entailment_prob,
            )
            rid = store.put(render_receipt_html(src.artifact_html, receipt))
            report = ClaimReport(claim=claim, status="cited", receipt_id=rid, url=src.url)
            break
        else:
            if verified_but_unanchored:
                report = ClaimReport(
                    claim=claim, status="unverified",
                    reason="a source supports this claim, but the exact supporting line could not be "
                           "re-located in the snapshot to anchor a receipt",
                )
        claims.append(report)

    cited = sum(1 for c in claims if c.status == "cited")
    return ResearchReport(
        question=question, draft=draft, claims=claims, sources=sources,
        ledger=Ledger(cited=cited, unverified=len(claims) - cited, excluded=excluded_count),
    )
