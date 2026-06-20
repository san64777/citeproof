"""The M1 spine: (url, claim) -> a verified receipt, or an honest exclusion.

Composition of the verified-fetch pipeline, every stage already built and tested on its own:

    fetch (SSRF-guarded, veriscrape verdict)
      -> EMPTY_SHELL-only render escalation (rendering, not circumvention)
      -> cite-gate: anything not strictly OK is EXCLUDED here, with the verdict as the reason
      -> snapshot (one sha256-pinned artifact)
      -> extract (from the artifact ONLY, hash-checked)
      -> bind (the M0 binder: entailment + symbolic + optional second signal, abstain-first)
      -> anchor (re-find the cited span in the artifact's visible text, abstain over mis-highlight)
      -> Receipt

TWO-FETCH HONESTY: SingleFile re-fetches a page to inline its resources, so its artifact can differ
from the bytes the verdict was computed on. The spine therefore snapshots THE VERIFIED BYTES
(snapshot_raw of the fetched body) by default - the one-artifact invariant holds exactly. A
SingleFile fidelity upgrade is only trustworthy if the new artifact is itself re-classified OK;
that upgrade path is deferred until receipts need offline resource fidelity (M2).

The binder is injected (any EntailmentBinder), so the spine composes with FakeEntailment in tests
and MiniCheck + DeBERTa in production - same code path either way.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from selectolax.parser import HTMLParser
from veriscrape import FetchRecord

from citeproof.anchor import TextAnchor, anchor_quote, proportional_near
from citeproof.binder.binder import EntailmentBinder
from citeproof.eval.models import Bucket, ClaimSourcePair, Fold
from citeproof.extract import extract_text
from citeproof.fetch import fetch
from citeproof.render import RenderUnavailableError, render_and_reclassify, should_escalate
from citeproof.snapshot import SnapshotArtifact, snapshot_raw


class Receipt(BaseModel):
    """One verified citation: the claim, the page that backs it, and the pinned evidence trail."""

    claim: str
    url: str
    verdict: str
    tactic: str | None  # how the content was obtained ("rendered" when escalated), for disclosure
    artifact_path: str
    artifact_sha256: str
    quote: str  # the binder's cited span (verbatim from the extracted text)
    anchor_exact: str  # the quote AS IT APPEARS in the artifact's visible text
    anchor_prefix: str
    anchor_suffix: str
    anchor_strategy: str
    entailment_prob: float | None


class SpineResult(BaseModel):
    """The outcome for one (url, claim): exactly one of receipt / exclusion reason is set."""

    url: str
    claim: str
    stage: str  # the last stage reached: excluded | no_content | abstained | unanchored | cited
    verdict: str
    receipt: Receipt | None = None
    reason: str | None = None


def _visible_text(html: str) -> str:
    """The artifact's human-visible text (script/style stripped) - the anchor target a receipt
    renderer re-finds in the live DOM.

    CRITICAL: this must concatenate text nodes THE SAME WAY the receipt's in-browser highlight script
    does (a TreeWalker doing `text += node.nodeValue`), i.e. with NO inter-node separator. Using
    separator=" " inserts spurious spaces around inline elements (a sentence crossing a Wikipedia
    link becomes "blast fishing , cyanide fishing"), so the anchor computed here would never be
    re-located in the live DOM and every such receipt would fail to highlight. separator="" matches
    the browser, so the natural spaces already in the text are kept and no phantom ones are added.
    """
    tree = HTMLParser(html)
    for tag in ("script", "style", "noscript", "template"):
        for node in tree.css(tag):
            node.decompose()
    body = tree.body
    return body.text(separator="") if body is not None else ""


def acquire(url: str, *, timeout: float = 20.0) -> FetchRecord:
    """Fetch with the SSRF guard; escalate ONLY an EMPTY_SHELL verdict through one headless render.
    If rendering is unavailable, the EMPTY_SHELL verdict stands (exclude, never guess).
    """
    record = fetch(url, timeout=timeout)
    if should_escalate(record):
        try:
            record = render_and_reclassify(url, timeout=max(timeout, 30.0))
        except RenderUnavailableError:
            pass  # no renderer installed: the shell verdict stands and the page is excluded
    return record


def run_spine(
    url: str,
    claim: str,
    binder: EntailmentBinder,
    out_dir: Path,
    *,
    timeout: float = 20.0,
    pair_id: str = "spine",
) -> SpineResult:
    """The full spine for one (url, claim). Every non-cited outcome is an HONEST, named stage:
    excluded (cite-gate), no_content (nothing extractable), abstained (binder), unanchored (the
    cited span could not be re-found in the artifact - dropped rather than mis-highlighted).
    """
    record = acquire(url, timeout=timeout)
    verdict = record.verdict.value

    # THE CITE-GATE: strictly OK or the page is out. UNVERIFIED is an abstention, excluded the same.
    if not record.ok:
        return SpineResult(url=url, claim=claim, stage="excluded", verdict=verdict,
                           reason=f"page is not verified-OK (verdict {verdict})")

    artifact: SnapshotArtifact = snapshot_raw(record.text or "", url, out_dir)
    extracted = extract_text(artifact)
    if not extracted:
        return SpineResult(url=url, claim=claim, stage="no_content", verdict=verdict,
                           reason="no main content extractable from the artifact")

    pair = ClaimSourcePair(
        id=pair_id, bucket=Bucket.CLEAN_ENTAILED, fold=Fold.TEST, claim=claim,
        source_url=url, source_text=extracted, verdict=record.verdict,
        gold_span=None, entailed=False, answerable=False,
    )
    out = binder.bind(pair)
    if not out.cited or not out.cited_span:
        return SpineResult(url=url, claim=claim, stage="abstained", verdict=verdict,
                           reason="binder abstained: no span passed the entailment + symbolic gates")

    visible = _visible_text(record.text or "")
    near = proportional_near(out.cited_span_start, len(extracted), len(visible))
    anchor: TextAnchor | None = anchor_quote(out.cited_span, visible, near=near)
    if anchor is None:
        return SpineResult(url=url, claim=claim, stage="unanchored", verdict=verdict,
                           reason="cited span could not be re-found in the artifact (dropped, not mis-highlighted)")

    receipt = Receipt(
        claim=claim, url=url, verdict=verdict, tactic=record.tactic,
        artifact_path=artifact.path, artifact_sha256=artifact.sha256,
        quote=out.cited_span,
        anchor_exact=anchor.exact, anchor_prefix=anchor.prefix, anchor_suffix=anchor.suffix,
        anchor_strategy=anchor.strategy,
        entailment_prob=out.entailment_prob,
    )
    return SpineResult(url=url, claim=claim, stage="cited", verdict=verdict, receipt=receipt)
