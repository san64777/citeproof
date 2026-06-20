"""The entailment seam: a Protocol, a deterministic fake, and a LAZY MiniCheck adapter.

The binder depends only on the EntailmentModel Protocol, so the heavy model is fully swappable and
the core stays testable with no model weights and no new runtime dependency. FakeEntailment is the
deterministic stand-in every test uses. MiniCheckEntailment is the real adapter, but it imports
minicheck/transformers ONLY lazily (inside __init__), so importing this module never pulls in the
optional 'binder' extra. If that extra is missing, it raises a clear, actionable ImportError.

The primary signal in the two-signal verifier is MiniCheck-RoBERTa-Large entailment probability;
the orthogonal second signal is the symbolic check in symbolic.py. This module owns only the
entailment side.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

_WORD = re.compile(r"[a-z0-9]+")


@runtime_checkable
class EntailmentModel(Protocol):
    """A model that scores how strongly a span entails a claim, in [0, 1]."""

    def score(self, claim: str, span: str) -> float: ...


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _jaccard(claim: str, span: str) -> float:
    a = _tokens(claim)
    b = _tokens(span)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class FakeEntailment:
    """A deterministic entailment stand-in for tests and the synthetic seed.

    By default it scores token-overlap Jaccard between claim and span (deterministic, in [0, 1]).
    A caller may pass an explicit `scores` dict keyed by (claim, span) for exact control in tests;
    any pair not present in the dict falls back to the Jaccard default.
    """

    def __init__(self, scores: dict[tuple[str, str], float] | None = None) -> None:
        self._scores = dict(scores) if scores else {}

    def score(self, claim: str, span: str) -> float:
        if (claim, span) in self._scores:
            return self._scores[(claim, span)]
        return _jaccard(claim, span)


class MiniCheckEntailment:
    """Lazy adapter for MiniCheck-RoBERTa-Large (MIT). NEVER imports the heavy deps at module top.

    The minicheck/transformers import happens inside __init__, so merely importing this module is
    free. When the optional 'binder' extra is not installed, __init__ raises an actionable
    ImportError pointing the user at `uv sync --extra binder`.
    """

    def __init__(self, model_name: str = "roberta-large", cache_dir: str = "ckpts") -> None:
        # VERIFIED 2026-06-08 against the MiniCheck README (github.com/Liyan06/MiniCheck):
        #   - model_name is the SHORT name "roberta-large" (MiniCheck maps it internally to the MIT
        #     checkpoint lytang/MiniCheck-RoBERTa-Large), NOT a bare HuggingFace id.
        #   - import: `from minicheck.minicheck import MiniCheck`.
        #   - score(docs=, claims=) -> (pred_label, raw_prob, _, _); raw_prob is the support prob.
        #   - install (NOT on PyPI, so a documented step, not a locked dep; see pyproject):
        #       uv pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"
        #   - LIVE-VERIFIED (wired + scored on an RTX 3060): clean-entail ~0.98, unrelated ~0.10,
        #     not-entailed ~0.10, near-miss-number ~0.03. Needs `accelerate` (transformers 5.x
        #     device_map) and the nltk 'punkt_tab' resource (fetched below).
        # cache_dir holds the downloaded checkpoint (model weights); keep it gitignored.
        try:
            from minicheck.minicheck import MiniCheck  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised via the ImportError test path
            raise ImportError(
                "MiniCheckEntailment requires the optional 'binder' deps plus the MiniCheck package. "
                "Install with:\n"
                "  uv sync --extra binder\n"
                '  uv pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"'
            ) from exc

        # MiniCheck tokenizes with nltk; newer nltk needs 'punkt_tab' (it ships 'punkt' only). Fetch
        # it once, quietly, so the first real score does not crash. nltk arrives with minicheck.
        # nltk.download() returns False (it does NOT raise) when offline or the resource is missing,
        # which would let construction "succeed" and then fail mid-run on the first score. So we only
        # download when the resource is ABSENT, and treat a failed download as fatal HERE - a
        # verification binder that cannot tokenize must fail loudly at load, never silently mis-score.
        import nltk

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            if not nltk.download("punkt_tab", quiet=True):
                raise RuntimeError(
                    "MiniCheck needs the nltk 'punkt_tab' tokenizer but it is absent and could not be "
                    "downloaded (offline?). Pre-fetch it once online: "
                    "python -c \"import nltk; nltk.download('punkt_tab')\""
                ) from None
        self._model = MiniCheck(model_name=model_name, cache_dir=cache_dir)

    def score(self, claim: str, span: str) -> float:
        # MiniCheck scores (doc, claim); it returns a support label and a probability. We map the
        # support probability into [0, 1]. Signature confirmed against the installed package at the
        # time the binder extra is wired (see the VERIFY note above).
        _pred, raw_prob, _raw_label, _ = self._model.score(  # type: ignore[no-untyped-call]
            docs=[span], claims=[claim]
        )
        prob = raw_prob[0] if isinstance(raw_prob, (list, tuple)) else raw_prob
        return float(prob)


_DB_MAX_TOKENS = 512  # DeBERTa-v3 trained position limit; the per-forward token budget (premise+claim)
_DB_PREMISE_WINDOW = 448  # premise tokens per window, leaving room for the claim + special tokens
_DB_MAX_WINDOWS = 24  # cap windows so a pathologically long source cannot run unbounded (~10k tokens)


class DebertaMnliEntailment:
    """Lazy adapter for a DeBERTa-v3 MNLI model: the NON-MiniCheck arbiter for the decompose
    round-trip. A different lineage from MiniCheck, so using it as the round-trip arbiter avoids the
    MiniCheck-judging-MiniCheck circularity the red-team forbids. NEVER imports the heavy deps at
    module top.

    score(claim, span) returns P(span entails claim): NLI with premise=span, hypothesis=claim, the
    entailment probability (softmax). The entailment label index is read from the model's own
    config.id2label, not assumed.
    """

    def __init__(self, model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli") -> None:
        # VERIFIED 2026-06-09 against the HF model card: tokenizer(premise, hypothesis) (premise
        # first), softmax over the last axis, label_names = [entailment, neutral, contradiction]
        # (the entailment index is read from config.id2label below, not hardcoded). Different lineage
        # from MiniCheck = the round-trip's required orthogonality.
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised via the ImportError test path
            raise ImportError(
                "DebertaMnliEntailment requires the optional 'binder' dependencies (transformers + "
                "torch), which are not installed. Install them with: uv sync --extra binder"
            ) from exc

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self._device)
        self._model.eval()
        id2label = {int(k): str(v).lower() for k, v in self._model.config.id2label.items()}
        entail = [i for i, label in id2label.items() if "entail" in label]
        self._entail_idx = entail[0] if entail else 0

    def score(self, claim: str, span: str) -> float:
        torch = self._torch
        # premise = span (the source / context), hypothesis = claim: does the span entail the claim?
        #
        # DeBERTa-v3 disentangled attention is O(seq^2) in MEMORY, and its tokenizer reports an
        # effectively-unbounded model_max_length, so `truncation=True` ALONE does not truncate - a
        # full real page (thousands of tokens) OOMs the GPU (build_relative_position allocates a
        # seq x seq tensor). So we WINDOW the premise into <= _DB_MAX_TOKENS pieces, score the claim
        # against each, and take the MAX entailment: the claim is supported if SOME part of the
        # source entails it. Per-forward memory is bounded to one window. On a SHORT source (one
        # window) this is identical to the old single-pass, so the M0-frozen thresholds are
        # unchanged - only long real pages (which M0 never had) now chunk instead of OOM.
        ids = self._tokenizer(span, add_special_tokens=False)["input_ids"]
        windows: list[str]
        if len(ids) <= _DB_PREMISE_WINDOW:
            windows = [span]
        else:
            windows = [
                str(self._tokenizer.decode(ids[i:i + _DB_PREMISE_WINDOW]))
                for i in range(0, min(len(ids), _DB_PREMISE_WINDOW * _DB_MAX_WINDOWS), _DB_PREMISE_WINDOW)
            ]
        best = 0.0
        for window in windows:
            inputs = self._tokenizer(
                window, claim, truncation=True, max_length=_DB_MAX_TOKENS, return_tensors="pt"
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**inputs).logits[0]
            prob = float(torch.softmax(logits, dim=-1)[self._entail_idx].item())
            best = max(best, prob)
        return best
