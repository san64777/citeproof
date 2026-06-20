"""RETRIEVE: top-k candidate spans for a claim, behind the cite-gate (OK snapshots only).

This is the binder's front half (the "RETRIEVE" stage): given a claim and a corpus of
candidate spans drawn from fetched pages, return the top-k most similar spans by cosine similarity.
It is the real recall bottleneck the M0 eval measures, and the oracle-retrieval ablation (gold span
treated as always retrieved) isolates the verifier's ceiling from retrieval loss.

Two seams keep the core testable with NO heavy model and NO new default runtime dependency:
  - Embedder is a Protocol; FakeEmbedder is a deterministic hashed bag-of-words that makes cosine
    ranks lexically meaningful, so retrieval-recall tests are real (a claim retrieves its
    lexically-overlapping gold span in top-k). numpy is already available (transitive via scipy).
  - SentenceTransformerEmbedder is a LAZY adapter: it imports sentence-transformers ONLY inside
    __init__ and raises an actionable ImportError when the optional 'binder' extra is missing.
    Importing this module never pulls in the heavy deps.

CITE-GATE, defense in depth: when ok_only is True
(the default), the Retriever INDEXES ONLY candidates whose verdict is Verdict.OK. A non-OK snapshot
is never even in the retrieval index, so a non-OK span can never be returned and never reach the
verifier. UNVERIFIED is excluded exactly like BLOCKED.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from veriscrape import Verdict

_WORD = re.compile(r"[a-z0-9]+")
_FAKE_DIM = 256


@runtime_checkable
class Embedder(Protocol):
    """Embeds each input text into a fixed-length vector (one vector per text)."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class FakeEmbedder:
    """Deterministic, lexically meaningful embedder: a fixed-dim hashed bag-of-words.

    For each text: lowercase, tokenize on word characters, hash each token into one of `dim`
    buckets (counting occurrences), then L2-normalize. The same input always yields the same
    vector, and texts that share words land closer in cosine space, so the retrieval-recall tests
    are genuinely meaningful (a claim ranks its lexically-overlapping gold span highly) with no
    model weights and no new runtime dependency.

    Hashing is stabilized with a fixed seed mixed into a stdlib blake2b digest, so it is
    deterministic across processes and Python's hash randomization does not affect it.
    """

    def __init__(self, dim: int = _FAKE_DIM) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim

    def _bucket(self, token: str) -> int:
        import hashlib

        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _tokens(text):
            vec[self._bucket(token)] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm == 0.0:
            return vec  # all-zero (no word tokens); cosine handles this safely
        return [v / norm for v in vec]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class SentenceTransformerEmbedder:
    """Lazy adapter for a sentence-transformers embedding model. NEVER imports it at module top.

    The sentence-transformers import happens inside __init__, so merely importing this module is
    free and the default `uv sync` (no extras) stays light. When the optional 'binder' extra is not
    installed, __init__ raises an actionable ImportError pointing the user at `uv sync --extra
    binder`.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B") -> None:
        # LIVE-VERIFIED (2026-06-09) on an RTX 3060 via the HF model card + a real run: the id
        # "Qwen/Qwen3-Embedding-0.6B" loads with a bare SentenceTransformer(model_name); 1024-dim,
        # cosine. Real scores: cos(claim, relevant) ~0.90 vs cos(claim, unrelated) ~0.36, and the
        # Retriever ranks the relevant span first. REFINEMENT (deferred, +1-5% recall per the model
        # card): Qwen3 benefits from a query instruction, encode(..., prompt_name="query"), for the
        # CLAIM while documents encode bare; that needs a query/document split at the Retriever level,
        # so the symmetric bare encode here is correct but slightly suboptimal for now.
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised via the ImportError test path
            raise ImportError(
                "SentenceTransformerEmbedder requires the optional 'binder' dependencies "
                "(sentence-transformers), which are not installed. Install them with: "
                "uv sync --extra binder"
            ) from exc

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(list(texts), normalize_embeddings=True)  # type: ignore[no-untyped-call]
        return [list(map(float, v)) for v in vectors]


@dataclass(frozen=True)
class Candidate:
    """A retrievable candidate span drawn from a fetched page, with its verdict and source offsets.

    verdict carries the page's veriscrape verdict so the cite-gate can be enforced at index time
    (ok_only). start/end are the char offsets of the span in the source page (0/0 when unknown).
    """

    span_text: str
    source_url: str
    verdict: Verdict
    start: int = 0
    end: int = 0


def _cosine_topk(query: np.ndarray, matrix: np.ndarray, k: int) -> list[int]:
    """Return the indices of the top-k rows of `matrix` by cosine similarity to `query`.

    Inputs are raw (not necessarily normalized) vectors; this normalizes safely, treating a
    zero-vector as similarity 0 to everything (never a division by zero). Ties break by lower
    index for determinism.
    """
    if matrix.shape[0] == 0 or k <= 0:
        return []

    q_norm = float(np.linalg.norm(query))
    row_norms = np.linalg.norm(matrix, axis=1)

    if q_norm == 0.0:
        sims = np.zeros(matrix.shape[0], dtype=float)
    else:
        denom = row_norms * q_norm
        dots = matrix @ query
        # Avoid divide-by-zero for zero rows; their similarity stays 0.
        sims = np.divide(dots, denom, out=np.zeros_like(dots, dtype=float), where=denom > 0)

    k = min(k, matrix.shape[0])
    # Stable order: sort by (-similarity, index) so ties resolve to the lower index deterministically.
    order = sorted(range(matrix.shape[0]), key=lambda i: (-float(sims[i]), i))
    return order[:k]


class Retriever:
    """Indexes candidate spans and returns the top-k most similar to a claim, behind the cite-gate.

    Args:
        embedder: any Embedder (FakeEmbedder in tests; SentenceTransformerEmbedder in anger).
        candidates: the corpus of candidate spans to index.
        ok_only: when True (default), ONLY candidates whose verdict is Verdict.OK are indexed.
            This is cite-gate defense in depth: a non-OK snapshot is never in the index, so a
            non-OK span can never be retrieved. UNVERIFIED is excluded exactly like BLOCKED. Set
            False only for diagnostics that deliberately need the full corpus.

    Candidate embeddings are pre-computed ONCE at construction. retrieve() embeds only the claim.
    """

    def __init__(
        self,
        embedder: Embedder,
        candidates: Sequence[Candidate],
        *,
        ok_only: bool = True,
    ) -> None:
        self.embedder = embedder
        self.ok_only = ok_only
        if ok_only:
            indexed = [c for c in candidates if c.verdict is Verdict.OK]
        else:
            indexed = list(candidates)
        self._candidates: list[Candidate] = indexed

        if indexed:
            vectors = embedder.embed([c.span_text for c in indexed])
            self._matrix = np.asarray(vectors, dtype=float)
        else:
            self._matrix = np.empty((0, 0), dtype=float)

    @property
    def size(self) -> int:
        """Number of indexed candidates (after the cite-gate filter)."""
        return len(self._candidates)

    def retrieve(self, claim: str, k: int = 5) -> list[Candidate]:
        """Return the top-k indexed candidates most similar to `claim`, by cosine similarity.

        Safe on an empty index (returns []), on k larger than the corpus (returns all, ranked),
        and on zero-vectors (a claim or candidate with no word tokens yields similarity 0, never a
        division error). Order is deterministic: ties break toward the lower index.
        """
        if not self._candidates:
            return []
        query = np.asarray(self.embedder.embed([claim])[0], dtype=float)
        top = _cosine_topk(query, self._matrix, k)
        return [self._candidates[i] for i in top]


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace, for robust substring/equality span matching."""
    return " ".join(text.lower().split())


def retrieval_hit(retrieved: Sequence[Candidate], gold_span: str) -> bool:
    """Did the gold span text appear among the retrieved candidates?

    Match is by normalized substring/equality (case-insensitive, whitespace-collapsed) so trivial
    formatting differences between the gold span and the retrieved span text do not cause a false
    miss. An empty gold_span never hits. This is the per-claim retrieval-recall metric the harness
    aggregates.
    """
    gold = _normalize(gold_span)
    if not gold:
        return False
    for cand in retrieved:
        span = _normalize(cand.span_text)
        if gold == span or gold in span or span in gold:
            return True
    return False


@dataclass(frozen=True)
class OracleRetrieval:
    """The oracle-retrieval ablation: the gold span is ALWAYS retrieved.

    Wrapping a real Retriever, retrieve() guarantees a Candidate carrying the gold span is present
    in the returned list (prepended if a real pass missed it), so the harness can measure the
    binder's VERIFIER-ONLY ceiling independent of retrieval loss. The oracle-vs-top-k gap diagnoses
    retrieval-vs-verifier. The gold candidate is synthesized as an OK candidate so it survives the
    cite-gate; this ablation is a diagnostic, never a shipping path.
    """

    retriever: Retriever
    gold_span: str
    gold_source_url: str = "oracle://gold-span"
    extra_fields: dict[str, int] = field(default_factory=dict)

    def retrieve(self, claim: str, k: int = 5) -> list[Candidate]:
        real = self.retriever.retrieve(claim, k=k)
        if retrieval_hit(real, self.gold_span):
            return real
        gold = Candidate(
            span_text=self.gold_span,
            source_url=self.gold_source_url,
            verdict=Verdict.OK,
        )
        # Prepend the oracle span and keep the list length at k (drop the weakest real tail).
        # k <= 0 returns [], mirroring Retriever (no surprising asymmetry).
        combined = [gold, *real]
        return combined[:k] if k > 0 else []
