"""The v0 local web app: the spine with a face. localhost-only by default.

One screen: ask a question (optionally paste source URLs), get the report with every claim either
CITED (click-to-verify receipt opens in the right pane, passage highlighted) or VISIBLY UNVERIFIED,
plus the verification ledger (N cited / M unverified / K excluded-with-verdict) and the source list
with verdicts - the "excluded the block page" moment is the EXCLUDED row, shown honestly.

Construction is dependency-injected (create_app takes the runner), so the API and UI are tested
against fakes and the production wiring lives in `main()`. Receipts are served from an in-memory
store under server-generated uuid ids - never from request-derived paths.
"""

from __future__ import annotations

import json
import queue
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from citeproof.receipt import RECEIPT_CSP
from citeproof.research import MemoryReceiptStore, ResearchReport

_STATIC = Path(__file__).resolve().parent / "static"

# A runner takes (question, urls, on_progress) and returns the report. on_progress (optional) is
# called with a human-readable status string at each pipeline stage, for live UI feedback.
ProgressCb = Callable[[str], None]
ResearchRunner = Callable[[str, "list[str] | None", "ProgressCb | None"], ResearchReport]


class ResearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    urls: list[str] = Field(default_factory=list, max_length=12)


def create_app(runner: ResearchRunner, store: MemoryReceiptStore) -> FastAPI:
    app = FastAPI(title="citeproof", docs_url=None, redoc_url=None)
    # FastAPI runs sync endpoints in a threadpool, but the binder's HuggingFace fast tokenizers are
    # NOT thread-safe (concurrent calls raise "RuntimeError: Already borrowed"), and there is one
    # GPU anyway - so research requests are SERIALIZED. A second query waits rather than crashing.
    research_lock = threading.Lock()

    @app.get("/", response_class=FileResponse)
    def index() -> FileResponse:
        return FileResponse(_STATIC / "index.html")

    @app.post("/api/research")
    def research(req: ResearchRequest) -> ResearchReport:
        with research_lock:
            return runner(req.question, req.urls or None, None)

    @app.post("/api/research/stream")
    def research_stream(req: ResearchRequest) -> StreamingResponse:
        # Stream progress as newline-delimited JSON: {"type":"progress","data":"..."} lines while the
        # pipeline runs, then one {"type":"result", data: <report>} (or {"type":"error"}). The work is
        # sync and GPU-bound, so it runs in a thread that pushes progress to a queue the generator
        # drains; the research_lock still serializes it (the tokenizers are not thread-safe).
        events: queue.Queue[tuple[str, object]] = queue.Queue()
        done = object()

        def work() -> None:
            try:
                with research_lock:
                    report = runner(req.question, req.urls or None,
                                    lambda m: events.put(("progress", m)))
                events.put(("result", report.model_dump()))
            except Exception as exc:  # surface a clean error line instead of a dropped stream
                events.put(("error", f"{type(exc).__name__}: {exc}"))
            finally:
                events.put(("__done__", done))

        threading.Thread(target=work, daemon=True).start()

        def gen() -> Iterator[str]:
            yield json.dumps({"type": "progress", "data": "Starting..."}) + "\n"
            while True:
                kind, payload = events.get()
                if kind == "__done__":
                    break
                yield json.dumps({"type": kind, "data": payload}) + "\n"

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    @app.get("/receipt/{rid}", response_class=HTMLResponse)
    def receipt(rid: str) -> HTMLResponse:
        html = store.get(rid)
        if html is None:
            raise HTTPException(status_code=404, detail="unknown receipt")
        # The CSP re-asserts the sandbox at the document level (defense in depth on top of the
        # iframe sandbox) and blocks all external network when the snapshot is viewed.
        return HTMLResponse(html, headers={"Content-Security-Policy": RECEIPT_CSP})

    return app


def build_production_runner(
    store: MemoryReceiptStore, *, model: str = "qwen3:8b", tau_mc: float = 0.5, tau_db: float = 0.3,
    k_sources: int = 5
) -> ResearchRunner:
    """The real wiring: MiniCheck + DeBERTa binder (the M0-frozen thresholds), the Ollama brain,
    the default search provider. Heavy models load once here, not per request.

    k_sources caps how many sources an auto-SEARCH query fetches+verifies (the local 8B brain +
    binder cost ~60-90s PER source, so the interactive default is small; pasted URLs are always all
    used). Raise it for thoroughness at the cost of latency.
    """
    from citeproof.binder.binder import EntailmentBinder
    from citeproof.binder.entailment import DebertaMnliEntailment, MiniCheckEntailment
    from citeproof.brain import OllamaBrain
    from citeproof.research import run_research
    from citeproof.search import default_provider

    binder = EntailmentBinder(
        MiniCheckEntailment(), tau_mc=tau_mc, second_signal=DebertaMnliEntailment(), tau_db=tau_db
    )
    brain = OllamaBrain(model=model)
    provider = default_provider()
    out_dir = Path(tempfile.mkdtemp(prefix="citeproof-"))

    def runner(question: str, urls: list[str] | None,
               on_progress: ProgressCb | None = None) -> ResearchReport:
        return run_research(question, binder=binder, brain=brain, provider=provider,
                            store=store, out_dir=out_dir, urls=urls, k_sources=k_sources,
                            on_progress=on_progress)

    return runner


def main() -> None:  # pragma: no cover - the production entry point
    import os

    import uvicorn

    store = MemoryReceiptStore()
    app = create_app(build_production_runner(store), store)
    # localhost-only by DEFAULT: this is a local product. Binding wider is an EXPLICIT opt-in via
    # CITEPROOF_HOST (e.g. 0.0.0.0 to reach it from the Windows host across the WSL2 boundary, or
    # from another machine) - which also exposes it to the local network, so it is never the default.
    host = os.environ.get("CITEPROOF_HOST", "127.0.0.1")
    port = int(os.environ.get("CITEPROOF_PORT", "8417"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    main()
