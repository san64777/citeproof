"""The FastAPI app: routing, receipt serving by uuid, input validation, no path traversal."""

import pytest

pytest.importorskip("fastapi", reason="the 'app' extra is not installed (uv sync --extra app)")

from fastapi.testclient import TestClient  # noqa: E402

from citeproof.app import create_app
from citeproof.research import (
    ClaimReport,
    Ledger,
    MemoryReceiptStore,
    ResearchReport,
    SourceReport,
)


def _runner_factory(store: MemoryReceiptStore):
    def runner(question: str, urls: list[str] | None, on_progress=None) -> ResearchReport:
        if on_progress is not None:
            on_progress("Verifying...")
        rid = store.put("<html><body>RECEIPT BODY</body></html>")
        return ResearchReport(
            question=question, draft="A fact.",
            claims=[ClaimReport(claim="A fact.", status="cited", receipt_id=rid, url="https://ok.test/a")],
            sources=[SourceReport(url="https://ok.test/a", verdict="OK", status="ok"),
                     SourceReport(url="https://wall.test/b", verdict="LOGIN_WALL", status="excluded",
                                  reason="not verified-OK (verdict LOGIN_WALL)")],
            ledger=Ledger(cited=1, unverified=0, excluded=1),
        )
    return runner


def _client() -> tuple[TestClient, MemoryReceiptStore]:
    store = MemoryReceiptStore()
    app = create_app(_runner_factory(store), store)
    return TestClient(app), store


def test_index_serves_the_ui() -> None:
    client, _ = _client()
    r = client.get("/")
    assert r.status_code == 200
    assert "citeproof" in r.text


def test_research_endpoint_returns_ledger_and_claims() -> None:
    client, _ = _client()
    r = client.post("/api/research", json={"question": "how big is it?", "urls": []})
    assert r.status_code == 200
    body = r.json()
    assert body["ledger"] == {"cited": 1, "unverified": 0, "excluded": 1}
    assert body["claims"][0]["status"] == "cited"
    assert body["sources"][1]["verdict"] == "LOGIN_WALL"


def test_receipt_served_by_id() -> None:
    client, _ = _client()
    rid = client.post("/api/research", json={"question": "a question", "urls": []}).json()["claims"][0]["receipt_id"]
    r = client.get(f"/receipt/{rid}")
    assert r.status_code == 200
    assert "RECEIPT BODY" in r.text


def test_receipt_carries_a_sandboxing_csp() -> None:
    client, _ = _client()
    rid = client.post("/api/research", json={"question": "a question", "urls": []}).json()["claims"][0]["receipt_id"]
    r = client.get(f"/receipt/{rid}")
    csp = r.headers.get("content-security-policy", "")
    assert "sandbox allow-scripts" in csp
    assert "default-src 'none'" in csp


def test_research_stream_emits_progress_then_result() -> None:
    import json as _json

    client, _ = _client()
    r = client.post("/api/research/stream", json={"question": "a real question", "urls": []})
    assert r.status_code == 200
    events = [_json.loads(line) for line in r.text.splitlines() if line.strip()]
    kinds = [e["type"] for e in events]
    assert "progress" in kinds  # at least one progress line streamed
    assert kinds[-1] == "result"  # the final event is the report
    assert events[-1]["data"]["ledger"] == {"cited": 1, "unverified": 0, "excluded": 1}


def test_unknown_receipt_is_404() -> None:
    client, _ = _client()
    assert client.get("/receipt/deadbeefdeadbeef").status_code == 404


def test_receipt_id_cannot_traverse_to_files() -> None:
    # The receipt id is a store key, not a path - traversal must 404, never read a file.
    client, _ = _client()
    for evil in ["..%2f..%2f..%2fetc%2fpasswd", "....//etc/passwd"]:
        assert client.get(f"/receipt/{evil}").status_code in (404, 400)


def test_question_too_short_is_rejected() -> None:
    client, _ = _client()
    assert client.post("/api/research", json={"question": "x", "urls": []}).status_code == 422


def test_too_many_urls_rejected() -> None:
    client, _ = _client()
    r = client.post("/api/research", json={"question": "a real question", "urls": [f"https://e{i}.test" for i in range(20)]})
    assert r.status_code == 422
