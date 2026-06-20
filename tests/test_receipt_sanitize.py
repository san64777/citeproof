"""Receipt sanitization: a fetched page's own JS/handlers must not run inside the receipt view."""

from citeproof.receipt import _sanitize_page


def test_page_scripts_are_stripped() -> None:
    html = "<html><body><p>real text</p><script>steal()</script></body></html>"
    out = _sanitize_page(html)
    assert "steal()" not in out
    assert "<script" not in out.lower()
    assert "real text" in out  # content preserved


def test_inline_event_handlers_are_stripped() -> None:
    html = '<html><body><img src="x" onerror="alert(1)"><div onclick="boom()">hi</div></body></html>'
    out = _sanitize_page(html).lower()
    assert "onerror" not in out
    assert "onclick" not in out
    assert "hi" in out


def test_javascript_urls_are_stripped() -> None:
    html = '<html><body><a href="javascript:evil()">x</a></body></html>'
    out = _sanitize_page(html).lower()
    assert "javascript:" not in out


def test_iframes_and_objects_are_stripped() -> None:
    html = '<html><body><iframe src="//evil"></iframe><object data="x"></object><p>keep</p></body></html>'
    out = _sanitize_page(html).lower()
    assert "<iframe" not in out
    assert "<object" not in out
    assert "keep" in out


def test_ordinary_content_and_links_survive() -> None:
    html = '<html><body><h1>Title</h1><a href="https://example.com">link</a><p>body</p></body></html>'
    out = _sanitize_page(html)
    assert "Title" in out
    assert "https://example.com" in out
    assert "body" in out


def test_meta_refresh_redirect_is_stripped() -> None:
    html = '<html><head><meta http-equiv="refresh" content="0;url=//evil"></head><body>x</body></html>'
    out = _sanitize_page(html).lower()
    assert "http-equiv" not in out
    assert "refresh" not in out


def test_xlink_href_javascript_is_stripped() -> None:
    html = '<html><body><svg><a xlink:href="javascript:evil()"><text>x</text></a></svg></body></html>'
    out = _sanitize_page(html).lower()
    assert "javascript:" not in out


def test_data_urls_are_stripped_from_navigational_attrs() -> None:
    html = '<html><body><a href="data:text/html,<script>evil()</script>">x</a></body></html>'
    out = _sanitize_page(html).lower()
    assert "data:text/html" not in out


def test_data_image_survives_on_img_src() -> None:
    # A data:image is legit inline CONTENT (CSP img-src data: constrains it to images); it must NOT
    # be stripped from a resource attr, only from navigational ones.
    html = '<html><body><img src="data:image/png;base64,iVBORw0KGgo="></body></html>'
    out = _sanitize_page(html)
    assert "data:image/png" in out


def test_external_stylesheet_link_is_stripped() -> None:
    html = '<html><head><link rel="stylesheet" href="//evil/x.css"></head><body>x</body></html>'
    out = _sanitize_page(html).lower()
    assert "<link" not in out


def test_inline_style_survives_for_fidelity() -> None:
    # <style> is KEPT (the receipt CSP blocks its external fetches); inline styling renders.
    html = "<html><head><style>p{color:red}</style></head><body><p>x</p></body></html>"
    out = _sanitize_page(html)
    assert "color:red" in out
