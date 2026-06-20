"""Receipt rendering: the snapshot, with the supporting passage highlighted - click-to-verify.

The receipt page IS the artifact (the sha256-pinned snapshot bytes) plus two injected pieces:
  1. a banner stating the claim, the source URL, the verdict, the artifact digest, and how the
     passage was re-found (the anchor strategy) - full disclosure, nothing hidden;
  2. a small script that re-locates the anchored passage in the LIVE DOM and highlights it via the
     CSS Custom Highlight API (supported by all three Tauri webview engines: Chromium, WebKit, and
     WebKitGTK), falling back to scroll-only when the API is unavailable.

The script re-finds the passage the same way anchor.py did (whitespace-collapsed search with an
index map back to real offsets), because DOM text-node whitespace differs from any server-side
extraction. If the passage cannot be re-found, the banner SAYS SO - a receipt never fakes a
highlight. Whole sentences/passages are highlighted, never token-exact fragments.
"""

from __future__ import annotations

import json
import re

from selectolax.parser import HTMLParser

from citeproof.spine import Receipt

# The receipt's Content-Security-Policy. `sandbox allow-scripts` re-asserts the iframe sandbox at the
# document level, so the policy travels with the receipt no matter how it is embedded (a future
# "open in new tab" cannot escalate to the app origin). default-src 'none' blocks ALL external
# network - no CSS/img/font/connect beacon can fire when a receipt is viewed, honoring the
# offline-and-provable promise - while inline style + the single inline highlight script are allowed,
# and img/font data: URIs render. No external origin is ever permitted.
RECEIPT_CSP = (
    "sandbox allow-scripts; default-src 'none'; style-src 'unsafe-inline'; "
    "script-src 'unsafe-inline'; img-src data:; font-src data:; form-action 'none'; base-uri 'none'"
)

_BANNER_CSS = """
#citeproof-banner { position: sticky; top: 0; z-index: 2147483647; background: #102a43;
  color: #f0f4f8; font: 14px/1.5 system-ui, sans-serif; padding: 10px 16px;
  border-bottom: 3px solid #2dd4bf; }
#citeproof-banner .cp-claim { font-weight: 600; }
#citeproof-banner .cp-meta { font-size: 12px; color: #9fb3c8; margin-top: 2px; word-break: break-all; }
#citeproof-banner .cp-warn { color: #fbbf24; font-weight: 600; }
::highlight(citeproof) { background-color: #fde68a; color: #1f2937; }
"""

# The re-anchoring script: TreeWalker over text nodes -> concatenated haystack with a node map ->
# whitespace-collapsed search -> Range over the matching node span -> CSS Custom Highlight.
_HIGHLIGHT_JS = """
(function () {
  var anchor = __CITEPROOF_ANCHOR__;
  function collect() {
    var walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
      acceptNode: function (n) {
        var p = n.parentNode && n.parentNode.nodeName;
        if (p === 'SCRIPT' || p === 'STYLE' || p === 'NOSCRIPT' || p === 'TEMPLATE') {
          return NodeFilter.FILTER_REJECT;
        }
        return NodeFilter.FILTER_ACCEPT;
      }
    });
    var nodes = [], text = '', node;
    while ((node = walker.nextNode())) { nodes.push({ node: node, start: text.length }); text += node.nodeValue; }
    return { nodes: nodes, text: text };
  }
  function normalize(s) {
    var out = '', map = [], inSpace = false;
    for (var i = 0; i < s.length; i++) {
      var ch = s[i];
      if (ch === '\\u00ad') continue;
      if (/\\s/.test(ch)) {
        if (!inSpace && out.length) { out += ' '; map.push(i); }
        inSpace = true; continue;
      }
      inSpace = false; out += ch; map.push(i);
    }
    if (out.endsWith(' ')) { out = out.slice(0, -1); map.pop(); }
    return { out: out, map: map };
  }
  function locate(doc, needle) {
    var h = normalize(doc.text), q = normalize(needle);
    if (!q.out) return null;
    var at = h.out.indexOf(q.out);
    if (at < 0) return null;
    return { start: h.map[at], end: h.map[at + q.out.length - 1] + 1 };
  }
  function nodeAt(doc, pos, preferStart) {
    for (var i = 0; i < doc.nodes.length; i++) {
      var e = doc.nodes[i], len = e.node.nodeValue.length;
      if (pos < e.start + len || (preferStart && pos === e.start + len && i + 1 === doc.nodes.length)) {
        return { node: e.node, offset: Math.min(Math.max(pos - e.start, 0), len) };
      }
    }
    var last = doc.nodes[doc.nodes.length - 1];
    return { node: last.node, offset: last.node.nodeValue.length };
  }
  function run() {
    var doc = collect();
    var hit = locate(doc, anchor.exact);
    var status = document.getElementById('citeproof-status');
    if (!hit) { if (status) { status.textContent = 'passage could not be re-located in this snapshot'; status.className = 'cp-warn'; } return; }
    var s = nodeAt(doc, hit.start, true), e = nodeAt(doc, hit.end, false);
    var range = document.createRange();
    range.setStart(s.node, s.offset); range.setEnd(e.node, e.offset);
    if (window.Highlight && CSS.highlights) {
      CSS.highlights.set('citeproof', new Highlight(range));
      if (status) status.textContent = 'supporting passage highlighted (' + anchor.strategy + ')';
    } else if (status) {
      status.textContent = 'highlight API unavailable; passage located (' + anchor.strategy + ')';
    }
    var el = s.node.parentElement; if (el && el.scrollIntoView) el.scrollIntoView({ block: 'center' });
  }
  if (document.readyState === 'loading') { document.addEventListener('DOMContentLoaded', run); } else { run(); }
})();
"""


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# Script-bearing URL schemes are stripped from EVERY url attribute. `data:` is dangerous only where
# it can NAVIGATE/execute (data:text/html in a link), so it is stripped from navigational attrs but
# KEPT on resource attrs (img/video src), where a data:image is legit inline content and the CSP's
# `img-src data:` already constrains it to images.
_SCRIPT_URL = re.compile(r"^\s*(javascript|vbscript)\s*:", re.IGNORECASE)
_DATA_URL = re.compile(r"^\s*data\s*:", re.IGNORECASE)

# Navigational attrs: a URL here can drive a navigation/submission, so data: is unsafe too.
_NAV_URL_ATTRS = ("href", "xlink:href", "action", "formaction")
# Resource attrs: a URL here loads a subresource; data: is allowed (CSP-constrained to images).
_RESOURCE_URL_ATTRS = ("src", "poster")

# THE security boundary is the iframe sandbox (sandbox="allow-scripts" WITHOUT allow-same-origin in
# index.html), which forces the receipt into a unique opaque origin: even script that slips past this
# function cannot read the app origin, its cookies, or storage, and cannot navigate the top frame.
# `_sanitize_page` is DEFENSE IN DEPTH, not the boundary - it must never be trusted enough to weaken
# the sandbox. It strips the page's own active content so the only script that runs is the highlight
# script citeproof injects.


def _sanitize_page(html: str) -> str:
    """Defang the fetched page for viewing: remove its scripts, framing/embedding elements, redirect
    and link elements, inline event handlers, javascript:/vbscript: URLs everywhere, and data: URLs
    in navigational attrs. The artifact FILE on disk stays byte-true (its hash must keep matching);
    only this viewable rendering changes.
    """
    tree = HTMLParser(html)
    # script (incl. inside <svg>), framing/embedding, <base>/<form> hijacks, <meta> (http-equiv
    # refresh redirects), and <link> (prefetch/import/stylesheet beacons). <style> is KEPT for visual
    # fidelity - its external url()/@import fetches are neutralized by the receipt's CSP, which allows
    # inline style but no external network (see RECEIPT_CSP / app.py).
    for tag in ("script", "iframe", "frame", "frameset", "object", "embed", "applet", "base",
                "form", "meta", "link"):
        for node in tree.css(tag):
            node.decompose()
    for node in tree.css("*"):
        attrs = node.attributes
        for name in list(attrs):
            low = name.lower()
            value = attrs.get(name) or ""
            drop = (
                low.startswith("on")
                or (low in _NAV_URL_ATTRS and (_SCRIPT_URL.match(value) or _DATA_URL.match(value)))
                or (low in _RESOURCE_URL_ATTRS and _SCRIPT_URL.match(value))
            )
            if drop:
                del node.attrs[name]
    return tree.html or ""


def render_receipt_html(artifact_html: str, receipt: Receipt) -> str:
    """The artifact (defanged: page scripts/handlers stripped) with the receipt banner + highlight
    script injected. The artifact FILE is never modified (its hash must keep matching); this
    produces a separate, viewable receipt document.
    """
    artifact_html = _sanitize_page(artifact_html)
    # The anchor text comes FROM THE FETCHED PAGE (attacker-controlled), and it is interpolated
    # inside a <script> block - so "</" must be neutralized or a page containing "</script>" would
    # break out of the block and inject markup into the receipt (XSS). U+2028/U+2029 are legal in
    # JSON but terminate lines in JS string literals, so they are escaped too.
    anchor_json = (
        json.dumps(
            {"exact": receipt.anchor_exact, "prefix": receipt.anchor_prefix,
             "suffix": receipt.anchor_suffix, "strategy": receipt.anchor_strategy}
        )
        .replace("</", "<\\/")
        # json.dumps emits U+2028/U+2029 RAW; they are legal in JSON but terminate a JS string
        # literal, so escape them. Written as \\u escapes (not the literal separator chars)
        # so the codepoints are unambiguous in source.
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    tactic = f" via {_esc(receipt.tactic)}" if receipt.tactic else ""
    banner = (
        f'<div id="citeproof-banner"><div class="cp-claim">{_esc(receipt.claim)}</div>'
        f'<div class="cp-meta">source: {_esc(receipt.url)} &middot; verdict {_esc(receipt.verdict)}{tactic} '
        f"&middot; snapshot sha256 {receipt.artifact_sha256[:16]}&hellip; &middot; "
        f'<span id="citeproof-status">locating passage&hellip;</span></div></div>'
    )
    injection = (
        f"<style>{_BANNER_CSS}</style>{banner}"
        f"<script>{_HIGHLIGHT_JS.replace('__CITEPROOF_ANCHOR__', anchor_json)}</script>"
    )
    lower = artifact_html.lower()
    at = lower.find("<body")
    if at != -1:
        gt = artifact_html.find(">", at)
        if gt != -1:
            return artifact_html[: gt + 1] + injection + artifact_html[gt + 1:]
    return injection + artifact_html
