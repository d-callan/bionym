"""Smoke tests for the self-contained HTML report.

These guard the two ways report generation silently breaks:
  1. template/asset plumbing — render must embed the bundle, the graph
     JSON, and no un-replaced tokens;
  2. stale assets — _static/components.iife.js is a build artifact of
     web/components/src; if sources are newer than the bundle, someone
     forgot `npm run build:report`.
"""

import json
import os
import time
from pathlib import Path

from bionym.graph import KnowledgeGraph
from bionym.report import render_html

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "src" / "bionym" / "_static"
COMPONENTS_SRC = ROOT / "web" / "components" / "src"

# A label crafted to probe the HTML/JS escaping: quotes, tags, and the
# script-breakout sequence must never reach the page raw.
_EVIL = 'x<"\'</script><script>alert(1)</script>'


def _graph() -> KnowledgeGraph:
    return KnowledgeGraph(
        nodes={
            "a": {"id": "a", "type": "Gene", "label": _EVIL},
            "b": {"id": "b", "type": "GOTerm"},
        },
        edges=[
            {"subject": "a", "predicate": "has_function", "object": "b",
             "confidence": 0.9,
             "evidence": [{"source": "test", "endpoint": "mem://x"}]},
        ],
        metadata={"query": {"input": _EVIL}},
    )


def test_render_embeds_everything():
    html = render_html(_graph())
    # no un-replaced tokens
    assert "__TITLE__" not in html and "__DATA__" not in html
    assert "__JS__" not in html and "__CSS__" not in html
    # bundle + driver + graph are inlined
    assert "BionymComponents" in html
    assert "window.__GRAPH__" in html
    assert "BionymNetwork" in html
    # and it is actually self-contained: no external script/link/src
    assert '<script src=' not in html and '<link' not in html


def test_evil_strings_cannot_break_the_page():
    html = render_html(_graph())
    # the raw payload must never reach the page — </ must be <\/-escaped
    # in the embedded JSON and HTML-escaped in the <title>/header
    assert _EVIL not in html
    assert "</script><script>alert(1)</script>" not in html


def test_static_assets_not_stale():
    """_static bundle must not be older than web/components sources.

    Only meaningful inside a dev checkout — skips if the source tree
    isn't present (e.g. tests run from an sdist), or under CI, where a
    fresh checkout restores all files at near-identical mtimes. CI
    verifies the bundle by rebuilding and diffing instead (ci.yml ui job).
    """
    if os.environ.get("CI") or not COMPONENTS_SRC.is_dir():
        return
    srcs = list(COMPONENTS_SRC.glob("*.js")) + list(COMPONENTS_SRC.glob("*.css"))
    newest_src = max(f.stat().st_mtime for f in srcs)
    bundle = STATIC / "components.iife.js"
    age = time.time() - bundle.stat().st_mtime
    assert bundle.stat().st_mtime > newest_src, (
        f"components.iife.js is stale (built {age / 60:.0f} min ago). "
        "Rebuild with: cd web/components && npm run build:report"
    )
