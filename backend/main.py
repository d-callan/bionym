"""FastAPI backend for bionym — thin wrapper over the installed package.

Run:  uvicorn main:app --reload --port 8000   (from backend/, with the
      bionym conda env active and .env loaded or vars exported)

The static frontend in ../web calls GET /api/resolve.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from bionym.clients.ncbi import NcbiClient
from bionym.clients.veupathdb import VEuPathDBClient
from bionym.jev import JevClient
from bionym.resolver import Resolver

load_dotenv()

app = FastAPI(title="bionym", version="0.1.0")

# Static frontend is served from a different origin (file://, GH Pages, or
# a separate port) — allow all origins for local dev; tighten on deploy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

_cache_dir = os.environ.get("BIONYM_CACHE_DIR") or None


def _resolver(mock_jev: bool = False) -> Resolver:
    return Resolver(
        jev=JevClient(mock=mock_jev, cache_dir=_cache_dir),
        ncbi=NcbiClient(cache_dir=_cache_dir),
        veupathdb=VEuPathDBClient(cache_dir=_cache_dir),
    )


@app.get("/api/resolve")
def resolve(
    identifier: str = Query(..., min_length=1),
    depth: int = Query(6, ge=0, le=6),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    mock_jev: bool = Query(False),
):
    if not mock_jev and not os.environ.get("TYPESAFE_API_KEY"):
        raise HTTPException(500, "TYPESAFE_API_KEY not configured on server")
    resolver = _resolver(mock_jev=mock_jev)
    graph = resolver.resolve(identifier, depth=depth)
    graph.filter_by_confidence(min_confidence)
    return graph.model_dump(mode="json")


@app.get("/api/health")
def health():
    return {"ok": True, "jev_configured": bool(os.environ.get("TYPESAFE_API_KEY"))}
