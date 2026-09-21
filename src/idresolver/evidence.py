"""Evidence records: provenance for every claim in the knowledge graph."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """A single piece of evidence supporting a claim.

    Every edge in the knowledge graph must carry at least one Evidence record
    so that "why do we believe this" is always answerable.
    """

    source: str  # e.g. "ncbi_datasets", "ncbi_eutils", "veupathdb"
    endpoint: str  # URL or endpoint identifier used to retrieve it
    retrieved_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    summary: str = ""  # short human-readable description of what was retrieved
    payload: dict[str, Any] = Field(default_factory=dict)  # normalized raw record
