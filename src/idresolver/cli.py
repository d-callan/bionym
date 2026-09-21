"""idresolver CLI: `idresolver resolve <id>` -> graph.json"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

from .clients.ncbi import NcbiClient
from .clients.veupathdb import VEuPathDBClient
from .jev import JevClient, JevError
from .resolver import Resolver

app = typer.Typer(
    help="Resolve bioinformatics identifiers into confidence-scored knowledge graphs.",
    no_args_is_help=True,
)


def _cache_dir() -> str | None:
    return os.environ.get("IDRESOLVER_CACHE_DIR") or None


@app.command()
def version() -> None:
    from . import __version__

    typer.echo(__version__)


@app.command()
def resolve(
    identifier: str = typer.Argument(..., help="Identifier to resolve (v1: gene IDs)."),
    out: Path = typer.Option(Path("graph.json"), "-o", "--out", help="Output JSON path."),
    depth: int = typer.Option(3, "--depth", help="Stages: 0=classify, 1=+resolve, 2=+assemblies, 3=+orthologs."),
    max_candidates: int = typer.Option(20, "--max-candidates"),
    mock_jev: bool = typer.Option(False, "--mock-jev", help="Offline dev: no API key needed."),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cache = _cache_dir()
    try:
        jev = JevClient(mock=mock_jev, cache_dir=cache)
    except JevError as e:
        typer.secho(str(e), err=True, fg=typer.colors.RED)
        raise typer.Exit(2)

    resolver = Resolver(
        jev=jev,
        ncbi=NcbiClient(cache_dir=cache),
        veupathdb=VEuPathDBClient(cache_dir=cache),
        max_candidates=max_candidates,
    )
    graph = resolver.resolve(identifier, depth=depth)
    graph.to_json(out)

    usage = graph.metadata.get("jev_usage", {}).get("total", {})
    typer.echo(
        f"wrote {out}: {len(graph.nodes)} nodes, {len(graph.edges)} edges, "
        f"jev tokens in={usage.get('input_tokens', 0)} out={usage.get('output_tokens', 0)}"
    )


if __name__ == "__main__":
    app()
