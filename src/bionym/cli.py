"""bionym CLI: `bionym resolve <id>` -> graph.json"""

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
from .llm import LlmClient
from .resolver import Resolver

app = typer.Typer(
    help="Resolve bioinformatics identifiers into confidence-scored knowledge graphs.",
    no_args_is_help=True,
)


def _cache_dir() -> str | None:
    return os.environ.get("BIONYM_CACHE_DIR") or None


@app.command()
def version() -> None:
    from . import __version__

    typer.echo(__version__)


@app.command()
def resolve(
    identifier: str = typer.Argument(..., help="Identifier to resolve (v1: gene IDs)."),
    out: Path = typer.Option(Path("graph.json"), "-o", "--out", help="Output JSON path."),
    depth: int = typer.Option(6, "--depth", help="Stages: 0=classify, 1=+resolve, 2=+assemblies, 3=+orthologs, 4=+annotation, 5=+expression, 6=+remap."),
    min_confidence: float = typer.Option(0.0, "--min-confidence", help="Drop edges below this confidence (and orphan nodes)."),
    mock_jev: bool = typer.Option(False, "--mock-jev", help="Offline dev: no API key needed."),
    mock_llm: bool = typer.Option(False, "--mock-llm", help="Offline dev: canned LLM proposals."),
    report: bool = typer.Option(False, "--report", help="Also write an HTML report next to the JSON."),
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
        llm=LlmClient(mock=mock_llm, cache_dir=cache),
    )
    graph = resolver.resolve(identifier, depth=depth)
    graph.filter_by_confidence(min_confidence)
    graph.to_json(out)

    usage = jev.total_usage()
    typer.echo(
        f"wrote {out}: {len(graph.nodes)} nodes, {len(graph.edges)} edges, "
        f"jev tokens in={usage['input_tokens']} out={usage['output_tokens']}"
    )
    if report:
        from .report import write_report

        report_path = out.with_suffix(".html")
        write_report(graph, report_path)
        typer.echo(f"wrote {report_path}")


if __name__ == "__main__":
    app()
