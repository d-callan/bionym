"""bionym CLI: `bionym resolve <id>` -> graph.json"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

from .ask import ask_graph
from .clients.ncbi import NcbiClient
from .clients.veupathdb import VEuPathDBClient
from .graph import KnowledgeGraph
from .jev import JevClient, JevError
from .llm import LlmClient, LlmError
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
    propose: bool = typer.Option(True, "--propose/--no-propose", help="LLM proposal passes (claims). --no-propose = quick scan."),
    summarize: bool = typer.Option(False, "--summarize", help="Also write a JEV-verified gene summary into metadata."),
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
    graph = resolver.resolve(
        identifier, depth=depth, summarize=summarize, propose=propose
    )
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


@app.command()
def report(
    graph_json: Path = typer.Argument(
        ..., help="Existing graph JSON from `bionym resolve`."
    ),
    out: Optional[Path] = typer.Option(
        None, "-o", "--out", help="Output path (default: input with .html)."
    ),
) -> None:
    """Render a self-contained HTML report from an existing graph JSON."""
    from .report import write_report

    graph = KnowledgeGraph.from_json(graph_json)
    dest = out or graph_json.with_suffix(".html")
    write_report(graph, dest)
    typer.echo(f"wrote {dest}")


@app.command()
def summarize(
    graph_json: Path = typer.Argument(
        ..., help="Existing graph JSON from `bionym resolve`."
    ),
    out: Optional[Path] = typer.Option(
        None, "-o", "--out", help="Output path (default: overwrite input)."
    ),
    mock_jev: bool = typer.Option(False, "--mock-jev", help="Offline dev: no API key needed."),
    mock_llm: bool = typer.Option(False, "--mock-llm", help="Offline dev: canned LLM proposals."),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Run the LLM+JEV summary pass over an existing graph JSON."""
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

    resolver = Resolver(jev=jev, llm=LlmClient(mock=mock_llm, cache_dir=cache))
    graph = KnowledgeGraph.from_json(graph_json)
    resolver.summarize(graph)  # match derived from the graph itself

    summary = graph.metadata.get("summary") or []
    for s in summary:
        typer.echo(f"[{s['confidence']:.2f}] {s['claim']}")
    if not summary:
        typer.echo("no summary claims produced")
    dest = out or graph_json
    graph.to_json(dest)
    typer.echo(f"wrote {dest}")


@app.command()
def ask(
    graph_json: Path = typer.Argument(
        ..., help="Existing graph JSON from `bionym resolve`."
    ),
    question: str = typer.Argument(..., help="Free-form question over the graph."),
    mock_jev: bool = typer.Option(False, "--mock-jev", help="Offline dev: no API key needed."),
    mock_llm: bool = typer.Option(False, "--mock-llm", help="Offline dev: canned LLM answer."),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Ask a free-form question over an existing graph JSON.

    JEV classifies the question into graph categories; the LLM answers
    over just that slice and JEV scores accuracy/completeness. Questions
    outside the categories are answered unscored with a warning.
    """
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

    graph = KnowledgeGraph.from_json(graph_json)
    try:
        res = ask_graph(
            jev, LlmClient(mock=mock_llm, cache_dir=cache), question, graph
        )
    except (JevError, LlmError) as e:
        typer.secho(str(e), err=True, fg=typer.colors.RED)
        raise typer.Exit(2)

    if res["categories"]:
        typer.echo(f"categories: {', '.join(res['categories'])}")
    typer.echo(res["answer"])
    if res["cited_nodes"]:
        cites = ", ".join(str(c) for c in res["cited_nodes"])
        typer.echo(f"cites: {cites}")
    if res["invalid_cited_nodes"]:
        typer.secho(
            "not in graph: " + ", ".join(str(c) for c in res["invalid_cited_nodes"]),
            fg=typer.colors.RED,
        )
    if res["scores"]:
        scores = ", ".join(
            f"{k} {v:.2f}" for k, v in res["scores"].items() if v is not None
        )
        typer.echo(f"scores: {scores}")
    if res["warning"]:
        typer.secho(f"warning: {res['warning']}", fg=typer.colors.YELLOW)


if __name__ == "__main__":
    app()
