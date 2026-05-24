"""CLI segundo cérebro (Typer)."""

from __future__ import annotations

import asyncio
from typing import Any

import typer
from pydantic import ValidationError
from rich.console import Console

from secondbrain.config import Settings
from secondbrain.ingestion.indexer import index_vault
from secondbrain.logging_config import configure_structlog
from secondbrain.models import AskRequest, SearchFilters, SearchRequest
from secondbrain.rag.pipeline import answer_question
from secondbrain.retrieval.retriever import semantic_search_service


def _print_json_pretty(payload: dict[str, Any]) -> None:
    """JSON indentado + realce no terminal (alternativa pip ao `jq` para leitura humana)."""
    Console().print_json(data=payload, indent=2)


app_cli = typer.Typer(no_args_is_help=True, help="Obsidian vault → segundo cérebro.")


@app_cli.callback()
def _configure_logs() -> None:
    configure_structlog()


@app_cli.command("index")
def index_command() -> None:
    settings = Settings()
    summary = asyncio.run(index_vault(settings))
    typer.echo(
        "\n".join(
            [
                "Indexação concluída.",
                f"total arquivos: {summary.files_total}",
                f"inalterados (skip): {summary.skipped_unchanged}",
                f"arquivos atualizados: {summary.files_indexed}",
                f"chunks gravados nesta corrida: {summary.chunks_written}",
            ],
        ),
    )


@app_cli.command("search")
def search_command(
    query: str,
    top_k: int = typer.Option(10),
    tag: str | None = typer.Option(None),
    path_prefix: str | None = typer.Option(None, help='Prefixo POSIX (ex.: "notas/projeto/")'),
    json_out: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Saída JSON indentada no terminal (via rich; não precisa do executável jq).",
    ),
) -> None:
    settings = Settings()
    filt: SearchFilters | None = None
    if tag or path_prefix:
        filt = SearchFilters(tag=tag, path_prefix=path_prefix)
    req = SearchRequest(query=query, top_k=top_k, filters=filt)
    hits = asyncio.run(semantic_search_service(settings, req))
    if json_out:
        _print_json_pretty({"hits": [h.model_dump() for h in hits]})
        return
    typer.echo(f"{len(hits)} hits")
    typer.echo("")
    for hit in hits:
        path = str((hit.metadata or {}).get("source_path", ""))
        score_txt = str(round(hit.score, 4))
        typer.echo(f"[{score_txt}] {path}")
        excerpt = (hit.text or "").replace("\n", " ")[:260]
        typer.echo(f"  {excerpt}...")
        typer.echo("")


@app_cli.command("ask")
def ask_command(
    query: list[str] = typer.Argument(
        ...,
        help="Pergunta; várias palavras sem aspas são juntadas (ex.: secondbrain ask onde falo de IA).",
    ),
    top_k: int = typer.Option(8, "--top-k", "-k"),
    max_context_chars: int = typer.Option(12_000, "--max-context-chars", "-c"),
    json_out: bool = typer.Option(
        False,
        "--json",
        "-j",
        help='Resposta como JSON (campos "answer" e "sources"); senão imprime texto legível.',
    ),
) -> None:
    """RAG: recupera trechos do vault e responde com o LLM configurado no .env (sem usar HTTP / curl)."""
    q = " ".join(query).strip()
    if not q:
        raise typer.BadParameter("Escreve uma pergunta após «ask».")
    settings = Settings()
    try:
        req = AskRequest(query=q, top_k=top_k, max_context_chars=max_context_chars)
    except ValidationError as exc:
        raise typer.BadParameter("; ".join(f"{e['loc']}: {e['msg']}" for e in exc.errors())) from exc
    out = asyncio.run(answer_question(settings, req))
    if json_out:
        _print_json_pretty(out.model_dump())
        return
    typer.echo(out.answer.strip())
    typer.echo("")
    typer.secho("Fontes (trechos usados):", dim=True)
    if not out.sources:
        typer.echo("  (nenhuma citação registada)")
        return
    for cite in out.sources:
        h = cite.heading_path.strip() if cite.heading_path else ""
        suffix = f" · {h}" if h else ""
        typer.echo(f"  • {cite.path}{suffix}")


def app_main() -> None:
    app_cli()


if __name__ == "__main__":
    app_cli()
