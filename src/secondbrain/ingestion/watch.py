"""Watch vault for changes and re-index incrementally."""

from __future__ import annotations

import asyncio

import structlog
from watchfiles import awatch

from secondbrain.config import Settings
from secondbrain.ingestion.indexer import index_vault, scan_vault_changes

_LOG = structlog.get_logger()


async def watch_vault(settings: Settings, *, debounce_ms: int = 500) -> None:
    vault = settings.obsidian_vault_path
    _LOG.info("watch.start", vault=vault, debounce_ms=debounce_ms)
    async for changes in awatch(vault, debounce=debounce_ms, recursive=True):
        md_changes = [p for _, p in changes if str(p).endswith(".md")]
        if not md_changes:
            continue
        _LOG.info("watch.detected", n_files=len(md_changes))
        report = await scan_vault_changes(settings)
        if report.has_changes:
            summary = await index_vault(settings, change_report=report, show_progress=True)
            _LOG.info(
                "watch.indexed",
                files_indexed=summary.files_indexed,
                chunks_written=summary.chunks_written,
            )


def run_watch(settings: Settings) -> None:
    asyncio.run(watch_vault(settings))
