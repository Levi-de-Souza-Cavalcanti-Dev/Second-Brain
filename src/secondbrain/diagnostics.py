"""Vault statistics and health diagnostics."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import httpx

from secondbrain.config import Settings
from secondbrain.ingestion.manifest import load_manifest, manifest_path
from secondbrain.ingestion.types import embed_model_fingerprint
from secondbrain.ingestion.vault_scanner import vault_markdown_paths


@dataclass(slots=True)
class VaultStats:
    files_in_vault: int
    files_in_manifest: int
    embed_model: str
    embed_dim: int
    vectorstore_bytes: int


@dataclass(slots=True)
class DoctorReport:
    vault_ok: bool
    vectorstore_ok: bool
    ollama_ok: bool | None
    embed_model_available: bool | None
    chat_model_available: bool | None
    manifest_embed_dim: int
    issues: list[str]


def collect_stats(settings: Settings) -> VaultStats:
    vs = Path(settings.vectorstore_path).expanduser().resolve()
    vr = Path(settings.obsidian_vault_path).expanduser().resolve()
    manifest = load_manifest(manifest_path(vs))
    files = vault_markdown_paths(vr, settings.ignore_globs)
    store_bytes = sum(f.stat().st_size for f in vs.rglob("*") if f.is_file())
    return VaultStats(
        files_in_vault=len(files),
        files_in_manifest=len(manifest.entries),
        embed_model=manifest.embed_model or embed_model_fingerprint(settings),
        embed_dim=manifest.embed_dim,
        vectorstore_bytes=store_bytes,
    )


async def run_doctor(settings: Settings) -> DoctorReport:
    issues: list[str] = []
    vr = Path(settings.obsidian_vault_path)
    vs = Path(settings.vectorstore_path)
    vault_ok = vr.is_dir()
    if not vault_ok:
        issues.append(f"Vault inválido: {vr}")

    vectorstore_ok = vs.is_dir()
    if not vectorstore_ok:
        issues.append(f"Vectorstore inválido: {vs}")

    manifest = load_manifest(manifest_path(vs))
    ollama_ok: bool | None = None
    embed_model_available: bool | None = None
    chat_model_available: bool | None = None

    if settings.embedding_provider == "ollama" or settings.chat_provider == "ollama":
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{settings.ollama_host.rstrip('/')}/api/tags")
                resp.raise_for_status()
                ollama_ok = True
                body = resp.json()
                models = {
                    m.get("name", "").split(":")[0]
                    for m in body.get("models", [])
                    if isinstance(m, dict)
                }
                if settings.embedding_provider == "ollama":
                    embed_model_available = settings.ollama_embed_model.split(":")[0] in models
                    if not embed_model_available:
                        issues.append(
                            f"Modelo de embedding não encontrado no Ollama: {settings.ollama_embed_model}",
                        )
                if settings.chat_provider == "ollama":
                    chat_model_available = settings.ollama_chat_model.split(":")[0] in models
                    if not chat_model_available:
                        issues.append(
                            f"Modelo de chat não encontrado no Ollama: {settings.ollama_chat_model}",
                        )
        except Exception as exc:
            ollama_ok = False
            issues.append(f"Ollama inacessível em {settings.ollama_host}: {exc}")

    return DoctorReport(
        vault_ok=vault_ok,
        vectorstore_ok=vectorstore_ok,
        ollama_ok=ollama_ok,
        embed_model_available=embed_model_available,
        chat_model_available=chat_model_available,
        manifest_embed_dim=manifest.embed_dim,
        issues=issues,
    )


def doctor_sync(settings: Settings) -> DoctorReport:
    return asyncio.run(run_doctor(settings))
