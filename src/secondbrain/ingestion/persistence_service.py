"""Manifest and BM25 persistence helpers."""

from __future__ import annotations

import pathlib

from secondbrain.ingestion.manifest import (
    FileStatMeta,
    VaultManifest,
    load_manifest,
    load_manifest_meta,
    manifest_meta_path,
    manifest_path,
    save_manifest,
    save_manifest_meta,
)
from secondbrain.retrieval.bm25 import BM25Index, bm25_index_path, load_bm25_index, save_bm25_index


class PersistenceService:
    def __init__(self, vectorstore_root: pathlib.Path) -> None:
        self._root = vectorstore_root
        self.manifest_path = manifest_path(vectorstore_root)
        self.meta_path = manifest_meta_path(vectorstore_root)
        self.bm25_path = bm25_index_path(vectorstore_root)

    def load_manifest(self) -> VaultManifest:
        return load_manifest(self.manifest_path)

    def load_stat_meta(self) -> dict[str, FileStatMeta]:
        return load_manifest_meta(self.meta_path)

    def save_manifest(self, manifest: VaultManifest) -> None:
        save_manifest(self.manifest_path, manifest)

    def save_stat_meta(self, meta: dict[str, FileStatMeta]) -> None:
        save_manifest_meta(self.meta_path, meta)

    def load_bm25(self) -> BM25Index:
        return load_bm25_index(self.bm25_path)

    def save_bm25(self, index: BM25Index) -> None:
        save_bm25_index(self.bm25_path, index)
