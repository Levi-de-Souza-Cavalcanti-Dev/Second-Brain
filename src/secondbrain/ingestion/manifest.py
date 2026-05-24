"""Versioned vault manifest and mtime/size sidecar for incremental indexing."""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass, field

MANIFEST_VERSION = 1


@dataclass(slots=True)
class VaultManifest:
    """On-disk manifest (v1); v0 flat dict migrates on load."""

    entries: dict[str, str] = field(default_factory=dict)
    version: int = MANIFEST_VERSION
    embed_model: str = ""
    embed_dim: int = 0

    def to_json_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "embed_model": self.embed_model,
            "embed_dim": self.embed_dim,
            "entries": dict(sorted(self.entries.items())),
        }


@dataclass(frozen=True, slots=True)
class FileStatMeta:
    mtime_ns: int
    size: int

    def to_json(self) -> dict[str, int]:
        return {"mtime_ns": self.mtime_ns, "size": self.size}

    @classmethod
    def from_json(cls, raw: object) -> FileStatMeta | None:
        if not isinstance(raw, dict):
            return None
        try:
            return cls(mtime_ns=int(raw["mtime_ns"]), size=int(raw["size"]))
        except (KeyError, TypeError, ValueError):
            return None


def manifest_path(vectorstore_root: pathlib.Path) -> pathlib.Path:
    return vectorstore_root / "manifest.json"


def manifest_meta_path(vectorstore_root: pathlib.Path) -> pathlib.Path:
    return vectorstore_root / "manifest_meta.json"


def load_manifest(path: pathlib.Path) -> VaultManifest:
    if not path.is_file():
        return VaultManifest()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return VaultManifest()

    if "version" in data and "entries" in data:
        entries_raw = data.get("entries")
        entries: dict[str, str] = {}
        if isinstance(entries_raw, dict):
            for k, v in entries_raw.items():
                if isinstance(k, str) and isinstance(v, str):
                    entries[k] = v
        embed_model = str(data.get("embed_model") or "")
        try:
            embed_dim = int(data.get("embed_dim") or 0)
        except (TypeError, ValueError):
            embed_dim = 0
        return VaultManifest(
            entries=entries,
            version=int(data.get("version") or MANIFEST_VERSION),
            embed_model=embed_model,
            embed_dim=embed_dim,
        )

    # v0: flat path -> hash
    entries_v0: dict[str, str] = {}
    for k, v in data.items():
        if isinstance(k, str) and isinstance(v, str):
            entries_v0[k] = v
    return VaultManifest(entries=entries_v0, version=0, embed_model="", embed_dim=0)


def save_manifest(path: pathlib.Path, manifest: VaultManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if manifest.version < MANIFEST_VERSION:
        manifest = VaultManifest(
            entries=manifest.entries,
            version=MANIFEST_VERSION,
            embed_model=manifest.embed_model,
            embed_dim=manifest.embed_dim,
        )
    payload = json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def load_manifest_meta(path: pathlib.Path) -> dict[str, FileStatMeta]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    out: dict[str, FileStatMeta] = {}
    for k, v in data.items():
        if not isinstance(k, str):
            continue
        meta = FileStatMeta.from_json(v)
        if meta is not None:
            out[k] = meta
    return out


def save_manifest_meta(path: pathlib.Path, meta: dict[str, FileStatMeta]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        k: meta[k].to_json()
        for k in sorted(meta.keys())
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def file_stat_meta(path: pathlib.Path) -> FileStatMeta:
    st = path.stat()
    return FileStatMeta(mtime_ns=st.st_mtime_ns, size=st.st_size)
