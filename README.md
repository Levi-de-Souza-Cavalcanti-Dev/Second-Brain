# Second Brain

Semantic index and **RAG** over an **Obsidian** Markdown vault (YAML frontmatter, `#` and YAML tags, `[[wikilinks]]`, paths relative to the vault).

Core stack: Python 3.11+, **Typer**, **Rich**, **ChromaDB**, **httpx** (Ollama + OpenAI-compatible), **rank-bm25** (hybrid search), **watchfiles** (watch mode).

## Quick start

```bash
cd /path/to/repo
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"

cp .env.example .env   # OBSIDIAN_VAULT_PATH must be an existing directory
secondbrain doctor
secondbrain index
secondbrain ask Which notes mention neural networks?
```

With **uv**: `uv venv .venv && source .venv/bin/activate && uv pip install -e ".[dev]"`.

## CLI commands

| Command | Purpose |
|---------|---------|
| `secondbrain index` | Incremental reindex with progress bar |
| `secondbrain search "…"` | Semantic search (+ optional `--tag`, `--path-prefix`, `--json`) |
| `secondbrain ask …` | RAG with auto-index; `--stream/-s` for token streaming |
| `secondbrain doctor` | Health check (vault, vectorstore, Ollama models) |
| `secondbrain stats` | Index stats as JSON |
| `secondbrain watch` | Watch vault and re-index on changes |
| `secondbrain chat` | Conversational REPL (`--persist` saves session) |
| `secondbrain serve` | HTTP API on `:8765` (requires `pip install -e ".[api]"`) |
| `secondbrain version` | Package version |

## Optional features (`.env`)

- **Hybrid search**: `HYBRID_SEARCH=true` (BM25 + dense via RRF)
- **Reranker**: `RERANKER_ENABLED=true` (requires `pip install -e ".[embeddings-st]"`)
- **MMR diversification**: `MMR_LAMBDA=0.5`
- **Token chunking**: `CHUNK_BY_TOKENS=true` (optional `pip install -e ".[tokenizer]"`)
- **OpenAI embeddings**: `EMBEDDING_PROVIDER=openai`
- **Wikilink expansion in RAG**: `RAG_LINK_EXPANSION=true`
- **OpenTelemetry**: `OTEL_EXPORTER_OTLP_ENDPOINT=…` (requires `pip install -e ".[otel]"`)

## Development

```bash
ruff check src tests && ruff format --check src tests
mypy src/secondbrain
pytest -q
pre-commit install   # optional hooks
```

CI runs on Python 3.11 and 3.12 via GitHub Actions.

## Project choices

- Single vault via `OBSIDIAN_VAULT_PATH` (validated at startup)
- Chroma collection `secondbrain_notes` at `VECTORSTORE_PATH`
- Manifest v1 + `manifest_meta.json` for fast incremental indexing
- Deterministic `chunk_id` and whole-file SHA-256 hashing
