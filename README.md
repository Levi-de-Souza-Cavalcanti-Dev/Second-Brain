# Second Brain

Semantic index and **RAG** over an **Obsidian** Markdown vault (YAML frontmatter, `#` and YAML tags, `[[wikilinks]]`, paths relative to the vault).

Core stack: Python 3.11+, **Typer**, **Rich** (indented/highlighted JSON in the terminal), **ChromaDB** (on-disk persistence via `VECTORSTORE_PATH`), **httpx** (Ollama + OpenAI-compatible), **python-frontmatter**, **pathspec** (negated globs like `.gitignore`).

## Project choices

- **Packages**: single **`requirements.txt`** (runtime + tests + `sentence-transformers` in the same list; the last is only used if you configure that provider).
- **Vector DB**: **Chroma** `PersistentClient` at `VECTORSTORE_PATH` (collection `secondbrain_notes`).
- **Embeddings via HTTP**: calls `POST ${OLLAMA_HOST}/api/embed` (current API; `/api/embeddings` is legacy and uses a different contract).
- **Embedding fallback**: `sentence-transformers` is already in `requirements.txt`; set `EMBEDDING_PROVIDER=sentence_transformers` when you want it.
- **`file_hash`**: `SHA-256` of the whole Unicode file (including frontmatter), after normalizing line endings `\r`/`\r\n → \n` and `rstrip` per line.
- **`chunk_id`**: `SHA-256` hex of `{source_path}\0{heading_path|__root__}\0{ordinal}` (monotonic ordinal per file under the current ordering).

## Quick requirements

- Python ≥ 3.11
- Ollama service (local or remote HTTP) when `EMBEDDING_PROVIDER=ollama` or `CHAT_PROVIDER=ollama`

### Suggested local models (`ollama pull …`)

| Use case      | Example model      |
|---------------|--------------------|
| Embedding     | `nomic-embed-text` |
| Chat / RAG    | `llama3.2` or `mistral` |

## How to run

```bash
cd /path/to/repo
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .

cp .env.example .env   # set OBSIDIAN_VAULT_PATH and other vars
secondbrain index
secondbrain ask Which notes mention neural networks?
```

With **uv**: `uv venv .venv`, activate, then `uv pip install -r requirements.txt`, then `uv pip install -e .`.

## Useful CLI

| Command                        | Purpose |
|--------------------------------|---------|
| `secondbrain index`            | Incremental reindex (manifest + hashing; removes stale chunks). |
| `secondbrain search "…"`       | Semantic search over the vector store + optional filters. |
| `secondbrain search "…" --json` | Same search as indented JSON (`-j`; **Rich**). |
| `secondbrain ask …question…`   | **RAG**: asks the LLM with note context (uses `.env`). Before retrieval, runs incremental indexing if the vault changed. |

Quotes optional: `secondbrain ask Which notes mention neural networks?`  
Optional flags on `ask`: `-k/--top-k`, `-c/--max-context-chars`, `-j/--json`.

Useful flags on `secondbrain search`: `--top-k`, `--tag foo`, `--path-prefix notes/area/`, `--json`/`-j`.

On **CPU**, the first `ask` can **take several minutes** (embedding + model load + generation). Optional in `.env`: **`OLLAMA_CHAT_TIMEOUT_SECONDS`** (default 900), **`OLLAMA_EMBED_TIMEOUT_SECONDS`** (default 300), **`OPENAI_COMPAT_TIMEOUT_SECONDS`** (OpenAI-compatible chat).

## Git and vault in the same repo

**`.gitignore`** skips Obsidian metadata (`**/.obsidian/`, `.trash`, `*.excalidraw.md`, etc.) and typical vault folders at the repo root (`vault/`, `obsidian/`, `Obsidian Vault/`). If you use another folder name, **add it to `.gitignore`** so notes are not committed by mistake.

## Observability / errors

- Logging with **structlog** and plain text (`rag.*` during `ask`).
- Embedding or chat failures are raised as clear exceptions in the CLI.
- Secrets only via environment (e.g. local `.env` ignored by git); use `.env.example` as a template.

## Tests

```bash
pytest -q
```

Coverage includes, among other things: parsers/chunkers/hashing, Ollama embedding client (HTTP mock), retrieval with ephemeral Chroma + synthetic embedder, and CLI **`search`/`ask`** (`typer.testing`).

## Optional extra layers

- `LexicalRetriever` already has a **`NoopLexicalRetriever`** implementation for a future swap (`rank_bm25`/FTS/etc.).

---

Pull requests welcome: keep changes incremental (avoid touching unrelated modules when adding features).
