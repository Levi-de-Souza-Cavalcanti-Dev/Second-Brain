# Second Brain

Semantic index and **RAG** over an **Obsidian** Markdown vault (YAML frontmatter, `#` and YAML tags, `[[wikilinks]]`, paths relative to the vault).

Core stack: Python 3.11+, **FastAPI**, **Typer**, **Rich** (indented/highlighted JSON in the terminal, a `pip`-only alternative to `jq`), **ChromaDB** (on-disk persistence via `VECTORSTORE_PATH`), **httpx** (Ollama + OpenAI-compatible), **python-frontmatter**, **pathspec** (negated globs like `.gitignore`).

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
secondbrain serve
```

With **uv**: `uv venv .venv`, activate, then `uv pip install -r requirements.txt`, then `uv pip install -e .`.

## Useful CLI

| Command                        | Purpose |
|--------------------------------|---------|
| `secondbrain index`            | Incremental reindex (manifest + hashing; removes stale chunks). |
| `secondbrain search "…"`       | Semantic search over the vector store + optional filters. |
| `secondbrain search "…" --json` | Same search as indented JSON (`-j`; **Rich**, no `jq`). |
| `secondbrain ask …question…`   | **RAG**: asks the LLM with note context (uses `.env`; no `curl`). Before retrieval, runs incremental indexing if the vault changed. |
| `secondbrain serve`            | FastAPI/Uvicorn on `API_HOST`:`API_PORT`. |

Quotes optional: `secondbrain ask Which notes mention neural networks?`  
Optional flags on `ask`: `-k/--top-k`, `-c/--max-context-chars`, `-j/--json`.

Useful flags on `secondbrain search`: `--top-k`, `--tag foo`, `--path-prefix notes/area/`, `--json`/`-j`.

## API & `curl` examples

| Route          | Purpose |
|----------------|---------|
| `GET /health`  | Basic metadata. |
| `POST /search` | Semantic hits with score + metadata. |
| `POST /ask`    | RAG pipeline (retrieval + configurable LLM). |
| `POST /reindex`| Triggers async reindex (simple FastAPI `BackgroundTask`). |

Search (`POST /search`):

- Compact JSON or **`"pretty": true`** in the body; **`?pretty=true`** forces indentation even without that field. **`X-SecondBrain-Json-Pretty: true`** does the same. Indented responses include **`X-SecondBrain-Pretty: 1`** (handy with `curl -v`).
- With **`curl`**, `-d '...'` alone usually sends `application/x-www-form-urlencoded`, not JSON — the server may ignore `"pretty"` (and even the full body). Use **`-H 'content-type: application/json'`** or **`curl --json '{...}'`** (curl ≥ 7.82).
- In the shell, **`| jq .`** also pretty-prints JSON (`pacman -S jq` on Arch).
- On startup the server logs **`secondbrain.api.loaded`** with the path to `main.py`; if JSON stays compact even with `"pretty": true`, run **`pip install -e .`**, restart `serve`, and confirm with **`curl -v`** that **`X-SecondBrain-Pretty: 1`** appears on the response.

```bash
curl -sf http://127.0.0.1:8000/search \
  -H 'content-type: application/json' \
  -d '{"query":"stacks in Python","top_k":5,"pretty":true}'
```

Query string only (pretty without a field in the JSON):

```bash
curl -sf 'http://127.0.0.1:8000/search?pretty=true' \
  -H 'content-type: application/json' \
  -d '{"query":"stacks in Python","top_k":5}'
```

Same request with optional filters and `jq`:

```bash
curl -sf http://127.0.0.1:8000/search \
  -H 'content-type: application/json' \
  -d '{"query":"stacks in Python","top_k":5,"filters":{"tag":"study","path_prefix":"notes/lang/"}}' \
  | jq .
```

RAG (`POST /ask`):

- On **CPU**, the first request can **take several minutes** (embedding + model load + generation). Watch the **`secondbrain serve`** terminal: `rag.ask.start` → `retrieval_done` → `chat_request` → `done`. While it shows `chat_request`, the model is busy in Ollama.
- Optional in `.env`: **`OLLAMA_CHAT_TIMEOUT_SECONDS`** (default 900), **`OLLAMA_EMBED_TIMEOUT_SECONDS`** (default 300), **`OPENAI_COMPAT_TIMEOUT_SECONDS`** (OpenAI-compatible chat).
- In `curl`, raise the client limit: **`curl --max-time 960 ...`**.

```bash
curl -sf --max-time 960 http://127.0.0.1:8000/ask \
  -H 'content-type: application/json' \
  -H 'x-secondbrain-json-pretty: true' \
  -d '{"query":"Summarize ideas about RAG from my notes","top_k":8,"max_context_chars":12000,"pretty":true}'
```

Or: `... | jq .` without using `pretty`.

To consume API JSON **from Python** (also via pip, like `jq` in the shell):

```python
import httpx
from rich.console import Console

# `/search`: a short timeout is usually enough; for `/ask` on CPU use a high timeout (e.g. 960).
resp = httpx.post(
    "http://127.0.0.1:8000/search",
    json={"query": "neural networks", "top_k": 5},
    timeout=120.0,
)
Console().print_json(resp.text)
```

Health:

```bash
curl -sf http://127.0.0.1:8000/health | jq .
```

To explore notes **without hand-building JSON**: **`secondbrain search "…"`** (snippets + paths) or **`secondbrain ask …`** (terminal RAG, same engine as `POST /ask`).

## Git and vault in the same repo

**`.gitignore`** skips Obsidian metadata (`**/.obsidian/`, `.trash`, `*.excalidraw.md`, etc.) and typical vault folders at the repo root (`vault/`, `obsidian/`, `Obsidian Vault/`). If you use another folder name, **add it to `.gitignore`** so notes are not committed by mistake.

## Observability / errors

- Logging with **structlog** and plain text (`rag.*` during `POST /ask`; `secondbrain.api.loaded` at startup with the path to `main.py`).
- Embedding or chat failures surface as **HTTP 502** with text details.
- Secrets only via environment (e.g. local `.env` ignored by git); use `.env.example` as a template.

## Tests

```bash
pytest -q
```

Coverage includes, among other things: parsers/chunkers/hashing, Ollama embedding client (HTTP mock), retrieval with ephemeral Chroma + synthetic embedder, `/search`/`/ask` JSON shape, and CLI **`search`/`ask`** (`typer.testing`).

## Optional extra layers

- `LexicalRetriever` already has a **`NoopLexicalRetriever`** implementation for a future swap (`rank_bm25`/FTS/etc.).

---

Pull requests welcome: keep changes incremental (avoid touching unrelated modules when adding features).
