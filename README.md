# secondbrain

Índice semântico e RAG sobre o vault Markdown do **Obsidian** (YAML frontmatter, tags `#` e YAML, wikilinks `[[ligações]]`, caminho relativo ao vault).

Stack principal: Python 3.11+, **FastAPI**, **Typer**, **Rich** (JSON indentado/realçado no terminal, alternativa ao `jq` só com pip), **ChromaDB** (persistência em disco via `VECTORSTORE_PATH`), **httpx** (Ollama + OpenAI-compat), **python-frontmatter**, **pathspec** (glob negado estilo `.gitignore`).

## Escolhas de projeto

- **Pacotes**: único **`requirements.txt`** (runtime + testes + `sentence-transformers` na mesma lista; a última só é usada se configurar o provider).
- **Vector DB**: **Chroma** `PersistentClient` em `VECTORSTORE_PATH` (coleção `secondbrain_notes`).
- **Embeddings obrigatórios via HTTP**: chamadas `POST ${OLLAMA_HOST}/api/embed` (API atual; a rota `/api/embeddings` é legada e usa outro contrato).
- **Fallback embeddings**: `sentence-transformers` já está em `requirements.txt`; use `EMBEDDING_PROVIDER=sentence_transformers` quando quiser.
- **`file_hash`**: `SHA-256` Unicode do arquivo inteiro (inclui frontmatter), após normalizar quebras `\r`/`\r\n → \n` e `rstrip` por linha.
- **`chunk_id`**: `SHA-256(hex)` de `{source_path}\0{heading_path|__root__}\0ordinal}` (ordinal monotônico por arquivo conforme ordenação atual).

## Dependências rápidas

- Python ≥ 3.11
- Serviço Ollama (local ou remoto HTTP) quando `EMBEDDING_PROVIDER=ollama` ou `CHAT_PROVIDER=ollama`

### Sugestões de modelos locais (`ollama pull …`)

| Uso           | Modelo exemplo            |
|---------------|---------------------------|
| Embedding     | `nomic-embed-text`        |
| Chat / RAG    | `llama3.2` ou `mistral`   |

## Como rodar

```bash
cd /caminho/para/repo
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .

cp .env.example .env   # personalize OBSIDIAN_VAULT_PATH e demais vars
secondbrain index
secondbrain serve
```

Com **uv**: `uv venv .venv`, ative e `uv pip install -r requirements.txt`, depois `uv pip install -e .`.

## CLI útil

| Comando                             | Função                                                              |
|-------------------------------------|---------------------------------------------------------------------|
| `secondbrain index`                 | Reindex incremental (manifest + hashing; remove chunks antigos)     |
| `secondbrain search "…"`            | Semantic search usando vector store + filtros opcionais             |
| `secondbrain search "…" --json`      | Mesma busca em JSON indentado (`-j`; formatação via **Rich**, sem `jq`) |
| `secondbrain ask …pergunta…`        | **RAG**: pergunta ao LLM com contexto das notas (usa `.env`; sem `curl`) |
| `secondbrain serve`                 | FastAPI/Uvicorn em `API_HOST`:`API_PORT`                            |

Sem aspas também funciona: `secondbrain ask Em que notas falo de redes neurais?`  
Flags opcionais: `ask` aceita `-k/--top-k`, `-c/--max-context-chars`, `-j/--json`.

Flags úteis de `secondbrain search`: `--top-k`, `--tag foo`, `--path-prefix notas/area/`, `--json`/`-j`.

## API & exemplos `curl`

| Rota                      | Finalidade                                                       |
|---------------------------|------------------------------------------------------------------|
| `GET /health`             | Metadados básicos                                                |
| `POST /search`            | Hits semânticos com score + metadados                            |
| `POST /ask`               | Pipeline RAG (recuperação + LLM configurável)                     |
| `POST /reindex`           | Dispara nova indexação assíncrona (simples BackgroundTask FastAPI)|

Busca (`POST /search`):

- JSON compacto ou **`"pretty": true`** no body; **`?pretty=true`** força indent mesmo sem esse campo no JSON. **`X-SecondBrain-Json-Pretty: true`** no pedido faz o mesmo. Respostas indentadas trazem o cabeçalho **`X-SecondBrain-Pretty: 1`** (útil com `curl -v`).
- Com **`curl`**, o `-d '...'` sozinho costuma enviar `application/x-www-form-urlencoded`, não JSON — o servidor pode ignorar `"pretty"` (e até o modelo completo). Use **` -H 'content-type: application/json'`** ou **`curl --json '{...}'`** (curl ≥ 7.82).
- No terminal, **`| jq .`** também formata o JSON (`pacman -S jq` no Arch).
- Ao arranque, o servidor regista **`secondbrain.api.loaded`** com o caminho de `main.py`; se o JSON compacto persistir mesmo com `"pretty": true`, faz **`pip install -e .`**, reinicia o `serve` e confirma com **`curl -v`** o cabeçalho **`X-SecondBrain-Pretty: 1`** na resposta.

```bash
curl -sf http://127.0.0.1:8000/search \
  -H 'content-type: application/json' \
  -d '{"query":"pilhas em Python","top_k":5,"pretty":true}'
```

Ou só pela query (pretty sem campo no JSON):

```bash
curl -sf 'http://127.0.0.1:8000/search?pretty=true' \
  -H 'content-type: application/json' \
  -d '{"query":"pilhas em Python","top_k":5}'
```

Mesmo pedido com filtro opcional e `jq`:

```bash
curl -sf http://127.0.0.1:8000/search \
  -H 'content-type: application/json' \
  -d '{"query":"pilhas em Python","top_k":5,"filters":{"tag":"study","path_prefix":"notas/lang/"}}' \
  | jq .
```

RAG (`POST /ask`):

- Em **CPU**, o primeiro pedido pode **demorar vários minutos** (embedding + modelo a carregar e a gerar). Olha para o terminal do **`secondbrain serve`**: aparecem `rag.ask.start` → `retrieval_done` → `chat_request` → `done`. Enquanto estiver em `chat_request`, o modelo está no Ollama.
- Opcional no `.env`: **`OLLAMA_CHAT_TIMEOUT_SECONDS`** (padrão 900), **`OLLAMA_EMBED_TIMEOUT_SECONDS`** (padrão 300), **`OPENAI_COMPAT_TIMEOUT_SECONDS`** (chat compat OpenAI).
- No `curl`, podes aumentar limite cliente: **`curl --max-time 960 ...`**.

```bash
curl -sf --max-time 960 http://127.0.0.1:8000/ask \
  -H 'content-type: application/json' \
  -H 'x-secondbrain-json-pretty: true' \
  -d '{"query":"Resume ideias sobre RAG nas minhas notas","top_k":8,"max_context_chars":12000,"pretty":true}'
```

Ou: `... | jq .` sem usar `pretty`.

Para ver JSON da API **a partir de Python** (também com pip, como o `jq` no terminal):

```python
import httpx
from rich.console import Console

# `/search`: timeout curto costuma bastar; para `/ask` em CPU usa timeout alto (ex.: 960).
resp = httpx.post(
    "http://127.0.0.1:8000/search",
    json={"query": "neural networks", "top_k": 5},
    timeout=120.0,
)
Console().print_json(resp.text)
```

Saúde:

```bash
curl -sf http://127.0.0.1:8000/health | jq .
```

Para explorar notas **sem montar JSON à mão**: **`secondbrain search "…"`** (trechos + caminhos) ou **`secondbrain ask …`** (RAG pelo terminal, mesmo motor que `POST /ask`).

## Git e vault dentro do mesmo repositório

O **`.gitignore`** ignora metadados do Obsidian (`**/.obsidian/`, `.trash`, `*.excalidraw.md`, etc.) e pastas típicas de vault na raiz (`vault/`, `obsidian/`, `Obsidian Vault/`). Quem usar outro nome de pasta deve **acrescentá-lo ao `.gitignore`** para não comitar notas por engano.

## Observabilidade / erros

- Logging via **structlog** com renderização texto simples (`rag.*` durante `POST /ask`; `secondbrain.api.loaded` no arranque com caminho do `main.py`).
- Exceções de embedding ou chat aparecem como **HTTP 502** com detalhes textuais.
- Secrets via ambiente apenas (ex.: `.env` local ignorado pelo git); use `.env.example` como modelo.

## Testes

```bash
pytest -q
```

Cobertos entre outros: parsers/chunkers/hashing, cliente Ollama embeddings (mock HTTP), retrieval com Chroma efémero + embedder sintético, formato JSON `/search`/`/ask` e comandos **`search`/`ask`** na CLI (`typer.testing`).

## Camadas extras opcionais

- `LexicalRetriever` já tem implementação **`NoopLexicalRetriever`** pronta para troca futura (`rank_bm25`/FTS/etc.).

---

Pull requests bem-vindos: mantenha o escopo incremental (evite mexer em módulos sem necessidade ao adicionar features).
