# Saudi Vision 2030 · Day 2 chatbot

**Start here: [interactive Day 2 presentation](docs/index.html).** Download/open the HTML in your browser, or serve it at `/guide/`. It is a standalone file with 12 stages, keyboard navigation, adjustable chunking, cosine similarity, retrieval top-k, parameter-memory and context-budget demonstrations. Use “Show all steps” for reading or “Print / PDF” for handouts. Demonstrations are explicitly simulated; the Python app runs the real pipeline.

## Recommended provider: Groq

1. Sign up at [Groq Console](https://console.groq.com/).
2. Create your own [API key](https://console.groq.com/keys) and put it in `.env` as `GROQ_API_KEY`.
3. Use `GROQ_MODEL=openai/gpt-oss-20b`, or another chat model available to your organization. This model ID is served **by Groq**; it does not require an OpenAI account.
4. Stay on Groq's Free plan for the workshop and check your [organization limits](https://console.groq.com/settings/limits). Free use has per-model request/token limits per minute and per day, not unlimited tokens. Limits apply to the organization, so avoid a shared class key. On HTTP 429, wait for the limit to reset; repeated retries do not fix a daily cap. Paid plans are optional.

The LLM runs on Groq infrastructure; students do not host it. To avoid a separate paid embedding API, the app downloads `intfloat/multilingual-e5-small` from Hugging Face once and runs it locally on CPU. Allow download time, disk space and memory before class. It needs no Hugging Face key for this public model. After download, retrieval works offline; answers still need Groq internet access.

The E5 model uses `passage: ` for documents and `query: ` for queries, with normalized embeddings. It supports multilingual text including Arabic. Its input limit is 512 tokens; longer inputs can be truncated. Character chunking does not guarantee a token limit: inspect retrieval and reduce chunk size for unusually token-dense text. This embedding limit is separate from the LLM context window.

**Existing installations:** install the updated `requirements-simple.txt`, add the three new variables from `.env.example`, and rerun `python -m simple.ingest` to rebuild the old OpenAI vector index. The signature detects this change. Keep your existing secrets private; do not overwrite your `.env` blindly.

Sources: [Groq free-plan limits](https://console.groq.com/docs/rate-limits), [ChatGroq integration](https://docs.langchain.com/oss/python/integrations/chat/groq), [E5 model card](https://huggingface.co/intfloat/multilingual-e5-small).

## Simple LangChain teaching path

Use Python **3.11 or 3.12**. The original PDF and extracted Arabic TXT are already in `data/`.

```bash
git clone https://github.com/Yazeed-Moha/AI-Workshop-Tuwaiq-KSU-lab.git
cd AI-Workshop-Tuwaiq-KSU-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-simple.txt
cp .env.example .env
# Edit .env: set GROQ_API_KEY. Never commit the key.
python -m simple.ingest --preview
python -m simple.ingest
python -m simple.chatbot "ما هي محاور الرؤية؟" --retrieve-only
python -m simple.chatbot "ما هي محاور الرؤية؟"
python -m uvicorn simple.app:app --host 127.0.0.1 --port 8000
```

Windows PowerShell: use `py -3.12 -m venv .venv`, `.\.venv\Scripts\Activate.ps1`, and `Copy-Item .env.example .env`. If activation is blocked, use `.\.venv\Scripts\python.exe` directly instead of `python`.

Open **http://127.0.0.1:8000** for the chatbot and **http://127.0.0.1:8000/guide/** for the presentation. Each question is independent. Questions and retrieved evidence go to Groq for generation. Embeddings run locally on CPU and Chroma runs locally without a database account. No OpenAI key is needed for this path.

| Small component | Responsibility |
|---|---|
| `simple/config.py` | Paths, model names, 1000-character chunks / 250 overlap |
| `simple/ingest.py` | Exact slices → LangChain Documents → embeddings → persistent Chroma |
| `simple/chatbot.py` | Search → context → prompt / model / parser → answer with evidence |
| `simple/app.py` | FastAPI endpoints and existing plain HTML frontend |

Chroma uses **cosine distance**: smaller is closer. The frontend displays `1 - distance` as cosine similarity, never a confidence percentage. The index is in `artifacts/chroma/`, separate from the previous JSON index. Unchanged ingestion reuses the index. Changed source, chunk settings or embedding model rebuilds this dedicated collection; generation-model changes do not require re-embedding.

Optional extraction: see [data provenance and OCR instructions](data/README.md). OCR can misread text and figures. Check claims against the original PDF. The document describes targets, not verified current progress. Citation-ID validation is not factual verification.

## Check the simple implementation

```bash
python -m pip install httpx
python -m unittest tests.test_simple -v
```

These tests use real local Chroma and fake embeddings/model responses, with no API cost. Test actual Arabic retrieval and answer quality separately with your key. The local teaching app has no authentication, conversation memory or production deployment controls.

## Earlier implementation (reference)

The framework-free `rag_lab/` implementation is retained for comparison. Its original setup follows below; for Day 2 use the **simple** path above. The earlier guide is [docs/reference.html](docs/reference.html).

---

# Saudi Vision 2030 Chatbot

Ask questions in Arabic or English about the included Arabic Saudi Vision 2030 document. Answers are grounded in retrieved passages with chunk citations. This independent chatbot is not an official service and does not track current progress. Each question is independent (no conversation memory).

The repository includes the original **[`data/saudi_vision2030_ar.pdf`](data/saudi_vision2030_ar.pdf)** and the extracted **[`data/saudi_vision2030_ar.txt`](data/saudi_vision2030_ar.txt)**. The TXT is the default knowledge source; you do not need to provide another document or run extraction to start.

**Setup guide:** open [`docs/index.html`](docs/index.html). It explains the existing pipeline: static 1,000-character chunks with 250-character overlap → embeddings → retrieval → generation → frontend.

Requires Git, Python 3.10+, and an OpenAI API key. Document chunks and questions are sent to OpenAI; API calls incur usage charges. No key or generated index is committed.

## Source quality

The PDF contains 81 pages. Its embedded Arabic text has broken character mappings and reading order, so the committed TXT uses local Arabic OCR. Page markers identify PDF page order, which can differ from printed page numbers. OCR may misread spelling, numbers, or columns; check important claims against the original PDF. The chatbot distinguishes targets in the document from actual present-day results.

See [`data/README.md`](data/README.md) for provenance, extraction details, and the reproducible extraction command. Existing installations should update `DATA_PATH` in `.env` to `data/saudi_vision2030_ar.txt`, then rerun `chunk` and `embed`.

## Quick start

```bash
git clone https://github.com/Yazeed-Moha/AI-Workshop-Tuwaiq-KSU-lab.git rag-workshop
cd rag-workshop
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

On Windows PowerShell, use `py -m venv .venv`, `.\.venv\Scripts\Activate.ps1`, and `Copy-Item .env.example .env`. If activation is blocked, run `.\.venv\Scripts\python.exe` instead of `python` throughout the lab.

Edit `.env` and set `OPENAI_API_KEY`. The key never belongs in frontend JavaScript or Git.

Run each stage separately:

```bash
# 1. Fixed 1,000-character chunks, 250-character overlap, stride 750.
python -m rag_lab chunk

# 2. Embed and persist the vectors. Reuses a valid index on repeat runs.
python -m rag_lab embed

# Example questions about the included Vision 2030 source.
# 3. Search without generating an answer.
python -m rag_lab retrieve "ما هي محاور رؤية السعودية 2030؟" --top-k 3

# 4. Chain retrieval, context, and generation.
python -m rag_lab ask "ما هي محاور رؤية السعودية 2030؟" --top-k 3

# 5. Serve the backend and browser interface.
python -m rag_lab serve
```

Open **http://127.0.0.1:8000**. The served guide is at `/guide/` and the API reference is at `/docs`. Stop the server with Ctrl+C. To change the port: `python -m rag_lab serve --port 8001`.

## Inspect the checkpoints

| Stage | Generated artifact | Inspect |
|---|---|---|
| Chunking | `artifacts/chunks.json` | Text, IDs, exact offsets, source hash |
| Embedding | `artifacts/index.json` | Model, vector dimensions, vectors and chunks |
| Retrieval | `artifacts/last_retrieve.json` | Ranked passages and similarity scores |
| Generation | `artifacts/last_ask.json` | Answer, evidence, citations and pipeline trace |

Offsets use Python Unicode code points after UTF-8 BOM removal and newline normalization. End offsets are exclusive. No recursive splitter, tokenizer, or semantic chunker is used. The final window can be shorter than 1,000 characters; no redundant overlap-only window is added.

The vector index is readable JSON with an in-memory cosine scan. This keeps the lesson transparent and requires no external database. It is suitable for a small workshop document, not a large production collection. The default input limit is 5 MB.

## Files to explore

- `rag_lab/chunking.py`: text loading and exact fixed windows.
- `rag_lab/storage.py`: checkpoints and source freshness validation.
- `rag_lab/provider.py`: OpenAI embedding and Responses API calls.
- `rag_lab/retrieval.py`: batching, index persistence, query embedding, cosine ranking.
- `rag_lab/pipeline.py`: ordinary Python function composition (no LangChain requirement).
- `rag_lab/api.py`: FastAPI routes and static file serving.
- `frontend/`: plain HTML, CSS, and JavaScript; no Node build step.

If you change the document, rerun **chunk**, then **embed**. Changing the embedding model also requires re-embedding. Changing only the generation model does not. `embed --force` intentionally repeats embedding API calls.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -v
```

Tests use a fake provider and make no API calls. They cover chunk boundaries, Unicode, overlap, stale indexes, vector validation, citation checks, and HTTP integration. To verify actual semantic quality, run the supported / boundary-spanning / unsupported question exercises in the guide with the Vision 2030 document and a real API key.

## Limitations

- Similarity scores are not confidence percentages.
- Retrieval always ranks available passages; high rank does not prove answerability.
- The prompt asks for source-grounded answers and abstention. Verify results yourself.
- Citation validation checks IDs, not whether the cited passage supports a claim.
- Source text and model output render as plain text in the browser.
- The server binds to localhost. It has no user authentication, rate limits, or production access controls.
- The repository ignores `.env` and generated artifacts. The source PDF and OCR text are included.

Official references are linked in the HTML guide and provider code follows the OpenAI embeddings and Responses APIs.
