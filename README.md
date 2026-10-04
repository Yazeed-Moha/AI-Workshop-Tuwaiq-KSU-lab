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
