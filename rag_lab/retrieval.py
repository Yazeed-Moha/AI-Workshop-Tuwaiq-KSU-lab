"""Steps 2–3: embed chunks once, then rank them by cosine similarity."""
import math
from .settings import Settings, LabError
from .storage import load_chunks, read_json, write_json


def normalized(vector: list[float]) -> list[float]:
    if not isinstance(vector, list) or not vector:
        raise LabError("Invalid empty embedding vector. Rebuild the index.")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in vector):
        raise LabError("Embedding vectors must contain finite numbers. Rebuild the index.")
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0 or not math.isfinite(norm):
        raise LabError("Invalid embedding magnitude. Rebuild the index.")
    return [x / norm for x in vector]


def load_index(settings: Settings) -> dict:
    chunks = load_chunks(settings)
    index = read_json(settings.artifacts / "index.json", "Run: python -m rag_lab embed")
    if (index.get("schema_version") != 1
        or index.get("source_sha256") != chunks["source_sha256"]
        or index.get("embedding_model") != settings.embedding_model
        or index.get("chunks") != chunks["chunks"]):
        raise LabError("The index is stale or uses another embedding model. Run: python -m rag_lab embed")
    vectors = index.get("vectors")
    dim = index.get("dimensions")
    if not isinstance(vectors, list) or len(vectors) != len(chunks["chunks"]) or not isinstance(dim, int) or dim < 1:
        raise LabError("Invalid index structure. Run: python -m rag_lab embed --force")
    for vector in vectors:
        normalized(vector)
        if len(vector) != dim:
            raise LabError("Index dimensions do not match. Run: python -m rag_lab embed --force")
    return index


def build_index(settings: Settings, provider, force: bool = False) -> tuple[dict, bool]:
    chunks = load_chunks(settings)
    if not force:
        try:
            return load_index(settings), True
        except LabError:
            pass
    vectors = []
    records = chunks["chunks"]
    for start in range(0, len(records), 32):
        batch = records[start:start + 32]
        # A whitespace-only window has no semantic content. Keep its exact offsets
        # but provide a visible marker to the embedding API instead of an empty input.
        embedded = provider.embed([c["text"] if c["text"].strip() else "[blank text]" for c in batch])
        if len(embedded) != len(batch):
            raise LabError("Embedding count does not match the chunk count.")
        vectors.extend(normalized(v) for v in embedded)
    dim = len(vectors[0])
    if any(len(v) != dim for v in vectors):
        raise LabError("Embedding dimensions changed between batches. Retry with one model.")
    index = {"schema_version": 1, "source": chunks["source"],
             "source_sha256": chunks["source_sha256"], "embedding_model": settings.embedding_model,
             "dimensions": dim, "chunks": records, "vectors": vectors}
    write_json(settings.artifacts / "index.json", index)
    return index, False


def validate_question(question: str, top_k: int) -> str:
    if not isinstance(question, str) or not question.strip() or len(question) > 2000:
        raise LabError("Enter a question between 1 and 2,000 characters.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 8:
        raise LabError("top_k must be an integer between 1 and 8.")
    return question.strip()


def retrieve(settings: Settings, provider, question: str, top_k: int = 3) -> dict:
    question = validate_question(question, top_k)
    index = load_index(settings)
    queries = provider.embed([question])
    if len(queries) != 1:
        raise LabError("Expected one query embedding.")
    query = normalized(queries[0])
    if len(query) != index["dimensions"]:
        raise LabError("Query and document dimensions differ. Rebuild the index with the same embedding model.")
    hits = []
    for chunk, vector in zip(index["chunks"], index["vectors"]):
        score = sum(a * b for a, b in zip(query, normalized(vector)))
        # Blank windows are retained for exact chunking but not useful evidence.
        if chunk["text"].strip():
            hits.append({**chunk, "source": index["source"], "score": max(-1.0, min(1.0, score))})
    hits.sort(key=lambda row: (-row["score"], row["start"]))
    result = {"question": question, "top_k": top_k, "embedding_model": index["embedding_model"],
              "matches": hits[:top_k], "score_note": "Cosine similarity, not answer confidence."}
    return result
