"""Readable JSON checkpoints with atomic writes and source freshness checks."""
import json
import os
from pathlib import Path
import tempfile
from .chunking import chunk_text, read_document, text_hash
from .settings import CHUNK_SIZE, CHUNK_OVERLAP, Settings, LabError


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as f:
            name = f.name
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def read_json(path: Path, instruction: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LabError(f"Missing or unreadable {path.name}. {instruction}") from exc
    if not isinstance(value, dict):
        raise LabError(f"Invalid {path.name}. {instruction}")
    return value


def prepare_chunks(settings: Settings) -> dict:
    text = read_document(settings.source)
    result = {
        "schema_version": 1, "source": settings.source.name,
        "source_sha256": text_hash(text), "character_count": len(text),
        "chunk_size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP,
        "offset_convention": "zero-based Unicode code points; end exclusive; newlines normalized to LF",
        "chunks": chunk_text(text),
    }
    write_json(settings.artifacts / "chunks.json", result)
    return result


def load_chunks(settings: Settings) -> dict:
    data = read_json(settings.artifacts / "chunks.json", "Run: python -m rag_lab chunk")
    text = read_document(settings.source)
    valid = (data.get("schema_version") == 1
             and data.get("source") == settings.source.name
             and data.get("source_sha256") == text_hash(text)
             and data.get("chunk_size") == CHUNK_SIZE
             and data.get("overlap") == CHUNK_OVERLAP
             and data.get("chunks") == chunk_text(text))
    if not valid:
        raise LabError("Chunks are stale or modified. Run: python -m rag_lab chunk, then embed.")
    return data
