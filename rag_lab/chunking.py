"""Step 1: fixed character windows, not tokens or recursive splitting."""
import hashlib
from pathlib import Path
from .settings import CHUNK_SIZE, CHUNK_OVERLAP, LabError


def read_document(path: Path) -> str:
    if path.suffix.lower() != ".txt":
        raise LabError("DATA_PATH must point to a UTF-8 .txt file.")
    if not path.is_file():
        raise LabError(f"Document not found: {path.name}. Add data/knowledge.txt or set DATA_PATH in .env.")
    if path.stat().st_size > 5_000_000:
        raise LabError("Use a document smaller than 5 MB for this workshop.")
    try:
        # Decode BOM if present; normalize CRLF/CR to LF on every platform.
        text = path.read_text(encoding="utf-8-sig")
    except (UnicodeError, OSError) as exc:
        raise LabError("Could not read the document. Save it as UTF-8 plain text.") from exc
    if not text.strip():
        raise LabError("The source document is empty or contains only whitespace.")
    return text


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size.")
    if not text.strip():
        raise ValueError("Text must contain a non-whitespace character.")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append({
            "id": f"chunk-{len(chunks) + 1:04d}",
            "start": start, "end": end, "text": text[start:end],
        })
        if end == len(text):
            break  # Do not add a redundant final chunk that is only overlap.
        start += size - overlap
    return chunks
