"""Configuration stays on the backend. Paths are anchored to the repo root."""
from dataclasses import dataclass
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 250


class LabError(Exception):
    """A useful, safe-to-display message for the workshop participant."""


@dataclass(frozen=True)
class Settings:
    source: Path
    artifacts: Path
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4.1-mini"
    api_key: str = ""


def load_settings() -> Settings:
    # Core unit tests do not need any third-party packages or credentials.
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass
    def path(name: str, default: str) -> Path:
        raw = Path(os.getenv(name, default)).expanduser()
        return (ROOT / raw).resolve() if not raw.is_absolute() else raw.resolve()
    return Settings(
        source=path("DATA_PATH", "data/knowledge.txt"),
        artifacts=path("ARTIFACTS_DIR", "artifacts"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        generation_model=os.getenv("GENERATION_MODEL", "gpt-4.1-mini"),
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )
