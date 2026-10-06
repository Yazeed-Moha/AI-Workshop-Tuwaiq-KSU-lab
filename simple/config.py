"""Shared paths and model names. Secrets stay on the server."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
SOURCE = ROOT / os.getenv('DATA_PATH', 'data/saudi_vision2030_ar.txt')
DB = ROOT / 'artifacts/chroma'
EMBED_MODEL = os.getenv('LOCAL_EMBEDDING_MODEL', 'intfloat/multilingual-e5-small')
CHAT_MODEL = os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b')
SIZE, OVERLAP = 1000, 250
