"""Serve with: python -m uvicorn simple.app:app --host 127.0.0.1 --port 8000"""
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .config import ROOT, DB, SOURCE
from .ingest import require_index
from .chatbot import ask, retrieve

app = FastAPI(title='Vision 2030 · Simple LangChain chatbot')
app.mount('/static', StaticFiles(directory=ROOT / 'frontend'), name='static')
app.mount('/guide', StaticFiles(directory=ROOT / 'docs', html=True), name='guide')


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=8, strict=True)


@app.get('/')
def home():
    return FileResponse(ROOT / 'frontend/index.html')


@app.get('/source.pdf')
def pdf():
    return FileResponse(ROOT / 'data/saudi_vision2030_ar.pdf', media_type='application/pdf')


@app.get('/api/health')
def health():
    try:
        require_index()
        return {'ready': True, 'source': SOURCE.name,
                'chunks': json.loads((DB / 'ready.json').read_text())['chunks'],
                'index_type': 'Chroma · cosine'}
    except (ValueError, OSError, KeyError) as exc:
        return {'ready': False, 'message': str(exc)}


def run(payload, generate):
    try:
        return ask(payload.question, payload.top_k) if generate else {'matches': retrieve(payload.question, payload.top_k)}
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, 'Model request failed. Check API key, model access, quota and connection.') from exc


@app.post('/api/ask')
def answer(payload: Question):
    return run(payload, True)


@app.post('/api/retrieve')
def search(payload: Question):
    return run(payload, False)
