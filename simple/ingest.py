"""Run once: python -m simple.ingest. Inspect: python -m simple.ingest --preview."""
import argparse
import hashlib
import json
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from .config import SOURCE, DB, EMBED_MODEL, SIZE, OVERLAP


def documents(text, size=SIZE, overlap=OVERLAP):
    if not 0 <= overlap < size:
        raise ValueError('Require 0 <= overlap < size.')
    result = []
    for start in range(0, len(text), size - overlap):
        end = min(start + size, len(text))
        result.append(Document(page_content=text[start:end], metadata={
            'id': f'chunk-{len(result)+1:04}', 'source': SOURCE.name,
            'start': start, 'end': end}))
        if end == len(text):
            break
    return result


def fingerprint():
    text = SOURCE.read_text(encoding='utf-8-sig')
    if not text.strip():
        raise ValueError('The source text is empty.')
    signature = hashlib.sha256((text + EMBED_MODEL + str((SIZE, OVERLAP))).encode()).hexdigest()
    return text, signature


def open_store(embeddings=None):
    return Chroma(collection_name='vision2030', persist_directory=str(DB),
                  embedding_function=embeddings or OpenAIEmbeddings(model=EMBED_MODEL),
                  collection_metadata={'hnsw:space': 'cosine'})


def require_index():
    _, signature = fingerprint()
    marker = DB / 'ready.json'
    if not marker.exists() or json.loads(marker.read_text())['signature'] != signature:
        raise ValueError('Index missing or stale. Run: python -m simple.ingest')


def ingest(preview=False):
    text, signature = fingerprint()
    docs = documents(text)
    print(f'{len(text)} characters → {len(docs)} chunks; size={SIZE}, overlap={OVERLAP}')
    print(docs[0])
    if preview:
        return
    marker = DB / 'ready.json'
    if marker.exists() and json.loads(marker.read_text()).get('signature') == signature:
        print('Index is current; no embedding request made.')
        return
    # Remove readiness before rebuilding so an interrupted build cannot be used.
    marker.unlink(missing_ok=True)
    store = open_store()
    store.reset_collection()
    for start in range(0, len(docs), 32):
        batch = docs[start:start+32]
        store.add_documents(batch, ids=[d.metadata['id'] for d in batch])
    marker.write_text(json.dumps({'signature': signature, 'chunks': len(docs)}))
    print('Saved vectors, text and metadata to artifacts/chroma. Ready to retrieve.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', action='store_true', help='Inspect chunks without an API call')
    ingest(parser.parse_args().preview)
