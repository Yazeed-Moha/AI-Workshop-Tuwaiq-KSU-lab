"""Step 5: local backend. No credentials are ever sent to the browser."""
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .settings import ROOT, LabError, load_settings
from .provider import OpenAIProvider
from .retrieval import load_index, retrieve
from .pipeline import answer_question


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=8, strict=True)


def create_app(settings=None, provider_factory=OpenAIProvider) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="Saudi Vision 2030 Chatbot", version="1.0.0")
    app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")
    app.mount("/guide", StaticFiles(directory=ROOT / "docs", html=True), name="guide")

    @app.get("/")
    def home():
        return FileResponse(ROOT / "frontend" / "index.html")

    @app.get("/source.pdf", include_in_schema=False)
    def source_pdf():
        return FileResponse(ROOT / "data" / "saudi_vision2030_ar.pdf", media_type="application/pdf")

    @app.get("/api/health")
    def health():
        try:
            index = load_index(settings)
            return {"ready": True, "source": index["source"], "chunks": len(index["chunks"]),
                    "dimensions": index["dimensions"], "embedding_model": index["embedding_model"]}
        except LabError as exc:
            return {"ready": False, "message": str(exc)}

    def run(payload: Question, generate: bool):
        try:
            # Validate local artifacts before spending money on any API request.
            load_index(settings)
            provider = provider_factory(settings)
            fn = answer_question if generate else retrieve
            return fn(settings, provider, payload.question, payload.top_k)
        except LabError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/retrieve")
    def search(payload: Question):
        return run(payload, False)

    @app.post("/api/ask")
    def ask(payload: Question):
        return run(payload, True)

    return app


app = create_app()
