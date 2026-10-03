"""One command per checkpoint. Run all commands from the repository root."""
import argparse
import json
import sys
from .settings import load_settings, LabError
from .storage import prepare_chunks, load_chunks, write_json
from .retrieval import build_index, load_index, retrieve
from .pipeline import answer_question
from .provider import OpenAIProvider


def main():
    parser = argparse.ArgumentParser(description="Step-by-step RAG lab")
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("chunk", help="1: save exact 1000-character windows with 250-character overlap")
    embed = sub.add_parser("embed", help="2: embed and save the local vector index")
    embed.add_argument("--force", action="store_true", help="re-embed even if the index is up to date (API charges apply)")
    for name in ["retrieve", "ask"]:
        command = sub.add_parser(name)
        command.add_argument("question")
        command.add_argument("--top-k", type=int, default=3)
    serve = sub.add_parser("serve", help="5: run backend and frontend together")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    settings = load_settings()
    try:
        if args.step == "chunk":
            result = prepare_chunks(settings)
            print(f"Saved {len(result['chunks'])} chunks from {result['character_count']} characters.")
            for c in result["chunks"][:5]:
                print(f"{c['id']}: [{c['start']}, {c['end']}) — {len(c['text'])} characters")
            print("Checkpoint: artifacts/chunks.json")
        elif args.step == "embed":
            load_chunks(settings)
            # Reusing a valid index does not need an API key or an API call.
            if not args.force:
                try:
                    index = load_index(settings)
                    print(f"Reused {len(index['chunks'])} embeddings; no API call made.")
                    return
                except LabError:
                    pass
            index, reused = build_index(settings, OpenAIProvider(settings), force=args.force)
            print(f"Saved {len(index['vectors'])} embeddings, {index['dimensions']} dimensions each.")
            print("Checkpoint: artifacts/index.json")
        elif args.step in {"retrieve", "ask"}:
            load_index(settings)
            provider = OpenAIProvider(settings)
            fn = retrieve if args.step == "retrieve" else answer_question
            result = fn(settings, provider, args.question, args.top_k)
            write_json(settings.artifacts / f"last_{args.step}.json", result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.step == "serve":
            if not 1 <= args.port <= 65535:
                raise LabError("Choose a port between 1 and 65535.")
            try:
                import uvicorn
            except ImportError as exc:
                raise LabError("Install dependencies: python -m pip install -r requirements.txt") from exc
            print(f"Frontend: http://127.0.0.1:{args.port}")
            print(f"Guide:    http://127.0.0.1:{args.port}/guide/")
            uvicorn.run("rag_lab.api:app", host="127.0.0.1", port=args.port)
    except (LabError, OSError) as exc:
        print(f"Lab error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
