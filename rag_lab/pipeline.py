"""Step 4: question → retrieval → context → generation → answer and evidence."""
import re
from .retrieval import retrieve
from .settings import Settings


def answer_question(settings: Settings, provider, question: str, top_k: int = 3) -> dict:
    search = retrieve(settings, provider, question, top_k)
    evidence = search["matches"]
    answer = provider.generate(search["question"], evidence)
    cited_ids = sorted(set(re.findall(r"\[(chunk-\d+)\]", answer)))
    retrieved_ids = {hit["id"] for hit in evidence}
    invalid = [item for item in cited_ids if item not in retrieved_ids]
    warnings = []
    if invalid:
        warnings.append("The answer cites a chunk that was not retrieved. Check the evidence.")
    if not cited_ids:
        warnings.append("No chunk citation found. This can be appropriate for an unsupported question; review the answer.")
    return {**search, "answer": answer, "generation_model": settings.generation_model,
            "cited_ids": cited_ids, "citation_warnings": warnings,
            "trace": ["Embed the question", "Rank document chunks", "Build evidence context", "Generate a grounded answer"]}
