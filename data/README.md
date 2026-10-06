# Saudi Vision 2030 source documents

- `saudi_vision2030_ar.pdf`: original 81-page Arabic PDF supplied by the repository owner; copied without modification.
- `saudi_vision2030_ar.txt`: UTF-8 Arabic OCR text, used by the chatbot by default.
- `extraction.json`: source/text SHA-256 checksums, extraction method, and per-page character offsets.

The PDF's embedded text has malformed Arabic glyph mappings and reversed reading order. Text was therefore extracted locally with Tesseract's Arabic OCR model at 300 DPI. Unicode is normalized to NFC and decorative tatweel is removed. Page markers such as `[صفحة PDF 21]` refer to the 1-based PDF page order, not printed page numbers. Blank or image-only pages retain their markers.

**Quality:** this is machine OCR, not a certified transcription. It can omit or misread words, figures, headings, or columns. Verify important answers against the original PDF. The source describes Vision 2030 ambitions and targets; it is not evidence that a target has been achieved. Chunk IDs identify retrieved text, not verified claims.

## Reproduce extraction (optional)

The TXT is already committed; normal chatbot setup does not require OCR tools.

Install Tesseract with Arabic language data (`ara`) using your operating system package manager, then:

```bash
python -m pip install -r requirements-extraction.txt
python scripts/extract_pdf.py
python -m rag_lab chunk
python -m rag_lab embed
```

The extraction script uses local OCR only and does not send the PDF to an external service. The Day 2 simple/ path embeds locally and sends questions and evidence to Groq for generation. The legacy rag_lab/ path uses OpenAI APIs. OCR output can vary with Tesseract versions and language models.
