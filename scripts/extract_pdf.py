"""Recreate the Arabic knowledge source locally (Tesseract + ara language pack)."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/saudi_vision2030_ar.pdf'
OUTPUT = ROOT / 'data/saudi_vision2030_ar.txt'


def extract(item):
    number, image = item
    result = subprocess.run(['tesseract', str(image), 'stdout', '-l', 'ara', '--psm', '3', '--dpi', '300'], check=True, capture_output=True, text=True, encoding='utf-8')
    if len(result.stdout.strip()) < 100:
        fallback = subprocess.run(['tesseract', str(image), 'stdout', '-l', 'ara', '--psm', '6', '--dpi', '300'], check=True, capture_output=True, text=True, encoding='utf-8')
        if len(fallback.stdout.strip()) > len(result.stdout.strip()):
            result = fallback
    text = unicodedata.normalize('NFC', result.stdout).replace('\u0640', '')
    text = '\n'.join(line.rstrip() for line in text.splitlines()).strip()
    return number, text


def main():
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(SOURCE)
    with tempfile.TemporaryDirectory() as directory:
        jobs = []
        for number in range(len(pdf)):
            path = Path(directory) / f'{number + 1:03}.png'
            pdf[number].render(scale=300/72).to_pil().convert('L').save(path)
            jobs.append((number + 1, path))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            pages = list(pool.map(extract, jobs))
    parts, offsets = [], []
    position = 0
    for number, text in pages:
        part = f'[صفحة PDF {number}]\n{text}\n\n'
        offsets.append({'pdf_page': number, 'start': position, 'end': position + len(part), 'ocr_characters': len(text)})
        parts.append(part)
        position += len(part)
    OUTPUT.write_text(''.join(parts), encoding='utf-8')
    report = {'source': SOURCE.name, 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(), 'text_sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest(), 'method': 'Tesseract Arabic OCR, 300 DPI, PSM 3 with PSM 6 fallback for sparse pages; NFC; tatweel removed', 'page_numbering': '1-based PDF page order, not printed folio numbers', 'pages': offsets, 'limitations': 'Machine OCR; spelling, numbers, and reading order may contain errors. Verify answers against the PDF.'}
    (ROOT / 'data/extraction.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'Extracted {len(pages)} pages; {position:,} characters to {OUTPUT.name}')

if __name__ == '__main__':
    main()
