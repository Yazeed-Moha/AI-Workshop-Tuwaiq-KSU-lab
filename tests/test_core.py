import json
import tempfile
import unittest
from pathlib import Path
from rag_lab.chunking import chunk_text, read_document
from rag_lab.settings import Settings, LabError
from rag_lab.storage import prepare_chunks
from rag_lab.retrieval import build_index, load_index, retrieve
from rag_lab.pipeline import answer_question


class FakeProvider:
    def __init__(self):
        self.embedded = []
        self.evidence = None
    def embed(self, texts):
        self.embedded.extend(texts)
        return [[1.0, 0.0] if 'library' in t.lower() else [0.0, 1.0] for t in texts]
    def generate(self, question, evidence):
        self.evidence = evidence
        return f"The library is open on weekdays. [{evidence[0]['id']}]"


class ChunkTests(unittest.TestCase):
    def test_exact_boundaries_and_full_coverage(self):
        for length in [1, 249, 250, 999, 1000, 1001, 1750, 1751, 2500, 4001]:
            with self.subTest(length=length):
                text = ''.join(chr(0x0620 + i % 30) for i in range(length))
                chunks = chunk_text(text)
                self.assertEqual(chunks[0]['start'], 0)
                self.assertEqual(chunks[-1]['end'], length)
                rebuilt = chunks[0]['text']
                for left, right in zip(chunks, chunks[1:]):
                    self.assertEqual(right['start'] - left['start'], 750)
                    self.assertEqual(left['text'][-250:], right['text'][:250])
                    rebuilt += right['text'][250:]
                self.assertEqual(rebuilt, text)
                self.assertTrue(all(len(c['text']) <= 1000 for c in chunks))
                self.assertTrue(all(c['end'] < length for c in chunks[:-1]))
    def test_invalid_chunk_arguments(self):
        for size, overlap in [(0, 0), (1000, 1000), (1000, -1)]:
            with self.assertRaises(ValueError): chunk_text('x', size, overlap)
    def test_empty_text(self):
        with self.assertRaises(ValueError): chunk_text(' \n\t')


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.settings = Settings(root / 'knowledge.txt', root / 'artifacts')
        self.settings.source.write_text('Library opening hours. ' * 60 + 'Parking permit rules. ' * 80, encoding='utf-8')
        self.provider = FakeProvider()
        prepare_chunks(self.settings)
    def tearDown(self): self.temp.cleanup()
    def test_newlines_and_bom(self):
        self.settings.source.write_bytes(b'\xef\xbb\xbfOne\r\nTwo\rThree')
        self.assertEqual(read_document(self.settings.source), 'One\nTwo\nThree')
    def test_index_reuse_does_not_call_api(self):
        _, reused = build_index(self.settings, self.provider)
        self.assertFalse(reused)
        calls = len(self.provider.embedded)
        _, reused = build_index(self.settings, self.provider)
        self.assertTrue(reused)
        self.assertEqual(len(self.provider.embedded), calls)
    def test_source_change_rejected(self):
        build_index(self.settings, self.provider)
        self.settings.source.write_text('New content', encoding='utf-8')
        with self.assertRaisesRegex(LabError, 'stale'): load_index(self.settings)
    def test_embedding_model_change_rejected(self):
        build_index(self.settings, self.provider)
        changed = Settings(self.settings.source, self.settings.artifacts, embedding_model='other')
        with self.assertRaisesRegex(LabError, 'stale'): load_index(changed)
    def test_retrieval_and_pipeline_keep_source_evidence(self):
        build_index(self.settings, self.provider)
        result = answer_question(self.settings, self.provider, 'When is the library open?', 2)
        self.assertEqual(len(result['matches']), 2)
        self.assertEqual(result['matches'][0]['score'], 1.0)
        self.assertIn('Library', result['matches'][0]['text'])
        self.assertEqual(self.provider.evidence, result['matches'])
        self.assertEqual(result['citation_warnings'], [])
    def test_dimension_mismatch_rejected(self):
        build_index(self.settings, self.provider)
        self.provider.embed = lambda texts: [[1, 0, 0]]
        with self.assertRaisesRegex(LabError, 'dimensions differ'): retrieve(self.settings, self.provider, 'library')
    def test_corrupt_vectors_rejected(self):
        build_index(self.settings, self.provider)
        path = self.settings.artifacts / 'index.json'
        index = json.loads(path.read_text())
        index['vectors'][0] = [0, 0]
        path.write_text(json.dumps(index))
        with self.assertRaisesRegex(LabError, 'magnitude'): load_index(self.settings)
    def test_bad_citations_flagged(self):
        build_index(self.settings, self.provider)
        self.provider.generate = lambda q, evidence: 'Invented claim [chunk-9999]'
        result = answer_question(self.settings, self.provider, 'library')
        self.assertIn('not retrieved', result['citation_warnings'][0])
    def test_invalid_question_and_top_k(self):
        build_index(self.settings, self.provider)
        for question, k in [(' ', 3), ('x', 0), ('x', 9), ('x' * 2001, 3)]:
            with self.assertRaises(LabError): retrieve(self.settings, self.provider, question, k)


if __name__ == '__main__': unittest.main()
