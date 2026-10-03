import importlib.util
import unittest
from . import test_core
from rag_lab.retrieval import build_index

AVAILABLE = all(importlib.util.find_spec(m) for m in ['fastapi', 'httpx'])


@unittest.skipUnless(AVAILABLE, 'Install requirements-dev.txt to run HTTP integration tests')
class ApiTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from rag_lab.api import create_app
        self.fixture = test_core.PipelineTests()
        self.fixture.setUp()
        build_index(self.fixture.settings, self.fixture.provider)
        self.client = TestClient(create_app(self.fixture.settings, lambda settings: test_core.FakeProvider()))
    def tearDown(self):
        self.client.close()
        self.fixture.tearDown()
    def test_frontend_guide_and_health(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        self.assertEqual(self.client.get('/guide/').status_code, 200)
        self.assertTrue(self.client.get('/api/health').json()['ready'])
    def test_retrieve_and_ask(self):
        payload = {'question': 'When is the library open?', 'top_k': 2}
        search = self.client.post('/api/retrieve', json=payload)
        self.assertEqual(search.status_code, 200)
        self.assertNotIn('answer', search.json())
        result = self.client.post('/api/ask', json=payload)
        self.assertEqual(result.status_code, 200)
        self.assertIn('[chunk-', result.json()['answer'])
        self.assertEqual(len(result.json()['matches']), 2)
    def test_validation_and_stale_index(self):
        self.assertEqual(self.client.post('/api/ask', json={'question': 'x', 'top_k': 99}).status_code, 422)
        self.assertEqual(self.client.post('/api/ask', json={'question': '   '}).status_code, 400)
        self.fixture.settings.source.write_text('Changed document')
        self.assertFalse(self.client.get('/api/health').json()['ready'])
        self.assertEqual(self.client.post('/api/ask', json={'question': 'x'}).status_code, 400)
