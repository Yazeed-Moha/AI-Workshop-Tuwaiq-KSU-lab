"""Real local Chroma + fake model/embeddings: no paid API requests."""
import tempfile
import unittest
from pathlib import Path
import importlib.util
if not importlib.util.find_spec("langchain_chroma"):
    raise unittest.SkipTest("Install requirements-simple.txt for LangChain tests")
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_chroma import Chroma
from simple.ingest import documents
from simple.chatbot import ask, make_chain, retrieve


class TinyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]
    def embed_query(self, text):
        return [1., 0.] if 'business' in text.lower() else [0., 1.]


class SimpleTests(unittest.TestCase):
    def test_exact_unicode_windows(self):
        for length in [1, 1000, 1001, 1750, 2500]:
            text = ('رؤية السعودية ' * 300)[:length]
            docs = documents(text)
            self.assertEqual(docs[-1].metadata['end'], len(text))
            self.assertEqual(docs[0].page_content + ''.join(d.page_content[250:] for d in docs[1:]), text)
            for a,b in zip(docs,docs[1:]):
                self.assertEqual(a.page_content[-250:], b.page_content[:250])
    def test_chroma_distance_and_chain(self):
        with tempfile.TemporaryDirectory() as folder:
            store=Chroma(collection_name='test_vision',persist_directory=folder,
                         embedding_function=TinyEmbeddings(),collection_metadata={'hnsw:space':'cosine'})
            a=documents('Support business');b=documents('Healthy living')
            b[0].metadata['id']='chunk-0002'
            store.add_documents(a+b,ids=['a','b'])
            matches=retrieve('business',2,store)
            self.assertEqual(matches[0]['id'],'chunk-0001')
            self.assertAlmostEqual(matches[0]['distance'],0,places=5)
            self.assertAlmostEqual(matches[1]['score'],0,places=5)
            def fake(prompt):
                messages=prompt.to_messages()
                self.assertIn('untrusted',messages[0].content)
                self.assertIn('Support business',messages[1].content)
                return AIMessage(content='Support business [chunk-0001]')
            result=ask('business',2,store,make_chain(RunnableLambda(fake)))
            self.assertEqual(result['citation_warnings'],[])
            self.assertEqual(len(result['matches']),2)
            with self.assertRaises(ValueError): retrieve(' ',3,store)
            with self.assertRaises(ValueError): retrieve('business',9,store)
    def test_http_contract(self):
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        from simple.app import app
        with TestClient(app) as client:
            self.assertEqual(client.get('/').status_code,200)
            self.assertIn('Build a Vision 2030 Chatbot',client.get('/guide/').text)
            self.assertEqual(client.post('/api/ask',json={'question':'x','top_k':9}).status_code,422)
            with patch('simple.app.ask',return_value={'answer':'test','matches':[]}):
                self.assertEqual(client.post('/api/ask',json={'question':'test'}).json()['answer'],'test')
