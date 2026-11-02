import unittest
import asyncio

import httpx

from knowledge import BM25Index, chunk_document
from webapp.engine import Settings, answer_language, chat_messages, with_history
from webapp.main import create_app
from webapp.retrieval import KnowledgeHub, infer_topic
from webapp.schemas import ChatRequest, ChatResponse


class FakeEngine:
    def __init__(self):
        self.hub = type("Hub", (), {"corpora": {"law": [1], "culture": [], "wiki": []}})()
        self.settings = Settings(adapter="/fake/adapter")
        self.requests = []

    def reply(self, request):
        self.requests.append(request)
        return ChatResponse(answer="계약 상대방을 확인하세요.", category=request.category,
                            language="ko", used_adapter=True, source_type="official",
                            sources=[{"title": "공식 안내", "url": "https://example.org/law",
                                      "license": "attribution"}])


class WebAppTests(unittest.TestCase):
    @staticmethod
    def run_async(coroutine):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coroutine)
        finally:
            loop.close()

    def test_http_chat_health_and_static_home(self):
        engine = FakeEngine()
        async def check():
            app = create_app(engine=engine)
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                    self.assertEqual((await client.get("/")).status_code, 200)
                    self.assertIn("갓을 쓴 말맥", (await client.get("/")).text)
                    self.assertEqual((await client.get("/static/mascot.svg")).status_code, 200)
                    self.assertEqual((await client.get("/api/health")).json()["corpora"]["law"], 1)
                    response = await client.post("/api/chat", json={"category": "life", "language": "ko",
                        "history": [{"role": "user", "content": "안녕"},
                                    {"role": "assistant", "content": "안녕하세요"}],
                        "message": "계약 전에 뭘 확인해요?"})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()["sources"][0]["title"], "공식 안내")
                    self.assertEqual(len(engine.requests[0].history), 2)
                    self.assertEqual(response.headers["cache-control"], "no-store")
                    self.assertEqual((await client.post("/api/chat", json={"message": "   "})).status_code, 422)
        self.run_async(check())

    def test_unconfigured_server_reports_unavailable(self):
        async def check():
            app = create_app(settings=Settings())
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                    self.assertFalse((await client.get("/api/health")).json()["ready"])
                    self.assertEqual((await client.post("/api/chat", json={"message": "Hello"})).status_code, 503)
        self.run_async(check())

    def test_history_is_short_and_evidence_stays_in_latest_turn(self):
        request = ChatRequest(message="What is banchan?", history=[
            {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}])
        turns = with_history(chat_messages(request, "en", [{"title": "Banchan", "url": "https://example.org",
                "text": "Banchan are side dishes in Korean meals."}], "food"), request)
        self.assertEqual([item["role"] for item in turns], ["system", "user", "assistant", "user"])
        self.assertIn("Banchan are side dishes", turns[-1]["content"])
        self.assertEqual(answer_language(request), "en")
        self.assertEqual(answer_language(ChatRequest(message="반찬이 뭐야?")), "ko")

    def test_lease_route_prefers_lease_guide(self):
        student = {"source_id": "easylaw:2853:2.3.1", "title": "외국인유학생 · 외국인등록 · 법제처",
                   "url": "https://example.org/student", "license": "attribution",
                   "text": "유학생은 외국인등록과 체류지 변경 신고를 해야 합니다. " * 12}
        lease = {"source_id": "easylaw:629:2.2.1", "title": "주택임대차 · 임대차계약의 당사자 확인 등 · 법제처",
                 "url": "https://example.org/lease", "license": "attribution",
                 "text": "계약 전 등기부상 소유자나 적법한 대리인인지 확인해야 합니다. " * 12}
        chunks = [part for row in (student, lease) for part in chunk_document(row)]
        hub = KnowledgeHub.__new__(KnowledgeHub)
        hub.corpora = {"law": chunks, "culture": [], "wiki": []}
        hub.index = BM25Index(chunks)
        found = hub.select("What should I check before signing a housing lease in Korea?", "all")
        self.assertEqual(infer_topic("housing lease"), "life")
        self.assertEqual(found.source_type, "official")
        self.assertEqual(found.passages[0]["source_id"], lease["source_id"])


if __name__ == "__main__":
    unittest.main()
