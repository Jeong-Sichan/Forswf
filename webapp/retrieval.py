"""Select the right local corpus before sending evidence to KoRMo."""

import re
from dataclasses import dataclass

from knowledge import BM25Index, E5Index, load_chunks, search_guide


@dataclass(frozen=True)
class Selection:
    topic: str
    source_type: str
    passages: list[dict]
    requires_source: bool = False


class ScopedIndex:
    def __init__(self, index, chunks: list[dict]):
        self.index = index
        self.chunks = chunks
        self.allowed = {row["source_id"] for row in chunks}

    def search(self, query: str, limit: int = 3) -> list[dict]:
        # Search the shared E5 index once, then retain only this corpus.
        found = self.index.search(query, limit=len(self.index.chunks))
        return [row for row in found if row["source_id"] in self.allowed][:limit]


def infer_topic(question: str) -> str:
    text = question.lower()
    if re.search(r"외국인등록|체류|비자|건강보험|임대차|전세|월세|전입신고|보증금|lease|landlord|tenant|visa|immigration|health insurance|residence card", text):
        return "life"
    if re.search(r"버스|지하철|교통카드|환승|대중교통|강남역|홍대입구|에서.{1,20}가려면|무임승차|부정승차|운임|subway|bus|transit|transport|fare card|fare evasion|directions|how do i get", text):
        return "transit"
    if re.search(r"반찬|식당|음식|먹|김치|한식|banchan|restaurant|food|meal|kimchi", text):
        return "food"
    if re.search(r"예절|인사|존댓말|높임말|교수님|etiquette|greet|honorific|polite", text):
        return "etiquette"
    if re.search(r"문화|명절|설날|추석|문양|유물|상징|culture|holiday|seollal|chuseok|symbol|motif|heritage", text):
        return "culture"
    return "all"


def is_heritage_question(question: str) -> bool:
    return bool(re.search(r"문양|유물|상징|상징성|motif|symbol|artifact|heritage pattern", question.lower()))


def is_transit_law_question(question: str) -> bool:
    """The current Easy Law corpus covers fare evasion, not journey planning or boarding."""
    return bool(re.search(r"무임승차|부정승차|요금.{0,8}(?:안 내|미납)|fare evasion|fare dodging|not paying.{0,12}fare",
                          question.lower()))


class KnowledgeHub:
    def __init__(self, law_path: str = "", culture_path: str = "", wiki_path: str = "",
                 retriever: str = "e5", cache_path: str = ""):
        self.corpora = {
            "law": load_chunks(law_path) if law_path else [],
            "culture": load_chunks(culture_path) if culture_path else [],
            "wiki": load_chunks(wiki_path) if wiki_path else [],
        }
        all_chunks = [row for corpus in self.corpora.values() for row in corpus]
        if not all_chunks:
            self.index = None
        elif retriever == "bm25":
            self.index = BM25Index(all_chunks)
        elif retriever == "e5":
            self.index = E5Index(all_chunks, cache_path=cache_path or None)
        else:
            raise ValueError("MALMAEK_RETRIEVER must be e5 or bm25")

    def _scope(self, corpus: str) -> ScopedIndex | None:
        if self.index is None or not self.corpora[corpus]:
            return None
        return ScopedIndex(self.index, self.corpora[corpus])

    def select(self, question: str, category: str) -> Selection:
        topic = infer_topic(question) if category == "all" else category
        if topic == "transit":
            if not is_transit_law_question(question):
                return Selection(topic, "general", [])
            scope = self._scope("law")
            if not scope:
                return Selection(topic, "none", [], True)
            found = [row for row in search_guide(scope, question)
                     if row["source_id"].startswith("easylaw:661:")
                     and "부정승차" in row["title"]]
            return Selection(topic, "official" if found else "none", found, True)
        if topic == "life":
            if infer_topic(question) != "life":
                return Selection(topic, "general", [])
            scope = self._scope("law")
            if not scope:
                return Selection(topic, "none", [], True)
            found = search_guide(scope, question)
            found = [row for row in found if row["source_id"].startswith(("easylaw:2853:", "easylaw:629:"))]
            return Selection(topic, "official" if found else "none", found, True)
        if topic == "culture" and is_heritage_question(question):
            scope = self._scope("culture")
            if scope:
                found = scope.search(question, limit=2)
                if found:
                    return Selection(topic, "heritage", found)
        if topic in {"culture", "food", "etiquette"}:
            scope = self._scope("wiki")
            if scope:
                found = scope.search(question, limit=2)
                if found:
                    return Selection(topic, "encyclopedia", found)
        return Selection(topic, "general", [])
