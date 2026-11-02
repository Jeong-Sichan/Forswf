"""Small, dependency-free source retrieval for the first Malmaek prototype."""

import json
import hashlib
import math
import re
from collections import Counter
from pathlib import Path

from prompting import require_local_dataset


def terms(value: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+|[가-힣]+", value.lower())
    result = []
    for word in words:
        if re.search(r"[가-힣]", word):
            result.extend(word[i:i + 2] for i in range(len(word) - 1))
            if len(word) == 1:
                result.append(word)
        elif len(word) > 2:
            result.append(word)
    return result


def chunk_document(document: dict, size: int = 750, overlap: int = 80) -> list[dict]:
    if size <= overlap or overlap < 0:
        raise ValueError("chunk size must exceed overlap")
    required = ("source_id", "title", "url", "license", "text")
    if any(not isinstance(document.get(key), str) or not document[key].strip() for key in required):
        raise ValueError(f"document requires {', '.join(required)}")
    text = re.sub(r"\s+", " ", document["text"]).strip()
    if len(text) < 80:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(". ", start + size // 2, end)
            if boundary > start:
                end = boundary + 1
        chunks.append({key: document[key] for key in required if key != "text"} | {
            "updated_at": document.get("updated_at", ""),
            "chunk_id": f"{document['source_id']}:{len(chunks)}",
            "text": text[start:end].strip(),
        })
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def load_chunks(path: str) -> list[dict]:
    require_local_dataset(path)
    chunks = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not chunks:
        raise ValueError("검색할 지식 문서가 없습니다")
    for row in chunks:
        if any(not row.get(key) for key in ("source_id", "title", "url", "license", "chunk_id", "text")):
            raise ValueError("검색 문서에 필수 출처 정보가 없습니다")
    return chunks


class BM25Index:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.counts = [Counter(terms(row["title"] + " " + row["text"])) for row in chunks]
        self.lengths = [sum(counts.values()) for counts in self.counts]
        self.avg_length = sum(self.lengths) / max(len(chunks), 1)
        self.df = Counter()
        for counts in self.counts:
            self.df.update(counts.keys())

    def search(self, query: str, limit: int = 3) -> list[dict]:
        query_terms = set(terms(query))
        scored = []
        for row, counts, length in zip(self.chunks, self.counts, self.lengths):
            score = 0.0
            for token in query_terms & counts.keys():
                frequency = counts[token]
                idf = math.log(1 + (len(self.chunks) - self.df[token] + 0.5) / (self.df[token] + 0.5))
                score += idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / self.avg_length))
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        unique = []
        seen = set()
        for score, row in scored:
            if row["source_id"] in seen:
                continue
            seen.add(row["source_id"])
            unique.append(dict(row, score=round(score, 3)))
            if len(unique) == limit:
                break
        return unique


class E5Index:
    """Multilingual semantic search on CPU; keeps KoRMo's GPU memory free."""

    def __init__(self, chunks: list[dict], model_name: str = "intfloat/multilingual-e5-small", cache_path: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.chunks = chunks
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to("cpu").eval()
        fingerprint = hashlib.sha256(json.dumps([(row["chunk_id"], row["text"]) for row in chunks], ensure_ascii=False).encode()).hexdigest()
        cache = Path(cache_path) if cache_path else None
        if cache:
            require_local_dataset(str(cache))
        saved = None
        if cache and cache.exists():
            saved = torch.load(cache, map_location="cpu", weights_only=True)
        if saved and saved.get("model") == model_name and saved.get("fingerprint") == fingerprint:
            self.vectors = saved["vectors"]
        else:
            texts = ["passage: " + row["title"] + ". " + row["text"] for row in chunks]
            self.vectors = torch.cat([self._encode(texts[i:i + 12]) for i in range(0, len(texts), 12)])
            if cache:
                torch.save({"model": model_name, "fingerprint": fingerprint, "vectors": self.vectors}, cache)

    def _encode(self, texts: list[str]):
        torch = self.torch
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.inference_mode():
            states = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            vectors = (states * mask).sum(dim=1) / mask.sum(dim=1)
            return torch.nn.functional.normalize(vectors, p=2, dim=1)

    def search(self, query: str, limit: int = 3) -> list[dict]:
        query_vector = self._encode(["query: " + query])[0]
        scores = (self.vectors @ query_vector).tolist()
        selected = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        unique = []
        seen = set()
        for index in selected:
            row = self.chunks[index]
            if row["source_id"] in seen:
                continue
            seen.add(row["source_id"])
            unique.append(dict(row, score=round(scores[index], 3)))
            if len(unique) == limit:
                break
        return unique


def evidence_for_prompt(chunks: list[dict]) -> list[dict]:
    return [{key: row[key] for key in ("source_id", "title", "url", "license", "text")} for row in chunks]


def guide_source_matches(question: str, row: dict) -> bool:
    """Screen obvious off-topic Easy Law hits for Korean student questions.

    Route clearly named English topics to the matching official guide too.
    """
    source_id = row.get("source_id", "")
    if not source_id.startswith("easylaw:"):
        return True
    english = question.lower()
    if re.search(r"\b(lease|rental|renting|landlord|tenant|housing contract)\b", english):
        return source_id.startswith("easylaw:629:")
    if re.search(r"\b(bus|subway|transit|transportation|fare card)\b", english):
        return source_id.startswith("easylaw:661:")
    if not re.search(r"[가-힣]", question):
        return True
    if source_id.startswith("easylaw:661:"):
        transit_terms = ("버스", "지하철", "대중교통", "교통카드", "승차권", "부정승차")
        return any(term in question and term in row["text"] for term in transit_terms)
    title_parts = row["title"].split("·")
    topic = title_parts[1].strip() if len(title_parts) >= 3 else row["title"]
    return len(set(terms(question)) & set(terms(topic))) >= 2


def search_guide(index, question: str, situation: str = "") -> list[dict]:
    """Retrieve one relevant Easy Law passage without generic student context bias."""
    if re.search(r"\b(lease|rental|renting)\b", question.lower()) and re.search(
        r"\b(before|sign|signing|check)\b", question.lower()
    ):
        # This section starts with what to verify before a rental contract.
        lead = next((row for row in index.chunks
                     if row["source_id"].startswith("easylaw:629:")
                     and "임대차계약의 당사자 확인" in row["title"]
                     and row["chunk_id"].endswith(":0")), None)
        if lead:
            return [lead]
    query = question if re.search(r"\b(lease|rental|renting|landlord|tenant|bus|subway|transit|transportation)\b",
                                  question.lower()) else question + " " + situation
    found = index.search(query, limit=len(index.chunks))
    return [row for row in found if guide_source_matches(question, row)][:1]


def source_footer(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    lines = ["\n\nSources / 자료 출처:"]
    for row in chunks:
        date = f" · 자료 날짜: {row['updated_at']}" if row.get("updated_at") else ""
        lines.append(f"[{row['title']}]({row['url']}) · {row['license']}{date}")
    return "\n".join(lines)
