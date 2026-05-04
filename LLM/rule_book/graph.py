import asyncio
import os
import re
import time
import logging
from typing import TypedDict, List, Dict, Optional

from langgraph.graph import StateGraph, END
from openai import AsyncOpenAI

from LLM.rule_book.index import get_index
from LLM.rule_book.logger import log_rule_book

logger = logging.getLogger(__name__)

_async_client = AsyncOpenAI(
    base_url=os.getenv("OSS_BASE_URL"),
    api_key=os.getenv("OSS_API_KEY"),
)
OSS_MODEL = os.getenv("OSS_MODEL")

TOP_K = int(os.getenv("RULE_BOOK_TOP_K", "5"))
LLM_TOP_K = int(os.getenv("RULE_BOOK_LLM_TOP_K", "2"))
LLM_MAX_TOKENS = int(os.getenv("RULE_BOOK_LLM_MAX_TOKENS", "192"))
LLM_TIMEOUT_SECONDS = float(os.getenv("RULE_BOOK_LLM_TIMEOUT_SECONDS", "45"))
DIRECT_RULE_TERMS_RE = re.compile(r"(임기|기간|자격|권한|의무|선출|선임|구성|정족수|징계|사퇴|해임|소집)")
RULE_BOOK_NOISE_RE = re.compile(r"(규정집|규정|학칙|준칙|회칙|규약|세칙|강령|운영규칙|찾아줘|알려줘|에서)")
ARTICLE_HEADING_RE = re.compile(r"^제\s*\d+\s*조(?:의\d+)?(?:\([^)]*\))?$")

class RuleState(TypedDict):
    query: str
    chunks: List[Dict]
    answer: str
    error: Optional[str]


async def retrieve(state: RuleState) -> RuleState:
    index = get_index()
    if not index._built:
        return {**state, "chunks": [], "error": "인덱스가 아직 준비되지 않았습니다."}
    try:
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(None, lambda: index.search(state["query"], top_k=TOP_K))
        return {**state, "chunks": chunks, "error": None}
    except Exception as e:
        logger.exception("규정집 검색 중 오류 발생: %s", e)
        return {**state, "chunks": [], "error": "규정집 검색 중 오류가 발생했습니다."}


def _query_terms(query: str) -> list[str]:
    cleaned = RULE_BOOK_NOISE_RE.sub(" ", query or "")
    terms = [t for t in re.findall(r"[가-힣A-Za-z0-9]{2,}", cleaned) if len(t) >= 2]
    return list(dict.fromkeys(terms))


def _clean_text(text: str, max_chars: int = 360) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _split_rule_sentences(text: str) -> list[str]:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    candidates = []
    for line in lines:
        candidates.extend(part.strip() for part in re.split(r"(?<=[다요함음)])\s+", line) if part.strip())
    if not candidates:
        candidates = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text or "") if part.strip()]
    return candidates


def _source_label(chunk: Dict) -> str:
    source = (chunk.get("source") or "").strip()
    article = (chunk.get("article") or "").strip()
    return f"{source} {article}".strip()


def _direct_answer_from_chunks(query: str, chunks: List[Dict]) -> Optional[str]:
    if not chunks or not DIRECT_RULE_TERMS_RE.search(query or ""):
        return None

    terms = _query_terms(query)
    if not terms:
        return None

    def chunk_score(chunk: Dict) -> int:
        haystack = f"{chunk.get('source', '')} {chunk.get('article', '')} {chunk.get('text', '')}"
        return sum(1 for term in terms if term in haystack)

    ranked = sorted(chunks, key=chunk_score, reverse=True)
    for chunk in ranked[:3]:
        text = chunk.get("text", "")
        label = _source_label(chunk)
        sentences = _split_rule_sentences(text)

        scored_sentences = []
        for idx, sentence in enumerate(sentences):
            if ARTICLE_HEADING_RE.match(sentence) and idx + 1 < len(sentences):
                sentence = f"{sentence} {sentences[idx + 1]}"
            score = sum(1 for term in terms if term in sentence)
            if DIRECT_RULE_TERMS_RE.search(sentence):
                score += 2
            if score > 0:
                scored_sentences.append((score, sentence))

        if scored_sentences:
            scored_sentences.sort(key=lambda item: item[0], reverse=True)
            snippet = _clean_text(scored_sentences[0][1])
            return f"{label}에 따르면, {snippet} (출처: {label})"

        if chunk_score(chunk) >= max(1, min(2, len(terms))):
            snippet = _clean_text(text)
            return f"관련 조문은 {label}입니다. 확인된 내용: {snippet} (출처: {label})"

    return None


def _fallback_answer_from_chunks(chunks: List[Dict]) -> str:
    if not chunks:
        return "해당 규정을 찾을 수 없습니다."
    lines = []
    for chunk in chunks[:2]:
        label = _source_label(chunk)
        snippet = _clean_text(chunk.get("text", ""), max_chars=220)
        lines.append(f"- {label}: {snippet}")
    return "확인된 규정집 자료 기준으로 관련 조문은 다음과 같습니다.\n" + "\n".join(lines)


async def generate(state: RuleState) -> RuleState:
    if state.get("error") and not state.get("chunks"):
        return {**state, "answer": "규정집 검색 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."}

    chunks = state.get("chunks", [])
    if not chunks:
        return {**state, "answer": "해당 규정을 찾을 수 없습니다."}

    direct_answer = _direct_answer_from_chunks(state["query"], chunks)
    if direct_answer:
        return {**state, "answer": direct_answer}

    context_parts = []
    for c in chunks[:LLM_TOP_K]:
        text = c['text'][:450]
        context_parts.append(f"[출처: {c['source']} {c['article']}]\n{text}")
    context = "\n\n".join(context_parts)

    system_prompt = (
        "당신은 동양미래대학교 규정집 전문 안내 챗봇입니다.\n"
        "반드시 아래 <규정집 내용> 안의 텍스트만을 근거로 질문에 답하세요.\n"
        "<규정집 내용> 외의 정보는 절대 사용하지 마세요.\n"
        "답변은 간결하고 명확하게, 관련 조문 번호와 출처를 함께 안내하세요.\n"
        "<규정집 내용>에 질문과 관련된 내용이 없으면 '해당 규정을 찾을 수 없습니다'라고만 답하세요.\n"
        "추측하거나 임의로 정보를 만들지 마세요."
    )
    user_prompt = f"질문: {state['query']}\n\n<규정집 내용>\n{context}\n</규정집 내용>"

    try:
        resp = await asyncio.wait_for(
            _async_client.chat.completions.create(
                model=OSS_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=LLM_MAX_TOKENS,
            ),
            timeout=LLM_TIMEOUT_SECONDS,
        )
        choice = resp.choices[0]
        finish_reason = choice.finish_reason
        answer = (choice.message.content or "").strip()
        logger.info("LLM finish_reason=%s, answer_len=%d", finish_reason, len(answer))
        if not answer:
            logger.warning("LLM 빈 응답 반환 finish_reason=%s", finish_reason)
            answer = "답변을 생성하지 못했습니다. 다시 시도해 주세요."
        return {**state, "answer": answer}
    except Exception as e:
        logger.exception("LLM 답변 생성 중 오류 발생: %s", e)
        return {**state, "answer": _fallback_answer_from_chunks(chunks)}


def _build_graph():
    g = StateGraph(RuleState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


_graph = _build_graph()


async def run_rule_book(query: str) -> str:
    initial: RuleState = {
        "query": query,
        "chunks": [],
        "answer": "",
        "error": None,
    }
    start = time.time()
    result = await _graph.ainvoke(initial)
    elapsed_ms = int((time.time() - start) * 1000)

    await log_rule_book(
        query=query,
        chunks=result.get("chunks", []),
        answer=result.get("answer", ""),
        error=result.get("error"),
        elapsed_ms=elapsed_ms,
    )

    return result.get("answer", "답변을 생성하지 못했습니다.")
