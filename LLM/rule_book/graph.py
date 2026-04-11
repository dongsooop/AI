import os
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
        chunks = index.search(state["query"], top_k=TOP_K)
        return {**state, "chunks": chunks, "error": None}
    except Exception as e:
        logger.exception("규정집 검색 중 오류 발생: %s", e)
        return {**state, "chunks": [], "error": "규정집 검색 중 오류가 발생했습니다."}


async def generate(state: RuleState) -> RuleState:
    if state.get("error") and not state.get("chunks"):
        return {**state, "answer": "규정집 검색 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."}

    chunks = state.get("chunks", [])
    if not chunks:
        return {**state, "answer": "관련 규정을 찾지 못했습니다. 더 구체적인 키워드로 질문해 주세요."}

    context_parts = []
    for c in chunks:
        context_parts.append(f"[출처: {c['source']} {c['article']}]\n{c['text']}")
    context = "\n\n".join(context_parts)

    system_prompt = (
        "당신은 동양미래대학교 규정집 전문 안내 챗봇입니다.\n"
        "아래 <규정집 내용>만을 근거로 질문에 답하세요.\n"
        "답변은 간결하고 명확하게, 관련 조문 번호와 출처를 함께 안내하세요.\n"
        "규정집에 없는 내용은 '해당 규정을 찾을 수 없습니다'라고 답하세요.\n"
        "임의로 정보를 만들지 마세요."
    )
    user_prompt = f"질문: {state['query']}\n\n<규정집 내용>\n{context}\n</규정집 내용>"

    try:
        resp = await _async_client.chat.completions.create(
            model=OSS_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        answer = (resp.choices[0].message.content or "").strip()
        if not answer:
            answer = "답변을 생성하지 못했습니다. 다시 시도해 주세요."
        return {**state, "answer": answer}
    except Exception as e:
        logger.exception("LLM 답변 생성 중 오류 발생: %s", e)
        return {**state, "answer": "답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."}


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
