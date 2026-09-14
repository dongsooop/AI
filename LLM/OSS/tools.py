from dataclasses import dataclass
from typing import Optional
import logging
from urllib.parse import urlparse

from core.settings import get_settings
from LLM.OSS.formatter import (
    dept_clarification_message,
    professor_room_clarification_message,
    render_chatty_schedule,
)
from LLM.OSS.modes import ambiguous_academic_topic, is_academic_procedure_query, is_ambiguous_graduation_query, looks_like_schedule, looks_like_topic
from LLM.OSS.postprocess import run_postprocess
from LLM.OSS.support_guidance import support_guidance
from LLM.sub_model.query_index import build_answer, confident_search_answer, metadata_direct_answer
from LLM.sub_model.schedule_index import schedule_search


settings = get_settings()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolResult:
    name: str
    text: str = ""
    url: Optional[str] = None
    engine: str = "fast"
    confidence: float = 0.0
    decision_source: str = "rule"
    source_urls: tuple[str, ...] = ()
    llm_required: bool = False
    reason: str = ""

    @property
    def resolved(self) -> bool:
        return bool(self.text.strip()) and not self.llm_required

    @property
    def handled(self) -> bool:
        return self.name != "none" and not self.llm_required

    def to_response(self) -> dict:
        response = {"engine": self.engine, "text": self.text}
        if self.url:
            response["url"] = self.url
        return response


EMPTY_TOOL_RESULT = ToolResult(
    name="none",
    confidence=0.0,
    decision_source="none",
    reason="no tool matched",
)


def _source_urls(url: Optional[str]) -> tuple[str, ...]:
    return (url,) if url else ()


def call_submodel(user_text: str) -> str:
    base = ""
    try:
        result = build_answer(user_text, top_k=12)
        base = (result or {}).get("answer", "").strip()
    except Exception as exc:
        logger.warning("build_answer failed: %s", exc, exc_info=True)
        base = ""

    if settings.json_only_mode:
        return base

    try:
        schedule = schedule_search(user_text, top_k=8)
    except Exception as exc:
        logger.warning("schedule_search failed: %s", exc, exc_info=True)
        schedule = ""

    if schedule and base:
        return schedule + "\n" + base
    return schedule or base


def _direct_answer_tool(user_text: str, engine: str) -> ToolResult:
    try:
        direct = metadata_direct_answer(user_text)
    except Exception:
        return EMPTY_TOOL_RESULT
    if not direct:
        return EMPTY_TOOL_RESULT
    needs_clarification = bool(direct.get("needs_clarification"))
    return ToolResult(
        name="graduation_scope_clarification" if needs_clarification else "metadata_direct_answer",
        text=direct["answer"],
        url=direct.get("url"),
        engine=engine,
        confidence=0.95,
        decision_source="rule" if needs_clarification else "retrieval",
        source_urls=_source_urls(direct.get("url")),
        reason="graduation applicability needs clarification" if needs_clarification else "metadata direct answer matched",
    )


def _schedule_tool(user_text: str, *, ceremonial_first: bool = False) -> ToolResult:
    if is_academic_procedure_query(user_text):
        return EMPTY_TOOL_RESULT
    if ceremonial_first and not any(keyword in user_text for keyword in ("종강", "졸업식", "종업식", "학위수여식")):
        return EMPTY_TOOL_RESULT

    try:
        schedule = schedule_search(user_text, top_k=8)
    except Exception:
        return EMPTY_TOOL_RESULT
    if not schedule:
        return EMPTY_TOOL_RESULT
    return ToolResult(
        name="schedule_search",
        text=render_chatty_schedule(schedule, user_text),
        engine="fast",
        confidence=0.9,
        decision_source="retrieval",
        reason="schedule intent matched",
    )


def _clarification_tool(user_text: str) -> ToolResult:
    try:
        clarification = dept_clarification_message(user_text)
    except Exception:
        return EMPTY_TOOL_RESULT
    if not clarification:
        return EMPTY_TOOL_RESULT
    return ToolResult(
        name="dept_clarification",
        text=clarification,
        engine="fast",
        confidence=0.85,
        reason="department query needs clarification",
    )


def _professor_room_clarification_tool(user_text: str) -> ToolResult:
    try:
        clarification = professor_room_clarification_message(user_text)
    except Exception:
        return EMPTY_TOOL_RESULT
    if not clarification:
        return EMPTY_TOOL_RESULT
    return ToolResult(
        name="professor_room_clarification",
        text=clarification,
        engine="fast",
        confidence=0.95,
        reason="professor room query needs professor or department clarification",
    )


def _postprocess_tool(mode: str, user_text: str) -> ToolResult:
    try:
        sub_answer = call_submodel(user_text)
        text, url = run_postprocess(mode, user_text, sub_answer)
    except Exception:
        return EMPTY_TOOL_RESULT
    return ToolResult(
        name=f"{mode}_search_postprocess",
        text=text,
        url=url,
        engine=mode,
        confidence=0.65 if text else 0.0,
        decision_source="retrieval",
        source_urls=_source_urls(url),
        reason=f"{mode} postprocess generated answer" if text else f"{mode} postprocess returned empty answer",
    )


def _confident_search_tool(user_text: str) -> ToolResult:
    try:
        confident = confident_search_answer(user_text, top_k=2)
    except Exception:
        return EMPTY_TOOL_RESULT
    if not confident:
        return EMPTY_TOOL_RESULT
    return ToolResult(
        name="confident_search_answer",
        text=confident["answer"],
        url=confident.get("url"),
        engine="fast",
        confidence=0.85,
        decision_source="retrieval",
        source_urls=_source_urls(confident.get("url")),
        reason="high confidence search answer matched",
    )


def run_support_guidance_tool(user_text: str) -> ToolResult:
    guidance = support_guidance(user_text)
    if guidance is None:
        return EMPTY_TOOL_RESULT
    topic, mode, page, message = guidance
    source = _postprocess_tool(mode, topic)
    # Only attach the matching source page returned by retrieval. A fallback
    # homepage or an unrelated result must not become evidence for this guide.
    parsed = urlparse(source.url or "")
    url = source.url if (parsed.scheme in ('http', 'https')
                        and parsed.hostname == 'www.dongyang.ac.kr'
                        and parsed.path == f'/dmu/{page}/subview.do') else None
    if url:
        message += f"\n공식 안내: {url}"
    else:
        message += "\n관련 공식 안내 링크를 현재 검색 결과에서 확인하지 못했어요."
    return ToolResult(name="support_topic_guidance", text=message, url=url,
                      engine=mode, confidence=0.8, source_urls=_source_urls(url),
                      reason="support query needs topic or applicability selection")


def run_academic_clarification_tool(user_text: str) -> ToolResult:
    topic = ambiguous_academic_topic(user_text)
    if not topic:
        return EMPTY_TOOL_RESULT
    examples = {
        "등록": "'등록금 납부 방법', '등록금 분할납부 방법', '등록금 납부 기간'",
        "등록금": "'등록금 납부 방법', '등록금 분할납부 방법', '등록금 납부 기간'",
        "성적": "'성적열람 기간', '성적 이의신청 방법', '성적증명서 발급'",
        "수강신청": "'수강신청 방법', '수강신청 기간', '수강정정 방법'",
    }
    return ToolResult(
        name="academic_topic_clarification",
        text=f"{topic} 관련해서 어떤 내용이 궁금한가요? {examples[topic]}처럼 구체적으로 입력해 주세요.",
        engine="policy",
        confidence=0.95,
        reason="academic query needs a specific purpose",
    )


def run_graduation_clarification_tool(user_text: str) -> ToolResult:
    if not is_ambiguous_graduation_query(user_text):
        return EMPTY_TOOL_RESULT
    return ToolResult(
        name="graduation_clarification",
        text=(
            "졸업 관련해서 어떤 내용이 궁금한가요? "
            "'졸업학점', '졸업식 일정', '졸업유예', '졸업보류'처럼 구체적으로 입력해 주세요."
        ),
        engine="grad",
        confidence=0.95,
        reason="graduation query needs a specific topic",
    )


def run_mode_tools(mode: str, user_text: str) -> ToolResult:
    if mode == "fast" and is_academic_procedure_query(user_text):
        mode = "policy"
    professor_room_clarification = _professor_room_clarification_tool(user_text)
    if professor_room_clarification.resolved:
        return professor_room_clarification

    if mode == "fast":
        for tool in (
            lambda: _schedule_tool(user_text, ceremonial_first=True),
            lambda: _direct_answer_tool(user_text, "fast"),
            lambda: _schedule_tool(user_text),
            lambda: _clarification_tool(user_text),
        ):
            tool_result = tool()
            if tool_result.resolved:
                return tool_result
        return _postprocess_tool("fast", user_text)

    if mode in {"policy", "grad"}:
        direct = _direct_answer_tool(user_text, mode)
        if direct.resolved:
            return direct
        return _postprocess_tool(mode, user_text)

    if mode in {"dorm", "topic"}:
        return _postprocess_tool(mode, user_text)

    return EMPTY_TOOL_RESULT


def run_oss_fast_path_tools(user_text: str) -> ToolResult:
    for tool in (
        lambda: _direct_answer_tool(user_text, "fast"),
        lambda: _confident_search_tool(user_text),
    ):
        tool_result = tool()
        if tool_result.resolved:
            return tool_result
    return EMPTY_TOOL_RESULT


def run_empty_oss_fallback_tools(user_text: str) -> ToolResult:
    if looks_like_schedule(user_text):
        schedule = _schedule_tool(user_text)
        if schedule.resolved:
            return schedule

    sub_answer = call_submodel(user_text)
    if looks_like_schedule(user_text) and sub_answer:
        return ToolResult(
            name="schedule_submodel_fallback",
            text=render_chatty_schedule(sub_answer, user_text),
            engine="fast",
            confidence=0.7,
            decision_source="retrieval",
            reason="schedule-like query recovered from rag context",
        )
    if sub_answer:
        mode = "topic" if looks_like_topic(user_text) else "fast"
        text, url = run_postprocess(mode, user_text, sub_answer)
        return ToolResult(
            name=f"{mode}_oss_empty_fallback",
            text=text,
            url=url,
            engine="oss",
            confidence=0.6 if text else 0.0,
            decision_source="retrieval",
            source_urls=_source_urls(url),
            reason="oss returned empty answer; recovered from rag context" if text else "oss returned empty answer; rag fallback empty",
        )
    return EMPTY_TOOL_RESULT


def run_final_fallback_tools(mode: str, user_text: str) -> ToolResult:
    for tool in (
        lambda: _direct_answer_tool(user_text, "fast"),
        lambda: _confident_search_tool(user_text),
    ):
        tool_result = tool()
        if tool_result.resolved:
            return tool_result
    return ToolResult(
        name="rag_context_required",
        text=call_submodel(user_text),
        engine=mode,
        confidence=0.5,
        decision_source="retrieval",
        llm_required=True,
        reason="no deterministic tool matched; rag context collected for grounded llm answer",
    )
