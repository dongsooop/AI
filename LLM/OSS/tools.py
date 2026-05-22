from dataclasses import dataclass
from typing import Optional

from core.settings import get_settings
from LLM.OSS.formatter import (
    dept_clarification_message,
    render_chatty_schedule,
)
from LLM.OSS.modes import looks_like_schedule, looks_like_topic
from LLM.OSS.postprocess import run_postprocess
from LLM.sub_model.query_index import build_answer, confident_search_answer, metadata_direct_answer
from LLM.sub_model.schedule_index import schedule_search


settings = get_settings()


@dataclass(frozen=True)
class ToolResult:
    name: str
    text: str = ""
    url: Optional[str] = None
    engine: str = "fast"
    confidence: float = 0.0
    llm_required: bool = False

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


EMPTY_TOOL_RESULT = ToolResult(name="none", confidence=0.0)


def call_submodel(user_text: str) -> str:
    base = ""
    try:
        result = build_answer(user_text, top_k=12)
        base = (result or {}).get("answer", "").strip()
    except Exception:
        base = ""

    if settings.json_only_mode:
        return base

    try:
        schedule = schedule_search(user_text, top_k=8)
    except Exception:
        schedule = ""

    if schedule and base:
        return schedule + "\n" + base
    return schedule or base


def _direct_answer_tool(user_text: str, engine: str) -> ToolResult:
    direct = metadata_direct_answer(user_text)
    if not direct:
        return EMPTY_TOOL_RESULT
    return ToolResult(
        name="metadata_direct_answer",
        text=direct["answer"],
        url=direct.get("url"),
        engine=engine,
        confidence=0.95,
    )


def _schedule_tool(user_text: str, *, ceremonial_first: bool = False) -> ToolResult:
    if ceremonial_first and not any(keyword in user_text for keyword in ("종강", "졸업식", "종업식", "학위수여식")):
        return EMPTY_TOOL_RESULT

    schedule = schedule_search(user_text, top_k=8)
    if not schedule:
        return EMPTY_TOOL_RESULT
    return ToolResult(
        name="schedule_search",
        text=render_chatty_schedule(schedule, user_text),
        engine="fast",
        confidence=0.9,
    )


def _clarification_tool(user_text: str) -> ToolResult:
    clarification = dept_clarification_message(user_text)
    if not clarification:
        return EMPTY_TOOL_RESULT
    return ToolResult(
        name="dept_clarification",
        text=clarification,
        engine="fast",
        confidence=0.85,
    )


def _postprocess_tool(mode: str, user_text: str) -> ToolResult:
    sub_answer = call_submodel(user_text)
    text, url = run_postprocess(mode, user_text, sub_answer)
    return ToolResult(
        name=f"{mode}_search_postprocess",
        text=text,
        url=url,
        engine=mode,
        confidence=0.65 if text else 0.0,
    )


def _confident_search_tool(user_text: str) -> ToolResult:
    confident = confident_search_answer(user_text, top_k=2)
    if not confident:
        return EMPTY_TOOL_RESULT
    return ToolResult(
        name="confident_search_answer",
        text=confident["answer"],
        url=confident.get("url"),
        engine="fast",
        confidence=0.85,
    )


def run_mode_tools(mode: str, user_text: str) -> ToolResult:
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
        )
    if sub_answer:
        mode = "topic" if looks_like_topic(user_text) else "fast"
        text, _ = run_postprocess(mode, user_text, sub_answer)
        return ToolResult(
            name=f"{mode}_oss_empty_fallback",
            text=text,
            engine="oss",
            confidence=0.6 if text else 0.0,
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
        llm_required=True,
    )
