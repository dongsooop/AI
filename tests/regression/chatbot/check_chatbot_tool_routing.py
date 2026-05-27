#!/usr/bin/env python3
import importlib
import json
import sys
import types
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def install_fake_query_index() -> None:
    module = types.ModuleType("LLM.sub_model.query_index")

    def build_answer(query: str, top_k: int = 12) -> dict:
        if "애매" in query:
            return {"answer": "검색 근거: 학교 생활 안내 페이지에 관련 정보를 확인해야 합니다."}
        return {"answer": ""}

    def confident_search_answer(query: str, top_k: int = 2) -> dict | None:
        if "전화번호" in query:
            return {
                "answer": "학생성공지원팀 전화번호는 02-2610-1234입니다.",
                "url": "https://www.dongyang.ac.kr/contact",
            }
        return None

    def metadata_direct_answer(query: str) -> dict | None:
        if "졸업학점" in query:
            return {
                "answer": "3년제 졸업이수 학점은 총 120학점입니다.",
                "url": "https://www.dongyang.ac.kr/grad",
            }
        return None

    module.build_answer = build_answer
    module.confident_search_answer = confident_search_answer
    module.metadata_direct_answer = metadata_direct_answer
    sys.modules["LLM.sub_model.query_index"] = module


def install_fake_schedule_index() -> None:
    module = types.ModuleType("LLM.sub_model.schedule_index")

    def schedule_search(query: str, top_k: int = 8) -> str:
        if "개강" in query:
            return "2026-03-02 개강"
        return ""

    module.schedule_search = schedule_search
    sys.modules["LLM.sub_model.schedule_index"] = module


def check_tool_routing() -> list[str]:
    errors = []
    install_fake_query_index()
    install_fake_schedule_index()
    sys.modules.pop("LLM.OSS.tools", None)
    tools = importlib.import_module("LLM.OSS.tools")

    schedule = tools.run_mode_tools("fast", "개강 언제야?")
    if schedule.name != "schedule_search" or not schedule.resolved:
        errors.append(f"schedule_route_failed:{schedule}")
    if not schedule.reason:
        errors.append("schedule_reason_missing")

    contact = tools.run_oss_fast_path_tools("학생성공지원팀 전화번호 알려줘")
    if contact.name != "confident_search_answer" or not contact.resolved:
        errors.append(f"contact_route_failed:{contact}")
    if not contact.reason:
        errors.append("contact_reason_missing")

    fallback = tools.run_final_fallback_tools("oss", "애매한 학교 생활 질문")
    if fallback.name != "rag_context_required" or not fallback.llm_required:
        errors.append(f"fallback_route_failed:{fallback}")
    if not fallback.text.strip():
        errors.append("fallback_context_missing")
    if "rag context" not in fallback.reason:
        errors.append(f"fallback_reason_unexpected:{fallback.reason}")

    return errors


def main() -> int:
    errors = check_tool_routing()
    print(json.dumps({"ok": not errors, "errors": errors}, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
