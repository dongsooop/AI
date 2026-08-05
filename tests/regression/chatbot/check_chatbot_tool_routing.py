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
            return "- 개강: 2026-03-02"
        if "학사일정" in query:
            return "- 수강신청: 2026-02-10 ~ 2026-02-12"
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
    if schedule.decision_source != "retrieval":
        errors.append(f"schedule_decision_source_unexpected:{schedule.decision_source}")
    if schedule.source_urls:
        errors.append(f"schedule_sources_unexpected:{schedule.source_urls}")
    if not schedule.text.startswith("개강 일정\n"):
        errors.append(f"schedule_heading_unexpected:{schedule.text}")

    generic_schedule = tools.run_mode_tools("fast", "학사일정 알려줘")
    if not generic_schedule.text.startswith("학사일정 안내\n"):
        errors.append(f"generic_schedule_heading_unexpected:{generic_schedule.text}")

    professor_room = tools.run_mode_tools("oss", "교수연구실이 어디야?")
    if professor_room.name != "professor_room_clarification" or not professor_room.resolved:
        errors.append(f"professor_room_clarification_failed:{professor_room}")
    if "교수명이나 학과명" not in professor_room.text:
        errors.append(f"professor_room_clarification_text_unexpected:{professor_room.text}")
    if professor_room.decision_source != "rule":
        errors.append(f"professor_room_decision_source_unexpected:{professor_room.decision_source}")

    specific_professor_room = tools.run_mode_tools("oss", "홍길동 교수님 연구실이 어디야?")
    if specific_professor_room.name == "professor_room_clarification":
        errors.append(f"specific_professor_room_overblocked:{specific_professor_room}")

    contact = tools.run_oss_fast_path_tools("학생성공지원팀 전화번호 알려줘")
    if contact.name != "confident_search_answer" or not contact.resolved:
        errors.append(f"contact_route_failed:{contact}")
    if not contact.reason:
        errors.append("contact_reason_missing")
    if contact.decision_source != "retrieval":
        errors.append(f"contact_decision_source_unexpected:{contact.decision_source}")
    if contact.source_urls != ("https://www.dongyang.ac.kr/contact",):
        errors.append(f"contact_sources_unexpected:{contact.source_urls}")
    if contact.to_response().get("url") != "https://www.dongyang.ac.kr/contact":
        errors.append(f"contact_response_url_changed:{contact.to_response()}")

    direct = tools.run_mode_tools("fast", "졸업학점 알려줘")
    if direct.name != "metadata_direct_answer" or not direct.resolved:
        errors.append(f"direct_answer_route_failed:{direct}")
    if direct.decision_source != "retrieval":
        errors.append(f"direct_answer_decision_source_unexpected:{direct.decision_source}")
    if direct.source_urls != ("https://www.dongyang.ac.kr/grad",):
        errors.append(f"direct_answer_sources_unexpected:{direct.source_urls}")

    fallback = tools.run_final_fallback_tools("oss", "애매한 학교 생활 질문")
    if fallback.name != "rag_context_required" or not fallback.llm_required:
        errors.append(f"fallback_route_failed:{fallback}")
    if not fallback.text.strip():
        errors.append("fallback_context_missing")
    if "rag context" not in fallback.reason:
        errors.append(f"fallback_reason_unexpected:{fallback.reason}")
    if fallback.decision_source != "retrieval":
        errors.append(f"fallback_decision_source_unexpected:{fallback.decision_source}")
    if fallback.source_urls:
        errors.append(f"fallback_unverified_sources:{fallback.source_urls}")

    empty = tools.run_oss_fast_path_tools("출처 없는 질문")
    if empty.name != "none" or empty.decision_source != "none" or empty.source_urls:
        errors.append(f"empty_tool_metadata_unexpected:{empty}")

    return errors


def main() -> int:
    errors = check_tool_routing()
    print(json.dumps({"ok": not errors, "errors": errors}, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
