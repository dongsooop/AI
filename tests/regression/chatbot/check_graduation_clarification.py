#!/usr/bin/env python3
"""Exercise graduation clarification through the service with external I/O mocked."""
import asyncio
import hashlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))


def fake_module(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


retrieval = Mock(return_value=None)
with patch.dict(sys.modules, {
    "LLM.sub_model.query_index": fake_module(
        "LLM.sub_model.query_index", build_answer=retrieval,
        confident_search_answer=retrieval, metadata_direct_answer=retrieval,
    ),
    "LLM.sub_model.schedule_index": fake_module("LLM.sub_model.schedule_index", schedule_search=retrieval),
    "LLM.OSS.chat_log_store": fake_module("LLM.OSS.chat_log_store", log_chatbot=Mock()),
    "LLM.OSS.llm_client": fake_module("LLM.OSS.llm_client", call_oss_async=AsyncMock()),
    "LLM.rule_book.graph": fake_module("LLM.rule_book.graph", run_rule_book=Mock()),
}):
    from LLM.OSS import service
    from LLM.OSS.modes import is_ambiguous_graduation_query
    from LLM.OSS.tools import ToolResult


class GraduationClarificationTests(unittest.TestCase):
    def setUp(self):
        retrieval.reset_mock()
        service._CACHE_GENERAL.clear()
        service._CACHE_RULE_BOOK.clear()
        self.block = self.enterContext(patch.object(service, "should_block_profanity", AsyncMock(return_value=False)))
        self.summary = self.enterContext(patch.object(service, "_log_chatbot_summary"))
        self.log = self.enterContext(patch.object(service, "log_chatbot"))
        self.llm = self.enterContext(patch.object(service, "call_oss_async", AsyncMock()))

    def request(self, **kwargs):
        return asyncio.run(service.chat_with_oss(service.ChatReq(**kwargs)))

    def assert_clarification(self, response):
        self.assertEqual(response["engine"], "grad")
        self.assertNotIn("url", response)
        for phrase in ("어떤 내용", "졸업학점", "졸업식 일정", "졸업유예", "졸업보류"):
            self.assertIn(phrase, response["text"])
        self.assertNotIn("https://", response["text"])
        retrieval.assert_not_called()
        self.llm.assert_not_called()

    def test_ambiguous_variants_use_clarification_without_search(self):
        for text in ("졸업", "졸업?", " 졸 업 ", "졸업 알려줘", "졸업에 대해 알려주세요",
                     "졸업 관련 정보", "졸업 안내해 주세요", "졸업이 궁금해요", "졸업 좀 알려줄래요?"):
            with self.subTest(text=text):
                self.assert_clarification(self.request(text=text))

    def test_specific_queries_are_not_overblocked(self):
        for text in ("졸업학점 알려줘", "졸업 요건", "졸업식 일정", "졸업유예", "졸업보류",
                     "졸업 담당자 전화번호", "컴퓨터공학부 졸업요건", "졸업증명서 발급",
                     "졸업 안 해도 돼?", "졸업생 취업률", "휴학", "장학금", "안녕", ""):
            with self.subTest(text=text):
                self.assertFalse(is_ambiguous_graduation_query(text))

    def test_concrete_menu_inputs_reach_existing_routes(self):
        for text, mode in (("졸업학점", "grad"), ("졸업식 일정", "fast"),
                           ("졸업유예", "grad"), ("졸업보류", "grad")):
            with self.subTest(text=text), patch.object(service, "run_mode_tools", return_value=ToolResult(
                name="existing_route", text="기존 경로 응답", engine=mode,
            )) as route:
                self.assertEqual(self.request(text=text)["text"], "기존 경로 응답")
                route.assert_called_once_with(mode, text)

    def test_old_cached_link_cannot_bypass_clarification(self):
        key = "grad:" + hashlib.sha256("졸업".encode()).hexdigest()
        service._CACHE_GENERAL[key] = {"engine": "grad", "text": "이전 졸업보류 안내", "url": "https://example.com"}
        self.assert_clarification(self.request(text="졸업"))

    def test_engine_override_cannot_turn_graduation_into_short_greeting(self):
        for engine in ("oss", "fast", "grad", "topic"):
            with self.subTest(engine=engine):
                self.assert_clarification(self.request(text="졸업", engine=engine))

    def test_messages_only_request(self):
        self.assert_clarification(self.request(messages=[{"role": "user", "content": "졸업"}]))

    def test_specific_followup_uses_latest_message(self):
        with patch.object(service, "run_mode_tools", return_value=ToolResult(
            name="existing_route", text="학점 안내", engine="grad",
        )) as route:
            response = self.request(messages=[
                {"role": "user", "content": "졸업"},
                {"role": "assistant", "content": "졸업 관련해서 어떤 내용이 궁금한가요?"},
                {"role": "user", "content": "졸업학점"},
            ])
            self.assertEqual(response["text"], "학점 안내")
            route.assert_called_once_with("grad", "졸업학점")

    def test_clarification_metadata_has_no_sources(self):
        self.request(text="졸업")
        routing = self.summary.call_args.args[3]
        self.assertEqual(routing.stage, "clarification")
        self.assertEqual(routing.tool, "graduation_clarification")
        self.assertEqual(routing.decision_source, "rule")
        self.assertEqual(routing.source_urls, ())
        self.assertFalse(routing.llm_required)
        self.log.assert_called_once()

    def test_guard_still_precedes_clarification(self):
        self.block.return_value = True
        self.assertEqual(self.request(text="졸업")["engine"], "guard")


if __name__ == "__main__":
    unittest.main()
