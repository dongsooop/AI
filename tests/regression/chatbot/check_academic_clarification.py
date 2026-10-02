"""Academic clarification through the service without external I/O."""
import asyncio
import hashlib
import unittest
from unittest.mock import AsyncMock, patch

import check_graduation_clarification as fixture
from LLM.OSS.modes import ambiguous_academic_topic, decide_mode

service = fixture.service


class AcademicClarificationTests(unittest.TestCase):
    def setUp(self):
        service._CACHE_GENERAL.clear()
        service._CACHE_RULE_BOOK.clear()
        self.block = self.enterContext(patch.object(service, 'should_block_profanity', AsyncMock(return_value=False)))
        self.log = self.enterContext(patch.object(service, 'log_chatbot'))
        self.summary = self.enterContext(patch.object(service, '_log_chatbot_summary'))
        self.llm = self.enterContext(patch.object(service, 'call_oss_async', AsyncMock()))
        self.route = self.enterContext(patch.object(service, 'run_mode_tools'))

    def request(self, **kwargs):
        return asyncio.run(service.chat_with_oss(service.ChatReq(**kwargs)))

    def test_broad_variants_return_choices_without_search(self):
        for text in ('등록', '등록금?', '성적 알려줘', '성적이 궁금해요',
                     '수강 신청', '수강신청 안내해 주세요', '등록에 대해서 알려주세요'):
            with self.subTest(text=text):
                response = self.request(text=text)
                self.assertEqual(response['engine'], 'policy')
                self.assertIn('어떤 내용', response['text'])
                self.assertNotIn('url', response)
                self.assertNotRegex(response['text'], r'\d|https?://')
                self.route.assert_not_called()
                self.llm.assert_not_called()
        routing = self.summary.call_args.args[3]
        self.assertEqual(routing.tool, 'academic_topic_clarification')
        self.assertEqual(routing.source_urls, ())
        self.assertEqual(routing.decision_source, 'rule')
        self.assertEqual(self.log.call_args.args[1], 'policy')

    def test_explicit_requests_and_followups_reach_existing_routes(self):
        for text in ('등록금 납부 방법', '2026년 1학기 재학생 등록금 납부 기간', '2026년 1학기 성적열람 기간',
                     '성적 이의신청 방법', '성적증명서 발급', '수강신청 방법',
                     '2026년 1학기 2학년 재학생 수강신청 기간', '수강정정 방법', '등록 담당자 연락처'):
            with self.subTest(text=text):
                self.assertFalse(ambiguous_academic_topic(text))
                self.route.reset_mock()
                self.route.return_value = fixture.ToolResult(name='existing_route', text='기존 안내', engine=decide_mode(text))
                response = self.request(messages=[{'role': 'user', 'content': '등록'},
                    {'role': 'assistant', 'content': '어떤 내용이 궁금한가요?'},
                    {'role': 'user', 'content': text}])
                self.assertEqual(response['text'], '기존 안내')
                self.route.assert_called_once_with(decide_mode(text), text)

    def test_cache_and_engine_override_cannot_bypass_clarification(self):
        for engine in ('fast', 'oss', 'policy'):
            key = engine + ':' + hashlib.sha256('등록'.encode()).hexdigest()
            service._CACHE_GENERAL[key] = {'engine': engine, 'text': '이전 일정'}
            self.assertIn('어떤 내용', self.request(text='등록', engine=engine)['text'])
        self.route.assert_not_called()

    def test_guard_still_precedes_clarification(self):
        self.block.return_value = True
        self.assertEqual(self.request(text='성적')['engine'], 'guard')
        self.route.assert_not_called()


if __name__ == '__main__':
    unittest.main()
