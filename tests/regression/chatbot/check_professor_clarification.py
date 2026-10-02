"""Professor topic selection before greetings/cache without blocking lookups."""
import asyncio
import hashlib
import unittest
from unittest.mock import AsyncMock, patch
import check_graduation_clarification as fixture
from LLM.OSS.modes import is_ambiguous_professor_query, decide_mode

service = fixture.service


class ProfessorClarificationTests(unittest.TestCase):
    def setUp(self):
        service._CACHE_GENERAL.clear()
        self.block = self.enterContext(patch.object(service, 'should_block_profanity', AsyncMock(return_value=False)))
        self.enterContext(patch.object(service, 'log_chatbot'))
        self.summary = self.enterContext(patch.object(service, '_log_chatbot_summary'))
        self.llm = self.enterContext(patch.object(service, 'call_oss_async', AsyncMock()))
        self.route = self.enterContext(patch.object(service, 'run_mode_tools'))

    def request(self, **kwargs):
        return asyncio.run(service.chat_with_oss(service.ChatReq(**kwargs)))

    def test_neutral_variants_without_search_or_sources(self):
        for q in ('교수', '교수님?', '교수 알려줘', '교수님에 대해서 알려주세요', '교수 정보', '교수님 안내해 주세요'):
            with self.subTest(q=q):
                response = self.request(text=q)
                self.assertEqual(response['engine'], 'fast')
                for phrase in ('연락처', '연구실 위치', '교수소개', '교수명이나 학과명'):
                    self.assertIn(phrase, response['text'])
                self.assertNotIn('url', response)
                self.assertNotIn('https://', response['text'])
        self.route.assert_not_called()
        self.llm.assert_not_called()
        routing = self.summary.call_args.args[3]
        self.assertEqual(routing.tool, 'professor_topic_clarification')
        self.assertEqual(routing.source_urls, ())

    def test_specific_questions_and_followups_keep_existing_route(self):
        for q in ('교수연구실이 어디야?', '김가람 교수 연락처', '교수 채용',
                  '컴퓨터소프트웨어공학과 교수소개', '컴퓨터소프트웨어공학과 교수연구실 위치',
                  '컴퓨터소프트웨어공학과 교수 연락처'):
            self.assertFalse(is_ambiguous_professor_query(q))
            self.route.reset_mock()
            self.route.return_value = fixture.ToolResult(name='existing', text='기존 조회', engine=decide_mode(q))
            response = self.request(messages=[{'role': 'user', 'content': '교수'},
                {'role': 'assistant', 'content': '어떤 내용이 궁금한가요?'},
                {'role': 'user', 'content': q}])
            self.assertEqual(response['text'], '기존 조회')
            self.route.assert_called_once_with(decide_mode(q), q)

    def test_cache_and_override_do_not_bypass_prompt(self):
        for engine in ('oss', 'fast', 'topic'):
            service._CACHE_GENERAL[engine + ':' + hashlib.sha256('교수'.encode()).hexdigest()] = {'text': '이전 답변', 'engine': engine}
            self.assertIn('어떤 내용', self.request(text='교수', engine=engine)['text'])
        self.route.assert_not_called()

    def test_guard_precedes_prompt(self):
        self.block.return_value = True
        self.assertEqual(self.request(text='교수')['engine'], 'guard')
        self.route.assert_not_called()


if __name__ == '__main__':
    unittest.main()
