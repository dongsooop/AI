"""Exercise support guides, source binding and concrete followups."""
import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch
import check_graduation_clarification as fixture
from LLM.OSS.support_guidance import support_guidance

service = fixture.service


class SupportGuidanceTests(unittest.TestCase):
    def setUp(self):
        service._CACHE_GENERAL.clear()
        self.enterContext(patch.object(service, 'should_block_profanity', AsyncMock(return_value=False)))
        self.enterContext(patch.object(service, 'log_chatbot'))
        self.enterContext(patch.object(service, '_log_chatbot_summary'))
        self.llm = self.enterContext(patch.object(service, 'call_oss_async', AsyncMock()))

    def test_broad_and_missing_scope_queries(self):
        for query, phrase in [('장학금 알려줘', '교내'), ('장학금 신청 조건', '대상 학기'),
                              ('기숙사', '시설'), ('기숙사비 얼마야?', '실 유형'),
                              ('휴학 안내해 주세요', '일반휴학'), ('휴학 신청 방법', '희망 학기')]:
            with self.subTest(query=query):
                topic, mode, page, _ = support_guidance(query)
                url = f'https://www.dongyang.ac.kr/dmu/{page}/subview.do'
                with patch.dict(service.run_support_guidance_tool.__globals__, {'_postprocess_tool': Mock(return_value=fixture.ToolResult(name='source', url=url))}):
                    result = asyncio.run(service.chat_with_oss(service.ChatReq(text=query, engine='oss')))
                self.assertEqual(result['engine'], mode)
                self.assertIn(phrase, result['text'])
                self.assertEqual(result['url'], url)
                self.assertEqual(result['text'].count(url), 1)
                self.assertNotRegex(result['text'].split('\n')[0], r'\d')
        self.llm.assert_not_called()

    def test_unrelated_source_is_not_attached(self):
        for url in (None, 'https://www.dongyang.ac.kr/dmu/4778/subview.do',
                    'https://example.org/dmu/4794/subview.do'):
            with patch.dict(service.run_support_guidance_tool.__globals__, {'_postprocess_tool': Mock(return_value=fixture.ToolResult(name='source', url=url))}):
                result = service.run_support_guidance_tool('장학금')
            self.assertIsNone(result.url)
            self.assertEqual(result.source_urls, ())
            self.assertNotIn('https://', result.text)

    def test_concrete_queries_are_not_intercepted(self):
        for query in ('국가장학금 신청 조건', '교내 장학금 안내', '성적우수 장학금',
                      '기숙사 위치', '기숙사 시설', '2026년 2학기 기숙사 2인실 비용',
                      '일반휴학 신청 방법', '군휴학 필요서류', '휴학연기 안내',
                      '기숙사 담당자 연락처', '장학금과 휴학', '휴학 말고 복학'):
            self.assertIsNone(support_guidance(query), query)

    def test_specific_followup_reaches_existing_route(self):
        with patch.object(service, 'run_mode_tools', return_value=fixture.ToolResult(name='existing', text='기존 안내', engine='policy')) as route:
            result = asyncio.run(service.chat_with_oss(service.ChatReq(messages=[
                {'role': 'user', 'content': '휴학'},
                {'role': 'assistant', 'content': '휴학 종류를 알려주세요'},
                {'role': 'user', 'content': '일반휴학 신청 방법'}])))
        self.assertEqual(result['text'], '기존 안내')
        route.assert_called_once_with('policy', '일반휴학 신청 방법')


if __name__ == '__main__':
    unittest.main()
