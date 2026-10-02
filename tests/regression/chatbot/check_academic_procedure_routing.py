"""Keep academic procedures out of calendar tools, including fast overrides."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from check_chatbot_tool_routing import install_fake_query_index, install_fake_schedule_index

install_fake_query_index()
install_fake_schedule_index()
from LLM.OSS import tools
from LLM.OSS.modes import decide_mode, looks_like_schedule


class AcademicProcedureRoutingTests(unittest.TestCase):
    def test_procedures_and_spelling_variants(self):
        for query in ('등록금 납부 방법', '등록금 분할납부 절차', '등록금 어떻게 내요',
                      '성적 증명서 발급', '수강 신청 방법', '수강정정 하는 법',
                      '수강신청 조건', '수강신청 기간과 방법', '성적증명서 발급 언제 가능해?'):
            with self.subTest(query=query):
                self.assertEqual(decide_mode(query), 'policy')
                self.assertFalse(looks_like_schedule(query))

    def test_calendar_queries_remain_calendar_queries(self):
        for query in ('수강신청 언제야?', '성적열람 기간', '등록금 납부 기간',
                      '수강정정 일정', '졸업식 일정'):
            self.assertEqual(decide_mode(query), 'fast')
            self.assertTrue(looks_like_schedule(query))

    def test_rule_book_and_contacts_keep_priority(self):
        self.assertEqual(decide_mode('수강신청 규정'), 'rule_book')
        self.assertEqual(decide_mode('성적증명서 발급 담당자 전화번호'), 'fast')

    def test_fast_override_cannot_call_calendar_for_procedures(self):
        answer = tools.ToolResult(name='policy_search_postprocess', text='공식 안내', engine='policy')
        with patch.object(tools, 'schedule_search') as schedule, \
             patch.object(tools, 'metadata_direct_answer', return_value=None), \
             patch.object(tools, '_postprocess_tool', return_value=answer) as policy:
            query = '성적증명서 발급'
            self.assertEqual(tools.run_mode_tools('fast', query), answer)
            policy.assert_called_once_with('policy', query)
            self.assertFalse(tools._schedule_tool(query).resolved)
            schedule.assert_not_called()


if __name__ == '__main__':
    unittest.main()
