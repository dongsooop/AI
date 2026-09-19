"""Missing fields and actual calendar filtering for supplied conditions."""
import asyncio
import datetime as dt
import os
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch
import pandas as pd
import check_graduation_clarification as fixture
from LLM.OSS.query_conditions import calendar_conditions
from LLM.OSS.support_guidance import support_guidance

os.environ.setdefault('SCHEDULE_CSV_PATH', str(Path(__file__).resolve().parents[3] / 'data/schedule/학사일정_년도추가.csv'))
from LLM.sub_model import schedule_index


class QueryConditionsTests(unittest.TestCase):
    def test_missing_fields_only(self):
        self.assertEqual(calendar_conditions('2026년 2학기 수강신청 언제야?')[1],
                         ['학년', '학생 유형(신입생·재학생·복학생·편입생·전공심화)'])
        self.assertEqual(calendar_conditions('2026년 2학기 성적열람 기간')[1], [])
        self.assertEqual(calendar_conditions('2026년 2학기 재학생 등록금 납부 기간')[1], [])
        self.assertIsNone(calendar_conditions('수강신청 방법'))
        for suffix in ('조건', '자격', '준비서류', '필요서류', '하는 법'):
            self.assertIsNone(calendar_conditions('2026년 2학기 수강신청 ' + suffix))
        self.assertIsNone(calendar_conditions('2026년 전체 수강신청 일정'))

    def test_support_does_not_repeat_known_fields(self):
        for query, wanted, forbidden in [
            ('2026년 국가장학금 신청 조건', '대상 학기', '대상 연도'),
            ('2026년 2학기 기숙사 비용', '실 유형', '이용 연도'),
            ('2026년 2학기 일반휴학 등록금 처리', '납부 여부', '휴학 종류')]:
            message = support_guidance(query)[3]
            self.assertIn(wanted, message)
            self.assertNotIn(forbidden, message)
        self.assertIsNone(support_guidance('2026년 2학기 국가장학금 신청 조건'))
        self.assertIsNone(support_guidance('일반휴학 신청 방법'))

    def test_named_scholarships_are_not_limited_to_allowlist(self):
        for name in ('봉사장학금', '다문화가족 장학금', '튜터링장학금', '성적향상 장학금'):
            with self.subTest(name=name):
                self.assertIsNone(support_guidance(f'2026년 2학기 {name} 신청 조건'))
                missing = support_guidance(f'2026년 {name} 신청 조건')[3]
                self.assertIn('대상 학기', missing)
                self.assertNotIn('장학금 이름', missing)
        for q in ('2026년2학기교내장학금조건', '2026년 2학기 교외 장학금 자격'):
            self.assertIn('구체적인 장학금 이름', support_guidance(q)[3])

    def test_service_clarifies_before_retrieval_and_cache(self):
        service = fixture.service
        with patch.object(service, 'should_block_profanity', AsyncMock(return_value=False)), \
             patch.object(service, 'log_chatbot'), patch.object(service, '_log_chatbot_summary'), \
             patch.object(service, 'run_mode_tools') as route:
            response = asyncio.run(service.chat_with_oss(service.ChatReq(text='2026년 2학기 수강신청 언제야?', engine='oss')))
        self.assertIn('학년', response['text'])
        self.assertNotIn('url', response)
        route.assert_not_called()

    def test_supplied_conditions_filter_actual_calendar_rows(self):
        titles = ['1학기 수강신청(재학생) : 2, 3학년',
                  '2학기 수강신청(재학생) : 2, 3학년',
                  '1학기 수강신청(재학생) : 1학년',
                  '1학기 수강신청(신입생) : 2학년', '수강신청']
        rows = [dict(학년도=2026, 일정명=title, start_date=pd.Timestamp(2026, 2, i + 1),
                     end_date=pd.Timestamp(2026, 2, i + 1), tags=schedule_index._tag_title(title))
                for i, title in enumerate(titles)]
        with patch.object(schedule_index, '_DF', pd.DataFrame(rows)):
            answer = schedule_index.schedule_search('2026년 1학기 2학년 재학생 수강신청 기간', today=dt.date(2026, 1, 1))
            self.assertIn('2026-02-01', answer)
            for day in ('02', '03', '04', '05'):
                self.assertNotIn('2026-02-' + day, answer)
            unknown = schedule_index.schedule_search('2026년 1학기 4학년 재학생 수강신청 기간')
            self.assertIn('확인하지 못했어요', unknown)
            self.assertNotRegex(unknown, r'\d{4}-\d{2}-\d{2}')


if __name__ == '__main__':
    unittest.main()
