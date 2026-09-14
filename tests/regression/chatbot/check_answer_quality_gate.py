#!/usr/bin/env python3
import asyncio
import contextlib
import io
import json
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from run_chatbot_regression import check_case, load_cases
import evaluate_chatbot_answers as evaluator

URL = 'https://www.dongyang.ac.kr/dmu/4778/subview.do'
CASE = dict(id='correct_scope', text='졸업학점', behavior='answer', engine_in=['grad'],
            all_of_text_contains=['일반학생', '3년제', '총 졸업학점 110학점', '전공최저이수 78학점'],
            none_of_text_contains=['자유전공', '연락처'], allowed_credit_values=[110, 78],
            url_policy='required', expected_urls=[URL], single_source=True)
RESPONSE = dict(engine='grad', text=f'일반학생 3년제 총 졸업학점 110학점 / 전공최저이수 78학점 (출처: {URL})', url=URL)


class AnswerQualityGateTests(unittest.TestCase):
    def test_correct_scope_and_layout_query(self):
        response = {**RESPONSE, 'url': URL + '?layout=unknown'}
        self.assertTrue(check_case(CASE, 200, response)['passed'])

    def test_wrong_facts_and_contamination_fail(self):
        for old, new in [('110학점', '120학점'), ('3년제', '2년제'), ('일반학생', '자유전공 학생')]:
            with self.subTest(new=new):
                self.assertFalse(check_case(CASE, 200, {**RESPONSE, 'text': RESPONSE['text'].replace(old, new)})['passed'])
        self.assertFalse(check_case(CASE, 200, {**RESPONSE, 'text': RESPONSE['text'] + ' 추가 999학점'})['passed'])

    def test_screenshot_mixed_source_fails(self):
        result = check_case(CASE, 200, {**RESPONSE, 'url': URL.replace('4778', '4781')})
        self.assertIn('body_button_source_mismatch', result['reasons'])

    def test_both_urls_changed_still_fail_expected_source(self):
        wrong = URL.replace('4778', '4781')
        result = check_case(CASE, 200, {**RESPONSE, 'text': RESPONSE['text'].replace(URL, wrong), 'url': wrong})
        self.assertIn('wrong_source_url', result['reasons'])

    def test_additional_body_source_fails(self):
        result = check_case(CASE, 200, {**RESPONSE, 'text': RESPONSE['text'] + ' https://example.com/other'})
        self.assertIn('body_button_source_mismatch', result['reasons'])

    def test_clarification_cannot_hide_a_url_or_credit_answer(self):
        case = dict(id='clarify', text='졸업', behavior='clarification', url_policy='forbidden',
                    all_of_text_contains=['어떤 내용'], none_of_text_regex=[r'\d+\s*학점'])
        valid = dict(engine='grad', text='어떤 내용이 궁금한가요?')
        self.assertTrue(check_case(case, 200, valid)['passed'])
        for extra in ({'url': URL}, {'text': valid['text'] + ' 110 학점입니다.'}):
            self.assertFalse(check_case(case, 200, {**valid, **extra})['passed'])

    def test_bad_payloads_and_http_errors_fail(self):
        for payload in ([], None, {}, {'text': None}, {'text': '   '}, {**RESPONSE, 'url': 'https://['}):
            self.assertFalse(check_case(CASE, 200, payload)['passed'])
        self.assertFalse(check_case(CASE, 503, RESPONSE)['passed'])

    def test_only_layout_is_ignored_in_url_comparison(self):
        for suffix in ('?year=2020', ';different', '?year='):
            self.assertIn('wrong_source_url', check_case(CASE, 200, {**RESPONSE, 'url': URL + suffix})['reasons'])

    def test_case_schema_rejects_empty_duplicate_and_invalid_regex(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cases.json'
            for value in ([], [CASE, CASE], [{**CASE, 'none_of_text_regex': ['[']} ]):
                path.write_text(json.dumps(value))
                with self.assertRaises((ValueError, re.error)):
                    load_cases(path)

    def test_strict_mode_returns_failure_and_writes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cases.json'; path.write_text(json.dumps([CASE]))
            report = Path(directory) / 'report.json'
            bad = check_case(CASE, 200, {**RESPONSE, 'text': '졸업 링크만 제공'})
            with patch.object(sys, 'argv', ['gate', '--cases', str(path), '--out', str(report), '--strict']), \
                 patch.object(evaluator, 'evaluate', AsyncMock(return_value=[bad])), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evaluator.main(), 2)
            self.assertEqual(json.loads(report.read_text())['summary']['failed'], 1)

    def test_unexpected_llm_call_fails_even_if_service_recovers(self):
        import LLM.OSS
        fake = types.SimpleNamespace(_CACHE_GENERAL={}, _CACHE_RULE_BOOK={}, should_block_profanity=AsyncMock(),
                                     log_chatbot=Mock(), _log_chatbot_summary=Mock(), call_oss_async=AsyncMock(),
                                     ChatReq=lambda **kwargs: kwargs)
        async def chat(req):
            try:
                await fake.call_oss_async([])
            except RuntimeError:
                pass
            return RESPONSE
        fake.chat_with_oss = chat
        with patch.object(LLM.OSS, 'service', fake, create=True):
            result = asyncio.run(evaluator.evaluate([CASE]))[0]
        self.assertFalse(result['passed'])
        self.assertIn('unexpected_llm_call', result['reasons'])


if __name__ == '__main__':
    unittest.main()
