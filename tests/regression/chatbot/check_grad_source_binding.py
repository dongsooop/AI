#!/usr/bin/env python3
"""Check actual graduation formatters without .env, models, or network access."""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))

A = "https://www.dongyang.ac.kr/dmu/4778/subview.do"
B = "https://www.dongyang.ac.kr/dmu/4781/subview.do"
DEFAULT = "https://www.dongyang.ac.kr/graduation"
settings = types.SimpleNamespace(
    org_homepage_label="학교 홈페이지",
    org_homepage_url="https://www.dongyang.ac.kr",
    grad_page_url=DEFAULT,
    repo_root=ROOT_DIR,
    resolved_dept_map_path=ROOT_DIR / "data/department.txt",
    bot_aliases=("동냥이",),
    staff_url_pattern=None,
)
fake_settings = types.ModuleType("core.settings")
fake_settings.get_settings = lambda: settings
with patch.dict(sys.modules, {"core.settings": fake_settings}):
    from LLM.OSS.formatter import one_sentence_grad
    from LLM.OSS.postprocess.engine import run_postprocess


class GraduationSourceBindingTests(unittest.TestCase):
    def check_response(self, query, evidence, url, required=(), forbidden=()):
        expected_url = url + ("&" if "?" in url else "?") + "layout=unknown"
        for renderer in (one_sentence_grad, lambda q, e: run_postprocess("grad", q, e)):
            with self.subTest(renderer=renderer):
                text, actual_url = renderer(query, evidence)
                self.assertEqual(actual_url, expected_url)
                self.assertIn(expected_url, text)
                for phrase in required:
                    self.assertIn(phrase, text)
                for phrase in forbidden:
                    self.assertNotIn(phrase, text)

    def test_screenshot_title_is_not_used_as_body(self):
        evidence = (
            f"- 학사안내/졸업보류: {B}\n"
            f"- 학점 전공최저이수 학점은 자유전공학과 연락처: {A}"
        )
        self.check_response("졸업", evidence, B, forbidden=("자유전공", "전공최저", A))

    def test_selected_body_uses_its_own_inline_source(self):
        evidence = f"졸업보류 안내입니다. (출처: {B})\n총 졸업학점: 120학점 (출처: {A})"
        self.check_response("졸업학점", evidence, A, required=("120학점",), forbidden=(B, "졸업보류"))

    def test_multiline_body_keeps_conditions_and_warning(self):
        evidence = (
            "2023년 이후 대상자 기준\n총 졸업학점: **120학점**\n세부: 전공 60학점\n"
            f"(출처: 학사안내/졸업 · {A})\n※ 학과·학번에 따라 상이할 수 있습니다."
        )
        self.check_response("졸업학점", evidence, A,
                            required=("2023년 이후", "120학점", "전공 60학점", "※ 학과·학번"))

    def test_explicit_program_prefers_matching_source_block(self):
        evidence = (
            f"3년제 졸업이수 학점: 120학점\n(출처: {B})\n"
            f"2년제 졸업이수 학점: 75학점\n(출처: {A})"
        )
        self.check_response("2년제 졸업학점", evidence, A, required=("75학점",), forbidden=(B, "120학점"))

    def test_link_record_cannot_supply_source_to_uncited_body(self):
        evidence = f"총 졸업학점: 999학점\n- 학사안내/졸업: {A}"
        self.check_response("졸업", evidence, A, forbidden=("999",))

    def test_uncited_body_cannot_borrow_default_source(self):
        self.check_response("졸업학점", "총 졸업학점: 999학점", DEFAULT, forbidden=("999",))

    def test_uncited_trailing_body_cannot_borrow_previous_source(self):
        evidence = f"졸업보류 안내입니다. (출처: {B})\n총 졸업학점: 999학점"
        self.check_response("졸업", evidence, B, forbidden=("999",))

    def test_body_with_guidance_phrase_still_gets_source(self):
        evidence = f"졸업학점은 교육과정에서 확인할 수 있습니다.\n(출처: {A})"
        self.check_response("졸업학점", evidence, A, required=("교육과정",))

    def test_link_list_does_not_contaminate_following_cited_body(self):
        evidence = f"- 학사안내/졸업보류: {B}\n총 졸업학점: 120학점\n(출처: {A})"
        self.check_response("졸업학점", evidence, A, required=("120학점",), forbidden=(B, "졸업보류"))

    def test_empty_input_keeps_existing_default(self):
        for renderer in (one_sentence_grad, lambda q, e: run_postprocess("grad", q, e)):
            text, url = renderer("졸업", "")
            self.assertEqual(url, DEFAULT)
            self.assertIn(DEFAULT, text)

    def test_source_query_string_is_preserved(self):
        self.check_response("졸업학점", f"총 졸업학점: 120학점 (출처: {A}?year=2023)",
                            A + "?year=2023", required=("120학점",))


if __name__ == "__main__":
    unittest.main()
