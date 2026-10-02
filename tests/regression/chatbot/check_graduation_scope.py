#!/usr/bin/env python3
"""Regression fixtures for applicability, not live graduation advice."""
import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from LLM.sub_model.graduation_rules import answer_graduation_scope, parse_graduation_rules, query_scope, valid_rule
from LLM.sub_model.index_utils import chunk_document

SOURCE = "https://www.dongyang.ac.kr/dmu/4778/subview.do"
HEADER = "구분 졸업 학점 2022년 2월 및 8월 졸업대상자 2023년 2월 이후 졸업대상자 "
TABLE = (
    "졸업이수 학점 " + HEADER + "2년제 75 75 3년제 120 110 "
    "전공최저이수 학점 " + HEADER + "2년제 52 52 3년제 [별표1] 78 "
    "1 [별표 1] 학과별 전공최저이수 학점 범위 "
    "2006, 2007학년도 입학자 60학점 기계과, 기계설계과 90학점 건축과 "
    "2008학년도 입학자 60학점 기계과 90학점 건축과 "
    "2 [별표 2] 편입학생 전공최저이수 학점 및 졸업 학점 "
    "2021 학년도 2년 2학년 1학기 편입학생 26학점 36학점 75학점 "
    "3년 2학년 1학기 편입학생 52학점 36학점 110학점 "
    "3학년 1학기 편입학생 30학점 72~80학점 120학점 "
    "2022 학년도 이후 2년 2학년 1학기 편입학생 26학점 36학점 75학점 "
    "3년 2학년 1학기 편입학생 52학점 36학점 110학점 "
    "3학년 1학기 편입학생 26학점 72학점 110학점 "
    "3 [별표 3] 전과ㆍ자유전공학과 학생 기준 "
    "전과학생 2년 1학년 2학기 전과학생 39학점 수강학점 75학점 "
    "2학년 1학기 전과학생 26학점 "
    "3년 1학년 2학기 전과학생 65학점 110학점 "
    "3년 2학년 1학기 전과학생 52학점 2학년 2학기 전과학생 39학점 "
    "자유전공학과 전공배정학생 2년 자유전공학과 전공배정학생 39학점 75학점 "
)


class GraduationScopeTests(unittest.TestCase):
    def setUp(self):
        self.rules = parse_graduation_rules("학사안내/졸업 / 학사안내/전문학사", TABLE)
        for rule in self.rules:
            rule['url'] = SOURCE

    def answer(self, query):
        return answer_graduation_scope(query, self.rules)

    def check_fact(self, query, *facts):
        result = self.answer(query)
        self.assertFalse(result['needs_clarification'], result)
        self.assertEqual(result['url'], SOURCE)
        for fact in facts:
            self.assertIn(fact, result['answer'])
        return result['answer']

    def check_clarification(self, query, *phrases):
        result = self.answer(query)
        self.assertTrue(result['needs_clarification'], result)
        self.assertNotRegex(result['answer'], r"\d+학점")
        for phrase in phrases:
            self.assertIn(phrase, result['answer'])

    def test_general_current_and_previous_graduation_years(self):
        self.check_fact("일반학생 3년제 2026년 졸업학점", "총 졸업학점 110학점", "전공최저이수 78학점")
        text = self.check_fact("일반학생 3년제 2022년 졸업학점", "총 졸업학점 120학점", "별표1")
        self.assertNotIn("전공최저이수 78학점", text)

    def test_two_year_program_does_not_use_larger_three_year_number(self):
        self.check_fact("일반학생 2년제 2026년 졸업학점", "총 졸업학점 75학점", "전공최저이수 52학점")

    def test_conflicting_year_bases_do_not_depend_on_requested_credit_field(self):
        for suffix in ("졸업학점", "졸업 전공학점"):
            self.check_clarification(f"일반학생 기계과 2008학번 3년제 2026년 {suffix}", "적용 기준을 확정")

    def test_historical_appendix_does_not_cover_unsupported_program(self):
        for suffix in ("졸업학점", "졸업 전공학점"):
            self.check_clarification(f"일반학생 기계과 2008학번 4년제 {suffix}", "전문학사")

    def test_historical_appendix_cannot_bypass_unknown_graduation_date(self):
        for date in ("2020년", "2023년1월"):
            self.check_clarification(f"일반학생 기계과 2008학번 3년제 {date} 졸업 전공학점", "확인하지 못했어요")

    def test_historical_appendix_resolves_a_matching_parent_reference(self):
        self.check_fact("일반학생 기계과 2008학번 3년제 2022년 졸업 전공학점", "전공최저이수 60학점")

    def test_admission_year_cannot_be_used_as_graduation_year(self):
        self.check_clarification("일반학생 3년제 22학번 졸업학점", "졸업대상 연도")
        self.check_clarification("일반학생 3년제 2022년 입학 졸업학점", "졸업대상 연도")

    def test_missing_student_group_and_program(self):
        self.check_clarification("졸업학점", "일반학생/편입/전과/자유전공")
        self.check_clarification("일반학생 2026년 졸업학점", "2년제/3년제")

    def test_historical_major_depends_on_department_and_admission_year(self):
        self.check_fact("일반학생 기계과 2008학번 졸업 전공최저이수학점", "전공최저이수 60학점", "2008년 입학")
        self.check_fact("일반학생 건축과 2008학번 졸업 전공최저이수학점", "전공최저이수 90학점")
        self.check_clarification("일반학생 2008학번 졸업 전공최저이수학점", "학과명")

    def test_previous_general_major_requests_historical_conditions(self):
        self.check_clarification("일반학생 3년제 2022년 졸업 전공학점", "입학 당시 학과명", "입학 연도")

    def test_transfer_year_and_stage(self):
        self.check_fact("2021년 편입 3년제 3학년1학기 졸업학점", "총 졸업학점 120학점", "전공최저이수 30학점")
        self.check_fact("2022년 편입 3년제 3학년1학기 졸업학점", "총 졸업학점 110학점", "전공최저이수 26학점")
        self.check_fact("2022년 편입 3년제 2학년1학기 졸업학점", "전공최저이수 52학점")
        self.check_clarification("2022년 편입 3년제 졸업학점", "학년과 학기")

    def test_admission_year_cannot_be_used_as_transfer_year(self):
        self.check_clarification("22학번 편입 3년제 2학년1학기 졸업학점", "편입 연도")

    def test_changed_major_inherited_table_cells(self):
        self.check_fact("전과 2년제 2학년1학기 졸업학점", "총 졸업학점 75학점", "전공최저이수 26학점")
        self.check_fact("전과 3년제 2학년2학기 졸업학점", "총 졸업학점 110학점", "전공최저이수 39학점")

    def test_free_major_is_not_applied_to_other_programs(self):
        self.check_fact("자유전공학과 2년제 졸업학점", "전공최저이수 39학점", "전공 배정 학기부터")
        self.check_clarification("자유전공학과 3년제 졸업학점", "확인하지 못했어요")

    def test_unknown_year_and_month_are_not_guessed(self):
        for query in ("일반학생 3년제 2020년 졸업학점", "일반학생 3년제 2023년1월졸업학점"):
            self.check_clarification(query, "확인하지 못했어요")

    def test_unrepresented_education_tracks_do_not_use_general_rules(self):
        for word in ("복학생", "재입학", "외국인", "계약학과", "전공심화", "학사학위"):
            self.check_clarification(f"{word} 3년제 2026년 졸업학점", "교육과정")

    def test_conflicting_source_records_do_not_select_first_number(self):
        conflict = copy.deepcopy(self.rules[3]); conflict['total'] = 119
        self.rules.append(conflict)
        self.check_clarification("일반학생 3년제 2026년 졸업학점", "여러 개")

    def test_missing_or_unrecognized_layout_does_not_extract_numbers(self):
        self.assertEqual(parse_graduation_rules("졸업", "3년제 110학점 자유전공학과 39학점"), [])
        self.assertTrue(answer_graduation_scope("졸업학점", [])['needs_clarification'])

    def test_generated_chunks_keep_conditions_and_source(self):
        chunks = chunk_document({'title': '학사안내/졸업', 'url': SOURCE, 'content': TABLE})
        scoped = [c for c in chunks if c.get('graduation_scope')]
        self.assertEqual(len(scoped), len(self.rules))
        for chunk in scoped:
            self.assertEqual(chunk['url'], SOURCE)
            self.assertTrue(valid_rule(json.loads(chunk['graduation_scope'])))
            self.assertEqual(chunk['chunk_type'], 'graduation_rule')

    def test_short_admission_year_and_invalid_record(self):
        self.assertEqual(query_scope("22학번")['years'], {'admission': 2022})
        self.assertFalse(valid_rule([]))
        self.assertFalse(valid_rule({'total': 120}))

    def test_conflicting_or_negated_student_group_is_not_guessed(self):
        self.check_clarification("일반학생 편입 3년제 2026년졸업학점", "구분을 하나")
        self.check_clarification("자유전공 아니고 일반학생 2년제 졸업학점", "구분을 하나")

    def test_current_grade_is_not_assumed_to_be_transfer_stage(self):
        self.check_clarification("2022년 편입 현재 3년제 2학년1학기 졸업학점", "당시 학년")

    def test_undated_exception_cannot_claim_a_specific_year(self):
        self.check_clarification("자유전공학과 2년제 2008년 입학 졸업학점", "적용 연도가 명시")


if __name__ == '__main__':
    unittest.main()
