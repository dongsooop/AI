#!/usr/bin/env python3
"""Contact ownership regression tests; no model or server dependencies."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from LLM.sub_model.index_utils import extract_units_and_contacts


class ContactExtractionTests(unittest.TestCase):
    def extract(self, text, title="학사안내/졸업"):
        return extract_units_and_contacts({"title": title, "content": text, "url": "https://example.com/source"})

    def test_graduation_prose_does_not_own_footer_phone(self):
        rows = self.extract(
            "전공최저이수 학점은 자유전공학과 전공 배정 학기부터 취득한 학점 "
            "해당 영역은 편집모드에서만 보여지는 영역입니다. "
            "담당부서명과 전화번호만 수정해주시면 됩니다. 교육지원팀 02-2610-1707"
        )
        self.assertEqual([(r["unit"], r["phone"]) for r in rows], [("교육지원팀", "02-2610-1707")])

    def test_separate_units_keep_their_own_phone(self):
        rows = self.extract("교육지원팀 02-2610-1707 학생서비스센터 02-2610-1714")
        self.assertEqual([(r["unit"], r["phone"]) for r in rows],
                         [("교육지원팀", "02-2610-1707"), ("학생서비스센터", "02-2610-1714")])

    def test_unit_without_contact_does_not_borrow_next_unit(self):
        rows = self.extract("자유전공학과 교육지원팀 02-2610-1707")
        self.assertEqual([r["unit"] for r in rows], ["교육지원팀"])

    def test_trailing_unit_does_not_borrow_previous_number(self):
        rows = self.extract("교육지원팀 02-2610-1707 자유전공학과")
        self.assertEqual([r["unit"] for r in rows], ["교육지원팀"])

    def test_real_department_contact_is_retained(self):
        rows = self.extract("자유전공학과 1호관 402호 02-2610-5159 office@example.com",
                            "학부ㆍ학과/자유전공학과")
        self.assertEqual([(r["unit"], r["phone"], r["email"]) for r in rows],
                         [("자유전공학과", "02-2610-5159", "office@example.com")])

    def test_short_department_suffix_with_contact_label(self):
        rows = self.extract("시각디자인과 전화: 02-2610-1234")
        self.assertEqual([r["unit"] for r in rows], ["시각디자인과"])

    def test_korean_conjunction_is_not_a_department(self):
        rows = self.extract("전문 지식과 바른 인성을 양성합니다. 02-2610-1234")
        self.assertEqual(rows, [])

    def test_no_department_contact_from_footer_only(self):
        rows = self.extract(
            "자유전공학과 소개 해당 영역은 편집모드에서만 보여지는 영역입니다. 교육지원팀 02-2610-1707",
            "학부ㆍ학과/자유전공학과",
        )
        self.assertEqual([r["unit"] for r in rows], ["교육지원팀"])

    def test_title_fallback_preserves_department_and_not_footer(self):
        rows = self.extract(
            "경영 실무역량을 쌓는 과정 2호관 303호 02-2610-1986 "
            "해당 영역은 편집모드에서만 보여지는 영역입니다. 교육지원팀 02-2610-1707",
            "학부ㆍ학과/경영학과",
        )
        department = [r for r in rows if r["unit"] == "경영학과"]
        self.assertEqual([r["phone"] for r in department], ["02-2610-1986"])

    def test_email_only_contact(self):
        rows = self.extract("교육지원팀 이메일: office@example.com")
        self.assertEqual([(r["unit"], r["phone"], r["email"]) for r in rows],
                         [("교육지원팀", "없음", "office@example.com")])

    def test_policy_text_without_contact(self):
        self.assertEqual(self.extract("자유전공학과 학생의 졸업학점 안내"), [])


if __name__ == "__main__":
    unittest.main()
