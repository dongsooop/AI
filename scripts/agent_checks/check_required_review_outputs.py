"""Print required Dongsooop branch-review output sections."""

from __future__ import annotations


REQUIRED_SECTIONS = [
    "보안 이슈",
    "기능 또는 회귀 리스크",
    "프라이버시/GitHub 공개 이슈",
    "Docker/의존성 이슈",
    "환경변수 또는 배포 설정 불일치",
    "남은 리스크 또는 미검증 항목",
    ".github/pull_request_template.md 형식의 PR 본문 초안",
    "type: english summary 형식의 커밋 메시지 후보 3개",
]


def main() -> int:
    print("[review-output]")
    print("Branch review responses must include:")
    for section in REQUIRED_SECTIONS:
        print(f"- {section}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

