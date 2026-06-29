---
name: dongsooop-issue-template-writing
description: Write GitHub issue drafts for Dongsooop using the repository's .github/ISSUE_TEMPLATE forms. Use when the user asks to make an issue template, GitHub issue body, feature issue, bug issue, or to convert a plan into an issue for this repo.
---

# Dongsooop 이슈 템플릿 작성

## 개요

Dongsooop 저장소의 GitHub 이슈 초안을 작성할 때 사용합니다. 사용자가 "이슈 템플릿 만들어줘", "이슈로 정리해줘", "GitHub issue 양식으로 작성해줘"처럼 요청하면 실제 `.github/ISSUE_TEMPLATE/` 양식을 먼저 확인하고 그 형식에 맞춰 작성합니다.

## 필수 확인

1. `.github/ISSUE_TEMPLATE/` 아래 템플릿 목록을 확인합니다.
2. 요청 성격에 맞는 템플릿을 읽습니다.
   - 기능 추가/개선/고도화: `.github/ISSUE_TEMPLATE/신규-기능.md`
   - 장애/오류/회귀: `.github/ISSUE_TEMPLATE/버그-이슈.md`
3. 템플릿의 frontmatter와 섹션명을 유지해 Markdown 초안을 작성합니다.

## 작성 원칙

- 사용자가 파일 생성을 명시하지 않으면, 이슈 본문 초안을 답변으로 제공합니다.
- 템플릿에 있는 섹션을 임의로 크게 늘리지 않습니다. 필요한 맥락은 기존 섹션 안에 간결하게 넣습니다.
- 제목은 템플릿 prefix를 유지합니다. 예: `[FEAT] ...`, `[BUG] ...`
- 체크리스트는 실제 작업 순서가 보이도록 3-8개 정도로 작성합니다.
- 참고 자료에는 관련 로컬 경로를 넣습니다. 예: `text_filtering/service.py`, `tests/regression/text_filtering/`
- 비밀값, `.env` 값, 서버 접속 정보, 토큰, SSH 키를 이슈 본문에 넣지 않습니다.
- `AGENTS.md`, `PLANS.md`, `SKILL.md`, `CHECKLIST.md`는 로컬 에이전트 문서라는 전제를 유지합니다.

## Dongsooop 가드레일

- 텍스트 필터링, 시간표 OCR, 챗봇 OSS의 서비스 경계를 섞지 않습니다.
- 운영 API 계약이 있는 작업은 기존 응답 구조를 깨지 않는 방향을 체크리스트에 명시합니다.
- 테스트/리포트 선검증이 필요한 AI 품질 작업은 "운영 반영 전 회귀 리포트 확인"을 포함합니다.
- 텍스트 필터링 고도화 이슈는 기존 True/False 또는 `has_profanity` 운영 판정을 즉시 바꾸지 않고, shadow logic과 회귀 리포트로 먼저 검증하는 방향을 우선합니다.
- 생성 산출물(`model/artifacts/`, 모델 파일 등)을 직접 수정하는 작업으로 쓰지 말고 생성 로직이나 검증 흐름을 이슈화합니다.

## 출력 형식

답변은 붙여넣기 쉬운 하나의 Markdown 코드블록으로 제공합니다. 사용자가 별도 설명을 원하지 않으면 코드블록 밖 설명은 한 문장 이내로 줄입니다.
