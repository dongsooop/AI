---
name: reviewing-dongsooop-branch
description: Review Dongsooop repository branches before pushing, opening pull requests, or publishing changes. Use when the user asks for a branch review, current changes review, pre-push check, security/publication check, PR body draft, or commit message suggestions for the Dongsooop backend.
---

# Dongsooop 브랜치 리뷰

## 개요

Dongsooop 백엔드 브랜치를 푸시, PR 생성, 공개하기 전에 실제 점검할 때 이 스킬을 사용합니다. 변경 파일, 보안 노출, 런타임 리스크, Docker/의존성 영향, 환경변수 정합성, GitHub 공개 가능 여부를 함께 확인합니다.

## 필수 참고 문서

리뷰 전에 다음 파일을 순서대로 읽습니다.

1. `docs/AGENTS.md`: 프로젝트 공통 가드레일
2. `docs/PLANS.md`: 현재 아키텍처와 서비스 경계
3. `docs/BRANCH_REVIEW_GUIDE.md`: 브랜치 리뷰 기준
4. `docs/CHECKLIST.md`: 실제 점검 체크리스트

PR 본문 초안을 작성할 때는 `.github/pull_request_template.md`도 확인합니다.

## 작업 흐름

1. `git status --short`와 diff를 확인해 브랜치 범위를 파악합니다.
2. 영향 영역을 분류합니다: 텍스트 필터링/시간표 서비스, OSS 챗봇 서비스, 공통 코어, Docker/의존성, 데이터/아티팩트, GitHub 워크플로, 문서.
3. 보안을 먼저 봅니다: 비밀값, 토큰, `.env`, 비공개 URL, 서버 접속 정보, 안전하지 않은 파일/경로 처리, 로그, 추적된 생성물/비공개 파일.
4. `docs/PLANS.md`의 두 서비스 구조와 한국어 NLP/OCR/RAG 제약을 기준으로 동작 리스크를 검토합니다.
5. 관련 파일이 바뀌었다면 Docker, requirements, 워크플로, 환경변수 정합성을 확인합니다.
6. 검증 실행 여부를 판단합니다. 문서 변경만 있으면 가벼운 검증을 사용하고, 코드 변경이면 가능한 범위에서 컴파일/회귀 명령을 실행합니다.
7. 발견 사항을 심각도 순으로 먼저 보고합니다. 문제가 없으면 그 사실과 남은 미검증 리스크를 함께 적습니다.
8. `.github/pull_request_template.md` 형식에 맞춰 PR 본문 초안을 작성하고, 이슈 번호를 모르면 `Closes #이슈번호`를 유지합니다.
9. `type: english summary` 형식의 커밋 메시지 후보 3개를 제안합니다. type은 `feat`, `fix`, `refactor`, `chore`, `docs` 중에서 고릅니다.

## 리뷰 우선순위

- diff가 의도적으로 공통 코드를 건드린 경우가 아니면 챗봇 작업과 텍스트 필터링/시간표 작업을 분리해서 봅니다.
- `LLM/OSS/Open_AI_OSS.py`는 크고 핵심적인 파일이므로 고위험 변경으로 취급합니다.
- `model/artifacts/` 아래 생성 산출물은 직접 수정하지 않고 생성 로직을 검토합니다.
- 사용자가 명시적으로 요청하지 않는 한, Docker 이미지가 내부에 `.env`를 복사할 수 있다는 OCI 배포 전제를 유지합니다.
- 루트의 `AGENTS.md`는 로컬 에이전트 진입점으로 보고, 추가 로컬 보조 문서는 `.local-agent-docs/` 아래에 둡니다. 공개용 문서는 `docs/` 아래 파일을 기준으로 봅니다.

## 출력 형식

리뷰 결과는 발견 사항부터 시작합니다.

1. 보안 이슈
2. 기능 또는 회귀 리스크
3. 프라이버시/GitHub 공개 이슈
4. Docker/의존성 이슈
5. 환경변수 또는 배포 설정 불일치
6. 남은 리스크 또는 미검증 항목

그다음 간결한 PR 본문 초안과 커밋 메시지 후보 3개를 포함합니다.
