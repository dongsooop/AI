# 프로젝트 문서

이 디렉터리에는 GitHub에 공개 가능한 프로젝트 구조, 운영, 리뷰 문서를 보관합니다.
실제 토큰, 서버 주소, OCI 자원값, 프로세스 ID, 원본 운영 로그와 측정 리포트는
포함하지 않습니다.

## 시작 문서

- [에이전트 작업 원칙](AGENTS.md)
- [아키텍처 및 변경 계획](PLANS.md)

## 브랜치 리뷰

- [브랜치 리뷰 가이드](review/BRANCH_REVIEW_GUIDE.md)
- [브랜치 리뷰 체크리스트](review/CHECKLIST.md)
- [리뷰 심각도 기준](review/REVIEW_SEVERITY.md)

## 운영

- [장애 기준 정책](operations/OPERATIONS_FAILURE_POLICY.md)
- [운영 SLI 후보 맵](operations/OPERATIONS_SLI_MAP.md)
- [로그 기반 SLI 수집](operations/OPERATIONS_LOG_SLI_COLLECTION.md)
- [1차 운영 SLO 후보](operations/OPERATIONS_SLO_CANDIDATES.md)
- [릴리즈 품질 게이트](operations/OPERATIONS_QUALITY_GATES.md)
- [운영 알림 및 점검 흐름](operations/OPERATIONS_INCIDENT_WORKFLOW.md)
- [OCI AI 성능 기준선 수집](operations/OCI_AI_PERFORMANCE_BASELINE.md)
- [텍스트 필터 ONNX shadow 실험](operations/TEXT_FILTER_ONNX_SHADOW.md)

## 공개 범위

운영 기준선 문서에는 실행 방법과 지표 정의만 기록합니다. 실제 배포 환경의 shape,
OCPU, RAM, 서비스 배치, 배포 모델명, worker 수와 원본 측정 결과는 실행 시
인자로 전달하고 Git에 커밋하지 않습니다.
