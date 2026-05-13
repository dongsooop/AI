# CHECKLIST.md

이 문서는 이 저장소에서 GPT 기반 에이전트가 실제 리뷰와 점검을 수행할 때 사용하는 운영 체크리스트입니다.

다음 상황에서 사용합니다.

- 브랜치를 머지하기 전 리뷰할 때
- 변경 사항을 GitHub에 푸시해도 되는지 확인할 때
- 작업 결과가 인수인계나 배포 준비 상태인지 확인할 때

이 문서는 다음 문서와 함께 사용합니다.

- `docs/AGENTS.md` — 저장소 공통 작업 원칙
- `docs/PLANS.md` — 현재 코드 구조와 변경 방향
- `docs/BRANCH_REVIEW_GUIDE.md` — 브랜치 리뷰 기준

## 핵심 점검 목표

모든 리뷰에서 다음 항목을 확인합니다.

1. 보안이 후퇴하지 않았는지
2. 현재 아키텍처에서 코드가 정상 동작할 가능성이 높은지
3. 코드, 설정, 추적 파일에 민감한 정보가 노출되지 않는지
4. Docker 및 의존성 파일이 여전히 안전하고 타당한지
5. 환경변수 사용 방식이 코드와 배포 가정에 맞는지

## 실행 체크리스트

### 1. 범위 확인

- 어떤 서비스가 영향을 받는지 확인합니다.
  - `main.py` 서비스, 포트 `8888`
  - `app_oss_main.py` 서비스, 포트 `8010`
- 브랜치가 다음 중 무엇을 건드리는지 확인합니다.
  - 런타임 코드
  - 인증 또는 설정
  - 검색 또는 후처리 로직
  - OCR 또는 텍스트 필터링
  - Docker 또는 의존성 파일
  - `.env` 또는 환경변수 가정
  - 파일 이동, 패키지 이름 변경, 책임 경계 수정 같은 구조 리팩토링

구조 리팩토링 브랜치라면 추가로 다음도 확인합니다.

- 현재 구조를 문서화하는 작업인지, 목표 구조로 이동하는 작업인지, 또는 둘 다인지
- 의미 있는 경계/책임 변경이 있었다면 `docs/PLANS.md`가 함께 업데이트되었는지
- 구조 이동 중에도 런타임 동작 안정성을 유지하려 했는지

### 2. 보안 체크

- API 키, 토큰, 비밀번호, JWT 비밀값, 비공개 URL이 하드코딩되지 않았는지
- 인증 우회나 위험한 JWT 처리 변경이 없는지
- 위험한 셸 실행, 경로 탐색, 안전하지 않은 파일 처리가 없는지
- 민감한 요청/사용자 데이터가 로그에 추가되지 않았는지
- 내부 서버 정보, SSH 관련 정보, 비공개 엔드포인트가 추적 파일에 들어가지 않았는지
- 비밀값이 환경변수에서 소스코드로 옮겨지지 않았는지

특히 다음을 중점적으로 봅니다.

- `core/auth.py`
- `core/settings.py`
- 파일 업로드 처리 및 OCR 경로 처리
- 데이터베이스 및 Ollama 설정 사용 부분

### 3. 기능 체크

- import와 모듈 참조가 여전히 정상 동작하는지
- 두 서비스 분리가 유지되는지
- 챗봇 변경이 retrieval -> generation -> formatting -> postprocess 흐름에 맞는지
- RAG 인덱싱 변경이 정제/문서 타입 분류/청킹 -> 아티팩트 생성 -> 검색/재랭킹 흐름에 맞는지
- 새 `search_df.parquet` 컬럼을 쓰는 경우 구버전 아티팩트 fallback이 깨지지 않는지
- `search_df.parquet`, `embeddings.npy`, `bm25.pkl`의 row 수가 서로 일치하는지
- OCR 변경이 service -> OCR engine -> timetable parsing 흐름에 맞는지
- 텍스트 필터링 변경이 ML/규칙 기반 양쪽에서 일관성 있는지
- 변경된 코드가 `docs/PLANS.md`의 현재 구조 설명과 크게 어긋나지 않는지
- 파일 이동이나 패키지 이름 변경이 엔트리포인트, import, 경로 가정, 런타임 연결부를 조용히 깨뜨리지 않는지

특히 다음을 중점적으로 봅니다.

- `main.py`
- `app_oss_main.py`
- `LLM/OSS/Open_AI_OSS.py`
- `LLM/OSS/formatter.py`
- `LLM/OSS/postprocess/`
- `LLM/sub_model/`
- `image_analysis/`
- `text_filtering/`

### 4. GitHub 공개 / 프라이버시 체크

- `.env`가 커밋되지 않았는지
- 소스, 주석, fixture, 문서에 인증정보처럼 보이는 문자열이 없는지
- 로컬 머신 경로, 사용자명, 내부 전용 메모가 들어가지 않았는지
- 민감한 데이터셋, 백업, 생성 아티팩트, 디버그 덤프가 실수로 추적되지 않는지
- 회귀 테스트 케이스가 공개 가능한 질문과 기대 문구만 포함하는지
- `.gitignore`가 로컬 전용/프라이빗 파일을 계속 잘 보호하는지
- 서버 접속 명령, 사설 IP, 키 파일 경로가 추적 문서에 들어가지 않았는지

특히 다음 경로를 주의해서 봅니다.

- `data/`
- `model/`
- `database/`
- `debug/`
- `backup_file/`
- `.env`

### 5. Docker 및 의존성 체크

다음 파일을 확인합니다.

- `Dockerfile`
- `Dockerfiles_oss`
- `requirements.txt`
- `requirements_oss.txt`

다음 항목을 확인합니다.

- 추가된 패키지가 정말 필요한지
- 버전이 명확한 충돌을 만들지 않는지
- OCR, Java, 런타임 의존성이 앱 요구사항과 맞는지
- 빌드 단계에서 비밀값이 노출되지 않는지
- 컨테이너 기본 설정이 불필요하게 위험하지 않은지
- 배포 가정이 현재 서비스 구조와 맞는지
- 구조 리팩토링 후에도 빌드 컨텍스트와 `COPY` 경로가 유효한지

### 6. 환경변수 체크

- 필요한 환경변수가 정의되었거나, 로드되거나, 문서화되어 있는지
- 이름이 바뀐 변수는 모든 참조 지점이 같이 수정되었는지
- 제거된 변수는 더 이상 참조되지 않는지
- `.env`, `core/settings.py`, 런타임 코드, Docker 가정이 서로 맞는지
- `OSS_BASE_URL`, `OSS_MODEL`, JWT, DB, 아티팩트/데이터 경로가 여전히 유효한지
- 로컬 전용 값이 추적 파일로 올라가지 않는지
- 리팩토링 후에도 경로 기반 환경변수가 올바른 위치를 가리키는지

## 리뷰 순서

브랜치 리뷰 시 다음 순서를 따릅니다.

1. 범위 확인
2. 보안
3. 기능 정상 동작
4. GitHub 공개 가능 여부 / 프라이버시
5. Docker 및 requirements
6. 환경변수 정합성
7. 남은 리스크 및 미검증 항목

## 결과 작성 형식

리뷰 결과는 다음 순서로 정리합니다.

1. 보안 이슈
2. 기능 이슈
3. 프라이버시 / GitHub 공개 이슈
4. Docker / 의존성 이슈
5. 환경변수 이슈
6. 남은 리스크 또는 미검증 항목

문제가 없다면 다음을 명시합니다.

- 치명적 문제 없음
- 무엇을 확인했는지
- 아직 검증하지 못한 부분이 있다면 무엇인지

## 빠른 푸시 가능 여부 판단

브랜치가 푸시 가능하다고 말하기 전에 다음을 모두 확인합니다.

- 눈에 띄는 보안 문제가 없는지
- 눈에 띄는 런타임 깨짐이 없는지
- GitHub로 유출될 프라이빗 정보가 없는지
- Docker 및 requirement 파일이 여전히 타당한지
- 환경변수가 구현과 맞게 정리되어 있는지

## 회귀 테스트 실행 방식

검증은 기본적으로 `conda` 환경 `dl_study`에서 실행합니다. 단순 문법 검증도 같은 환경을 사용합니다.

```bash
conda run -n dl_study python -m py_compile LLM/OSS/service.py LLM/sub_model/query_index.py LLM/rule_book/graph.py
```

챗봇 회귀 테스트는 OSS FastAPI 서버를 먼저 띄운 뒤 API 요청으로 검증합니다.

```bash
conda run -n dl_study uvicorn app_oss_main:app --reload --port 8010
conda run -n dl_study python debug/regression/run_chatbot_regression.py
```

RAG 인덱스 생성 로직을 바꾼 경우에는 API 회귀 테스트 전에 다음도 확인합니다.

```bash
conda run -n dl_study python -m py_compile LLM/sub_model/index_utils.py LLM/sub_model/build_index.py LLM/sub_model/query_index.py
conda run -n dl_study python LLM/sub_model/build_index.py
```
