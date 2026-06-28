# 🪴 동숲 프로젝트 개요

동양미래대학교 숲의 줄임말로, 학생들의 교내 생활 편리성 증진 목적의 정보 제공 및 통합 커뮤니티 서비스입니다.  
사용자에게 공지사항, 학사 일정 등 정보를 실시간으로 제공하고, 시간표를 등록해 학업 관리를 돕습니다.  
팀원 모집, 맛집 추천, 과팅 기능 등 교내 다른 학우들과 소통할 수 있는 커뮤니티를 제공합니다.

<img width="975" height="456" alt="image" src="https://github.com/user-attachments/assets/bafe1cd2-f0ec-484d-9317-b484c3d7dfe5" />

# 🤖 AI Backend

동숲의 AI 백엔드는 **학생이 직접 입력하고 확인해야 했던 반복 작업을 자동화**하는 데 초점을 맞춰 설계했습니다.
시간표 이미지는 일정 데이터로, 학교 정보 질문은 근거 기반 답변으로, 커뮤니티 문장은 안전한 게시 환경을 위한 필터링 결과로 변환합니다.

<div align="center">
  <img src="https://img.shields.io/badge/RAG-LLM%20Chatbot-111827?style=for-the-badge" height="24px"/>
  <img src="https://img.shields.io/badge/OCR-Timetable%20Parser-2563EB?style=for-the-badge" height="24px"/>
  <img src="https://img.shields.io/badge/NLP-Text%20Filtering-059669?style=for-the-badge" height="24px"/>
  <img src="https://img.shields.io/badge/Quality-Regression%20Report-F97316?style=for-the-badge" height="24px"/>
</div>

<br/>

| AI 영역                   | 해결한 문제                                                       | 구현 포인트                                                                          |
| ------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| 🧠 학교 정보 RAG 챗봇     | 공지사항, 학사일정, 연락처처럼 흩어진 학교 정보를 대화형으로 조회 | BM25 + semantic search 기반 Hybrid Search, RAG context 구성, Ollama gpt-oss:20b 연동 |
| 📅 시간표 OCR 분석        | 학생이 시간표를 직접 입력해야 하는 불편함 감소                    | OpenCV 격자 검출, Tesseract OCR, 셀 분리, 강의명/교수명/강의실/교시 매핑             |
| 🛡️ 커뮤니티 텍스트 필터링 | 게시글/채팅에서 부적절한 표현을 사전에 감지                       | ELECTRA 기반 문장 분류, 규칙 기반 필터, 한국어 띄어쓰기/변형 표현 대응               |
| 📊 AI 품질 검증           | 모델 응답을 감으로 확인하지 않고 재현 가능한 기준으로 점검        | 질문 세트, Recall@k, 출처 URL 검증, JSON 리포트, 경량/전체 회귀 스크립트             |

# 🚀 핵심 기능

## 💘 과팅

> <a href="https://github.com/dongsooop/backend/wiki/%EA%B3%BC%ED%8C%85-%EB%AC%B8%EC%A0%9C-%ED%95%B4%EA%B2%B0-%EA%B2%BD%ED%97%98">동시성 및 스케줄링 문제 해결 경험 WIKI</a>

정해진 세션 정원 수에 따라 매칭 인원 충족 시 과팅 세션이 시작됩니다.  
과팅은 채팅봇의 안내에 따라 진행되며, 주어진 대화 주제로 사용자가 대화를 하게 됩니다.  
세션 마지막에는 마음에 드는 사람을 선택할 수 있으며, 서로를 선택한 경우 1:1 채팅방이 개설됩니다.

<img alt="과팅" height="600" src="https://github.com/user-attachments/assets/b65d43be-7bba-4c9e-a42c-4a325167f6d3">

<br />
<br />

## 🔔 학사 정보 알림

사용자는 자신이 선택한 공지사항과 대학 공지를 실시간으로 받아볼 수 있습니다.  
등록된 학사 일정이 있는 날은 매일 아침 알림으로 리마인드합니다.  
시간표를 등록한 경우도 학사 일정 알림과 같은 시간에 함께 알림으로 리마인드합니다.

<img width="300" alt="학사 정보 알림" src="https://github.com/user-attachments/assets/22200841-44ef-4ce1-aeb4-b6679fb205a3" />

<br />
<br />

## 🍖 학교 주변 맛집 추천

Kakao map 기반 학교 주변 1Km 이내 식당을 확인할 수 있습니다.  
사용자간 추천을 통해 많이 가는 맛집을 조회할 수 있습니다.

<img height="600" alt="학교 주변 맛집 추천" src="https://github.com/user-attachments/assets/6b44eee9-cbe8-429e-aa38-cdd19dca0b50" />

<br />
<br />

## 📅 시간표 관리

- 시간표 이미지를 업로드하면 OCR과 OpenCV 기반 분석 로직이 강의명, 교수명, 강의실, 요일/교시 정보를 추출해 일정을 자동 등록합니다.
- 이미지 품질 차이를 고려해 격자 검출, 셀 분리, 텍스트 보정 과정을 거쳐 Spring Boot 서버가 사용할 수 있는 시간표 데이터로 변환합니다.
- 캘린더 UI를 통해 학업 일정을 추가 및 관리할 수 있습니다.

<img alt="시간표 관리" height="600" src="https://github.com/user-attachments/assets/a6662139-ec4b-4cbe-8103-0ab52e1a9aed">

<br />
<br />

## 🛡️ AI 텍스트 필터링

커뮤니티 게시글과 채팅 문장을 분석해 부적절한 표현을 사전에 감지합니다.
파인튜닝된 ELECTRA 모델과 규칙 기반 필터를 함께 사용해 한국어/영어 비속어, 띄어쓰기 변형, 우회 표현을 안정적으로 처리합니다.

- ML 기반 필터와 규칙 기반 필터를 분리해 운영 환경에서 필요한 방식으로 선택할 수 있습니다.
- 한국어 문장 특성을 고려해 단순 키워드 매칭보다 문맥 기반 판별을 우선합니다.
- 회귀 케이스와 품질 리포트로 false positive/false negative를 지속적으로 확인합니다.

<br />
<br />

## 🎯 팀원 모집 및 장터 게시판

튜터링, 스터디, 프로젝트 등 다양한 모임을 모집하고 참여할 수 있습니다.  
원하는 물건을 이미지와 함께 등록해 거래할 수 있습니다.

<img alt="모집 게시판" height="600" src="https://github.com/user-attachments/assets/9d4fca1f-6d34-41a9-82fe-4c5692b4f1dd">
<img alt="장터 게시판" height="600" src="https://github.com/user-attachments/assets/ffa99a1b-177c-405c-9e02-6d74d42dea63">

<br />
<br />

## 🤖 AI 챗봇

학교 관련 궁금한 점을 챗봇에게 질문하면 학교 데이터에서 근거를 검색한 뒤 실시간 답변을 받을 수 있습니다.
챗봇은 공지사항, 학사일정, 연락처, 장학금, 수강신청, 졸업 요건 등 학교 정보에 특화된 RAG 흐름으로 동작합니다.

<img alt="챗봇" height="600" src="https://github.com/user-attachments/assets/fcb669e1-4cce-4b81-8568-b2748e919d2f">

### 챗봇 처리 흐름

```text
사용자 질문
→ 인사/의도 감지
→ 학교 문서 Hybrid Search(BM25 + semantic search)
→ RAG context 구성
→ Ollama 기반 gpt-oss:20b 응답 생성
→ 안전 필터 및 출처 기반 응답 반환
```

### RAG 품질 평가

챗봇은 단순 응답 확인이 아니라 질문 세트 기반으로 검색 품질과 답변 근거성을 함께 평가합니다.

- 질문 세트: `tests/regression/chatbot/rag_eval_cases.json`
- 평가 범위: 학사일정, 연락처, 장학금, 수강신청, 졸업, 학과/기숙사 안내
- 검색 평가: top-k 검색 결과에 기대 URL/문서가 포함되는지 `Recall@1`, `Recall@3` 측정
- 답변 평가: 필수 키워드, 날짜 형식, 출처 URL 포함 여부, 비공식 URL 환각 여부 점검
- 자동화: `RAG Light Check` GitHub Actions가 PR 라벨 `run-rag-check` 또는 수동 실행에서 질문 세트 스키마와 경량 query-index 회귀를 검증합니다. `RAG Full Evaluation`은 self-hosted runner가 준비되면 수동 실행으로 전체 RAG 리포트를 생성하며, GitHub artifact에는 상세 질문/답변을 제외한 날짜 prefix summary만 업로드합니다. runner가 준비되기 전에는 로컬에서 전체 RAG 평가를 수동 실행합니다.

```bash
# 질문 세트 스키마 검증
python tests/regression/chatbot/evaluate_rag_retrieval.py --validate-only

# 로컬 모델/아티팩트 기반 전체 RAG 평가
python tests/regression/chatbot/evaluate_rag_retrieval.py --out tests/reports/chatbot/rag_eval_report.json --fail-on-fail

# OSS FastAPI 서버 대상 API 회귀 평가
python tests/regression/chatbot/run_chatbot_regression.py --url http://127.0.0.1:8010/chatbot
```

| 구분           |          질문 세트 |               검색 Recall@3 |                    답변 근거성 | 비고                                                                     |
| -------------- | -----------------: | --------------------------: | -----------------------------: | ------------------------------------------------------------------------ |
| 개선 전 기준선 |               30개 |               리포트 기준값 |                  리포트 기준값 | `tests/reports/chatbot/rag_eval_report.json` 또는 저장된 baseline 리포트 |
| 현재 브랜치    |               30개 | `summary.top3_url_accuracy` | `summary.source_url_pass_rate` | `tests/regression/chatbot/evaluate_rag_retrieval.py` 실행 결과로 갱신    |
| CI 경량 회귀   | synthetic + schema |                   통과/실패 |                      통과/실패 | 무거운 임베딩 모델 다운로드 없이 PR에서 빠르게 검증                      |

# ✌️ 작업 및 역할 분담

| 이름   | 분담                                                                                                                                                                                             |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 주성준 | <ul><li>메인 로직 설계</li><li>팀 리딩 및 커뮤니케이션</li><li>어플리케이션 배포</li></ul>                                                                                                       |
| 백승민 | <ul><li>서버 인프라 구축 </li><li>학식, 채팅, 신고, 검색기능 구현</li><li>편의성 파이프라인 구축</li></ul>                                                                                       |
| 유제승 | <ul><li>ELECTRA 기반 언어 필터링 구현</li><li>Ollama 기반 AI 챗봇 LLM/RAG 구현</li><li>OpenCV + OCR 기반 이미지 분석 및 시간표 생성</li><li>AI 품질 평가용 회귀 스크립트와 리포트 구성</li></ul> |
| 우승원 | <ul><li>UI/UX 설계</li> <li>API 연동</li></ul>                                                                                                                                                   |
| 전승빈 | <ul><li>UI/UX 설계</li> <li>API 연동</li></ul>                                                                                                                                                   |

<br/>
<br/>

# 🚀 기술 스택

<markdown-accessiblity-table data-catalyst="">
  <table>
    <tbody>
      <tr>
        <td align="center" width="160px">FE</td>
        <td align="center" width="560px">
          <img src="https://img.shields.io/badge/Flutter 3.x-02569B?style=for-the-badge&logo=flutter&logoColor=white" height="24px"/>
          <img src="https://img.shields.io/badge/Firebase-DD2C00?style=for-the-badge&logo=firebase&logoColor=white" height="24px"/>
        </td>
      </tr>
      <tr>
        <td align="center" width="160px">BE</td>
        <td align="center" width="560px">
          <img src="https://img.shields.io/badge/Spring Boot 3.4.3-6DB33F?style=for-the-badge&logo=springboot&logoColor=white" height="24px"/>
          <img src="https://img.shields.io/badge/Java 17-red?style=for-the-badge&logo=java&logoColor=white" height="24px"/>
          <img src="https://img.shields.io/badge/Postgresql 17-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" height="24px"/>
          <img src="https://img.shields.io/badge/Redis-FF4438?style=for-the-badge&logo=redis&logoColor=white" height="24px"/>
          <img src="https://img.shields.io/badge/Nginx-009639?style=for-the-badge&logo=nginx&logoColor=white" height="24px"/>
          <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" height="24px"/>
          <img src="https://img.shields.io/badge/Elasticsearch-005571?style=for-the-badge&logo=elasticsearch&logoColor=white" height="24px"/>
          <img src="https://img.shields.io/badge/Firebase-DD2C00?style=for-the-badge&logo=firebase&logoColor=white" height="24px"/>
          <img src="https://img.shields.io/badge/New Relic-1CE783?style=for-the-badge&logo=newrelic&logoColor=white" height="24px"/>
        </td>
      </tr>
      <tr>
        <td align="center" width="160px">AI</td>
        <td align="center" width="560px">
          <img src="https://img.shields.io/badge/Python 3.11.3-3776AB?style=for-the-badge&logo=Python&logoColor=white" height="24px"/>
          <img src="https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=Ollama&logoColor=white" height="24px"/>
          <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=FastAPI&logoColor=white" height="24px"/>
          <img src="https://img.shields.io/badge/opencv-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" height="24px"/>
          <img src="https://img.shields.io/badge/scikitlearn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" height="24px"/>
          <img src="https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" height="24px"/>
          <img src="https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" height="24px"/>
          <img src="https://img.shields.io/badge/SentenceTransformers-4B5563?style=for-the-badge&logo=huggingface&logoColor=white" height="24px"/>
          <img src="https://img.shields.io/badge/Tesseract%20OCR-2563EB?style=for-the-badge&logoColor=white" height="24px"/>
          <img src="https://img.shields.io/badge/KoNLPy-7C3AED?style=for-the-badge&logoColor=white" height="24px"/>
          <img src="https://img.shields.io/badge/OpenAI%20SDK-412991?style=for-the-badge&logo=openai&logoColor=white" height="24px"/>
        </td>
    </tbody>
  </table>
</markdown-accessiblity-table>

<br/>

# 🏗️ Architecture Overview

<img width="100%" alt="image" src="https://github.com/user-attachments/assets/9fa1aa7f-77ed-41f2-906a-c6e295c2e0c7" />

# 🌱 Team Members (팀원 및 팀 소개)

|                                   [주성준](https://github.com/rdyjun)                                   |                                  [백승민](https://github.com/alpin87)                                   |                                 [유제승](https://github.com/Yu-JeSeung)                                 |                                   [우승원](https://github.com/wsw0922)                                   |                               [전승빈](https://github.com/JEON-SEUNGBHIN)                                |
| :-----------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------: |
| <img width="150" height="150" alt="image" src="https://avatars.githubusercontent.com/u/45596014?v=4" /> | <img width="150" height="150" alt="image" src="https://avatars.githubusercontent.com/u/35371121?v=4" /> | <img width="150" height="150" alt="image" src="https://avatars.githubusercontent.com/u/88806404?v=4" /> | <img width="150" height="150" alt="image" src="https://avatars.githubusercontent.com/u/107173046?v=4" /> | <img width="150" height="150" alt="image" src="https://avatars.githubusercontent.com/u/104238055?v=4" /> |
|                                                   BE                                                    |                                                   BE                                                    |                                                   AI                                                    |                                                    FE                                                    |                                                    FE                                                    |

<br><br>

# 💻 오픈소스 라이선스

본 서비스는 아래 오픈소스/오픈웨이트 소프트웨어를 사용합니다.  
배포 형태: **서버 호스팅 추론(가중치 미배포)**, **모델/파라미터 수정 없음**.

---

## gpt-oss-20b

- **License**: Apache License 2.0 · 원문: https://github.com/openai/gpt-oss/blob/main/LICENSE
- **Source**: OpenAI gpt-oss
- **Notes**: 원본 그대로 사용(파인튜닝/양자화/파라미터 변경 없음), 서버 내 추론만 수행

```text
Model: gpt-oss:20b
Digest: sha256:b112e727c6f18875636c56a779790a590d705aec9e1c0eb5a97d51fc2a778583
Short ID: aa4295ac10c3
Deployed at: 2025-10-10 KST
OpenAI SDK (Python): 2.2.0
Modifications: none (no finetune/quant/param changes)
```

---

## Python Runtime Libraries

- OpenAI Python SDK — Apache-2.0
- FastAPI — MIT
- Starlette — BSD-3-Clause
- Uvicorn — BSD-3-Clause
- Pydantic — MIT
- python-dotenv — BSD
- requests — Apache-2.0

**전체 목록: [`THIRD_PARTY_LICENSES.md`](https://github.com/dongsooop/AI/blob/main/THIRD_PARTY_LICENSES.md)**
