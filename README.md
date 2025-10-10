# 1. Project Overview (프로젝트 개요)

- 프로젝트 이름: 동양미래대학교 숲 (동숲)
- 프로젝트 설명: 동양미래대학교 학생들의 교내 생활 편리성 증진을 목적으로 개발된 통합 커뮤니티 및 정보 제공 애플리케이션

<br/>
<img width="975" height="456" alt="image" src="https://github.com/user-attachments/assets/bafe1cd2-f0ec-484d-9317-b484c3d7dfe5" />

<br/>
<br/>

# 2. Team Members (팀원 및 팀 소개)

|                                                [주성준](https://github.com/rdyjun)                                                 |                                                [백승민](https://github.com/alpin87)                                                |                                              [유제승](https://github.com/Yu-JeSeung)                                               |                                                [우승원](https://github.com/wsw0922)                                                |                                            [전승빈](https://github.com/JEON-SEUNGBHIN)                                             |
| :--------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------: |
| <img width="150" height="150" alt="image" src="https://github.com/user-attachments/assets/9026c7ce-f063-4377-9a43-044e72598056" /> | <img width="150" height="150" alt="image" src="https://github.com/user-attachments/assets/9026c7ce-f063-4377-9a43-044e72598056" /> | <img width="150" height="150" alt="image" src="https://github.com/user-attachments/assets/9026c7ce-f063-4377-9a43-044e72598056" /> | <img width="150" height="150" alt="image" src="https://github.com/user-attachments/assets/9026c7ce-f063-4377-9a43-044e72598056" /> | <img width="150" height="150" alt="image" src="https://github.com/user-attachments/assets/9026c7ce-f063-4377-9a43-044e72598056" /> |
|                                                                 BE                                                                 |                                                                 BE                                                                 |                                                                 BE                                                                 |                                                                 FE                                                                 |                                                                 FE                                                                 |

<br/>
<br/>

- **회원가입 & 로그인**:

  - 학과 정보를 포함한 회원가입 후 JWT 기반 인증으로 로그인합니다.

- **맞춤형 공지사항**:

  - 전체 학교 공지와 본인 학과 공지를 필터링하여 확인할 수 있습니다.

- **학식 정보**:

  - 실시간 학식 메뉴를 앱에서 바로 확인할 수 있습니다.

- **시간표 관리**:

  - 시간표 이미지를 업로드하면 AI가 자동으로 인식하여 일정을 등록합니다.
  - 캘린더 UI를 통해 학업 일정을 추가 및 관리할 수 있습니다.

- **도서관 스터디룸 예약**:

  - 앱 내에서 도서관 스터디룸 예약이 가능합니다.

- **커뮤니티 & 모임 모집**:

  - 튜터링, 스터디, 프로젝트 등 다양한 모임을 모집하고 참여할 수 있습니다.

- **중고 거래**:

  - 교재 및 학용품 중고 거래 게시판을 통해 물품을 사고팔 수 있습니다.

- **실시간 채팅**:

  - 1:1 채팅 및 그룹 채팅 기능을 지원합니다.
  - Socket.IO 기반 과팅(과 소개팅) 기능을 제공합니다.

- **AI 챗봇**:

  - 학교 관련 궁금한 점을 챗봇에게 질문하면 실시간 답변을 받을 수 있습니다.

- **통합 검색**:
  - ELK Stack 기반으로 게시글, 공지사항 등을 빠르게 검색할 수 있습니다.

<br/>
<br/>

# 4. Tasks & Responsibilities (작업 및 역할 분담)

| 이름   | 분담                                                                                                       |
| ------ | ---------------------------------------------------------------------------------------------------------- |
| 주성준 | <ul><li>메인 로직 설계</li><li>팀 리딩 및 커뮤니케이션</li><li>어플리케이션 배포</li></ul>                 |
| 백승민 | <ul><li>서버 인프라 구축 </li><li>학식, 채팅, 신고, 검색기능 구현</li><li>편의성 파이프라인 구축</li></ul> |
| 유제승 | <ul><li>언어 필터링 구현</li><li>AI 챗봇 LLM 구현</li> <li>OpenCV를 통한 이미지 분석 및 시간표 생성</li>   |
| 우승원 | <ul><li>UI/UX 설계</li> <li>API 연동</li></ul>                                                             |
| 전승빈 | <ul><li>UI/UX 설계</li> <li>API 연동</li></ul>                                                             |

<br/>
<br/>

# 5. Technology Stack (기술 스택)

## 5.1 Frontend

- **Flutter** 3.x (Dart)

## 5.2 Backend

- **Spring Boot** 3.x (Java 17)
- **NestJS** 10.x (TypeScript)
- **FastAPI** 0.x (Python)

## 5.3 Database

- **PostgreSQL** 14.x
- **Redis** 7.x

## 5.4 Infrastructure & DevOps

- **Oracle Cloud** - 클라우드 인프라
- **Docker** - 컨테이너화
- **Nginx** - 리버스 프록시 및 웹 서버
- **GitHub Actions** - CI/CD

## 5.5 Monitoring & Logging

- **New Relic** - APM 모니터링
- **Elasticsearch** - 로그 분석 및 검색

## 5.6 External Services

- **Firebase** - FCM 푸시 알림
- **AWS S3** - 파일 스토리지

## 5.7 Collaboration Tools

- **Git** - 버전 관리
- **Notion** - 문서화 및 협업
- **Discord** - 팀 커뮤니케이션 및 편의성 파이프라인

<br/>

## 아키텍처

```
backend (API Gateway)
│  REST/gRPC
▼
AI (this repo)
├─ LLM                  # LLM 파이프라인/클라이언트/프롬프트·유틸
├─ text_filtering       # 비속어/유해 표현 감지 API/룰·모델
├─ image_analysis       # OCR/시각 분석 노트북·스크립트
├─ main.py              # 기본 실행 엔트리포인트 (서비스 통합)
└─ app_oss_main.py      # OSS(OpenAI 호환) 엔드포인트 실행
```

---

## 디렉터리 구조

```
AI/
├── LLM/
├── image_analysis/
├── text_filtering/
├── .dockerignore
├── .gitignore
├── Dockerfile
├── Dockerfiles_oss/           # (옵션) OSS 전용
│
Dockerfile
├── app_oss_main.py
└── main.py
```

<br><br>

# 7. Architecture Overview

- 동숲 프로젝트는 마이크로서비스 기반 아키텍처로 설계되었으며, Oracle Cloud 인프라 위에서 구동됩니다.

<img width="1000" height="700" alt="image" src="https://github.com/user-attachments/assets/2b088207-9c65-4065-baaf-7dfb09d34d7d" />

<br/>
<br/>

**Client Layer**

- **Flutter**: Android, iOS 크로스 플랫폼 지원

**API Gateway & Application Layer (Oracle Cloud)**

- **Spring Boot (OpenJDK)**: 메인 REST API 서버
- **NestJS**: 실시간 과팅 채팅 서버
- **Python/FastAPI**: 챗봇 및 텍스트 필터링, OpenCV 이미지 전처리
- **Nginx**: 리버스 프록시
- **ELK Stack**: 로그 수집 및 검색

**Data Layer**

- **PostgreSQL**: 메인 관계형 데이터베이스
- **Redis**: 캐시 및 세션 관리

**External Services**

- **Firebase**: FCM 푸시 알림
- **Discord**: 실시간 백엔드 로그 확인 및 각종 편의성 파이프라인

**DevOps & Infrastructure**

- **Docker**: 컨테이너 기반 배포
- **GitHub**: 소스 코드 관리 및 CI/CD
- **New Relic**: APM 성능 모니터링
- **Oracle Cloud**: 클라우드 인프라 호스팅

<br/>

# 8. Development Workflow (개발 워크플로우)

## 브랜치 전략 (Branch Strategy)

우리의 브랜치 전략은 Git Flow를 기반으로 하며, 다음과 같은 브랜치를 사용합니다.

- Main Branch
  - 배포 가능한 상태의 코드를 유지합니다.
  - 모든 배포는 이 브랜치에서 이루어집니다.
- {Commit Message Convention Style} 기능이름
  - 팀원 각자의 개발 브랜치입니다.
  - 모든 기능 개발은 이 브랜치에서 이루어집니다.
