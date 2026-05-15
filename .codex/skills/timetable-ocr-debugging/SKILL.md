---
name: timetable-ocr-debugging
description: Debug Dongsooop timetable OCR behavior. Use when investigating timetable image upload failures, Tesseract Korean OCR issues, OpenCV grid detection, cell splitting, course/professor/location extraction, time-slot mapping, Spring timetable push behavior, or regressions in image_analysis/* and debug/timetable/*.
---

# 시간표 OCR 디버깅

## 개요

Dongsooop 시간표 이미지 분석 문제를 디버깅할 때 이 스킬을 사용합니다. 텍스트 필터링 서비스와 같은 `main.py` 서비스 안에 있지만, 챗봇 OSS 서비스와는 분리해서 봅니다. Tesseract OCR, OpenCV 격자 탐지, 셀 분할, 강의명/교수명/강의실 추출, 시간대 매핑, Spring 서버 전송 흐름을 중심으로 확인합니다.

## 필수 확인 문맥

코드를 수정하기 전에 다음 파일을 먼저 읽습니다.

1. `docs/AGENTS.md`: 저장소 가드레일
2. `docs/PLANS.md`: 두 FastAPI 서비스 구조와 OCR 영역 설명
3. 시간표 분석 관련 파일:
   - `image_analysis/service.py`
   - `image_analysis/ocr_engine.py`
   - `image_analysis/timetable_analysis.py`
   - `main.py`에서 라우터 연결이 관련될 때
4. 디버그/보조 데이터:
   - `debug/timetable/debug_timetable.py`
   - `debug/timetable/timetable.py`
   - `data/weekdays.txt`
   - `data/time_slots.txt`

환경이나 배포 문제가 관련될 때만 `Dockerfile`, `requirements.txt`, `.env.example`을 확인합니다.

## 가드레일

- 챗봇 OSS 서비스(`app_oss_main.py`, `LLM/`)와 섞지 않습니다.
- OCR 로직을 바꿀 때 한국어 인식 특성을 유지합니다. `kor+eng`, `--psm 6`, 샤프닝, 고정 리사이즈, SCALE 값은 결과에 직접 영향을 줍니다.
- `data/weekdays.txt`와 `data/time_slots.txt`의 순서와 길이는 요일/교시 매핑에 영향을 주므로 신중하게 다룹니다.
- 테스트 이미지, 원본 데이터, 디버그 덤프가 공개 가능한지 확인합니다. `data/`와 `debug/`는 민감하거나 로컬 전용 파일이 섞이기 쉽습니다.
- `.env`, JWT, Spring 서버 URL, AppCheck 토큰, 운영 서버 정보는 커밋하지 않습니다.
- 사용자가 명시적으로 요청하지 않는 한 `Dockerfile`의 Tesseract 설치와 `.env` 복사 전제를 제거하지 않습니다.

## 디버깅 흐름

1. 실패를 재현합니다. 가능하면 실패 이미지, API 응답, 로그, 기대 결과를 함께 확인합니다.
2. 실패 유형을 분류합니다.
   - 이미지 업로드/디코딩 실패
   - JWT 또는 요청 헤더 오류
   - 큐/동시 처리/타임아웃 문제
   - Tesseract 설치 또는 `kor` 언어팩 문제
   - OpenCV 격자 라인 검출 실패
   - 셀 ROI 분할 오류
   - OCR 텍스트 노이즈 또는 누락
   - 강의명/교수명/강의실 필터링 오류
   - 요일/시간대 매핑 오류
   - 인접 교시 병합 오류
   - Spring 서버 전송 실패
3. 요청 흐름을 따라갑니다.
   - API 라우터: `image_analysis/timetable_analysis.py`
   - 업로드, 큐, Spring 전송: `image_analysis/service.py`
   - 실제 OCR/격자/파싱: `image_analysis/ocr_engine.py`
4. 환경 의존성을 먼저 확인합니다.
   - `tesseract-ocr`
   - `tesseract-ocr-kor`
   - `pytesseract`
   - `opencv-python-headless`
   - `Pillow`
   - `SPRING_TIMETABLE_URL`
5. 코드 변경은 실패 유형에 맞춰 가장 좁게 적용합니다.
6. 가능한 경우 로컬 디버그 스크립트나 API 통합 스크립트로 검증합니다.

```bash
python debug/timetable/debug_timetable.py
python debug/timetable/timetable.py
```

## 격자 탐지 점검

격자 라인이 부족하거나 셀이 어긋날 때 확인합니다.

- `TARGET_WIDTH`, `TARGET_HEIGHT`, `SCALE`이 실제 입력 이미지 비율과 맞는지 봅니다.
- `_make_base_gray()`의 리사이즈, 업스케일, 샤프닝 순서가 유지되는지 확인합니다.
- 수직 라인은 `VDIV`, `NEAR_EPS`, `_prune_x_lines()` 영향을 받습니다.
- 수평 라인은 `HDIV`, `H_MIN_LEN_RATIO`, `HOUGH_THRESH`, `HOUGH_MINLINE`, `HOUGH_MAXGAP`, `BAND_MERGE` 영향을 받습니다.
- 외곽선 제거가 필요한 문제인지, 실제 시간표 내부 라인을 잘라내는 문제인지 구분합니다.
- `debug/timetable/debug_timetable.py`의 grid overlay로 라인이 실제 셀 경계에 맞는지 확인합니다.

## OCR 텍스트 점검

텍스트가 깨지거나 노이즈가 많을 때 확인합니다.

- Tesseract가 `kor+eng`로 실행되는지 확인합니다.
- `--psm 6 --oem 1` 변경은 셀 단위 OCR 결과를 크게 바꿀 수 있으므로 비교 샘플을 남깁니다.
- `_ocr_task()`가 공백 제거와 `|` 제거를 수행하므로, 후처리 때문에 유효 텍스트가 사라지는지 확인합니다.
- `_is_valid_course()`와 `_is_valid_professor()`는 한글 포함, 길이, 반복 문자, 숫자/기호 노이즈를 필터링합니다.
- 강의실은 OCR 품질이 불안정할 수 있으므로 강의명/교수명 필터와 같은 기준을 무리하게 적용하지 않습니다.

## 시간표 매핑 점검

요일이나 시간이 틀릴 때 확인합니다.

- `WEEKDAY_BASE_COL_1`과 `TIME_FIRST_ROW`가 현재 그리드 row/col 기준과 맞는지 확인합니다.
- `data/weekdays.txt`의 요일 순서가 결과의 `week` 값과 일치하는지 확인합니다.
- `data/time_slots.txt`의 순서가 교시 row와 일치하는지 확인합니다.
- `_split_slot()`, `_to_hms()`, `_parse_hms()`가 입력 형식 변화에 안전한지 봅니다.
- `_merge_adjacent_same_name()`이 같은 요일/강의명만 병합하는지, 서로 다른 수업을 과하게 합치지 않는지 확인합니다.

## API와 Spring 전송 점검

API 호출이나 결과 전송이 실패할 때 확인합니다.

- 업로드 파일이 `cv2.imdecode()`에서 정상 이미지로 읽히는지 확인합니다.
- `Authorization: Bearer ...` 헤더가 없거나 잘못되면 `invalid_authorization_header`가 발생합니다.
- 같은 사용자 중복 작업은 `timetable_already_processing`으로 막힙니다.
- 큐가 가득 차면 `timetable_queue_timeout`이 발생할 수 있습니다.
- `SPRING_TIMETABLE_URL`이 없으면 설정 오류가 납니다.
- Spring 전송 payload의 `name`, `professor`, `location`, `week`, `startAt`, `endAt`, `year`, `semester`가 기대 형식인지 확인합니다.

## 출력 형식

결과를 보고할 때 다음 내용을 포함합니다.

1. 실패 유형과 원인
2. 확인하거나 변경한 파일
3. 실행한 검증과 정확한 명령
4. 남은 리스크, 특히 특정 이미지 샘플, Tesseract 설치, Spring 서버 연동, 운영 환경 의존성 중 검증하지 못한 부분

코드 변경이 필요 없다면 필요한 환경 설치, 테스트 이미지 교체, 또는 운영 설정 수정 단계를 명확히 설명합니다.
