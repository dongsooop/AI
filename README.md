# AI

### 추후 개발 진행 사항 정리하기

1. ~~두 코드를 하나로 합치기 or 외부 코드 불러오기 방식으로 수정 <br>
   (second_certification, vertification)~~
2. ~~생성날짜, 수정날짜 인식해서 읽어온 파일 검증~~
3. ~~비속어 필터링 모델 구축 및 데이터 확보
   -> ElectraForSequenceClassification 모델 사용~~
4. 강의 시간표 입력 시 텍스트 인식하여 시간표 입력해주는 코드 작성
   - 트러블 슈팅에 작성한 내용 확인 해야함
5. 학교 사이트 내용 크롤링하기(~~학교공지~~, 학과별공지)

---

- 4/8 : 이제 명확한 개발 진행 사항은 노션에 정리할 것.
- 3/27 : 학교 공지 전체 내용 크롤링 성공, local db insert and env파일에 민감 정보 작성
- 3/25 : 라벨링한 데이터 기준으로 모델 학습
- 3/21 : 기존 만들어진 ElectraForSequenceClassification 모델 사용
- 3/11 : png 파일이나 pdf 읽어서 내용 확인 + 예외처리에 사용될 학과 리스트 파일로 검증 추가
