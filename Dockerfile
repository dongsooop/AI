# 베이스 이미지
FROM python:3.11-slim

# 필수 패키지 설치
RUN apt-get update && apt-get install -y git

# 작업 디렉토리 설정
WORKDIR /app

# 로컬 파일 복사
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 모델 및 코드 복사
COPY ./model ./model
COPY ./data ./data
COPY ./text_filtering ./text_filtering

# FastAPI 실행
CMD ["uvicorn", "text_filltering.text_filltering:app", "--host", "0.0.0.0", "--port", "8888"]