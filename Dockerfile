# 베이스 이미지
FROM python:3.11-slim

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    tesseract-ocr \
    tesseract-ocr-kor \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 복사 및 설치
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 모델 및 데이터 복사
COPY model/my_electra_finetuned ./model/my_electra_finetuned
COPY data/bad_text_sample.txt ./data/bad_text_sample.txt

# 스크립트 복사
COPY text_filtering/text_filtering_rule.py ./text_filtering/text_filtering_rule.py
COPY text_filtering/text_filtering.py ./text_filtering/text_filtering.py
COPY text_filtering/text_filtering_main.py ./text_filtering/text_filtering_main.py

# FastAPI 실행
CMD ["uvicorn", "text_filtering.text_filtering_main:app", "--host", "0.0.0.0", "--port", "8888"]