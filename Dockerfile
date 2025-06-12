# 베이스 이미지
FROM python:3.11-slim

# 필수 패키지 설치
RUN apt-get update && apt-get install -y git

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 복사 및 설치
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY model/my_electra_finetuned ./model/my_electra_finetuned

COPY data/bad_text_sample.txt ./data/bad_text_sample.txt

# Tile, Tag, Content
COPY text_filtering/text_filtering_rule.py ./text_filtering/text_filtering_rule.py

# Content
COPY text_filtering/text_filtering.py ./text_filtering/text_filtering.py

# FastAPI 실행
CMD ["uvicorn", "text_filtering.text_filtering_rule:app", "--host", "0.0.0.0", "--port", "8888"]