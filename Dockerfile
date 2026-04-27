FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    git \
    tesseract-ocr \
    tesseract-ocr-kor \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY core/ ./core/
COPY text_filtering/ ./text_filtering/
COPY image_analysis/ ./image_analysis/
COPY model/my_electra_finetuned ./model/my_electra_finetuned
COPY data/bad_text_sample.txt ./data/bad_text_sample.txt
COPY data/eng_bad_text.txt ./data/eng_bad_text.txt
COPY main.py ./main.py
COPY .env ./.env

EXPOSE 8888
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8888"]
