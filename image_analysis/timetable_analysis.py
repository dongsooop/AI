# 한글로 출력은 하나, 한줄로 표시되는 문제점이 있는 코드

import cv2
import pytesseract
import numpy as np
from PIL import Image
import re

# Tesseract 실행 경로 설정
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

def extract_lecture_from_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    # 이미지 확대 (2배)
    scale = 2
    h, w = gray.shape
    resized = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # 이미지 선명도 향상 (옵션)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(resized, -1, sharpen_kernel)

    # PIL 변환
    pil_image = Image.fromarray(sharpened)

    # Tesseract 설정
    tesseract_config = r"--psm 6 --oem 1 -l kor+eng"
    ocr_text = pytesseract.image_to_string(pil_image, config=tesseract_config)

    print("\n[✅ 추출된 전체 OCR 텍스트]")
    print("=" * 60)
    print(ocr_text)

    # 한 줄로 변환
    flattened_text = " ".join(ocr_text.split())  # 모든 공백 문자 기준으로 하나의 줄로 만듦

    # 출력
    print("\n[📌 한 줄로 정리된 OCR 텍스트]")
    print("=" * 60)
    print(flattened_text)

    # 강의 정보 추출
    results = parse_flattened_text(flattened_text)

    print("\n[📋 정제된 강의 정보]")
    print("=" * 60)
    for item in results:
        print(f"{item['시간']} | {item['강의명']} | {item['교수']} | {item['강의실']}")

def parse_flattened_text(text):
    import re

    # 1. 시간 패턴을 기준으로 강의 블록 구분
    time_pattern = re.compile(r"\d{1,2}:\d{2}~\d{2}:\d{2}")
    split_texts = time_pattern.split(text)
    time_matches = time_pattern.findall(text)

    parsed = []

    # 2. 각 블록에서 강의명/교수/강의실 추출 시도
    for i in range(1, len(split_texts)):
        time = time_matches[i - 1]
        block = split_texts[i].strip()

        # 교수와 강의실 추정
        room_pattern = r"\d{1,3}-\d{1,3}"
        room_matches = re.findall(room_pattern, block)
        names = re.findall(r"[가-힣]{2,}", block)

        if len(names) >= 1 and len(room_matches) >= 1:
            for prof, room in zip(names[-len(room_matches):], room_matches):
                # 강의명은 나머지 문자열에서 추정
                course_candidates = block.split(prof)[0].strip()
                parsed.append({
                    "시간": time,
                    "강의명": course_candidates,
                    "교수": prof,
                    "강의실": room
                })

    return parsed

def main():
    image_path = "data/jeseung_4.png"
    extract_lecture_from_image(image_path)

if __name__ == "__main__":
    main()