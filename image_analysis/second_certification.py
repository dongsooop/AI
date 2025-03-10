import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
from pdf2image import convert_from_path

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

def ocr_extract_info_from_file(file_path: str):
    """
    file_path가 이미지(.png, .jpg 등)인지, PDF(.pdf)인지 확인 후,
    Tesseract OCR을 적용하여 '학과 / 학년 / 학기 / 학번 / 성명' 정보를 추출한다.
    매칭되는 정보를 찾으면 즉시 반환하고, 없으면 None 반환.
    """

    # 1) 파일 확장자 확인
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # 2) PDF -> 여러 개의 PIL Image
    pil_images = []
    if ext == ".pdf":
        # poppler_path는 pdfinfo 위치에 맞춰 지정. (Homebrew Poppler가 /opt/homebrew/bin/ 일 경우)
        pages = convert_from_path(file_path, dpi=300, poppler_path="/opt/homebrew/bin")
        pil_images = pages
    else:
        # 이미지로 처리
        img_cv = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img_cv is None:
            print(f"[오류] 이미지/파일을 불러오지 못했습니다: {file_path}")
            return None
        
        # 흑백 변환
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # 해상도 2배 확대
        scale = 2
        h, w = gray.shape[:2]
        resized = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

        # PIL 이미지로 변환
        pil_images = [Image.fromarray(resized)]

    # 3) Tesseract 옵션
    tesseract_config = r"--psm 6 --oem 1 -l kor"

    # 4) OCR 수행 - 모든 페이지(또는 이미지)에 대해
    full_text = ""
    for idx, pil_img in enumerate(pil_images):
        text = pytesseract.image_to_string(pil_img, config=tesseract_config)
        full_text += text + "\n---PAGE_BREAK---\n"

    # 5) 정규식 준비
    pattern = re.compile(r"(.+?)\s+(\d+)\s+(\d+)\s+[|:]?\s*(\d+)\s+(.*)")

    # 6) 첫 번째 매칭만 반환
    lines = full_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = pattern.search(line)
        if match:
            department = match.group(1).strip()
            year       = match.group(2).strip()
            semester   = match.group(3).strip()
            student_id = match.group(4).strip()
            name       = match.group(5).strip()

            # 매칭되면 즉시 결과 반환 (dict)
            return {
                "department": department,
                "year": year,
                "semester": semester,
                "student_id": student_id,
                "name": name
            }

    # 7) 여기까지 왔다면 매칭 실패
    return None


if __name__ == "__main__":
    # test_file_path: PDF나 이미지(.png, .jpg 등) 파일 경로
    test_file_path = "data/sungjun.png"
    # test_file_path = "data/jeseung.png"
    # test_file_path = "data/seungwon.pdf"
    result = ocr_extract_info_from_file(test_file_path)
    if result:
        print("[추출된 정보]")
        print(f"학과     : {result['department']}")
        print(f"학년     : {result['year']}")
        print(f"학기     : {result['semester']}")
        print(f"학번     : {result['student_id']}")
        print(f"성명     : {result['name']}")
    else:
        print("정규식에 매칭되는 정보가 없습니다.")