import os
import csv
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
from pdf2image import convert_from_path
from verification_file import is_file_modified

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

def load_valid_departments(csv_path: str):
    valid_depts = set()
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  # 빈 줄이 아닐 경우
                dept_name = row[0].strip()
                if dept_name:
                    valid_depts.add(dept_name)
    return valid_depts


def ocr_extract_info_from_file(file_path: str, dept_csv_path: str):

    valid_depts = load_valid_departments(dept_csv_path)

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    pil_images = []
    if ext == ".pdf":
        pages = convert_from_path(
            file_path, 
            dpi=300,
            poppler_path="/opt/homebrew/bin"  
        )
        pil_images = pages
    else:
        img_cv = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img_cv is None:
            print(f"[오류] 이미지/파일을 불러오지 못했습니다: {file_path}")
            return None
        
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        scale = 2
        h, w = gray.shape[:2]
        resized = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

        pil_images = [Image.fromarray(resized)]

    tesseract_config = r"--psm 6 --oem 1 -l kor"

    full_text = ""
    for pil_img in pil_images:
        text = pytesseract.image_to_string(pil_img, config=tesseract_config)
        full_text += text + "\n---PAGE_BREAK---\n"

    pattern = re.compile(r"(.+?)(?:\(\d+\))?\s+(\d+)\s+(\d+)\s+[|:]?\s*(\d+)\s+(.*)")

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

            if department not in valid_depts:
                print(f"[오류] CSV에 존재하지 않는 학과: '{department}'")
                return None

            return {
                "department": department,
                "year": year,
                "semester": semester,
                "student_id": student_id,
                "name": name
            }

    return None

def main():
    test_file_path = "data/jeseung.png"

    dept_csv_path = "data/Exception_department.csv"

    modified, created_dt, modified_dt, os_kind = is_file_modified(test_file_path)

    print(f"[OS 종류]     : {os_kind}")
    print(f"[파일 경로]   : {test_file_path}")
    print(f"[만든 날짜]   : {created_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[수정 날짜]   : {modified_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    if modified:
        print("결과 : 파일이 수정된 것으로 보입니다. OCR을 진행하지 않습니다.")
        return

    result = ocr_extract_info_from_file(test_file_path, dept_csv_path)
    if result:
        print("[추출된 정보]")
        print(f"학과   : {result['department']}")
        print(f"학년   : {result['year']}")
        print(f"학기   : {result['semester']}")
        print(f"학번   : {result['student_id']}")
        print(f"성명   : {result['name']}")
        print("\n결과 : 재학생이 맞습니다.")
    else:
        print("\n결과 : 재학생이 아닙니다.")

if __name__ == "__main__":
    main()