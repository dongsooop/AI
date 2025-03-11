import os
import csv
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
from pdf2image import convert_from_path

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

def load_valid_departments(csv_path: str):
    """
    CSV 파일에서 학과 이름을 모두 읽어와서 set 자료구조로 반환.
    CSV는 한 줄에 학과 이름이 하나씩 있다고 가정.
    """
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
    """
    1) file_path: 이미지(.png, .jpg 등) 또는 PDF(.pdf) 경로
    2) dept_csv_path: 학과 목록이 들어 있는 CSV 파일 경로

    Tesseract OCR을 적용하여 '학과 / 학년 / 학기 / 학번 / 성명' 정보를 추출.
    추출된 학과가 CSV 목록에 존재하지 않으면 오류 메시지 출력 후 None 반환.
    """

    # 우선 CSV에서 유효 학과 목록을 읽어옴
    valid_depts = load_valid_departments(dept_csv_path)

    # 1) 파일 확장자 확인
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # 2) PDF -> 여러 개의 PIL Image 생성
    pil_images = []
    if ext == ".pdf":
        # pdf2image 사용 (poppler_path 필요시 수정)
        pages = convert_from_path(
            file_path, 
            dpi=300,
            poppler_path="/opt/homebrew/bin"  # poppler가 /opt/homebrew/bin에 설치된 경우
        )
        pil_images = pages
    else:
        # 이미지 파일
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

        pil_images = [Image.fromarray(resized)]

    # 3) OCR 설정 (no_preprocess + scale=2, psm=6, oem=1)
    tesseract_config = r"--psm 6 --oem 1 -l kor"

    # 4) OCR 처리 (모든 페이지)
    full_text = ""
    for pil_img in pil_images:
        text = pytesseract.image_to_string(pil_img, config=tesseract_config)
        full_text += text + "\n---PAGE_BREAK---\n"

    # 5) 정규식
    pattern = re.compile(r"(.+?)(?:\(\d+\))?\s+(\d+)\s+(\d+)\s+[|:]?\s*(\d+)\s+(.*)")

    # 6) 줄 단위 확인 (첫 매칭만 반환)
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

            # CSV에 없는 학과인지 체크
            if department not in valid_depts:
                print(f"[오류] CSV에 존재하지 않는 학과: '{department}'")
                return None

            # 정상인 경우 결과 반환
            return {
                "department": department,
                "year": year,
                "semester": semester,
                "student_id": student_id,
                "name": name
            }

    # 매칭된 정보가 전혀 없으면 None 반환
    return None

from verification_file import is_file_modified
# 아래 함수는 이미 질문에서 제공된 OCR 함수 예시 (pytesseract + pdf2image)라고 가정
# "ocr_extract_info_from_file"가 들어있는 모듈을 import한다고 가정하세요.
# 여기서는 편의상 같은 파일에 있다고 가정해도 됩니다.


def main():
    # 1) 사용자 입력 파일 (이미지 or PDF)
    # test_file_path = "data/seungwon.pdf"
    # test_file_path = "data/jeseung_test.pdf"
    # test_file_path = "data/sungjun.png"
    # test_file_path = "data/jeseung.png"
    test_file_path = "data/jeseung_edit_test.png"

    # 2) 학과 예외 목록 CSV
    dept_csv_path = "data/Exception_department.csv"

    # 3) 먼저 수정 파일 여부 검사
    modified, created_dt, modified_dt, os_kind = is_file_modified(test_file_path)

    # 4) 결과 출력
    print(f"[OS 종류]     : {os_kind}")
    print(f"[파일 경로]   : {test_file_path}")
    print(f"[만든 날짜]   : {created_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[수정 날짜]   : {modified_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    if modified:
        print("결과 : 파일이 수정된 것으로 보입니다. OCR을 진행하지 않습니다.")
        return  # or raise an exception, etc.

    # 5) 수정되지 않은 파일이라면 OCR 진행
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