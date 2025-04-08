import cv2
import numpy as np
import pytesseract
from PIL import Image
import re

# Tesseract 설정 (Mac 예시)
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
tesseract_config = r"--psm 6 --oem 1 -l kor+eng"

def extract_night_schedule_optimized(image_path):
    # 1) 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"[ERROR] 이미지 로딩 실패: {image_path}")
    
    # Grayscale + 확대 + 선명화
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = 2
    resized = cv2.resize(gray, (gray.shape[1]*scale, gray.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(resized, -1, sharpen_kernel)
    debug_color = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    # 시간표 셀 위치 기준
    top_left_x = 305 * scale
    top_left_y = 1200 * scale
    cell_w = 177 * scale
    cell_h = 75 * scale

    weekdays = ["월요일", "화요일", "수요일", "목요일", "금요일"]
    time_slots = [
        "17:30~18:15", "18:20~19:05", "19:15~20:00",
        "20:05~20:50", "20:55~21:40", "21:45~22:30"
    ]

    rows = len(time_slots)
    cols = len(weekdays)
    results = []

    for r in range(rows):
        for c in range(cols):
            x1 = top_left_x + c * cell_w
            y1 = top_left_y + r * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h

            # 디버깅 박스
            cv2.rectangle(debug_color, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            # PIL 기반 OCR 처리 (Threshold 생략)
            roi = sharpened[int(y1):int(y2), int(x1):int(x2)]
            pil_image = Image.fromarray(roi)
            text = pytesseract.image_to_string(pil_image, config=tesseract_config).strip()
            lines = [l for l in text.split('\n') if l.strip()]

            if not lines:
                continue

            # 간단 정리
            course = lines[0].strip()
            professor = lines[1].strip() if len(lines) > 1 else ""
            room = lines[2].strip() if len(lines) > 2 else ""

            if course and professor:
                results.append(f"{weekdays[c]} {r+1} {course} {professor} {time_slots[r]} {room}")

    # 디버그 이미지 출력
    debug_resized = cv2.resize(debug_color, dsize=(1280, 980), interpolation=cv2.INTER_AREA)
    cv2.imshow("Debug Grid - Optimized", debug_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return results

def main():
    image_path = "data/jeseung_4.png"
    schedule = extract_night_schedule_optimized(image_path)

    print("\n[📋 최적화된 야간 강의 시간표]")
    print("=" * 60)
    for item in schedule:
        print(item)

if __name__ == "__main__":
    main()