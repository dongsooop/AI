# 빨간색 박스로 원하는 부분을 추출하긴 했으나, 정확도가 떨어지는 코드

import cv2
import numpy as np
import pytesseract
from PIL import Image
import re

# Tesseract 설정 (Mac 예시)
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
tesseract_config = r"--psm 6 --oem 1 -l kor+eng --dpi 300"


def extract_night_schedule(image_path):
    # 1) 원본 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"[ERROR] 이미지 로딩 실패: {image_path}")

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) 확대 + 선명화
    scale = 2
    height, width = gray.shape
    resized = cv2.resize(gray, (width*scale, height*scale), interpolation=cv2.INTER_CUBIC)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(resized, -1, sharpen_kernel)

    # ▼ 디버그용: sharpened를 BGR로 변환해 사각형을 그릴 예정
    debug_color = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    # 야간 시간표: 좌표(예시)
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

            # 빨간 사각형 디버그
            cv2.rectangle(debug_color, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            # ROI 추출
            roi = sharpened[int(y1):int(y2), int(x1):int(x2)]
            # 전처리
            roi_blur = cv2.GaussianBlur(roi, (3, 3), 0)
            roi_bin = cv2.adaptiveThreshold(
                roi_blur, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 3
            )

            text = pytesseract.image_to_string(
                Image.fromarray(roi_bin),
                config=tesseract_config
            ).strip()
            lines = [l for l in text.split('\n') if l.strip()]

            if not lines:
                continue

            # 간단히 (첫 줄=강의명, 둘째=교수, 셋째=강의실)
            course = lines[0].strip()
            professor = lines[1].strip() if len(lines) > 1 else ""
            room = lines[2].strip() if len(lines) > 2 else ""

            if course and professor:
                results.append(
                    f"{weekdays[c]} {r+1} {course} {professor} {time_slots[r]} {room}"
                )

    # 최종 디버그 이미지를 한 번만 띄워서 확인
    # 원본 sharpened 크기가 크면 창이 클 수 있음, 필요하면 resize
    dst = cv2.resize(debug_color, dsize=(1280, 980), interpolation=cv2.INTER_AREA)
    cv2.imshow("Debug Grid with Red Boxes", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return results

def main():
    path = "data/jeseung_4.png"
    schedule = extract_night_schedule(path)

    print("\n[📋 야간 강의 시간표]")
    print("=" * 60)
    for item in schedule:
        print(item)

if __name__ == "__main__":
    main()