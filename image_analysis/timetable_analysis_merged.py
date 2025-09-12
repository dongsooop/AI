import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
tesseract_config = r"--psm 6 --oem 1 -l kor+eng"
TARGET_WIDTH = 1170
TARGET_HEIGHT = 2532
SCALE = 2

def load_image_auto(path):
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".pdf":
        pages = convert_from_path(path, dpi=300)
        pil_image = pages[0]
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"[ERROR] 이미지 로딩 실패: {path}")
    return img

def extract_schedule_fixed_scaled(img):
    resized_fixed = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
    resized = cv2.resize(resized_fixed, (TARGET_WIDTH * SCALE, TARGET_HEIGHT * SCALE), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
    debug_color = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    top_left_x = 318 * SCALE
    top_left_y = 670 * SCALE
    cell_w = 160 * SCALE
    cell_h = 66 * SCALE

    weekdays = ["월요일", "화요일", "수요일", "목요일", "금요일"]
    time_slots = [
        "09:00~09:50", "10:00~10:50", "11:00~11:50", "12:00~12:50",
        "13:00~13:50", "14:00~14:50", "15:00~15:50", "16:00~16:50",
        "17:00~17:50", "18:00~18:50", "19:00~19:50", "20:00~20:50",
        "21:00~21:50", "22:00~22:50",
        "17:30~18:15", "18:20~19:05", "19:15~20:00",
        "20:05~20:50", "20:55~21:40", "21:45~22:30"
    ]

    results = []
    rows, cols = len(time_slots), len(weekdays)

    for r in range(rows):
        for c in range(cols):
            x1 = top_left_x + c * cell_w
            y1 = top_left_y + r * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h

            roi = sharpened[int(y1):int(y2), int(x1):int(x2)]
            pil_image = Image.fromarray(roi)
            text = pytesseract.image_to_string(pil_image, config=tesseract_config).strip()
            lines = [line.strip() for line in text.split("\n") if line.strip()]

            if not lines:
                continue
            course = lines[0]
            professor = lines[1] if len(lines) > 1 else ""
            room = lines[2] if len(lines) > 2 else ""

            if course and professor:
                results.append(f"{weekdays[c]} {r+1} {course} {professor} {time_slots[r]} {room}")
            cv2.rectangle(debug_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

    preview = cv2.resize(debug_color, (800, 1200))
    cv2.imshow("Debug Grid (Fixed + Scaled)", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return results

def main():
    input_path = "data/capture_heetae.png"
    img = load_image_auto(input_path)
    schedule = extract_schedule_fixed_scaled(img)

    print("\n[📋 추출된 강의 시간표]")
    print("=" * 60)
    for item in schedule:
        print(item)

if __name__ == "__main__":
    main()