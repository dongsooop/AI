# dnn까지 적용해서 PaddleOCR + Tesseract를 사용해봤으나, 정확도가 떨어지는 코드

import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
from paddleocr import PaddleOCR

# ============== 1) PaddleOCR 데모 코드 ==============
def run_paddle_ocr_demo(image_path):
    """
    PaddleOCR 이용:
    - 전체 이미지에 대해 OCR
    - 검출된 박스(4점)에 폴리라인 그려 표시
    """
    # PaddleOCR 객체 생성 (korean + 글자기울임각도 보정)
    ocr = PaddleOCR(lang='korean', use_angle_cls=True, show_log=False)

    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"[PaddleOCR] 이미지 로드 실패: {image_path}")

    # OCR 실행
    result = ocr.ocr(image_path, cls=True)

    # 디버그용 사본
    debug_img = img.copy()

    # 결과 박스 표시
    for line in result:
        for (box, (txt, conf)) in line:
            box = np.array(box).astype(int)  # 좌표 4점 (x,y)
            # 폴리라인 (초록색)
            cv2.polylines(debug_img, [box], True, (0,255,0), 2)

            x_min = min(box[:,0])
            y_min = min(box[:,1])
            # 텍스트와 신뢰도 표시
            cv2.putText(debug_img, f"{txt}({conf:.2f})", (x_min, y_min-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    # 표시
    cv2.imshow("PaddleOCR Demo", debug_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 콘솔 출력
    print("\n[PaddleOCR Output]")
    for line in result:
        for (box, (txt, conf)) in line:
            print(f"[TEXT] {txt}, [CONF] {conf:.2f}")

# ============== 2) Tesseract 야간 시간표 코드 ==============
# Tesseract 설정 (Mac 예시)
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
tesseract_config = r"--psm 6 --oem 1 -l kor+eng" # --dpi 300 추가하면 더 좋음

def extract_night_schedule(image_path):
    """
    Tesseract로 야간 시간표 부분만 좌표를 잡아 셀별 OCR.
    빨간 사각형 표시한 debug 이미지 창을 띄운다.
    """
    # 원본 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"[Tesseract] 이미지 로드 실패: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 확대 + 샤프닝
    scale = 2
    h, w = gray.shape
    resized = cv2.resize(gray, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(resized, -1, sharpen_kernel)

    # 디버그용 BGR 변환
    debug_color = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    # 야간 시간표 좌표
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
            x1 = top_left_x + c*cell_w
            y1 = top_left_y + r*cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h

            # 디버그 빨간 박스
            cv2.rectangle(debug_color, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)

            # ROI 추출
            roi = sharpened[int(y1):int(y2), int(x1):int(x2)]
            # Blur + AdaptiveThreshold
            roi_blur = cv2.GaussianBlur(roi, (3, 3), 0)
            roi_bin  = cv2.adaptiveThreshold(roi_blur, 255, cv2.THRESH_BINARY, 11, 3)

            text = pytesseract.image_to_string(Image.fromarray(roi_bin), config=tesseract_config).strip()
            lines = [l for l in text.split('\n') if l.strip()]

            if not lines:
                continue

            # 첫줄=강의명, 둘째=교수, 셋째=강의실
            course = lines[0].strip()
            professor = lines[1].strip() if len(lines) > 1 else ""
            room = lines[2].strip() if len(lines) > 2 else ""

            if course and professor:
                results.append(
                    f"{weekdays[c]} {r+1} {course} {professor} {time_slots[r]} {room}"
                )

    # 디버그 표시
    debug_show = cv2.resize(debug_color, (1280, 980), interpolation=cv2.INTER_AREA)
    cv2.imshow("Debug Tesseract (night cells)", debug_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return results

def main():
    image_path = "data/jeseung_4.png"

    # 1) PaddleOCR 데모
    print("[1) PaddleOCR Demo]")
    run_paddle_ocr_demo(image_path)
    
    # 2) Tesseract 야간 시간표
    print("[2) Tesseract Night Schedule]")
    schedule = extract_night_schedule(image_path)

    print("\n[📋 야간 강의 시간표]")
    print("=" * 60)
    for item in schedule:
        print(item)

if __name__ == "__main__":
    main()