# 워터마크를 확인하고 진정성있는 학교 파일인지 확인 하는 코드
# 트러블 슈팅 해결되면 나머지 작업 할 예정

import cv2
import numpy as np

# 1) 템플릿(워터마크 로고) 이미지 읽기
template = cv2.imread("data/school_logo.png", cv2.IMREAD_COLOR)
t_h, t_w = template.shape[:2]

# 2) 검사 대상 이미지 읽기
img = cv2.imread("test_image.jpg", cv2.IMREAD_COLOR)

# 3) 템플릿 매칭
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

# 4) 매칭 결과에서 최대값(max_val) 찾기
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# max_loc: 최댓값 위치 (탬플릿의 왼쪽 상단 위치)

print("매칭 스코어:", max_val)

# 5) 임계값 설정 후, 워터마크 인식 여부 판별
threshold = 0.8  # 예시
if max_val >= threshold:
    # 워터마크로 추정되는 위치 사각형 표시
    top_left = max_loc
    bottom_right = (top_left[0] + t_w, top_left[1] + t_h)

    # 디버깅 목적으로 빨간색 박스 그리기
    cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

    print("워터마크 발견. 위치:", top_left)
else:
    print("워터마크가 없거나 매칭 스코어 낮음.")

# 6) 결과 이미지 확인 (창으로 띄우거나 파일로 저장)
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()