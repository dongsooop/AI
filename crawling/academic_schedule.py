from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import csv
import pandas as pd
import os

def calculate_academic_year(event_year: int, event_month: int) -> int:
    """
    - 3월 ~ 12월: 해당 연도 = 학년도
    - 1월, 2월: 전년도 = 학년도
    """
    return event_year if event_month >= 3 else event_year - 1

def main():
    options = Options()
    options.add_argument("--headless")  # 브라우저 창 없이 실행하려면

    driver = webdriver.Chrome(options=options)
    BASE_URL = "https://www.dongyang.ac.kr/dmu/4749/subview.do#"
    driver.get(BASE_URL)
    time.sleep(2)

    # 전체일정 탭 클릭
    driver.find_element(By.XPATH, "//button[contains(@onclick,'typeChange') and contains(text(),'전체일정')]").click()
    time.sleep(2)

    result = []

    for year in range(2010, 2026):
        print(f"📅 {year}년 처리 중...")

        # 현재 선택된 연도 확인
        while True:
            try:
                year_value = driver.find_element(By.ID, "year").get_attribute("value")
                current_year = int(year_value)
                if current_year == year:
                    break
                elif current_year < year:
                    driver.find_element(By.CSS_SELECTOR, "a.DirectionRight.next").click()
                else:
                    driver.find_element(By.CSS_SELECTOR, "a.DirectionLeft.prev").click()
                time.sleep(1)
            except:
                time.sleep(1)

        time.sleep(2)  # 페이지 안정화

        # 학사일정 항목들 추출
        month_blocks = driver.find_elements(By.CSS_SELECTOR, "div.year-wrap ul li")
        for block in month_blocks:
            try:
                month_text = block.find_element(By.CLASS_NAME, "box-month").text.replace("월", "").strip()
                month = int(month_text)
                entries = block.find_elements(By.CLASS_NAME, "list-box")
                for entry in entries:
                    date = entry.find_element(By.CLASS_NAME, "list-date").text.strip().replace("\n", " ")
                    title = entry.find_element(By.CLASS_NAME, "list-content").text.strip()

                    # 학년도 계산
                    academic_year = calculate_academic_year(year, month)

                    result.append([year, month, date, title])
            except Exception as e:
                print(f"⚠️ 오류 발생: {e}")
                continue

    # 폴더 생성
    os.makedirs("data/schedule", exist_ok=True)

    # CSV 저장
    with open("data/schedule/학사일정.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["학년도", "월", "날짜", "일정명"])
        writer.writerows(result)

    print("✅ 학사일정 크롤링 완료")
    driver.quit()


# def calculate_academic_year(year: int, month: int) -> int:
#     return year if 3 <= month <= 12 else year - 1

def prepend_all_months(filepath):
        # 기존 파일 경로
    file_path = "data/schedule/학사일정.csv"

    # CSV 불러오기
    df = pd.read_csv(file_path)

    # '년도' 계산 함수
    def calculate_year(row):
        academic_year = int(row['학년도'])
        month = int(row['월'])
        return academic_year + 1 if month in [1, 2] else academic_year

    # 새로운 '년도' 열 추가
    df['년도'] = df.apply(calculate_year, axis=1)

    # 열 순서 정렬
    df = df[['학년도', '년도', '월', '날짜', '일정명']]

    # 저장
    output_path = "data/schedule/학사일정_년도추가.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"✅ 변환 완료! → '{output_path}'")


if __name__ == "__main__":
    # main()
    prepend_all_months("data/schedule/학사일정.csv")