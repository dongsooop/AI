import requests
from bs4 import BeautifulSoup
import csv

# URL 설정
BASE_URL = "https://www.dongyang.ac.kr"
PAGE_URL = "https://www.dongyang.ac.kr/dmu/4904/subview.do?page="

# 페이지 크롤링
def crawl_page(page):
    url = PAGE_URL + str(page)
    response = requests.get(url)
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")
    return soup

# 페이지 내용 파싱
def parse_page(soup):
    rows = soup.select("table.board-table tbody tr")
    data = []
    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 7:
            continue

        num = cols[0].text.strip()
        title_tag = cols[1].find("a")
        title = title_tag.get_text(strip=True) if title_tag else ""
        link = BASE_URL + title_tag['href'] if title_tag else ""

        department = cols[2].text.strip()
        writer = cols[3].text.strip()
        date = cols[4].text.strip()
        views = cols[5].text.strip()
        files = cols[6].text.strip()

        data.append([num, title, department, writer, date, views, files, link])
    return data

# CSV 파일로 저장
def save_to_csv(data, filename="data/dongyang_notices.csv"):
    with open(filename, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["번호", "제목", "부서", "작성자", "작성일", "조회수", "첨부파일 수", "링크"])
        writer.writerows(data)
    print(f"✅ 저장 완료: {filename}")
    remove_notice_rows("data/dongyang_notices.csv")

# '공지' 항목 제거
def remove_notice_rows(input_file, output_file=None):
    if output_file is None:
        output_file = input_file  # 덮어쓰기

    filtered_rows = []

    with open(input_file, mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row and row[0].isdigit():  # 번호가 숫자인 경우만 저장
                filtered_rows.append(row)

    with open(output_file, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(filtered_rows)

    print(f"✅ '공지' 항목 제거 완료 → {output_file}")

# 메인
def main():
    all_data = []
    for page in range(1, 487):  # 1~5페이지
        print(f"📄 {page} 페이지 크롤링 중...")
        soup = crawl_page(page)
        page_data = parse_page(soup)
        all_data.extend(page_data)

    save_to_csv(all_data)

# 실행
if __name__ == "__main__":
    main()