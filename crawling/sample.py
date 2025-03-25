import requests
from bs4 import BeautifulSoup
import csv

BASE_URL = "https://www.dongyang.ac.kr"
LIST_URL = f"{BASE_URL}/bbs/dmu/677/artclList.do"

all_data = []

def fetch_page(page_index):
    data = {
        "pageIndex": page_index,
        "menuSeq": "677",
        "boardSeq": "677"
    }
    response = requests.post(LIST_URL, data=data)
    response.encoding = 'utf-8'
    return BeautifulSoup(response.text, "html.parser")

def parse_notices(soup):
    rows = soup.select("table.board-table tbody tr")
    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 7:
            continue

        num = cols[0].get_text(strip=True)
        title_tag = cols[1].find("a")
        title = title_tag.get_text(strip=True) if title_tag else ""
        link = BASE_URL + title_tag['href'] if title_tag else ""

        department = cols[2].get_text(strip=True)
        writer = cols[3].get_text(strip=True)
        date = cols[4].get_text(strip=True)
        views = cols[5].get_text(strip=True)
        files = cols[6].get_text(strip=True)

        print(f"📌 [{num}] {title}")
        print(f"   🏢 부서: {department} | ✍ 작성자: {writer} | 📅 작성일: {date}")
        print(f"   👁 조회수: {views} | 📎 첨부파일: {files}")
        print(f"   🔗 링크: {link}\n")

        all_data.append([num, title, department, writer, date, views, files, link])

def save_to_csv(filename="data/dongyang_notices.csv"):
    with open(filename, mode='w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["번호", "제목", "부서", "작성자", "작성일", "조회수", "첨부파일 수", "링크"])
        writer.writerows(all_data)
    print(f"\n✅ CSV 파일 저장 완료: {filename}")
    remove_duplicates_from_csv()

def crawl_all_pages(start=1, end=3):
    for page in range(start, end + 1):
        print(f"\n===== ✅ {page} 페이지 =====")
        soup = fetch_page(page)
        parse_notices(soup)

    save_to_csv()

def remove_duplicates_from_csv(input_file="data/dongyang_notices.csv", output_file="data/dongyang_notices.csv"):
    seen = set()
    unique_rows = []

    with open(input_file, mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)  # 헤더는 따로 저장
        for row in reader:
            # 번호 + 제목 기준으로 중복 판단
            key = (row[0], row[1])
            if key not in seen:
                seen.add(key)
                unique_rows.append(row)

    with open(output_file, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(unique_rows)

    print(f"\n✅ 중복 제거 완료! → {output_file} 에 저장됨.")

# 실행
if __name__ == "__main__":
    crawl_all_pages(start=1, end=3)  # 페이지 범위는 필요에 따라 조절 가능