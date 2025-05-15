import requests
from bs4 import BeautifulSoup
import csv
import os

# URL 설정
BASE_URL = "https://www.dongyang.ac.kr"

# fnctNo 자동 추출 함수
def extract_fnctno_from_subview(url):
    response = requests.get(url)
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")
    form = soup.select_one("form[action*='/combBbs/']")
    if form:
        action_url = form.get("action", "")
        if "/combBbs/" in action_url and "/list.do" in action_url:
            fnct_no = action_url.split("/")[3]  # /combBbs/dmu/84/list.do -> 84
            return fnct_no
    return None

# department.txt 읽기 및 fnctNo 추출
def load_departments(file_path="data/department.txt"):
    departments = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            print(f"📄 처리 중: {line.strip()}")  # ← 로그 추가
            parts = line.strip().split(",")
            if len(parts) == 3:
                name, subview_url, page_count = parts
                subview_full_url = BASE_URL + subview_url
                
                if "학교공지" in name:  # ✅ 학교공지 예외 처리
                    print(f"📌 {name}: 학교 공지이므로 fnctNo 추출 생략")
                    departments.append((name, subview_url + "?page=", int(page_count), None))
                    continue

                fnct_no = extract_fnctno_from_subview(subview_full_url)
                print(f"🔍 {name}: 추출된 fnctNo = {fnct_no}")  # ← 로그 추가
                if fnct_no:
                    combbbs_url = f"/combBbs/dmu/{fnct_no}/list.do?page="
                    departments.append((name, combbbs_url, int(page_count), fnct_no))
    return departments

# 페이지 크롤링
def crawl_page(url):
    response = requests.get(url)
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")
    return soup

# 학교 공지 페이지 내용 파싱
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
        data.append([num, title, department, writer, date, link])
    return data

# 학과공지 페이지 내용 파싱 (fnctNo 포함)
def parse_department_page(soup, fnct_no):
    rows = soup.select("table.board-table.horizon tbody tr")
    data = []
    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 4:
            continue
        num = cols[0].text.strip()
        title_tag = cols[1].find("a")
        title = title_tag.get_text(strip=True) if title_tag else ""
        onclick = title_tag.get("href", "")
        nttId = None
        if "jf_combBbs_view" in onclick:
            try:
                parts = onclick.split(",")
                nttId = parts[-1].strip("'); ")
                link = f"https://www.dongyang.ac.kr/combBbs/dmu/{fnct_no}/view.do?nttId={nttId}"
            except:
                link = ""
        else:
            link = ""
        writer = cols[2].text.strip()
        date = cols[3].text.strip()
        department = "학과공지"
        data.append([num, title, department, writer, date, link])
    return data

# CSV 파일로 저장
def save_to_csv(data, department_name):
    filename = f"data/dept/{department_name}_notices.csv"
    with open(filename, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["번호", "제목", "부서", "작성자", "작성일", "링크"])
        writer.writerows(data)
    print(f"✅ 저장 완료: {filename}")
    remove_notice_rows(filename)

# '공지' 항목 제거
def remove_notice_rows(input_file, output_file=None):
    if output_file is None:
        output_file = input_file
    filtered_rows = []
    with open(input_file, mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row and row[0].isdigit():
                filtered_rows.append(row)
    with open(output_file, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(filtered_rows)
    print(f"✅ '공지' 항목 제거 완료 → {output_file}")

# 메인 실행
def main():
    departments = load_departments()
    for dept_name, dept_url, page_count, fnct_no in departments:
        print(f"\n📚 {dept_name} 크롤링 시작 ({page_count} 페이지)...")
        all_data = []
        for page in range(1, page_count + 1):
            print(f"📄 {page} 페이지 크롤링 중...")
            full_url = BASE_URL + dept_url + str(page)
            soup = crawl_page(full_url)
            if "combBbs" in dept_url:
                page_data = parse_department_page(soup, fnct_no)
            else:
                page_data = parse_page(soup)
            all_data.extend(page_data)
        save_to_csv(all_data, dept_name)

if __name__ == "__main__":
    main()
