from sshtunnel import SSHTunnelForwarder
import psycopg2
import os
from dotenv import load_dotenv
import csv
from datetime import datetime

# 환경 변수 로딩
load_dotenv()

# SSH 접속 정보
SSH_HOST = os.getenv("SSH_HOST")
SSH_PORT = 22
SSH_USER = os.getenv("SSH_USER")
SSH_KEY_PATH = os.getenv("SSH_KEY_PATH")

# DB 접속 정보
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# CSV 디렉토리
CSV_DIR = "data/dept"

# 학과 이름 → department_id 매핑
DEPT_ID_MAP = {
    "건축과": "DEPT_6003",
    "경영정보학과": "DEPT_7005",
    "경영학과": "DEPT_7001",
    "교양과": "DEPT_9001",
    "기계공학과": "DEPT_3001",
    "기계설계공학과": "DEPT_3002",
    "로봇소프트웨어과": "DEPT_4002",
    "바이오융합공학과": "DEPT_6002",
    "반도체전자공학과": "DEPT_5002",
    "빅데이터경영과": "DEPT_7006",
    "생명화학공학과": "DEPT_6001",
    "세무회계학과": "DEPT_7002",
    "소방안전관리과": "DEPT_5004",
    "시각디자인과": "DEPT_6005",
    "실내건축디자인과": "DEPT_6004",
    "웹응용소프트웨어공학과": "DEPT_2003",
    "유통마케팅학과": "DEPT_7003",
    "인공지능소프트웨어학과": "DEPT_2002",
    "자동화공학과": "DEPT_4001",
    "자유전공학과": "DEPT_8001",
    "전기공학과": "DEPT_5001",
    "정보통신공학과": "DEPT_5003",
    "컴퓨터소프트웨어공학과": "DEPT_2001",
    "호텔관광학과": "DEPT_7004",
    "AR·VR콘텐츠디자인과": "DEPT_6006",
    "학교공지": "DEPT_1001"
}

def insert_all_departments():
    with SSHTunnelForwarder(
        (SSH_HOST, SSH_PORT),
        ssh_username=SSH_USER,
        ssh_pkey=SSH_KEY_PATH,
        remote_bind_address=('localhost', 5432),
        local_bind_address=('localhost', 6543)
    ) as tunnel:
        print("✅ SSH 터널 연결됨")

        conn = psycopg2.connect(
            host='localhost',
            port=6543,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()

        insert_details_query = """
        INSERT INTO notice_details (id, created_at, link, title, writer)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING;
        """

        insert_notice_query = """
        INSERT INTO notice (notice_details_id, department_id)
        VALUES (%s, %s)
        ON CONFLICT (notice_details_id, department_id) DO NOTHING;
        """

        for filename in os.listdir(CSV_DIR):
            if not filename.endswith(".csv"):
                continue

            dept_name = filename.replace("_notices.csv", "")
            department_id = DEPT_ID_MAP.get(dept_name)

            if department_id is None:
                print(f"⚠️ 알 수 없는 학과: {dept_name}, 건너뜀")
                continue

            file_path = os.path.join(CSV_DIR, filename)
            print(f"\n📄 {dept_name} ({department_id}) → 파일: {file_path}")

            try:
                with open(file_path, mode="r", encoding="utf-8-sig") as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        post_id = int(row[1])
                        title = row[2].strip()
                        writer = row[4].strip()
                        date = datetime.strptime(row[5], "%Y.%m.%d").date()
                        link = row[6].replace("https://www.dongyang.ac.kr", "").strip()

                        cursor.execute(insert_details_query, (post_id, date, link, title, writer))
                        cursor.execute(insert_notice_query, (post_id, department_id))

                conn.commit()
                print(f"✅ {dept_name} 데이터 삽입 성공")

            except Exception as e:
                conn.rollback()
                print(f"❌ {dept_name} 삽입 오류: {e}")

        cursor.close()
        conn.close()
        print("\n🔌 전체 DB 연결 종료")

if __name__ == "__main__":
    insert_all_departments()