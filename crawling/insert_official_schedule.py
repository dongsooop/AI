from sshtunnel import SSHTunnelForwarder
import psycopg2
import os
from dotenv import load_dotenv
import csv
from datetime import datetime
import re

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

CSV_FILE = "data/학사일정.csv"

# 날짜 파싱 함수
def parse_date_range(year, raw_date):
    matches = re.findall(r"(\d{2})\.(\d{2})", raw_date)
    if not matches:
        return None, None
    start_date = datetime(year, int(matches[0][0]), int(matches[0][1])).date()
    end_date = start_date
    if len(matches) == 2:
        end_date = datetime(year, int(matches[1][0]), int(matches[1][1])).date()
    return start_date, end_date

def insert_schedule():
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

        insert_query = """
        INSERT INTO official_schedule (title, start_at, end_at, created_at, updated_at, is_deleted)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING;
        """

        with open(CSV_FILE, mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader)  # 헤더 건너뛰기

            for row in reader:
                try:
                    year = int(row[0])
                    raw_date = row[2].strip()
                    title = row[3].strip()
                    now = datetime.now()

                    start_at, end_at = parse_date_range(year, raw_date)
                    if start_at is None:
                        print(f"⚠️ 잘못된 날짜 형식: {raw_date} → 건너뜀")
                        continue

                    cursor.execute(insert_query, (
                        title,
                        start_at,
                        end_at,
                        now,
                        now,
                        False
                    ))

                except Exception as e:
                    conn.rollback()
                    print(f"❌ 삽입 오류: {e}")
                else:
                    conn.commit()

        cursor.close()
        conn.close()
        print("🔌 전체 DB 연결 종료")

if __name__ == "__main__":
    insert_schedule()