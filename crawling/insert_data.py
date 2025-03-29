import psycopg2
import csv
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# CSV 파일 경로
CSV_FILE_PATH = "data/dongyang_notices.csv"

def connect_to_db():
    """
    PostgreSQL 데이터베이스에 연결하는 함수
    """
    try:
        connection = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        return connection
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def insert_data_to_db(data):
    """
    CSV 파일 데이터를 PostgreSQL에 INSERT하는 함수
    """
    connection = connect_to_db()
    if connection is None:
        return

    cursor = connection.cursor()

    insert_query = """
    INSERT INTO main_school_notice (id, title, department, writer, date, link)
    VALUES (%s, %s, %s, %s, %s, %s);
    """

    try:
        for row in data:

            # INSERT 쿼리 실행
            cursor.execute(insert_query, (
                row[0], row[1], row[2], row[3], row[4], row[5]
            ))

        connection.commit()  # 커밋하여 변경사항 저장
        print("✅ 데이터가 성공적으로 삽입되었습니다.")
    except Exception as e:
        print(f"데이터 삽입 중 오류 발생: {e}")
    finally:
        cursor.close()
        connection.close()

def read_csv_and_insert():
    """
    CSV 파일을 읽고 데이터베이스에 삽입하는 함수
    """
    with open(CSV_FILE_PATH, mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)  # 헤더는 건너뛰기
        data = list(reader)

    insert_data_to_db(data)

if __name__ == "__main__":
    read_csv_and_insert()