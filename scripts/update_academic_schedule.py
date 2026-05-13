import argparse
import datetime as dt
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


BASE_URL = "https://www.dongyang.ac.kr/dmu/4749/subview.do"
OUTPUT_CSV = Path("data/schedule/학사일정_년도추가.csv")
REQUIRED_COLUMNS = ["학년도", "년도", "월", "날짜", "일정명"]


def calendar_year_for(academic_year: int, month: int) -> int:
    return academic_year + 1 if month in (1, 2) else academic_year


def build_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1280,1800")
    return webdriver.Chrome(options=options)


def wait_for_schedule(driver: webdriver.Chrome) -> WebDriverWait:
    return WebDriverWait(driver, 20)


def open_all_schedule(driver: webdriver.Chrome) -> None:
    wait = wait_for_schedule(driver)
    driver.get(BASE_URL)

    try:
        all_tab = wait.until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "//button[contains(@onclick,'typeChange') and contains(normalize-space(.),'전체일정')]",
                )
            )
        )
        all_tab.click()
    except TimeoutException:
        # The page can already render the all-schedule view.
        pass

    wait.until(EC.presence_of_element_located((By.ID, "year")))
    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.year-wrap ul li")))


def selected_academic_year(driver: webdriver.Chrome) -> int:
    raw = driver.find_element(By.ID, "year").get_attribute("value")
    return int(raw)


def move_to_academic_year(driver: webdriver.Chrome, target_year: int) -> None:
    wait = wait_for_schedule(driver)

    for _ in range(30):
        current_year = selected_academic_year(driver)
        if current_year == target_year:
            return

        if current_year < target_year:
            driver.find_element(By.CSS_SELECTOR, "a.DirectionRight.next").click()
        else:
            driver.find_element(By.CSS_SELECTOR, "a.DirectionLeft.prev").click()

        wait.until(lambda d: selected_academic_year(d) != current_year)
        wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.year-wrap ul li")))

    raise RuntimeError(f"Could not move academic schedule to {target_year}.")


def scrape_academic_year(target_year: int) -> pd.DataFrame:
    driver = build_driver()
    try:
        open_all_schedule(driver)
        move_to_academic_year(driver, target_year)

        rows = []
        month_blocks = driver.find_elements(By.CSS_SELECTOR, "div.year-wrap ul li")
        for block in month_blocks:
            month_text = block.find_element(By.CLASS_NAME, "box-month").text
            month = int(month_text.replace("월", "").strip())
            year = calendar_year_for(target_year, month)

            for entry in block.find_elements(By.CLASS_NAME, "list-box"):
                date_text = entry.find_element(By.CLASS_NAME, "list-date").text.strip()
                title = entry.find_element(By.CLASS_NAME, "list-content").text.strip()
                if not date_text or not title:
                    continue
                rows.append(
                    {
                        "학년도": target_year,
                        "년도": year,
                        "월": month,
                        "날짜": " ".join(date_text.split()),
                        "일정명": " ".join(title.split()),
                    }
                )
    finally:
        driver.quit()

    if not rows:
        raise RuntimeError(f"No academic schedule rows scraped for {target_year}.")

    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS).drop_duplicates()


def load_existing_schedule(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Existing schedule CSV is missing columns: {missing}")

    return df[REQUIRED_COLUMNS]


def write_updated_schedule(path: Path, target_year: int, scraped: pd.DataFrame) -> None:
    existing = load_existing_schedule(path)
    existing["학년도"] = existing["학년도"].astype(str)

    kept = existing[existing["학년도"] != str(target_year)].copy()
    updated = pd.concat([kept, scraped.astype(str)], ignore_index=True)
    updated = updated.drop_duplicates(subset=REQUIRED_COLUMNS, keep="last")

    path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(path, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update Dongyang Mirae academic schedule CSV.")
    parser.add_argument(
        "--target-year",
        type=int,
        default=dt.date.today().year,
        help="Academic year to refresh. Defaults to the current calendar year.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        help="Final schedule CSV path consumed by the chatbot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scraped = scrape_academic_year(args.target_year)
    write_updated_schedule(args.output, args.target_year, scraped)
    print(f"Updated {args.output} with {len(scraped)} rows for {args.target_year} academic year.")


if __name__ == "__main__":
    main()
