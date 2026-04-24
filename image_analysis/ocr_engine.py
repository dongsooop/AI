from datetime import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
import re
import cv2
import numpy as np
import pytesseract
from PIL import Image


WEEKDAYS_FILE = Path("data/weekdays.txt")
TIME_SLOTS_FILE = Path("data/time_slots.txt")

tesseract_config = r"--psm 6 --oem 1 -l kor+eng"
TARGET_WIDTH = 1170
TARGET_HEIGHT = 2532
SCALE = 2

WEEKDAY_BASE_COL_1 = 3
TIME_FIRST_ROW = 3

_hangul = re.compile(r"[가-힣]")
_only_digits_symbols = re.compile(r"^[\d\W_]+$")
_long_repeat = re.compile(r"(.)\1{3,}")

VDIV, HDIV = 50, 50
H_MIN_LEN_RATIO = 0.60
HOUGH_THRESH, HOUGH_MINLINE, HOUGH_MAXGAP = 120, 80, 10
NEAR_EPS, BAND_MERGE = 6, 8
TRIM_OUTER, TRIM_X_GAP, TRIM_Y_GAP = True, 20, 15
OCR_THREAD_WORKERS = 8


def _ocr_task(roi_bytes: bytes) -> List[str]:
    nparr = np.frombuffer(roi_bytes, np.uint8)
    roi_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    txt = pytesseract.image_to_string(Image.fromarray(roi_gray), config=tesseract_config).strip()
    return [re.sub(r"[|]", "", line.strip().replace(" ", "")) for line in txt.split("\n") if line.strip()]


def _is_valid_course(value: str) -> bool:
    value = value.strip()
    return (
        len(value) >= 2
        and _hangul.search(value) is not None
        and _only_digits_symbols.fullmatch(value) is None
        and _long_repeat.search(value) is None
        and value not in {"|", "-", "_"}
    )


def _is_valid_professor(value: str) -> bool:
    value = value.strip()
    return len(value) >= 2 and _hangul.search(value) is not None and _long_repeat.search(value) is None


def _split_slot(slot: str) -> Optional[tuple[str, str]]:
    if slot is None:
        return None
    parts = slot.split("~", 1)
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1].strip()


def _to_hms(value: str) -> str:
    if not value:
        return value
    parts = value.split(":")
    if len(parts) == 2:
        hour, minute = parts
        return f"{int(hour):02d}:{minute}:00"
    if len(parts) >= 3:
        hour, minute, second = parts[0], parts[1], parts[2]
        return f"{int(hour):02d}:{minute}:{second}"
    return value


def _parse_hms(value: str) -> time:
    parts = value.split(":")
    if len(parts) == 2:
        return time(int(parts[0]), int(parts[1]), 0)
    if len(parts) >= 3:
        return time(int(parts[0]), int(parts[1]), int(parts[2]))
    return time(0, 0, 0)


def _prune_x_lines(x_lines: List[int], base_gray: np.ndarray) -> List[int]:
    if not x_lines:
        return x_lines

    _, width = base_gray.shape[:2]
    margin = max(2, int(width * 0.02))
    xs = [x for x in x_lines if margin <= x <= (width - margin)]
    if len(xs) < 2:
        xs = x_lines[:]

    if len(xs) >= 4:
        gaps = np.diff(xs)
        median_gap = float(np.median(gaps[:-1])) if len(gaps) > 1 else float(np.median(gaps))
        if median_gap > 0 and gaps[-1] > 1.6 * median_gap:
            xs = xs[:-1]
    return xs


def _load_list_from_txt(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def _make_base_gray(img_bgr: np.ndarray) -> np.ndarray:
    resized_fixed = cv2.resize(img_bgr, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
    upscaled = cv2.resize(resized_fixed, (TARGET_WIDTH * SCALE, TARGET_HEIGHT * SCALE), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(gray, -1, sharpen_kernel)


def _vertical_lines(base_gray: np.ndarray) -> tuple[List[int], np.ndarray]:
    threshold = cv2.adaptiveThreshold(base_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 5)
    inverted = 255 - threshold
    vertical_length = max(20, base_gray.shape[0] // VDIV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_length))
    vertical = cv2.dilate(cv2.erode(inverted, kernel, 1), kernel, 1)
    vx = vertical.sum(axis=0)
    indices = np.where(vx > vx.max() * 0.35)[0]
    xs: List[int] = []
    if len(indices):
        current = [indices[0]]
        for index in indices[1:]:
            if index - current[-1] <= NEAR_EPS:
                current.append(index)
            else:
                xs.append(int(np.median(current)))
                current = [index]
        xs.append(int(np.median(current)))
    return sorted(xs), vertical


def _horizontal_lines(base_gray: np.ndarray, vertical_mask: Optional[np.ndarray]) -> tuple[List[int], np.ndarray]:
    threshold = cv2.adaptiveThreshold(base_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
    inverted = 255 - threshold
    horizontal_length = max(20, base_gray.shape[1] // HDIV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_length, 1))
    horizontal = cv2.dilate(cv2.erode(inverted, kernel, 1), kernel, 1)
    if vertical_mask is not None:
        horizontal = cv2.bitwise_and(horizontal, cv2.bitwise_not(vertical_mask))

    lines = cv2.HoughLinesP(
        horizontal,
        1,
        np.pi / 180,
        threshold=HOUGH_THRESH,
        minLineLength=HOUGH_MINLINE,
        maxLineGap=HOUGH_MAXGAP,
    )
    ys: List[int] = []
    if lines is not None:
        _, width = base_gray.shape[:2]
        inner_left, inner_right = int(width * 0.04), int(width * 0.96)
        min_len = int((inner_right - inner_left) * H_MIN_LEN_RATIO)
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(y1 - y2) <= 2 and abs(x2 - x1) >= min_len:
                ys.append((y1 + y2) // 2)
    ys.sort()

    merged: List[int] = []
    if ys:
        current = [ys[0]]
        for y in ys[1:]:
            if y - current[-1] <= NEAR_EPS:
                current.append(y)
            else:
                merged.append(int(np.median(current)))
                current = [y]
        merged.append(int(np.median(current)))

    final_y: List[int] = []
    idx = 0
    while idx < len(merged):
        if idx + 1 < len(merged) and (merged[idx + 1] - merged[idx]) <= BAND_MERGE:
            final_y.append((merged[idx] + merged[idx + 1]) // 2)
            idx += 2
        else:
            final_y.append(merged[idx])
            idx += 1
    return sorted(list(set(final_y))), horizontal


def _iterate_cells_autogrid(base_gray: np.ndarray, x_lines: List[int], y_lines: List[int]):
    for row in range(len(y_lines) - 1):
        for col in range(len(x_lines) - 1):
            x1, x2 = x_lines[col], x_lines[col + 1]
            y1, y2 = y_lines[row], y_lines[row + 1]
            pad_x = max(1, (x2 - x1) // 20)
            pad_y = max(1, (y2 - y1) // 20)
            roi = base_gray[y1 + pad_y:y2 - pad_y, x1 + pad_x:x2 - pad_x]
            yield row, col, roi


def _merge_adjacent_same_name(items: List[dict]) -> List[dict]:
    groups: dict[tuple[str, str], List[dict]] = {}
    for item in items:
        groups.setdefault((item["week"], item["name"]), []).append(item)

    merged: List[dict] = []
    for (week, name), group in groups.items():
        group.sort(key=lambda item: (item.get("period", 10**9), _parse_hms(item["startAt"]), _parse_hms(item["endAt"])))
        current = None
        for item in group:
            period = item["period"]
            start = _parse_hms(item["startAt"])
            end = _parse_hms(item["endAt"])

            if current is None:
                current = {
                    "week": week,
                    "name": name,
                    "professor": item.get("professor", "") or "",
                    "location": item.get("location", "") or "",
                    "startAt": item["startAt"],
                    "endAt": item["endAt"],
                    "periods": [period],
                    "period": period,
                }
                continue

            contiguous_by_period = period == current["periods"][-1] + 1
            contiguous_by_time = _parse_hms(current["endAt"]) <= start
            if contiguous_by_period or contiguous_by_time:
                if end > _parse_hms(current["endAt"]):
                    current["endAt"] = item["endAt"]
                current["periods"].append(period)
                if not current["professor"] and item.get("professor"):
                    current["professor"] = item["professor"]
                if not current["location"] and item.get("location"):
                    current["location"] = item["location"]
            else:
                merged.append(current)
                current = {
                    "week": week,
                    "name": name,
                    "professor": item.get("professor", "") or "",
                    "location": item.get("location", "") or "",
                    "startAt": item["startAt"],
                    "endAt": item["endAt"],
                    "periods": [period],
                    "period": period,
                }
        if current is not None:
            merged.append(current)

    merged.sort(key=lambda item: (item["week"], item["period"], item["startAt"]))
    return merged


def extract_schedule_fixed_scaled(img: np.ndarray) -> List[dict]:
    base_gray = _make_base_gray(img)
    x_lines, vertical_mask = _vertical_lines(base_gray)
    x_lines = _prune_x_lines(x_lines, base_gray)
    y_lines, horizontal_mask = _horizontal_lines(base_gray, vertical_mask=vertical_mask)

    if TRIM_OUTER:
        if len(x_lines) > 2 and (x_lines[1] - x_lines[0] < TRIM_X_GAP):
            x_lines = x_lines[1:]
        if len(x_lines) > 2 and (x_lines[-1] - x_lines[-2] < TRIM_X_GAP):
            x_lines = x_lines[:-1]
        if len(y_lines) > 2 and (y_lines[1] - y_lines[0] < TRIM_Y_GAP):
            y_lines = y_lines[1:]
        if len(y_lines) > 2 and (y_lines[-1] - y_lines[-2] < TRIM_Y_GAP):
            y_lines = y_lines[:-1]

    if len(x_lines) - 1 < 3 or len(y_lines) - 1 < 3:
        return []

    weekdays = _load_list_from_txt(WEEKDAYS_FILE)
    time_slots = _load_list_from_txt(TIME_SLOTS_FILE)
    if not weekdays or not time_slots:
        return []

    tasks: List[bytes] = []
    meta_map: List[tuple[int, int]] = []
    for row, col, roi_gray in _iterate_cells_autogrid(base_gray, x_lines, y_lines):
        if row == 0:
            continue
        ok, buf = cv2.imencode(".png", roi_gray)
        if not ok:
            continue
        tasks.append(buf.tobytes())
        meta_map.append((row, col))

    lines_list: List[List[str]] = []
    if tasks:
        max_workers = min(OCR_THREAD_WORKERS, len(tasks)) or 1
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ocr_cell") as executor:
            lines_list = list(executor.map(_ocr_task, tasks))

    raw_cells: List[dict] = []
    for (row, col), lines in zip(meta_map, lines_list):
        if not lines:
            continue
        course = lines[0]
        professor = lines[1] if len(lines) > 1 else ""
        room = lines[2] if len(lines) > 2 else ""

        if len(course) > 15 or len(professor) > 8 or len(room) > 10:
            continue

        room = room.replace("=", "-")
        if not (_is_valid_course(course) and _is_valid_professor(professor)):
            continue

        raw_cells.append({
            "row": row,
            "col": col + 1,
            "name": course,
            "professor": professor,
            "room": room,
        })

    if not raw_cells:
        return []

    schedule_rows: List[dict] = []
    for cell in raw_cells:
        row = cell["row"]
        col = cell["col"]
        week_idx = col - WEEKDAY_BASE_COL_1
        if not (0 <= week_idx < len(weekdays)):
            continue

        period0 = row - TIME_FIRST_ROW
        if not (0 <= period0 < len(time_slots)):
            continue

        slot = _split_slot(time_slots[period0])
        if not slot:
            continue

        if period0 > 14:
            start_time, end_time = slot
            schedule_rows.append({
                "week": weekdays[week_idx],
                "period": period0,
                "name": cell["name"],
                "professor": cell["professor"],
                "startAt": _to_hms(start_time),
                "endAt": _to_hms(end_time),
                "location": cell["room"],
            })
            continue

        next_slot = _split_slot(time_slots[period0 + 1])
        if not next_slot:
            continue
        start_time, end_time = next_slot
        schedule_rows.append({
            "week": weekdays[week_idx],
            "period": period0 + 1,
            "name": cell["name"],
            "professor": cell["professor"],
            "startAt": _to_hms(start_time),
            "endAt": _to_hms(end_time),
            "location": cell["room"],
        })

    if not schedule_rows:
        return []

    merged = _merge_adjacent_same_name(schedule_rows)
    del base_gray
    del vertical_mask
    del horizontal_mask
    return merged
