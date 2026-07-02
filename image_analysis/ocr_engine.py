from datetime import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional
import logging
import re
import time as perf_time
import cv2
import numpy as np
import pytesseract
from PIL import Image

from core.logging import (
    RuntimeComponent,
    RuntimeOperation,
    RuntimeStatus,
    configure_logging,
    get_logger,
    runtime_log_message,
)


logger = get_logger(__name__)
OCR_WORKER_SERVICE_NAME = "main-api"

WEEKDAYS_FILE = Path("data/weekdays.txt")
TIME_SLOTS_FILE = Path("data/time_slots.txt")

tesseract_config = r"--psm 6 --oem 1 -l kor+eng"
fallback_tesseract_config = r"--psm 11 --oem 1 -l kor+eng"
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
LOW_OCR_CONFIDENCE = 55.0
MAX_REJECTED_CELLS_IN_DIAGNOSTICS = 120
EMPTY_CELL_FOREGROUND_DENSITY_THRESHOLD = 0.005
MAX_RUNTIME_FALLBACK_CELLS = 5
EMPTY_FALLBACK_FOREGROUND_DENSITY_THRESHOLD = 0.05
MIN_TEXT_COMPONENTS_IN_CELL = 1


def configure_ocr_worker_logging() -> None:
    configure_logging(OCR_WORKER_SERVICE_NAME)


def _ensure_ocr_engine_logging() -> None:
    if not logging.getLogger().handlers:
        configure_ocr_worker_logging()


def _clean_ocr_lines(text: str) -> List[str]:
    return [re.sub(r"[|]", "", line.strip().replace(" ", "")) for line in text.split("\n") if line.strip()]


def _ocr_confidence(roi_gray: np.ndarray, config: str) -> Optional[float]:
    try:
        data = pytesseract.image_to_data(Image.fromarray(roi_gray), config=config, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractError:
        return None

    values: List[float] = []
    for value in data.get("conf", []):
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            continue
        if confidence >= 0:
            values.append(confidence)
    if not values:
        return None
    return round(float(np.mean(values)), 2)


def _run_ocr(roi_gray: np.ndarray, config: str, config_name: str) -> Dict[str, Any]:
    text = pytesseract.image_to_string(Image.fromarray(roi_gray), config=config).strip()
    lines = _clean_ocr_lines(text)
    return {
        "config": config_name,
        "lines": lines,
        "raw_text": text,
        "confidence": _ocr_confidence(roi_gray, config),
    }


def _score_ocr_candidate(candidate: Dict[str, Any]) -> float:
    confidence = candidate.get("confidence")
    score = float(confidence) if confidence is not None else 0.0
    lines = candidate.get("lines") or []
    if lines:
        score += 8.0
    if len(lines) >= 2:
        score += 10.0
    if lines and _is_valid_course(lines[0]):
        score += 12.0
    if len(lines) >= 2 and _is_valid_professor(lines[1]):
        score += 12.0
    return score


def _ocr_task(roi_bytes: bytes) -> List[str]:
    nparr = np.frombuffer(roi_bytes, np.uint8)
    roi_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    text = pytesseract.image_to_string(Image.fromarray(roi_gray), config=tesseract_config).strip()
    return _clean_ocr_lines(text)


def _ocr_fallback_task(roi_bytes: bytes) -> Dict[str, Any]:
    nparr = np.frombuffer(roi_bytes, np.uint8)
    roi_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    text = pytesseract.image_to_string(Image.fromarray(roi_gray), config=fallback_tesseract_config).strip()
    return {
        "lines": _clean_ocr_lines(text),
        "confidence": None,
        "config": "psm11",
        "fallback_used": True,
    }


def _is_acceptable_ocr_candidate(candidate: Dict[str, Any]) -> bool:
    lines = candidate.get("lines") or []
    if not lines:
        return False
    course = lines[0]
    professor = lines[1] if len(lines) > 1 else ""
    room = lines[2] if len(lines) > 2 else ""
    if len(course) > 15 or len(professor) > 8 or len(room) > 10:
        return False
    return _is_valid_course(course) and _is_valid_professor(professor)


def _ocr_diagnostic_task(roi_bytes: bytes) -> Dict[str, Any]:
    nparr = np.frombuffer(roi_bytes, np.uint8)
    roi_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    primary = _run_ocr(roi_gray, tesseract_config, "psm6")
    needs_fallback = (
        not primary["lines"]
        or len(primary["lines"]) < 2
        or (primary["confidence"] is not None and primary["confidence"] < LOW_OCR_CONFIDENCE)
    )
    if not needs_fallback:
        primary["fallback_used"] = False
        return primary

    fallback = _run_ocr(roi_gray, fallback_tesseract_config, "psm11")
    primary_ok = _is_acceptable_ocr_candidate(primary)
    fallback_ok = _is_acceptable_ocr_candidate(fallback)
    if primary_ok != fallback_ok:
        best = fallback if fallback_ok else primary
    else:
        best = fallback if _score_ocr_candidate(fallback) > _score_ocr_candidate(primary) else primary
    best["fallback_used"] = best is fallback
    best["primary_confidence"] = primary["confidence"]
    best["fallback_confidence"] = fallback["confidence"]
    return best


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


def _reject_reason_for_lines(lines: List[str]) -> Optional[str]:
    if not lines:
        return "empty_ocr"

    course = lines[0]
    professor = lines[1] if len(lines) > 1 else ""
    room = lines[2] if len(lines) > 2 else ""
    if len(course) > 15 or len(professor) > 8 or len(room) > 10:
        return "field_too_long"
    if not (_is_valid_course(course) and _is_valid_professor(professor)):
        return "invalid_course_or_professor"
    return None


def _runtime_fallback_score(lines: List[str], foreground_density: float) -> float:
    if not lines:
        return 20.0 + foreground_density

    score = 40.0 + min(len(lines), 3) * 5.0 + foreground_density
    course_valid = _is_valid_course(lines[0])
    professor_valid = len(lines) >= 2 and _is_valid_professor(lines[1])
    if course_valid != professor_valid:
        score += 50.0
    elif course_valid or professor_valid:
        score += 25.0
    elif any(_hangul.search(line) for line in lines):
        score += 10.0
    return score


def _should_consider_runtime_fallback(lines: List[str], foreground_density: float) -> bool:
    if not lines:
        return foreground_density >= EMPTY_FALLBACK_FOREGROUND_DENSITY_THRESHOLD

    if not any(_hangul.search(line) for line in lines):
        return False

    course_valid = _is_valid_course(lines[0])
    professor_valid = len(lines) >= 2 and _is_valid_professor(lines[1])
    return course_valid != professor_valid or _reject_reason_for_lines(lines) in {
        "field_too_long",
        "invalid_course_or_professor",
    }


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


def assess_image_quality(img_bgr: np.ndarray) -> Dict[str, Any]:
    height, width = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brightness = round(float(np.mean(gray)), 2)
    contrast = round(float(np.std(gray)), 2)
    blur_score = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)

    flags: List[str] = []
    if width < 900 or height < 1400:
        flags.append("low_resolution")
    if brightness < 60:
        flags.append("too_dark")
    elif brightness > 220:
        flags.append("too_bright")
    if contrast < 35:
        flags.append("low_contrast")
    if blur_score < 80:
        flags.append("possible_blur")

    return {
        "width": width,
        "height": height,
        "brightness": brightness,
        "contrast": contrast,
        "blur_score": blur_score,
        "flags": flags,
    }


def _cell_foreground_mask(roi_gray: np.ndarray) -> np.ndarray:
    height, width = roi_gray.shape[:2]
    if height <= 0 or width <= 0:
        return np.zeros((0, 0), dtype=np.uint8)

    margin_y = max(1, int(height * 0.06))
    margin_x = max(1, int(width * 0.06))
    if height > margin_y * 2 and width > margin_x * 2:
        roi_gray = roi_gray[margin_y:height - margin_y, margin_x:width - margin_x]

    blurred = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    _, dark_mask = cv2.threshold(blurred, 205, 255, cv2.THRESH_BINARY_INV)
    if min(dark_mask.shape[:2]) >= 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return dark_mask


def _foreground_density(roi_gray: np.ndarray) -> float:
    dark_mask = _cell_foreground_mask(roi_gray)
    area = dark_mask.shape[0] * dark_mask.shape[1]
    if area <= 0:
        return 0.0
    return round(float(cv2.countNonZero(dark_mask)) / float(area), 6)


def _text_component_count(roi_gray: np.ndarray) -> int:
    dark_mask = _cell_foreground_mask(roi_gray)
    if dark_mask.size == 0:
        return 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    height, width = dark_mask.shape[:2]
    area = height * width
    text_components = 0
    for label in range(1, count):
        x, y, component_width, component_height, component_area = stats[label]
        if component_area < 4:
            continue
        if component_area > area * 0.18:
            continue
        if component_width > width * 0.75 or component_height > height * 0.75:
            continue
        if component_width < 2 or component_height < 2:
            continue
        aspect_ratio = component_width / max(component_height, 1)
        if aspect_ratio > 12.0 or aspect_ratio < 0.08:
            continue
        text_components += 1
    return text_components


def _has_text_presence(roi_gray: np.ndarray) -> bool:
    if _foreground_density(roi_gray) <= EMPTY_CELL_FOREGROUND_DENSITY_THRESHOLD:
        return False
    return _text_component_count(roi_gray) >= MIN_TEXT_COMPONENTS_IN_CELL


def _is_empty_cell_candidate(roi_gray: np.ndarray) -> bool:
    return not _has_text_presence(roi_gray)


def _save_grid_debug_image(base_gray: np.ndarray, x_lines: List[int], y_lines: List[int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    if x_lines and y_lines:
        for x in x_lines:
            cv2.line(overlay, (x, y_lines[0]), (x, y_lines[-1]), (0, 255, 0), 2)
        for y in y_lines:
            cv2.line(overlay, (x_lines[0], y), (x_lines[-1], y), (0, 0, 255), 2)
    cv2.imwrite(str(output_path), overlay)


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


def _reject_cell(
    rejected_cells: List[dict],
    row: int,
    col: int,
    reason: str,
    ocr_result: Optional[Dict[str, Any]] = None,
) -> None:
    if len(rejected_cells) >= MAX_REJECTED_CELLS_IN_DIAGNOSTICS:
        return
    item: Dict[str, Any] = {
        "row": row,
        "col": col + 1,
        "reason": reason,
    }
    if ocr_result:
        item["lines"] = ocr_result.get("lines", [])
        item["confidence"] = ocr_result.get("confidence")
        item["config"] = ocr_result.get("config")
        item["fallback_used"] = ocr_result.get("fallback_used", False)
    rejected_cells.append(item)


def _extract_schedule_core(
    img: np.ndarray,
    debug_dir: Optional[Path] = None,
    diagnostics_enabled: bool = False,
) -> Dict[str, Any]:
    start = perf_time.monotonic()
    diagnostics: Dict[str, Any] = {
        "image_quality": assess_image_quality(img) if diagnostics_enabled else {},
        "grid": {},
        "ocr": {
            "total_cells": 0,
            "ocr_task_cells": 0,
            "skipped_empty_cells": 0,
            "empty_cell_skip_threshold": EMPTY_CELL_FOREGROUND_DENSITY_THRESHOLD,
            "min_text_components_in_cell": MIN_TEXT_COMPONENTS_IN_CELL,
            "runtime_fallback_limit": MAX_RUNTIME_FALLBACK_CELLS,
            "runtime_fallback_candidates": 0,
            "runtime_fallback_attempted_cells": 0,
            "text_cells": 0,
            "accepted_cells": 0,
            "rejected_cells": [],
            "fallback_cells": 0,
            "low_confidence_cells": 0,
            "average_confidence": None,
        },
        "runtime": {
            "grid_detection_duration_ms": 0,
            "ocr_duration_ms": 0,
            "total_duration_ms": 0,
        },
        "debug_images": {},
    }

    grid_start = perf_time.monotonic()
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

    diagnostics["runtime"]["grid_detection_duration_ms"] = int((perf_time.monotonic() - grid_start) * 1000)
    diagnostics["grid"] = {
        "x_line_count": len(x_lines),
        "y_line_count": len(y_lines),
        "cell_columns": max(0, len(x_lines) - 1),
        "cell_rows": max(0, len(y_lines) - 1),
    }
    if diagnostics_enabled and debug_dir is not None:
        grid_path = debug_dir / "grid_overlay.png"
        _save_grid_debug_image(base_gray, x_lines, y_lines, grid_path)
        diagnostics["debug_images"]["grid_overlay"] = str(grid_path)

    if len(x_lines) - 1 < 3 or len(y_lines) - 1 < 3:
        diagnostics["failure_reason"] = "insufficient_grid_lines"
        diagnostics["runtime"]["total_duration_ms"] = int((perf_time.monotonic() - start) * 1000)
        _log_ocr_engine_runtime(diagnostics, 0)
        return {"schedules": [], "diagnostics": diagnostics}

    weekdays = _load_list_from_txt(WEEKDAYS_FILE)
    time_slots = _load_list_from_txt(TIME_SLOTS_FILE)
    if not weekdays or not time_slots:
        diagnostics["failure_reason"] = "missing_weekdays_or_time_slots"
        diagnostics["runtime"]["total_duration_ms"] = int((perf_time.monotonic() - start) * 1000)
        _log_ocr_engine_runtime(diagnostics, 0)
        return {"schedules": [], "diagnostics": diagnostics}

    tasks: List[bytes] = []
    meta_map: List[tuple[int, int]] = []
    foreground_density_map: List[float] = []
    for row, col, roi_gray in _iterate_cells_autogrid(base_gray, x_lines, y_lines):
        if row == 0:
            continue
        diagnostics["ocr"]["total_cells"] += 1
        if _is_empty_cell_candidate(roi_gray):
            diagnostics["ocr"]["skipped_empty_cells"] += 1
            if diagnostics_enabled:
                _reject_cell(diagnostics["ocr"]["rejected_cells"], row, col, "skipped_empty_cell")
            continue
        ok, buf = cv2.imencode(".png", roi_gray)
        if not ok:
            if diagnostics_enabled:
                _reject_cell(diagnostics["ocr"]["rejected_cells"], row, col, "roi_encode_failed")
            continue
        tasks.append(buf.tobytes())
        meta_map.append((row, col))
        foreground_density_map.append(_foreground_density(roi_gray))
        diagnostics["ocr"]["ocr_task_cells"] += 1

    ocr_results: List[Dict[str, Any]] = []
    ocr_start = perf_time.monotonic()
    if tasks:
        max_workers = min(OCR_THREAD_WORKERS, len(tasks)) or 1
        ocr_func = _ocr_diagnostic_task if diagnostics_enabled else _ocr_task
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ocr_cell") as executor:
            for result in executor.map(ocr_func, tasks):
                if diagnostics_enabled:
                    ocr_results.append(result)
                else:
                    ocr_results.append({
                        "lines": result,
                        "confidence": None,
                        "config": "psm6",
                        "fallback_used": False,
                    })

    if not diagnostics_enabled and ocr_results:
        fallback_candidates: List[tuple[float, int, bytes]] = []
        for index, (ocr_result, roi_bytes) in enumerate(zip(ocr_results, tasks)):
            lines = ocr_result.get("lines") or []
            if _reject_reason_for_lines(lines) is None:
                continue
            foreground_density = foreground_density_map[index]
            if not _should_consider_runtime_fallback(lines, foreground_density):
                continue
            fallback_candidates.append((
                _runtime_fallback_score(lines, foreground_density),
                index,
                roi_bytes,
            ))

        diagnostics["ocr"]["runtime_fallback_candidates"] = len(fallback_candidates)
        fallback_candidates.sort(key=lambda item: item[0], reverse=True)
        fallback_candidates = fallback_candidates[:MAX_RUNTIME_FALLBACK_CELLS]
        diagnostics["ocr"]["runtime_fallback_attempted_cells"] = len(fallback_candidates)
        if fallback_candidates:
            max_workers = min(OCR_THREAD_WORKERS, len(fallback_candidates)) or 1
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ocr_fallback") as executor:
                fallback_results = executor.map(_ocr_fallback_task, [item[2] for item in fallback_candidates])
                for (_, index, _), fallback_result in zip(fallback_candidates, fallback_results):
                    fallback_lines = fallback_result.get("lines") or []
                    if _reject_reason_for_lines(fallback_lines) is None:
                        ocr_results[index] = fallback_result

    diagnostics["runtime"]["ocr_duration_ms"] = int((perf_time.monotonic() - ocr_start) * 1000)

    raw_cells: List[dict] = []
    confidence_values: List[float] = []
    for (row, col), ocr_result in zip(meta_map, ocr_results):
        lines = ocr_result.get("lines") or []
        if not lines:
            if diagnostics_enabled:
                _reject_cell(diagnostics["ocr"]["rejected_cells"], row, col, "empty_ocr", ocr_result)
            continue
        diagnostics["ocr"]["text_cells"] += 1
        confidence = ocr_result.get("confidence")
        if diagnostics_enabled and confidence is not None:
            confidence_values.append(float(confidence))
            if confidence < LOW_OCR_CONFIDENCE:
                diagnostics["ocr"]["low_confidence_cells"] += 1
        if ocr_result.get("fallback_used"):
            diagnostics["ocr"]["fallback_cells"] += 1

        course = lines[0]
        professor = lines[1] if len(lines) > 1 else ""
        room = lines[2] if len(lines) > 2 else ""

        if len(course) > 15 or len(professor) > 8 or len(room) > 10:
            if diagnostics_enabled:
                _reject_cell(diagnostics["ocr"]["rejected_cells"], row, col, "field_too_long", ocr_result)
            continue

        room = room.replace("=", "-")
        if not (_is_valid_course(course) and _is_valid_professor(professor)):
            if diagnostics_enabled:
                _reject_cell(diagnostics["ocr"]["rejected_cells"], row, col, "invalid_course_or_professor", ocr_result)
            continue

        raw_cells.append({
            "row": row,
            "col": col + 1,
            "name": course,
            "professor": professor,
            "room": room,
            "confidence": confidence,
            "ocr_config": ocr_result.get("config"),
            "fallback_used": ocr_result.get("fallback_used", False),
        })

    if diagnostics_enabled:
        diagnostics["ocr"]["accepted_cells"] = len(raw_cells)
        if confidence_values:
            diagnostics["ocr"]["average_confidence"] = round(float(np.mean(confidence_values)), 2)
    else:
        diagnostics["ocr"]["accepted_cells"] = len(raw_cells)

    if not raw_cells:
        diagnostics["failure_reason"] = "no_valid_cells"
        diagnostics["runtime"]["total_duration_ms"] = int((perf_time.monotonic() - start) * 1000)
        _log_ocr_engine_runtime(diagnostics, 0)
        return {"schedules": [], "diagnostics": diagnostics}

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
        diagnostics["failure_reason"] = "no_mapped_schedule_rows"
        diagnostics["runtime"]["total_duration_ms"] = int((perf_time.monotonic() - start) * 1000)
        _log_ocr_engine_runtime(diagnostics, 0)
        return {"schedules": [], "diagnostics": diagnostics}

    merged = _merge_adjacent_same_name(schedule_rows)
    del base_gray
    del vertical_mask
    del horizontal_mask
    diagnostics["schedule_count"] = len(merged)
    diagnostics["runtime"]["total_duration_ms"] = int((perf_time.monotonic() - start) * 1000)
    _log_ocr_engine_runtime(diagnostics, len(merged))
    return {"schedules": merged, "diagnostics": diagnostics}


def _log_ocr_engine_runtime(diagnostics: Dict[str, Any], schedule_count: int) -> None:
    _ensure_ocr_engine_logging()
    failure_reason = diagnostics.get("failure_reason")
    ocr = diagnostics.get("ocr", {})
    runtime = diagnostics.get("runtime", {})
    fallback_cell_count = int(ocr.get("fallback_cells", 0) or 0)
    logger.info(
        runtime_log_message(
            "timetable_ocr_engine_runtime",
            component=RuntimeComponent.OCR,
            operation=RuntimeOperation.OCR,
            status=RuntimeStatus.SUCCESS if not failure_reason and fallback_cell_count == 0 else RuntimeStatus.FALLBACK,
            duration_ms=runtime.get("total_duration_ms", 0),
            result_count=schedule_count,
            fallback=bool(failure_reason) or fallback_cell_count > 0,
            fallback_reason=failure_reason,
            error_code=None,
            grid_detection_duration_ms=runtime.get("grid_detection_duration_ms", 0),
            ocr_duration_ms=runtime.get("ocr_duration_ms", 0),
            extracted_cell_count=ocr.get("accepted_cells", 0),
            total_cell_count=ocr.get("total_cells", 0),
            ocr_task_cell_count=ocr.get("ocr_task_cells", 0),
            text_cell_count=ocr.get("text_cells", 0),
            skipped_empty_cell_count=ocr.get("skipped_empty_cells", 0),
            ocr_fallback_cell_count=fallback_cell_count,
            runtime_fallback_candidate_count=ocr.get("runtime_fallback_candidates", 0),
            runtime_fallback_attempted_cell_count=ocr.get("runtime_fallback_attempted_cells", 0),
        )
    )


def extract_schedule_fixed_scaled(img: np.ndarray) -> List[dict]:
    return _extract_schedule_core(img, diagnostics_enabled=False)["schedules"]


def extract_schedule_runtime_report(img: np.ndarray) -> Dict[str, Any]:
    return _extract_schedule_core(img, diagnostics_enabled=False)


def extract_schedule_with_diagnostics(img: np.ndarray, debug_dir: Optional[Path] = None) -> Dict[str, Any]:
    return _extract_schedule_core(img, debug_dir=debug_dir, diagnostics_enabled=True)
