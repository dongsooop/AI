from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import Response, JSONResponse
import cv2, re, numpy as np
import pytesseract
from PIL import Image
import os
from jose import JWTError, jwt
from dotenv import load_dotenv
import base64
from jose.exceptions import ExpiredSignatureError
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Optional

router = APIRouter()
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")

WEEKDAYS_FILE = "data/weekdays.txt"
TIME_SLOTS_FILE = "data/time_slots.txt"

tesseract_config = r"--psm 6 --oem 1 -l kor+eng"

TARGET_WIDTH = 1170
TARGET_HEIGHT = 2532
SCALE = 2

_hangul = re.compile(r"[가-힣]")
_only_digits_symbols = re.compile(r"^[\d\W_]+$")
_long_repeat = re.compile(r"(.)\1{3,}")
_time_pat = re.compile(r"\d{1,2}:\d{2}~\d{1,2}:\d{2}")
_kv_time = re.compile(r"(\d{1,2}:\d{2})\D+(\d{1,2}:\d{2})")

# 헤더 한글 요일
_KOR_WEEKDAYS = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]

VDIV, HDIV = 50, 50
NEAR_EPS, BAND_MERGE = 6, 8
HOUGH_THRESH, HOUGH_MINLINE, HOUGH_MAXGAP = 120, 80, 10
H_MIN_LEN_RATIO = 0.60
TRIM_OUTER, TRIM_X_GAP, TRIM_Y_GAP = True, 20, 15

# ====== 전역 프로세스 풀 ======
_POOL: Optional[Pool] = None
def _get_pool() -> Pool:
    global _POOL
    if _POOL is None:
        _POOL = Pool(processes=max(2, min(8, cpu_count())))
    return _POOL

def _is_valid_course(s: str) -> bool:
    s = s.strip()
    return (
        len(s) >= 2
        and _hangul.search(s) is not None
        and _only_digits_symbols.fullmatch(s) is None
        and _long_repeat.search(s) is None
        and s not in {"|", "-", "_"}
    )

def _is_valid_professor(s: str) -> bool:
    s = s.strip()
    return (
        len(s) >= 2
        and _hangul.search(s) is not None
        and _long_repeat.search(s) is None
    )

def _ocr_lines(img_gray: np.ndarray) -> List[str]:
    txt = pytesseract.image_to_string(Image.fromarray(img_gray), config=tesseract_config).strip()
    return [re.sub(r"[|]", "", ln.strip().replace(" ", "")) for ln in txt.split("\n") if ln.strip()]

def _ocr_lines_from_png_bytes(png_bytes: bytes) -> List[str]:
    nparr = np.frombuffer(png_bytes, np.uint8)
    roi_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    txt = pytesseract.image_to_string(Image.fromarray(roi_gray), config=tesseract_config).strip()
    return [re.sub(r"[|]", "", ln.strip().replace(" ", "")) for ln in txt.split("\n") if ln.strip()]

def _load_list_from_txt(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# ====== 고정 리사이즈 → 스케일×2 → 샤프닝 ======
def _make_base_gray(img_bgr: np.ndarray) -> np.ndarray:
    resized_fixed = cv2.resize(img_bgr, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
    up = cv2.resize(resized_fixed, (TARGET_WIDTH * SCALE, TARGET_HEIGHT * SCALE), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(gray, -1, sharpen_kernel)

def _vertical_lines(base_gray: np.ndarray) -> Tuple[List[int], np.ndarray]:
    thr = cv2.adaptiveThreshold(base_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 5)
    binv = 255 - thr
    v_len = max(20, base_gray.shape[0] // VDIV)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    vertical = cv2.dilate(cv2.erode(binv, k, 1), k, 1)
    vx = vertical.sum(axis=0); idx = np.where(vx > vx.max() * 0.35)[0]
    xs: List[int] = []
    if len(idx):
        cur = [idx[0]]
        for i in idx[1:]:
            if i - cur[-1] <= NEAR_EPS: cur.append(i)
            else: xs.append(int(np.median(cur))); cur = [i]
        xs.append(int(np.median(cur)))
    return sorted(xs), vertical

def _horizontal_lines(base_gray: np.ndarray, vertical_mask: Optional[np.ndarray]) -> Tuple[List[int], np.ndarray]:
    thr = cv2.adaptiveThreshold(base_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
    binv = 255 - thr
    h_len = max(20, base_gray.shape[1] // HDIV)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    horizontal = cv2.dilate(cv2.erode(binv, k, 1), k, 1)
    if vertical_mask is not None:
        horizontal = cv2.bitwise_and(horizontal, cv2.bitwise_not(vertical_mask))
    lines = cv2.HoughLinesP(horizontal, 1, np.pi/180, HOUGH_THRESH, minLineLength=HOUGH_MINLINE, maxLineGap=HOUGH_MAXGAP)

    ys: List[int] = []
    if lines is not None:
        H, W = base_gray.shape[:2]
        inner_left, inner_right = int(W * 0.04), int(W * 0.96)
        min_len = int((inner_right - inner_left) * H_MIN_LEN_RATIO)
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(y1 - y2) <= 2 and abs(x2 - x1) >= min_len: ys.append((y1 + y2) // 2)
    ys.sort()
    merged: List[int] = []
    if ys:
        cur = [ys[0]]
        for y in ys[1:]:
            if y - cur[-1] <= NEAR_EPS: cur.append(y)
            else: merged.append(int(np.median(cur))); cur = [y]
        merged.append(int(np.median(cur)))
    final_y: List[int] = []
    i = 0
    while i < len(merged):
        if i + 1 < len(merged) and (merged[i+1] - merged[i]) <= BAND_MERGE:
            final_y.append((merged[i] + merged[i+1]) // 2); i += 2
        else:
            final_y.append(merged[i]); i += 1
    return sorted(list(set(final_y))), horizontal

def verify_jwt_token(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or malformed")

    token = auth_header.split(" ")[1]
    try:
        padded_key = SECRET_KEY + '=' * (-len(SECRET_KEY) % 4)
        sc = base64.urlsafe_b64decode(padded_key)
        payload = jwt.decode(token, sc, algorithms=[ALGORITHM])

        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token: no subject")
        return username

    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")

    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ---------- 병렬 OCR 태스크 ----------
def _ocr_task(args):
    (period_idx, week_str, start_time, end_time), png_bytes = args
    lines = _ocr_lines_from_png_bytes(png_bytes) if png_bytes else []
    return (period_idx, week_str, start_time, end_time, lines)

# ---------- 요일 시작 열 찾기 ----------
def _detect_day_start_col(base_gray: np.ndarray, x_lines: List[int], y_lines: List[int]) -> Optional[int]:
    if len(y_lines) < 2:
        return None
    head_y1, head_y2 = y_lines[0], y_lines[1]
    pad_y = max(1, (head_y2 - head_y1) // 8)
    head_y1 += pad_y; head_y2 -= pad_y

    for c in range(len(x_lines) - 1):
        x1, x2 = x_lines[c], x_lines[c+1]
        pad_x = max(1, (x2 - x1) // 8)
        roi = base_gray[head_y1:head_y2, x1+pad_x:x2-pad_x]
        joined = "".join(_ocr_lines(roi))
        if not joined:
            continue
        for kw in _KOR_WEEKDAYS:
            if kw.replace(" ", "") in joined:
                return c
    return None

# ---------- 왼쪽 시간 라벨 정렬(shift) ----------
def _align_time_rows_by_label(base_gray: np.ndarray, x_lines: List[int], y_lines: List[int],
                              day_start: int, time_slots: List[str]) -> Optional[int]:
    if day_start is None or day_start <= 0 or len(y_lines) < 2:
        return None

    x_left = x_lines[0]
    x_right = x_lines[day_start]
    label_texts = []
    for r in range(1, len(y_lines) - 1):
        y1, y2 = y_lines[r], y_lines[r + 1]
        pad_x = max(1, (x_right - x_left)//10)
        pad_y = max(1, (y2 - y1)//10)
        roi = base_gray[y1 + pad_y:y2 - pad_y, x_left + pad_x:x_right - pad_x]
        txt = " ".join(_ocr_lines(roi))
        label_texts.append(txt)

    best_s, best_score = None, -1
    for s in range(-3, 4):
        score = 0
        for r_idx, lbl in enumerate(label_texts):
            m = _kv_time.search(lbl)
            if not m:
                continue
            l = f"{m.group(1)}~{m.group(2)}"
            ts_idx = r_idx - s
            if 0 <= ts_idx < len(time_slots):
                if l == time_slots[ts_idx]:
                    score += 1
        if score > best_score:
            best_score, best_s = score, s

    if best_s is None:
        return None
    return max(1, 1 + best_s)

# ---------- 간단 점수/국소 보정 ----------
def __count_hits_for_offsets(base_gray, x_lines, y_lines, day_start, time_start,
                             weekdays, time_slots, rows_total, cols_total,
                             sample_rows=4, sample_cols=4) -> int:
    hits = 0
    r0 = max(0, min(len(time_slots)-sample_rows, len(time_slots)-6))
    c0 = 0
    for r_idx in range(r0, r0 + sample_rows):
        r = r_idx + time_start
        if r >= rows_total: break
        y1, y2 = y_lines[r], y_lines[r+1]
        pad_y = max(1, (y2 - y1) // 20)
        for c_idx in range(c0, c0 + sample_cols):
            c = c_idx + day_start
            if c >= cols_total: break
            x1, x2 = x_lines[c], x_lines[c+1]
            pad_x = max(1, (x2 - x1) // 20)
            roi = base_gray[y1+pad_y:y2-pad_y, x1+pad_x:x2-pad_x]
            lines = _ocr_lines(roi)
            if not lines:
                continue
            course = lines[0]
            prof   = lines[1] if len(lines) > 1 else ""
            if _is_valid_course(course) and _is_valid_professor(prof):
                hits += 1
    return hits

def __refine_offsets_locally(base_gray, x_lines, y_lines, day_start, time_start,
                             weekdays, time_slots) -> tuple[int,int]:
    cols_total = len(x_lines) - 1
    rows_total = len(y_lines) - 1
    best = (day_start, time_start)
    best_hits = -1
    for dx in (-1, 0, 1):
        ds = day_start + dx
        if ds < 0 or ds >= cols_total:
            continue
        for dy in (-1, 0, 1):
            ts = time_start + dy
            if ts < 1 or ts >= rows_total-1:
                continue
            hits = __count_hits_for_offsets(
                base_gray, x_lines, y_lines, ds, ts,
                weekdays, time_slots, rows_total, cols_total,
                sample_rows=4, sample_cols=4
            )
            if hits > best_hits:
                best_hits = hits
                best = (ds, ts)
    return best

# ---------- 교시(숫자) 컬럼을 이용해 행→교시 매핑 ----------
_KOR_PERIOD_HDR = re.compile(r"교시|교\s*시")
_digit_pat = re.compile(r"\b([1-9]\d?)\b")

def _detect_period_col(base_gray: np.ndarray, x_lines: List[int], y_lines: List[int]) -> Optional[int]:
    """헤더 줄에서 '교시' 문구가 들어있는 컬럼 인덱스"""
    if len(y_lines) < 2:
        return None
    y1, y2 = y_lines[0], y_lines[1]
    pad_y = max(1, (y2 - y1)//8)
    for c in range(len(x_lines)-1):
        x1, x2 = x_lines[c], x_lines[c+1]
        pad_x = max(1, (x2 - x1)//8)
        roi = base_gray[y1+pad_y:y2-pad_y, x1+pad_x:x2-pad_x]
        txt = "".join(_ocr_lines(roi))
        if _KOR_PERIOD_HDR.search(txt):
            return c
    return None

def _read_period_number_cell(base_gray: np.ndarray, x1:int, x2:int, y1:int, y2:int) -> Optional[int]:
    """단일 칸에서 '교시 숫자'를 강하게 추출"""
    if x2 <= x1 or y2 <= y1:
        return None
    roi = base_gray[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    clahe = cv2.createCLAHE(2.0, (8,8))
    eq = clahe.apply(roi)
    thr = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    txt = pytesseract.image_to_string(Image.fromarray(thr), config="--psm 7 --oem 1 -l eng").strip()
    m = _digit_pat.search(txt.replace("O", "0"))
    if not m:
        return None
    try:
        n = int(m.group(1))
        return n if 1 <= n <= 40 else None
    except:
        return None

# ---------- 비대칭/완화 패딩 ----------
def _row_col_pads(r, c, y_lines, x_lines, time_start, day_start):
    h = y_lines[r+1] - y_lines[r]
    w = x_lines[c+1] - x_lines[c]
    base_pad_y = max(0, h // 30)
    base_pad_x = max(0, w // 20)

    is_first_content_row = (r == time_start)
    is_last_content_row  = (r == len(y_lines) - 2)

    top_pad = base_pad_y // 2 if is_first_content_row else base_pad_y
    bot_pad = base_pad_y // 2 if is_last_content_row  else base_pad_y

    is_first_content_col = (c == day_start)
    is_last_content_col  = (c == len(x_lines) - 2)

    left_pad  = base_pad_x // 2 if is_first_content_col else base_pad_x
    right_pad = base_pad_x // 2 if is_last_content_col  else base_pad_x

    return top_pad, bot_pad, left_pad, right_pad


def _apply_backshift_if_prev_empty_once(results: list[dict], time_slots: list[str],
                                        period_min: int = 1, period_max: int = 14) -> None:
    """
    - period_min 이상, period_max 이하에서만 보정
    - 앞 교시 비어있고 현재 시간이 표준 교시와 정확히 맞으면 1칸 당김
    - 아이템당 최대 1회만 보정
    """
    def _norm(t: str) -> str:
        p = t.split(":")
        hh = int(p[0]); mm = p[1]; ss = p[2] if len(p) > 2 else "00"
        return f"{hh:02d}:{mm}:{ss}"

    ts_pairs = [tuple(map(_norm, s.split("~"))) for s in time_slots]

    by_week: dict[str, list[dict]] = {}
    for it in results:
        by_week.setdefault(it["week"], []).append(it)

    processed_ids: set[int] = set()

    for week, items in by_week.items():
        items.sort(key=lambda x: x["period"])
        occupied = {it["period"] for it in items}

        for it in items:
            oid = id(it)
            if oid in processed_ids:
                continue

            p = it["period"]
            if p < period_min or p > period_max:  # ✅ 여기서 범위 제한
                continue

            prev_p = p - 1
            if prev_p in occupied:
                continue

            cur_start_end = (_norm(it["startAt"]), _norm(it["endAt"]))
            try:
                std_cur  = ts_pairs[p - 1]
                std_prev = ts_pairs[prev_p - 1]
            except IndexError:
                continue

            if cur_start_end == std_cur:
                it["period"]  = prev_p
                it["startAt"] = std_prev[0]
                it["endAt"]   = std_prev[1]
                occupied.discard(p)
                occupied.add(prev_p)
                processed_ids.add(oid)

# ================== 메인 ==================
def extract_schedule_fixed_scaled(img: np.ndarray) -> List[dict]:
    base_gray = _make_base_gray(img)
    x_lines, vmask = _vertical_lines(base_gray)
    y_lines, hmask = _horizontal_lines(base_gray, vertical_mask=vmask)

    # 외곽 트리밍(적응식 + 세이프티 마진)
    if TRIM_OUTER:
        row_heights = [y_lines[i+1] - y_lines[i] for i in range(len(y_lines)-1)] if len(y_lines) > 1 else []
        median_h = int(np.median(row_heights)) if row_heights else 0
        y_gap_min = max(6, int(median_h * 0.25))
        x_gap_min = TRIM_X_GAP

        if len(x_lines) > 2 and (x_lines[1] - x_lines[0] < x_gap_min):
            x_lines = x_lines[1:]
        if len(x_lines) > 2 and (x_lines[-1] - x_lines[-2] < x_gap_min):
            x_lines = x_lines[:-1]

        if len(y_lines) > 2 and (y_lines[1] - y_lines[0] < y_gap_min):
            y_lines = y_lines[1:]
        if len(y_lines) > 2 and (y_lines[-1] - y_lines[-2] < y_gap_min):
            y_lines = y_lines[:-1]

        H, W = base_gray.shape[:2]
        if y_lines:
            y_lines[0]  = max(0,      y_lines[0]  - 2)
            y_lines[-1] = min(H - 1,  y_lines[-1] + 2)

    cols_total = len(x_lines) - 1
    rows_total = len(y_lines) - 1
    if cols_total < 3 or rows_total < 3:
        return []

    weekdays   = _load_list_from_txt(WEEKDAYS_FILE)
    time_slots = _load_list_from_txt(TIME_SLOTS_FILE)
    if not weekdays or not time_slots:
        return []

    # 1) 요일 시작 컬럼
    day_start = _detect_day_start_col(base_gray, x_lines, y_lines)
    if day_start is None:
        day_start = max(0, cols_total - len(weekdays))

    # 2) 시간 행 정렬(왼쪽 라벨 shift)
    time_start = _align_time_rows_by_label(base_gray, x_lines, y_lines, day_start, time_slots)
    if time_start is None:
        # 폴백: 간략 샘플 점수
        def score_for_t(t):
            return __count_hits_for_offsets(base_gray, x_lines, y_lines, day_start, t,
                                            weekdays, time_slots, rows_total, cols_total,
                                            sample_rows=3, sample_cols=3)
        best_t, best_s = 1, -1
        for t in (1, 2, 3):
            s = score_for_t(t)
            if s > best_s:
                best_t, best_s = t, s
        time_start = best_t

    # 3) 국소 탐색 보정
    day_start, time_start = __refine_offsets_locally(
        base_gray, x_lines, y_lines, day_start, time_start, weekdays, time_slots
    )

    # 4) (신규) '교시 숫자' 컬럼에서 행→교시 매핑 생성
    period_col = _detect_period_col(base_gray, x_lines, y_lines)
    row_to_periodidx: dict[int, int] = {}
    if period_col is not None:
        for r in range(time_start, rows_total):
            y1, y2 = y_lines[r], y_lines[r+1]
            x1, x2 = x_lines[period_col], x_lines[period_col+1]
            pad_x = max(1, (x2 - x1)//10)
            pad_y = max(1, (y2 - y1)//10)
            n = _read_period_number_cell(base_gray, x1+pad_x, x2-pad_x, y1+pad_y, y2-pad_y)
            if n is not None:
                row_to_periodidx[r] = n - 1  # 0-based

    # 5) 병렬 OCR로 실제 추출
    tasks = []
    for r_idx, slot in enumerate(time_slots):
        r = r_idx + time_start
        if r >= rows_total:
            break

        # 기본값: 라벨 기반 r_idx
        period_idx = r_idx
        start_time, end_time = slot.split("~")

        # ★ 우선순위 1: 교시숫자 매핑 있으면 그것을 신뢰
        if r in row_to_periodidx:
            period_idx = row_to_periodidx[r]
            if 0 <= period_idx < len(time_slots):
                start_time, end_time = time_slots[period_idx].split("~")

        for c_idx, week in enumerate(weekdays):
            c = c_idx + day_start
            if c >= cols_total:
                break

            top_pad, bot_pad, left_pad, right_pad = _row_col_pads(r, c, y_lines, x_lines, time_start, day_start)
            y1c = max(0, y_lines[r]     + top_pad)
            y2c = min(base_gray.shape[0], y_lines[r+1] - bot_pad)
            x1c = max(0, x_lines[c]     + left_pad)
            x2c = min(base_gray.shape[1], x_lines[c+1] - right_pad)
            if y2c <= y1c or x2c <= x1c:
                continue

            roi = base_gray[y1c:y2c, x1c:x2c]
            ok, buf = cv2.imencode(".png", roi)
            png_bytes = buf.tobytes() if ok else b""
            meta = (period_idx, week, start_time, end_time)
            tasks.append((meta, png_bytes))

    results: List[dict] = []
    if tasks:
        mapped = _get_pool().map(_ocr_task, tasks, chunksize=max(8, len(tasks)//(cpu_count() or 1)))
        for (period_idx, week, start_time, end_time, lines) in mapped:
            if not lines or len(lines[0]) < 2:
                continue
            course = lines[0]
            professor = lines[1] if len(lines) > 1 else ""
            room = lines[2] if len(lines) > 2 else ""
            if not (_is_valid_course(course) and _is_valid_professor(professor)):
                continue
            results.append({
                "week": week,
                "period": period_idx + 1,   # 1-based
                "name": course,
                "professor": professor,
                "startAt": start_time,
                "endAt": end_time,
                "location": room
            })

        
    _apply_backshift_if_prev_empty_once(results, _load_list_from_txt(TIME_SLOTS_FILE),period_min=1, period_max=14)

    return results

@router.post("/timetable")
async def upload_timetable(request: Request, file: UploadFile = File(...)):
    _ = verify_jwt_token(request)
    try:
        file_bytes = await file.read()
        npimg = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        schedule = extract_schedule_fixed_scaled(img)

        if not schedule:
            return Response(status_code=204)
        
        return JSONResponse(content=schedule)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})