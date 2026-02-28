from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import Response, JSONResponse
import cv2, re, numpy as np
from PIL import Image
import os, asyncio, uuid, pytesseract, base64
from concurrent.futures import ThreadPoolExecutor
from jose import JWTError, jwt
from dotenv import load_dotenv
from jose.exceptions import ExpiredSignatureError
from typing import List, Optional, Dict, Any
from datetime import time, datetime, timedelta, timezone
from multiprocessing import Pool, cpu_count
from enum import Enum

router = APIRouter()
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")

WEEKDAYS_FILE = "data/weekdays.txt"
TIME_SLOTS_FILE = "data/time_slots.txt"

tesseract_config = r"--psm 6 --oem 1 -l kor+eng"
TARGET_WIDTH  = 1170
TARGET_HEIGHT = 2532
SCALE         = 2

WEEKDAY_BASE_COL_1 = 3
TIME_FIRST_ROW     = 3

_hangul = re.compile(r"[가-힣]")
_only_digits_symbols = re.compile(r"^[\d\W_]+$")
_long_repeat = re.compile(r"(.)\1{3,}")

VDIV, HDIV = 50, 50
H_MIN_LEN_RATIO = 0.60
HOUGH_THRESH, HOUGH_MINLINE, HOUGH_MAXGAP = 120, 80, 10
NEAR_EPS, BAND_MERGE = 6, 8
TRIM_OUTER, TRIM_X_GAP, TRIM_Y_GAP = True, 20, 15

JOB_TTL = timedelta(minutes=10)

_POOL: Optional[Pool] = None
def _get_pool() -> Pool:
    global _POOL
    if _POOL is None:
        _POOL = Pool(processes=os.cpu_count() or 8)
    return _POOL


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE    = "done"
    ERROR   = "error"

_job_store: Dict[str, Dict[str, Any]] = {}
_job_queue: asyncio.Queue = asyncio.Queue(maxsize=20)
_WORKER_CONCURRENCY = 2
_THREAD_EXECUTOR = ThreadPoolExecutor(max_workers=2)


def _cleanup_old_jobs():
    now = datetime.now(timezone.utc)
    expired = [
        job_id for job_id, job in _job_store.items()
        if now - job.get("created_at", now) > JOB_TTL
    ]
    for job_id in expired:
        _job_store.pop(job_id, None)


async def _queue_worker():
    loop = asyncio.get_running_loop()
    while True:
        job_id, img = await _job_queue.get()
        job = _job_store.get(job_id)
        if job is None:
            _job_queue.task_done()
            continue
        job["status"] = JobStatus.RUNNING
        try:
            result = await loop.run_in_executor(
                _THREAD_EXECUTOR, extract_schedule_fixed_scaled, img
            )
            job["status"] = JobStatus.DONE
            job["result"] = result
        except Exception as exc:
            job["status"] = JobStatus.ERROR
            job["error"]  = str(exc)
        finally:
            job["event"].set()
            _job_queue.task_done()
            _cleanup_old_jobs()


async def start_queue_workers():
    for _ in range(_WORKER_CONCURRENCY):
        asyncio.create_task(_queue_worker())


def _ocr_task(roi_bytes: bytes) -> List[str]:
    nparr = np.frombuffer(roi_bytes, np.uint8)
    roi_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    txt = pytesseract.image_to_string(Image.fromarray(roi_gray), config=tesseract_config).strip()
    return [re.sub(r"[|]", "", ln.strip().replace(" ", "")) for ln in txt.split("\n") if ln.strip()]


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


def _split_slot(slot: str) -> Optional[tuple[str, str]]:
    if slot is None:
        return None
    parts = slot.split("~", 1)
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1].strip()


def _to_hms(s: str) -> str:
    if not s:
        return s
    p = s.split(":")
    if len(p) == 2:
        h, m = p
        return f"{int(h):02d}:{m}:00"
    elif len(p) >= 3:
        h, m, s2 = p[0], p[1], p[2]
        return f"{int(h):02d}:{m}:{s2}"
    return s


def _parse_hms(t: str) -> time:
    p = t.split(":")
    if len(p) == 2:
        h, m = int(p[0]), int(p[1])
        return time(h, m, 0)
    elif len(p) >= 3:
        h, m, s = int(p[0]), int(p[1]), int(p[2])
        return time(h, m, s)
    return time(0, 0, 0)


def _prune_x_lines(x_lines: List[int], base_gray: np.ndarray) -> List[int]:
    if not x_lines:
        return x_lines

    H, W = base_gray.shape[:2]
    margin = max(2, int(W * 0.02))
    xs = [x for x in x_lines if margin <= x <= (W - margin)]
    if len(xs) < 2:
        xs = x_lines[:]

    if len(xs) >= 4:
        gaps = np.diff(xs)
        med = float(np.median(gaps[:-1])) if len(gaps) > 1 else float(np.median(gaps))
        if med > 0 and gaps[-1] > 1.6 * med:
            xs = xs[:-1]
    return xs


def _load_list_from_txt(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _make_base_gray(img_bgr: np.ndarray) -> np.ndarray:
    resized_fixed = cv2.resize(img_bgr, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
    up = cv2.resize(resized_fixed, (TARGET_WIDTH * SCALE, TARGET_HEIGHT * SCALE), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(gray, -1, sharpen_kernel)


def _vertical_lines(base_gray: np.ndarray):
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


def _horizontal_lines(base_gray: np.ndarray, vertical_mask: Optional[np.ndarray]):
    thr = cv2.adaptiveThreshold(base_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
    binv = 255 - thr
    h_len = max(20, base_gray.shape[1] // HDIV)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    horizontal = cv2.dilate(cv2.erode(binv, k, 1), k, 1)
    if vertical_mask is not None:
        horizontal = cv2.bitwise_and(horizontal, cv2.bitwise_not(vertical_mask))

    lines = cv2.HoughLinesP(horizontal, 1, np.pi/180,
                            threshold=HOUGH_THRESH,
                            minLineLength=HOUGH_MINLINE,
                            maxLineGap=HOUGH_MAXGAP)
    ys: List[int] = []
    if lines is not None:
        H, W = base_gray.shape[:2]
        inner_left, inner_right = int(W * 0.04), int(W * 0.96)
        min_len = int((inner_right - inner_left) * H_MIN_LEN_RATIO)
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(y1 - y2) <= 2 and abs(x2 - x1) >= min_len:
                ys.append((y1 + y2) // 2)
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


def _iterate_cells_autogrid(base_gray, x_lines, y_lines):
    for r in range(len(y_lines) - 1):
        for c in range(len(x_lines) - 1):
            x1, x2 = x_lines[c], x_lines[c + 1]
            y1, y2 = y_lines[r], y_lines[r + 1]
            pad_x = max(1, (x2 - x1) // 20)
            pad_y = max(1, (y2 - y1) // 20)
            roi = base_gray[y1 + pad_y:y2 - pad_y, x1 + pad_x:x2 - pad_x]
            yield (r, c, roi)


def _merge_adjacent_same_name(items: List[dict]) -> List[dict]:
    groups: dict[tuple[str, str], List[dict]] = {}
    for it in items:
        key = (it["week"], it["name"])
        groups.setdefault(key, []).append(it)

    merged: List[dict] = []

    for (week, name), group in groups.items():
        group.sort(key=lambda x: (x.get("period", 10**9), _parse_hms(x["startAt"]), _parse_hms(x["endAt"])))

        cur = None
        for it in group:
            p  = it["period"]
            st = _parse_hms(it["startAt"])
            et = _parse_hms(it["endAt"])

            if cur is None:
                cur = {
                    "week": week,
                    "name": name,
                    "professor": it.get("professor", "") or "",
                    "location": it.get("location", "") or "",
                    "startAt": it["startAt"],
                    "endAt": it["endAt"],
                    "periods": [p],
                    "period": p,
                }
                continue

            prev_last_p = cur["periods"][-1]
            contiguous_by_period = (p == prev_last_p + 1)
            contiguous_by_time   = (_parse_hms(cur["endAt"]) <= st)
            if contiguous_by_period or contiguous_by_time:
                if et > _parse_hms(cur["endAt"]):
                    cur["endAt"] = it["endAt"]
                cur["periods"].append(p)
                if not cur["professor"] and it.get("professor"):
                    cur["professor"] = it["professor"]
                if not cur["location"] and it.get("location"):
                    cur["location"] = it["location"]
            else:
                merged.append(cur)
                cur = {
                    "week": week,
                    "name": name,
                    "professor": it.get("professor", "") or "",
                    "location": it.get("location", "") or "",
                    "startAt": it["startAt"],
                    "endAt": it["endAt"],
                    "periods": [p],
                    "period": p,
                }
        if cur is not None:
            merged.append(cur)

    merged.sort(key=lambda x: (x["week"], x["period"], x["startAt"]))
    return merged


def extract_schedule_fixed_scaled(img: np.ndarray) -> List[dict]:
    base_gray = _make_base_gray(img)
    x_lines, vmask = _vertical_lines(base_gray)
    x_lines = _prune_x_lines(x_lines, base_gray)
    y_lines, hmask = _horizontal_lines(base_gray, vertical_mask=vmask)

    if TRIM_OUTER:
        if len(x_lines) > 2 and (x_lines[1] - x_lines[0] < TRIM_X_GAP):
            x_lines = x_lines[1:]
        if len(x_lines) > 2 and (x_lines[-1] - x_lines[-2] < TRIM_X_GAP):
            x_lines = x_lines[:-1]
        if len(y_lines) > 2 and (y_lines[1] - y_lines[0] < TRIM_Y_GAP):
            y_lines = y_lines[1:]
        if len(y_lines) > 2 and (y_lines[-1] - y_lines[-2] < TRIM_Y_GAP):
            y_lines = y_lines[:-1]

    cols_total = len(x_lines) - 1
    rows_total = len(y_lines) - 1
    if cols_total < 3 or rows_total < 3:
        return []

    weekdays   = _load_list_from_txt(WEEKDAYS_FILE)
    time_slots = _load_list_from_txt(TIME_SLOTS_FILE)
    if not weekdays or not time_slots:
        return []

    tasks = []
    meta_map = []
    for(r, c, roi_gray) in _iterate_cells_autogrid(base_gray, x_lines, y_lines):
        if r == 0:
            continue
        ok, buf = cv2.imencode(".png", roi_gray)
        if not ok:
            continue
        tasks.append(buf.tobytes())
        meta_map.append((r,c))
    
    lines_list: List[List[str]] = []
    if tasks:
        pool = _get_pool()
        chunk = max(16, len(tasks)//(cpu_count() or 1))
        lines_list = pool.map(_ocr_task, tasks, chunksize=chunk)

    raw_cells: List[dict] = []
    for (r, c), lines in zip(meta_map, lines_list):
        if not lines:
            continue

        course    = lines[0]
        professor = lines[1] if len(lines) > 1 else ""
        room      = lines[2] if len(lines) > 2 else ""

        if len(course) > 15 or len(professor) > 8 or len(room) > 10:
            continue
        
        room = room.replace("=", "-")

        if not (_is_valid_course(course) and _is_valid_professor(professor)):
            continue

        raw_cells.append({
            "row": r,
            "col": c + 1,
            "name": course,
            "professor": professor,
            "room": room,
        })

    if not raw_cells:
        return []

    schedule_rows: List[dict] = []
    for cell in raw_cells:
        r = cell["row"]
        c1 = cell["col"]

        week_idx = (c1 - WEEKDAY_BASE_COL_1)
        if 0 <= week_idx < len(weekdays):
            week = weekdays[week_idx]
        else:
            continue

        period0 = r - TIME_FIRST_ROW
        if not (0 <= period0 < len(time_slots)):
            continue

        slot = time_slots[period0]
        se = _split_slot(slot)
        if not se:
            continue
        start_time, end_time = se
        
        if period0 > 14:
            schedule_rows.append({
                "week": week,
                "period": period0,
                "name": cell["name"],
                "professor": cell["professor"],
                "startAt": _to_hms(start_time),
                "endAt": _to_hms(end_time),
                "location": cell["room"],
            })
        else:
            slot2 = time_slots[period0+1]
            se2 = _split_slot(slot2)
            mn_start_time, mn_end_time = se2
            schedule_rows.append({
                "week": week,
                "period": period0 + 1,
                "name": cell["name"],
                "professor": cell["professor"],
                "startAt": _to_hms(mn_start_time),
                "endAt": _to_hms(mn_end_time),
                "location": cell["room"],
            })

    if not schedule_rows:
        return []

    merged = _merge_adjacent_same_name(schedule_rows)

    # 메모리 확보
    del base_gray
    del vmask
    del hmask

    return merged


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


@router.post("/timetable_analysis")
async def upload_timetable(request: Request, file: UploadFile = File(...)):
    _ = verify_jwt_token(request)
    try:
        file_bytes = await file.read()
        npimg = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image format"})

        job_id = str(uuid.uuid4())
        _job_store[job_id] = {
            "status":     JobStatus.PENDING,
            "result":     None,
            "error":      None,
            "event":      asyncio.Event(),
            "created_at": datetime.now(timezone.utc),
        }
        try:
            await asyncio.wait_for(_job_queue.put((job_id, img)), timeout=30.0)
        except asyncio.TimeoutError:
            _job_store.pop(job_id, None)
            return JSONResponse(status_code=503, content={"error": "server is busy"})

        return JSONResponse(status_code=202, content={"job_id": job_id})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/timetable_result/{job_id}")
async def get_timetable_result(job_id: str, request: Request):
    _ = verify_jwt_token(request)

    job = _job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    try:
        await asyncio.wait_for(job["event"].wait(), timeout=120.0)
    except asyncio.TimeoutError:
        _job_store.pop(job_id, None)
        return JSONResponse(status_code=504, content={"error": "Processing timeout"})

    if job["status"] == JobStatus.ERROR:
        return JSONResponse(status_code=500, content={"error": job["error"]})

    result = job["result"]
    if not result:
        return Response(status_code=204)
    return JSONResponse(content=result)