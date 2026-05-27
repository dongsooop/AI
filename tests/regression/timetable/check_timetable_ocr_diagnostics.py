#!/usr/bin/env python3
import sys
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import cv2
    import numpy as np
    from image_analysis.ocr_engine import ( # noqa: E402
        assess_image_quality,
        extract_schedule_fixed_scaled,
        extract_schedule_with_diagnostics,
    )
except ModuleNotFoundError as exc:
    print(f"[SKIP] timetable OCR diagnostics dependency is missing: {exc.name}")
    raise SystemExit(2) from exc


def make_synthetic_timetable() -> np.ndarray:
    image = np.full((1600, 1000, 3), 255, dtype=np.uint8)
    for x in range(80, 921, 140):
        cv2.line(image, (x, 80), (x, 1480), (0, 0, 0), 3)
    for y in range(80, 1481, 100):
        cv2.line(image, (80, y), (920, y), (0, 0, 0), 3)
    cv2.putText(image, "class", (380, 410), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return image


def fake_image_to_string(*args, **kwargs) -> str:
    config = kwargs.get("config", "")
    if "--psm 6" in config:
        return "자료구조\n"
    return "자료구조\n김교수\nB101\n"


def fake_image_to_data(*args, **kwargs) -> dict:
    config = kwargs.get("config", "")
    if "--psm 6" in config:
        return {"conf": ["42.0"]}
    return {"conf": ["87.0", "90.0", "85.0"]}


def main() -> int:
    image = make_synthetic_timetable()
    quality = assess_image_quality(image)
    errors = []
    if quality["width"] != 1000 or quality["height"] != 1600:
        errors.append(f"unexpected_quality_size:{quality}")

    with patch("pytesseract.image_to_string", side_effect=fake_image_to_string), patch(
        "pytesseract.image_to_data",
        side_effect=fake_image_to_data,
    ):
        report = extract_schedule_with_diagnostics(image)

    diagnostics = report["diagnostics"]
    ocr = diagnostics["ocr"]
    if "image_quality" not in diagnostics:
        errors.append("missing_image_quality")
    if diagnostics["grid"]["x_line_count"] < 3 or diagnostics["grid"]["y_line_count"] < 3:
        errors.append(f"weak_grid_detection:{diagnostics['grid']}")
    if ocr["fallback_cells"] < 1:
        errors.append(f"fallback_not_used:{ocr}")
    if ocr["average_confidence"] is None:
        errors.append(f"missing_average_confidence:{ocr}")
    if "rejected_cells" not in ocr:
        errors.append("missing_rejected_cells")

    call_counts = {"string": 0, "data": 0}

    def fast_image_to_string(*args, **kwargs) -> str:
        call_counts["string"] += 1
        return "자료구조\n김교수\nB101\n"

    def forbidden_image_to_data(*args, **kwargs) -> dict:
        call_counts["data"] += 1
        return {"conf": ["1.0"]}

    with patch("pytesseract.image_to_string", side_effect=fast_image_to_string), patch(
        "pytesseract.image_to_data",
        side_effect=forbidden_image_to_data,
    ):
        extract_schedule_fixed_scaled(image)

    if call_counts["string"] < 1:
        errors.append("fast_path_ocr_not_called")
    if call_counts["data"] != 0:
        errors.append(f"fast_path_called_confidence_ocr:{call_counts}")

    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1

    print("[OK] timetable OCR diagnostics contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
