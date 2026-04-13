import asyncio
import json
import os
from datetime import datetime, timezone
import aiofiles
import logging

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "logs")
MAX_BYTES = 5 * 1024 * 1024  # 5MB
_logger = logging.getLogger(__name__)


def _log_path() -> str:
    date_str = datetime.now().strftime("%Y-%m-%d")
    base = os.path.join(LOG_DIR, f"rule_book_{date_str}")

    path = f"{base}.jsonl"
    index = 1
    while os.path.exists(path) and os.path.getsize(path) >= MAX_BYTES:
        path = f"{base}_{index}.jsonl"
        index += 1
    return path


async def _write_log(record: dict) -> None:
    await asyncio.to_thread(os.makedirs, LOG_DIR, exist_ok=True)
    path = _log_path()
    line = json.dumps(record, ensure_ascii=False)
    async with aiofiles.open(path, "a", encoding="utf-8") as f:
        await f.write(line + "\n")


async def log_rule_book(query: str, chunks: list, answer: str, error: str | None, elapsed_ms: int) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "query": query,
        "chunks_count": len(chunks),
        "chunks_sources": [f"{c.get('source', '')} {c.get('article', '')}".strip() for c in chunks],
        "answer": answer,
        "error": error,
        "elapsed_ms": elapsed_ms,
    }
    try:
        await _write_log(record)
    except Exception as e:
        _logger.warning("rule_book 로그 기록 실패: %s", e, exc_info=True)
