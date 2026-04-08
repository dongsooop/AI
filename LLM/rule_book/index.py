import os
import re
from pathlib import Path
from typing import List, Dict

import fitz
from rank_bm25 import BM25Okapi


def _rule_book_dir() -> Path:
    env = os.getenv("RULE_BOOK_PATH")
    if env:
        return Path(os.path.expanduser(env)).resolve()
    return (Path(__file__).resolve().parents[2] / "data" / "rule_book")


_ARTICLE_RE = re.compile(r"(제\s*\d+\s*조(?:의\d+)?(?:\([^)]*\))?)")

def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[가-힣A-Za-z0-9]{2,}", text)]


def _chunk_pdf(path: Path) -> List[Dict]:
    doc = fitz.open(str(path))
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()

    source = path.stem

    parts = _ARTICLE_RE.split(full_text)
    chunks = []

    if len(parts) > 3:
        i = 1
        while i < len(parts) - 1:
            article_id = parts[i].strip()
            content = (parts[i + 1] if i + 1 < len(parts) else "").strip()
            if content:
                chunks.append({
                    "source": source,
                    "article": article_id,
                    "text": article_id + " " + content,
                })
            i += 2
        header = parts[0].strip()
        if header:
            chunks.insert(0, {"source": source, "article": "머리말", "text": header})
    else:
        step = 200
        for start in range(0, len(full_text), step):
            chunk_text = full_text[start:start + step + 100].strip()
            if chunk_text:
                chunks.append({
                    "source": source,
                    "article": f"p{start}",
                    "text": chunk_text,
                })

    return chunks


class RuleBookIndex:
    def __init__(self):
        self.chunks: List[Dict] = []
        self.bm25: BM25Okapi | None = None
        self._built = False

    def build(self) -> None:
        rule_dir = _rule_book_dir()
        if not rule_dir.exists():
            raise FileNotFoundError(f"규정집 디렉토리 없음: {rule_dir}")

        all_chunks: List[Dict] = []
        for pdf_path in sorted(rule_dir.glob("*.pdf")):
            try:
                all_chunks.extend(_chunk_pdf(pdf_path))
            except Exception as e:
                print(f"[RuleBook] {pdf_path.name} 파싱 실패: {e}")

        if not all_chunks:
            raise ValueError("규정집 청크가 0개 — PDF 파싱 실패")

        self.chunks = all_chunks
        tokenized = [_tokenize(c["text"]) for c in all_chunks]
        self.bm25 = BM25Okapi(tokenized)
        self._built = True
        print(f"[RuleBook] 인덱스 빌드 완료: {len(self.chunks)}개 청크")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self._built or self.bm25 is None:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({**self.chunks[idx], "score": float(scores[idx])})
        return results


_index = RuleBookIndex()


def get_index() -> RuleBookIndex:
    return _index


def build_index() -> None:
    _index.build()
