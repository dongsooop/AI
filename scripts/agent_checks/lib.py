"""Shared helpers for Dongsooop agent preflight checks."""

from __future__ import annotations

import fnmatch
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ChangedFile:
    path: str
    status: str


def run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {message}")
    return result.stdout


def get_changed_files() -> list[ChangedFile]:
    output = run_git(["status", "--porcelain"])
    changed: list[ChangedFile] = []
    for line in output.splitlines():
        if not line:
            continue
        status = line[:2]
        path_text = line[3:]
        if status == "??":
            continue
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[1]
        if (REPO_ROOT / path_text).is_file():
            changed.append(ChangedFile(path=path_text, status=status))

    untracked = run_git(["ls-files", "--others", "--exclude-standard"])
    for path_text in untracked.splitlines():
        if path_text and (REPO_ROOT / path_text).is_file():
            changed.append(ChangedFile(path=path_text, status="??"))
    return changed


def match_any(path: str, patterns: list[str]) -> bool:
    return any(path == pattern or fnmatch.fnmatch(path, pattern) for pattern in patterns)


def is_probably_text(path: str) -> bool:
    file_path = REPO_ROOT / path
    if not file_path.is_file():
        return False
    try:
        with file_path.open("rb") as handle:
            sample = handle.read(4096)
    except OSError:
        return False
    return b"\0" not in sample


def read_text_file(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8", errors="replace")
