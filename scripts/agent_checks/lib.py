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


@dataclass(frozen=True)
class AddedLine:
    path: str
    line_no: int
    text: str


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


def add_target_args(parser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--staged",
        action="store_true",
        help="inspect staged changes only",
    )
    group.add_argument(
        "--base",
        metavar="REF",
        help="inspect committed branch changes against REF using REF...HEAD",
    )


def target_from_args(args) -> tuple[str, str | None]:
    if getattr(args, "staged", False):
        return "staged", None
    base = getattr(args, "base", None)
    if base:
        return "base", base
    return "worktree", None


def parse_name_status(output: str) -> list[ChangedFile]:
    changed: list[ChangedFile] = []
    for line in output.splitlines():
        if not line:
            continue
        parts = line.split("\t")
        status = parts[0]
        path_text = parts[-1]
        if (REPO_ROOT / path_text).is_file() or status.startswith("D"):
            changed.append(ChangedFile(path=path_text, status=status))
    return changed


def get_changed_files(mode: str = "worktree", base: str | None = None) -> list[ChangedFile]:
    if mode == "staged":
        return parse_name_status(run_git(["diff", "--cached", "--name-status"]))
    if mode == "base":
        if not base:
            raise ValueError("--base requires a ref")
        return parse_name_status(run_git(["diff", "--name-status", f"{base}...HEAD"]))

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


def get_untracked_files() -> list[str]:
    output = run_git(["ls-files", "--others", "--exclude-standard"])
    return [path for path in output.splitlines() if path and (REPO_ROOT / path).is_file()]


def iter_file_lines(path: str) -> list[AddedLine]:
    try:
        text = read_text_file(path)
    except OSError:
        return []
    return [AddedLine(path=path, line_no=line_no, text=line) for line_no, line in enumerate(text.splitlines(), start=1)]


def parse_added_lines_from_diff(diff_text: str) -> list[AddedLine]:
    added: list[AddedLine] = []
    current_path = ""
    new_line_no = 0

    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            path = line[4:]
            current_path = "" if path == "/dev/null" else path.removeprefix("b/")
            continue
        if line.startswith("@@ "):
            marker = line.split(" +", 1)[1].split(" ", 1)[0]
            start = marker.split(",", 1)[0]
            new_line_no = int(start)
            continue
        if not current_path:
            continue
        if line.startswith("+") and not line.startswith("+++ "):
            added.append(AddedLine(path=current_path, line_no=new_line_no, text=line[1:]))
            new_line_no += 1
        elif line.startswith("-") and not line.startswith("--- "):
            continue
        elif line.startswith("\\"):
            continue
        else:
            new_line_no += 1
    return added


def get_added_lines(mode: str = "worktree", base: str | None = None) -> list[AddedLine]:
    if mode == "staged":
        return parse_added_lines_from_diff(run_git(["diff", "--cached", "--unified=0", "--no-ext-diff"]))
    if mode == "base":
        if not base:
            raise ValueError("--base requires a ref")
        return parse_added_lines_from_diff(run_git(["diff", "--unified=0", "--no-ext-diff", f"{base}...HEAD"]))

    added = parse_added_lines_from_diff(run_git(["diff", "--cached", "--unified=0", "--no-ext-diff"]))
    added.extend(parse_added_lines_from_diff(run_git(["diff", "--unified=0", "--no-ext-diff"])))
    for path in get_untracked_files():
        if is_probably_text(path):
            added.extend(iter_file_lines(path))
    return added


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
