"""Run Dongsooop agent preflight checks as one command."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import shlex

from check_changed_scope import classify
from lib import REPO_ROOT, get_changed_files


CHECKS = [
    ("changed scope", "check_changed_scope.py"),
    ("secret patterns", "check_secret_leaks.py"),
    ("generated artifacts", "check_generated_artifacts.py"),
    ("review output requirements", "check_required_review_outputs.py"),
]


def run_check(script_name: str) -> int:
    script_path = Path(__file__).resolve().parent / script_name
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode


def print_recommended_checks() -> None:
    files = get_changed_files()
    scopes = classify(files)
    changed_paths = [item.path for item in files]

    print("[recommended-checks]")
    if not files:
        print("- no checks needed; working tree has no changes")
        return

    python_paths = [path for path in changed_paths if path.endswith(".py")]
    if python_paths:
        joined = " ".join(shlex.quote(path) for path in python_paths)
        print(f"- python -m compileall {joined}")
    if "chatbot" in scopes:
        print("- python debug/regression/run_chatbot_regression.py")
    if "text_filter_timetable" in scopes:
        if any(path.startswith("image_analysis/") for path in changed_paths):
            print("- python debug/timetable/debug_timetable.py")
        if any(path.startswith("text_filtering/") for path in changed_paths):
            print("- run focused text filtering import/API checks")
    if "deployment" in scopes or "dependencies" in scopes:
        print("- verify Docker, requirements, and environment variable consistency")
    if set(scopes) == {"agent_docs"}:
        print("- documentation-only change; verify paths and instructions")


def main() -> int:
    print("== Dongsooop Agent Preflight ==")
    print()
    exit_codes: list[int] = []
    for label, script_name in CHECKS:
        print(f"== {label} ==")
        exit_codes.append(run_check(script_name))
        print()

    print_recommended_checks()
    print()

    if any(code != 0 for code in exit_codes):
        print("[preflight] completed with warnings")
        return 1
    print("[preflight] completed without warnings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
