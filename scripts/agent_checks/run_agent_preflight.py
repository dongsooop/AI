"""Run Dongsooop agent preflight checks as one command."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from check_changed_scope import classify
from lib import REPO_ROOT, add_target_args, get_changed_files, run_git, target_from_args


CHECKS = [
    ("changed scope", "check_changed_scope.py"),
    ("secret patterns", "check_secret_leaks.py"),
    ("generated artifacts", "check_generated_artifacts.py"),
    ("review output requirements", "check_required_review_outputs.py"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Exit codes: 0=no warnings, 1=warnings need review, 2=execution error.",
    )
    add_target_args(parser)
    return parser.parse_args()


def child_target_args(mode: str, base: str | None) -> list[str]:
    if mode == "staged":
        return ["--staged"]
    if mode == "base":
        return ["--base", str(base)]
    return []


def run_check(script_name: str, target_args: list[str]) -> int:
    script_path = Path(__file__).resolve().parent / script_name
    result = subprocess.run(
        [sys.executable, str(script_path), *target_args],
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


def print_recommended_checks(mode: str, base: str | None) -> None:
    files = get_changed_files(mode=mode, base=base)
    scopes = classify(files)
    changed_paths = [item.path for item in files]

    print("[recommended-checks]")
    if not files:
        if mode == "staged":
            print("- no checks needed; staged changes are empty")
        elif mode == "base":
            print(f"- no checks needed; no changes found against {base}")
        else:
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
    args = parse_args()
    mode, base = target_from_args(args)
    if mode == "base":
        run_git(["rev-parse", "--verify", str(base)])
    target_args = child_target_args(mode, base)

    print("== Dongsooop Agent Preflight ==")
    target_label = f"base {base}...HEAD" if mode == "base" else mode
    print(f"target: {target_label}")
    print()
    exit_codes: list[int] = []
    for label, script_name in CHECKS:
        print(f"== {label} ==")
        exit_codes.append(run_check(script_name, target_args))
        print()

    print_recommended_checks(mode=mode, base=base)
    print()

    invalid_codes = [code for code in exit_codes if code not in (0, 1, 2)]
    if invalid_codes:
        print(f"[preflight] failed because a check returned invalid exit code(s): {invalid_codes}")
        return 2
    if any(code == 2 for code in exit_codes):
        print("[preflight] failed because a check could not run")
        return 2
    if any(code != 0 for code in exit_codes):
        print("[preflight] completed with warnings")
        return 1
    print("[preflight] completed without warnings")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(2)
