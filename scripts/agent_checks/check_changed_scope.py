"""Classify changed files by Dongsooop service area."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from lib import ChangedFile, add_target_args, get_changed_files, match_any, target_from_args


SCOPE_PATTERNS: dict[str, list[str]] = {
    "chatbot": [
        "app_oss_main.py",
        "LLM/OSS/*",
        "LLM/OSS/**/*",
        "LLM/sub_model/*",
        "LLM/sub_model/**/*",
    ],
    "text_filter_timetable": [
        "main.py",
        "text_filtering/*",
        "text_filtering/**/*",
        "image_analysis/*",
        "image_analysis/**/*",
    ],
    "common_core": [
        "core/*",
        "core/**/*",
    ],
    "deployment": [
        "Dockerfile",
        "Dockerfiles_oss",
        "docker-compose*.yml",
        ".github/workflows/*",
        ".github/workflows/**/*",
    ],
    "dependencies": [
        "requirements*.txt",
        "pyproject.toml",
        "poetry.lock",
    ],
    "generated_artifacts": [
        "model/artifacts/*",
        "model/artifacts/**/*",
    ],
    "agent_docs": [
        "AGENTS.md",
        "docs/*.md",
        "docs/**/*.md",
        ".codex/skills/*/SKILL.md",
        ".local-agent-docs/*.md",
    ],
    "agent_tools": [
        "scripts/agent_checks/*",
        "scripts/agent_checks/**/*",
    ],
}


def classify(files: list[ChangedFile]) -> dict[str, list[str]]:
    scopes: dict[str, list[str]] = defaultdict(list)
    for changed in files:
        matched = False
        for scope, patterns in SCOPE_PATTERNS.items():
            if match_any(changed.path, patterns):
                scopes[scope].append(changed.path)
                matched = True
                break
        if not matched:
            scopes["other"].append(changed.path)
    return dict(scopes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_target_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mode, base = target_from_args(args)
    files = get_changed_files(mode=mode, base=base)
    print("[scope]")
    if not files:
        print("- no changed files")
        return 0

    scopes = classify(files)
    for scope in sorted(scopes):
        print(f"- {scope}: {len(scopes[scope])} file(s)")
        for path in scopes[scope]:
            print(f"  - {path}")

    service_scopes = {"chatbot", "text_filter_timetable"}
    touched_services = service_scopes.intersection(scopes)
    if len(touched_services) > 1:
        print("[warning] Changes cross chatbot and text_filter/timetable service boundaries.")
    if "generated_artifacts" in scopes:
        print("[warning] Generated artifacts changed. Prefer updating generation logic.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(2)
