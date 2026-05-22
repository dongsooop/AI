"""Look for obvious sensitive-value signals in changed text files."""

from __future__ import annotations

import argparse
import re
import sys

from lib import add_target_args, get_added_lines, get_changed_files, target_from_args


SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("SECRET_KEY", re.compile(r"\bSECRET_KEY\b", re.IGNORECASE)),
    ("PASSWORD", re.compile(r"\b(PASSWORD|PASSWD|PWD)\b", re.IGNORECASE)),
    ("TOKEN", re.compile(r"\b(TOKEN|ACCESS_TOKEN|REFRESH_TOKEN)\b", re.IGNORECASE)),
    ("API_KEY", re.compile(r"\b(API_KEY|OPENAI_API_KEY)\b", re.IGNORECASE)),
    ("PRIVATE_KEY", re.compile(r"\bPRIVATE_KEY\b|BEGIN [A-Z ]*PRIVATE KEY", re.IGNORECASE)),
    ("DATABASE_URL", re.compile(r"\b(postgres|postgresql|mysql)://", re.IGNORECASE)),
    ("SSH_PUBLIC_KEY", re.compile(r"\bssh-rsa\b|\bssh-ed25519\b", re.IGNORECASE)),
]

PLACEHOLDER_HINTS = (
    "example",
    "placeholder",
    "dummy",
    "sample",
    "your_",
    "change_me",
    "이슈번호",
)


def scan_added_lines(mode: str, base: str | None) -> list[tuple[str, int, str, str]]:
    findings: list[tuple[str, int, str, str]] = []
    for added in get_added_lines(mode=mode, base=base):
        if added.path == "scripts/agent_checks/check_secret_leaks.py":
            continue
        lowered = added.text.lower()
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(added.text):
                hint = "placeholder-like" if any(token in lowered for token in PLACEHOLDER_HINTS) else "verify"
                findings.append((added.path, added.line_no, label, hint))
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_target_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mode, base = target_from_args(args)
    print("[secret-check]")
    findings: list[tuple[str, int, str, str]] = []
    for changed in get_changed_files(mode=mode, base=base):
        if changed.path.endswith(".env") or "/.env" in changed.path:
            findings.append((changed.path, 0, "ENV_FILE", "verify"))
    findings.extend(scan_added_lines(mode=mode, base=base))

    if not findings:
        print("- no obvious sensitive patterns found in added lines")
        return 0

    for path, line_no, label, hint in findings:
        location = path if line_no == 0 else f"{path}:{line_no}"
        print(f"- {location}: {label} ({hint})")
    print("[warning] Verify matches are placeholders or non-secret references before publishing.")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(2)
