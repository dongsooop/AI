"""Look for obvious sensitive-value signals in changed text files."""

from __future__ import annotations

import re

from lib import get_changed_files, is_probably_text, read_text_file


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


def scan_file(path: str) -> list[tuple[int, str, str]]:
    if not is_probably_text(path):
        return []
    findings: list[tuple[int, str, str]] = []
    try:
        text = read_text_file(path)
    except OSError:
        return []
    for line_no, line in enumerate(text.splitlines(), start=1):
        lowered = line.lower()
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                hint = "placeholder-like" if any(token in lowered for token in PLACEHOLDER_HINTS) else "verify"
                findings.append((line_no, label, hint))
    return findings


def main() -> int:
    print("[secret-check]")
    files = get_changed_files()
    findings: list[tuple[str, int, str, str]] = []
    for changed in files:
        if changed.path == "scripts/agent_checks/check_secret_leaks.py":
            continue
        if changed.path.endswith(".env") or "/.env" in changed.path:
            findings.append((changed.path, 0, "ENV_FILE", "verify"))
        for line_no, label, hint in scan_file(changed.path):
            findings.append((changed.path, line_no, label, hint))

    if not findings:
        print("- no obvious sensitive patterns found in changed text files")
        return 0

    for path, line_no, label, hint in findings:
        location = path if line_no == 0 else f"{path}:{line_no}"
        print(f"- {location}: {label} ({hint})")
    print("[warning] Verify matches are placeholders or non-secret references before publishing.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
