"""Warn when generated model artifacts are part of the current diff."""

from __future__ import annotations

from lib import get_changed_files


ARTIFACT_PREFIX = "model/artifacts/"


def main() -> int:
    print("[artifact-check]")
    changed = [item.path for item in get_changed_files() if item.path.startswith(ARTIFACT_PREFIX)]
    if not changed:
        print("- no generated artifacts changed")
        return 0
    for path in changed:
        print(f"- {path}")
    print("[warning] Do not edit generated artifacts directly. Update generation logic instead.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

