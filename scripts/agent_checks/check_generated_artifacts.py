"""Warn when generated model artifacts are part of the current diff."""

from __future__ import annotations

import argparse
import sys

from lib import add_target_args, get_changed_files, target_from_args


ARTIFACT_PREFIX = "model/artifacts/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_target_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mode, base = target_from_args(args)
    print("[artifact-check]")
    changed = [item.path for item in get_changed_files(mode=mode, base=base) if item.path.startswith(ARTIFACT_PREFIX)]
    if not changed:
        print("- no generated artifacts changed")
        return 0
    for path in changed:
        print(f"- {path}")
    print("[warning] Do not edit generated artifacts directly. Update generation logic instead.")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(2)
