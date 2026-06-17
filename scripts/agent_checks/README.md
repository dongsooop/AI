# Dongsooop Agent Checks

This directory contains advisory preflight checks for Codex/Agent workflows.

The checks help reviewers classify changed files, detect obvious sensitive patterns in newly added lines, warn about generated artifacts, and remember required branch-review output sections.

These scripts do not replace dedicated security scanners such as Gitleaks, TruffleHog, or GitHub Secret Scanning.

## Usage

Inspect the current working tree, including staged, unstaged, and untracked files:

```bash
python scripts/agent_checks/run_agent_preflight.py
```

Inspect only staged changes:

```bash
python scripts/agent_checks/run_agent_preflight.py --staged
```

Inspect committed branch changes against a base ref:

```bash
python scripts/agent_checks/run_agent_preflight.py --base main
```

Allow advisory warnings while still failing execution errors:

```bash
python scripts/agent_checks/run_agent_preflight.py --base main --warnings-ok
```

The same target options are supported by individual check scripts.

## Exit Codes

| Code | Meaning |
| --- | --- |
| `0` | No warnings were found. |
| `1` | Warnings were found and should be reviewed. |
| `2` | A check could not run because of an execution error. |

Warnings are advisory. Review the output and decide whether the branch needs changes before publishing.
Use `--warnings-ok` for lightweight CI gates that should display advisory warnings without blocking the PR.
