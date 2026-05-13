---
name: reviewing-dongsooop-branch
description: Review Dongsooop repository branches before pushing, opening pull requests, or publishing changes. Use when the user asks for a branch review, current changes review, pre-push check, security/publication check, PR body draft, or commit message suggestions for the Dongsooop backend.
---

# Reviewing Dongsooop Branch

## Overview

Use this skill to perform a practical branch review for the Dongsooop backend. The review should check changed files, security exposure, runtime risks, Docker/dependency effects, environment variable consistency, and GitHub publication readiness.

## Required References

Read these files in order before reviewing:

1. `docs/AGENTS.md` for project guardrails
2. `docs/PLANS.md` for current architecture and service boundaries
3. `docs/BRANCH_REVIEW_GUIDE.md` for review criteria
4. `docs/CHECKLIST.md` for the operational checklist

Also inspect `.github/pull_request_template.md` when drafting a PR body.

## Workflow

1. Check `git status --short` and the diff to understand the branch scope.
2. Classify affected areas: text filtering/timetable service, OSS chatbot service, shared core, Docker/dependencies, data/artifacts, GitHub workflow, or documentation.
3. Review security first: secrets, tokens, `.env`, private URLs, server access details, unsafe file/path handling, logs, and tracked generated/private files.
4. Review behavior risks against the two-service architecture and Korean NLP/OCR/RAG constraints in `docs/PLANS.md`.
5. Check Docker, requirements, workflow, and environment-variable consistency when related files changed.
6. Decide whether available validation should run. Use existing lightweight checks for documentation-only changes; use focused compile/regression commands for code changes when practical.
7. Report findings first, ordered by severity. If no issues are found, say so and list residual unverified risks.
8. Draft a PR body using `.github/pull_request_template.md`, preserving unknown issue numbers as `Closes #이슈번호`.
9. Suggest three commit messages using `type: english summary`, choosing from `feat`, `fix`, `refactor`, `chore`, or `docs`.

## Review Priorities

- Keep chatbot work separate from text filtering/timetable work unless the diff intentionally touches shared code.
- Treat `LLM/OSS/Open_AI_OSS.py` as high-risk because it is large and central.
- Do not edit generated files under `model/artifacts/`; review the generation logic instead.
- Preserve the OCI deployment assumption that Docker images may copy `.env` internally unless the user explicitly asks to change it.
- Keep `AGENTS.md`, `PLANS.md`, `SKILL.md`, and `CHECKLIST.md` as local ignored files; public documentation belongs under `docs/`.

## Output Shape

Start with review findings:

1. Security issues
2. Functional or regression risks
3. Privacy/GitHub publication issues
4. Docker/dependency issues
5. Environment variable or deployment mismatches
6. Residual risks or unverified items

Then include a concise PR body draft and three commit message candidates.
