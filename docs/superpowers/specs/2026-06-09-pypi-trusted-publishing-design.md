# Design: Publish longcombat-py to PyPI via Trusted Publishing

**Date:** 2026-06-09
**Status:** approved, in progress
**Supersedes the manual-upload path in** `notes/2026-04-24_handoff_pypi_publish.md`

## Goal

`pip install longcombat-py` installs the package from PyPI. Publishing is done
by GitHub Actions using **PyPI Trusted Publishing (OIDC)** so no API token is
ever created, stored, or pasted into a terminal. The GitHub repo is flipped
public as part of the same effort.

## Context (verified 2026-06-09)

- Package is release-ready at **v0.1.1**; `pyproject.toml` fully configured
  (PEP 621 metadata, PEP 639 license, src-layout, `py.typed`).
- PyPI names `longcombat-py` and `longcombat` are both free (404).
- CI (`tests.yml`) is green on the release commit `cdf9df8`.
- Local pre-flight all pass: `python -m build`, `twine check` (both
  artifacts), clean sdist (no venv / R-lib / caches), and a fresh-venv wheel
  install + full-API import with `__version__ == 0.1.1`.
- The `v0.1.1` tag is **already pushed**, so a "publish on tag push" trigger
  cannot fire for this release.
- Repo is currently **private**.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Publish mechanism | Trusted Publishing (OIDC) | No token to handle; PyPA-recommended; automates future releases. |
| Workflow trigger | `release: published` + `workflow_dispatch` (with required `tag` input) | `v0.1.1` already pushed, so tag-push won't fire; Release-triggered also folds in the "make a proper GitHub Release" step. |
| Tag handling | Recreate `v0.1.1` on a new commit that contains `publish.yml` | A `release` event reads the workflow **from the tag's commit** (`release` is not one of the four default-branch-only events per the 2019-10-24 GitHub changelog). `v0.1.1` at `cdf9df8` predates `publish.yml`, so the trigger would not fire. `v0.1.1` was never published anywhere, so moving the tag is safe. |
| README links | Rewrite relative → absolute GitHub URLs | Relative links (`DIFFERENCES_FROM_R.md`, `LICENSE`, `NOTICE`) 404 on the PyPI page; 0.1.1's page is immutable. Folded into the same release commit. |
| TestPyPI dry-run | Skip | Local `twine check` + fresh-venv install already cover "does it build / install / import". |
| GitHub environment | `pypi` | Enables later required-reviewer/branch protection on releases. |
| Repo visibility | Public | sdist exposes all source on publish anyway; makes PyPI project URLs resolve; unlimited Actions minutes. |
| Tests in publish job | No | CI already gated this exact commit; keep the publish path minimal. |

## Components

### `.github/workflows/publish.yml`

Canonical PyPA two-job recipe:

- **build** job: checkout pinned to the tag (`ref: release.tag_name ||
  inputs.tag` — never a branch HEAD) → set up Python 3.13 → `python -m build` →
  `twine check dist/*` → upload `dist/` as an artifact.
- **publish** job: `needs: build`, `environment: pypi`,
  `permissions: id-token: write` only → download the artifact →
  `pypa/gh-action-pypi-publish@release/v1`.
- Top-level `permissions: contents: read` (least privilege); the publish job
  opts in to `id-token: write`.
- The explicit `ref` pin closes the audit's `workflow_dispatch` footgun (a
  manual run can never publish whatever is on the default branch); the
  `tag` input is therefore required.

Action versions pinned to major refs (`@v4`/`@v5`, `@release/v1`) to match the
existing `tests.yml` convention.

## Division of responsibility

**Automated / in-repo (done by Claude):**
1. Write `publish.yml` (tag-pinned build) and this spec; fix README links.
2. Commit `publish.yml` + spec + README fix to `main`, then recreate the
   `v0.1.1` tag on that commit (so the tag commit contains `publish.yml`) and
   push both.
3. Rebuild + verify artifacts locally at 0.1.1 after the README change.
4. Adversarial pre-flight audit of the workflow + release readiness before the
   first (irreversible) upload (done — unanimous GO, zero blockers).

**Credential-gated / outward-facing (done by the user, with exact steps):**
1. Create/confirm a PyPI account.
2. Add a PyPI **pending publisher**: project `longcombat-py`, owner
   `NoiseFilterT`, repository `longcombat-py`, workflow `publish.yml`,
   environment `pypi`.
3. Flip the repo to **public**.
4. Draft + publish a **GitHub Release** on the existing `v0.1.1` tag → fires
   the workflow → OIDC upload to PyPI.
5. Confirm `pip install longcombat-py` works.

## Risks

- **PyPI versions are immutable** — 0.1.1 can never be re-uploaded. Mitigated by
  the full local pre-flight and the pre-trigger adversarial audit.
- **Pending-publisher misconfiguration** (owner/repo/workflow/environment must
  match exactly) is the most common OIDC failure mode → the runbook lists the
  exact values.
- **`twine check` ≠ Warehouse upload validation.** The first 0.1.1 upload was
  rejected (400) because a `project.urls` label contained a comma: Warehouse
  partitions each `Project-URL` on `", "` into `label, url`, so the comma split
  the URL. `twine check` does not replicate this. Fixed by removing the comma
  (labels must also be ≤32 chars). The upload failed closed (nothing stored),
  so 0.1.1 was re-publishable. A TestPyPI dry-run would have caught this.

## Out of scope (follow-ups)

- Auto-CHANGELOG via commitizen (`update_changelog_on_bump`).
- Broadening CI to minimum dependency floors.
- Upstream license clarification (MIT vs Artistic-2.0).
- Opening an issue on `jcbeer/longCombat` linking this port.
