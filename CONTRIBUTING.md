# Contributing to AutoResearch

Thanks for your interest! This project is a reference implementation of an
autonomous-research loop, and its whole point is **evaluation integrity**. That
shapes how contributions work — please read the "Protected files" section below
before you start editing.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (dependency and Python manager)
- Python 3.13+ (the repo pins a version in `.python-version`; `uv` will fetch it
  for you)
- Git

Optional (only for opt-in LLM features): a working local
[Claude Code](https://claude.com/claude-code) login (the Claude Agent SDK reuses
it — no separate API key), and Docker if you want the hardened `container`
sandbox backend.

## Setup

```bash
git clone <your-fork-url>
cd autoresearch
uv sync
uv run python orchestrator.py init   # git baseline + protection manifest + held-out seeds
```

## Running the checks

Everything below is **offline and deterministic** — no network, no Docker, no
LLM calls.

```bash
# Lint
uvx ruff check .

# The unit drills (each is a self-contained script, not pytest)
uv run python tests/test_phase2.py
uv run python tests/test_phase3.py
uv run python tests/test_phase4.py
uv run python tests/test_phase5.py
uv run python tests/test_phase6.py
uv run python tests/test_phase6b.py
uv run python tests/test_phase6c.py
```

CI (`.github/workflows/ci.yml`) runs exactly these. A PR must be green.

## Protected files (read this first)

The core of the system is **frozen** and covered by a SHA-256 manifest in
`protection/hashes.json`. These paths are protected:

```
orchestrator.py  research_contract.yaml  pyproject.toml  uv.lock  .python-version
evaluation/**    literature/**  assurance/**  sandbox/**  protection/**
```

At runtime the orchestrator verifies this manifest every generation and refuses
to proceed if it changed — this is what keeps a candidate (or a bug) from
tampering with the evaluator, the contract, or the held-out seeds. Consequences
for contributors:

- **Do not edit a protected file casually.** If a change genuinely belongs
  there, you must intentionally unlock and re-baseline:
  ```bash
  chmod u+w <file>
  # edit
  uv run python orchestrator.py init --force   # regenerates seeds + manifest + baseline
  ```
  `init --force` **wipes `experiments/`** — back it up first if you need prior
  campaign records. Call this out explicitly in your PR.
- The **editable surface** for hypotheses is `src/train.py` (the `HYPERPARAMS`
  block + `FEATURE_SPEC`). `tests/**` and `docs/**` are also freely editable.
- If a change to a protected file is not accompanied by a manifest rebuild, CI
  and `verify-protection` will fail — that is intended.

## What does *not* get committed

Runtime state is provenance, but it lives in git branches and the ledger, not in
tracked files. `.gitignore` already excludes:

- `experiments/`, `artifacts/`, `.worktrees/`, `insight_memory.json` — per-run
  state
- `evaluation/heldout_config.json` — the held-out seeds (deliberately untracked
  so they are physically absent from candidate worktrees)

Per-experiment provenance lives on `hyp/<campaign>/rNNNN-*` branches. **These are
local runtime artifacts — do not push them.** Publish `main` only
(`git push origin main`).

## Pull requests

1. Fork and branch from `main`.
2. Keep changes focused; match the surrounding code's style, naming, and comment
   density.
3. Make sure `uvx ruff check .` and all seven drills pass locally.
4. If you touched a protected file, say so and describe the `init --force`
   rebuild in the PR description.
5. Describe *what* and *why*; link any related issue.

## Reporting bugs / requesting features

Open an issue using the templates. For anything security-relevant (sandbox
escape, evaluator tampering, seed leakage), please follow
[SECURITY.md](SECURITY.md) instead of filing a public issue.
