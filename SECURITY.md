# Security Policy

AutoResearch runs untrusted, machine- and LLM-generated code as part of its
normal operation, so its threat model is central to the project rather than an
afterthought. This document summarizes what the system defends against and how
to report a problem.

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Instead, report
privately via GitHub:

1. Go to the repository's **Security** tab → **Report a vulnerability** (GitHub
   private security advisories).
2. Describe the issue, the affected version/commit, and a reproduction if you
   have one.

You can expect an acknowledgement and a discussion of next steps. Coordinated
disclosure is appreciated.

## What is in scope

The interesting attack surface is the loop's ability to keep a
"plausible-looking failure" or a tampered candidate from being scored as
"verified progress." Reports are especially valuable if they show a way to:

- **Leak or read the held-out seeds** (`evaluation/heldout_config.json`,
  dev/gate/test seeds) from inside a candidate workspace.
- **Forge or tamper with evaluator metrics** (bypass the nonce echo, the
  evaluator self-hash, or the SHA-256 protection manifest).
- **Escape the coder worktree confinement** (read outside the worktree or write
  outside `src/`, evade the PreToolUse guard hook, or defeat the
  root-fingerprint snapshot comparison).
- **Escape the execution sandbox** under `sandbox.backend: container` (break
  `--network none`, the read-only rootfs, seed/ledger masking, or the ephemeral
  PID namespace).
- **Leak gate scores** into proposer context, insight, or any deterministic
  output (the blindness invariant).
- **Launder unsupported claims** through the literature path into a
  support-granting stance.

## Trust grades (important)

Isolation strength depends on the configured backend:

- **`sandbox.backend: container`** (opt-in) — real OS isolation: held-out seeds
  are masked and absent from the container filesystem; network, out-of-workspace
  writes, and TOCTOU daemon persistence are blocked at the OS level.
- **`sandbox.backend: subprocess`** (default) — policy-level protection only.
  The training subprocess can, in principle, read seed files by absolute path.
  This mode is intended for dev/smoke use. Running gate/test scoring under
  `subprocess` makes the orchestrator **always warn**, and
  `sandbox.require_container_for_trusted_splits: true` fail-closes it entirely.
  The trust grade is stamped into the `report.md` header.

If you are running campaigns whose results you intend to trust or publish, use
the `container` backend with a digest-pinned image. See the "Protection model"
and "Honest limitations" sections of the [README](README.md) and the canary in
`tests/test_phase6c.py` for the full threat model.

## Secrets

No secrets are committed to this repository, and none should be. The Semantic
Scholar key (`S2_API_KEY`) is **environment-only** and used only during
`ground --refresh`; it must never be placed in the contract or committed. The
LLM paths reuse a local Claude Code login via the Claude Agent SDK rather than a
stored API key.
