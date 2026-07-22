# AutoResearch — an executable autonomous-research loop

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](pyproject.toml)

*🇰🇷 한국어 원문은 [README.ko.md](README.ko.md) 에 있습니다.*

An implementation of a 2026 SOTA blueprint (a synthesis of **Arbor / Gome / ERA /
SciNav**). It starts from a **Karpathy-style keep/reject loop plus Arbor-style state
management** and grows into a **parallel hypothesis portfolio with a blind admission
gate and LLM coding workers** (Phase 2), **claim-level literature grounding via an
evidence graph** (Phase 3), and **directional branch refinement — Gome search
momentum + evidence-based steering + successive halving + a SciNav pairwise gate**
(Phase 4). Later phases add an assurance & reporting layer, real OS execution
isolation, a real literature API, and a real research domain (Euclidean-TSP).

The organizing principle: **build the evaluator, the research contract, and
provenance *before* the agent swarm.**

> **Core thesis:** *Without a trustworthy evaluator, adding more agents does not make
> you more scientific.* Every structure in this repo exists to separate
> "plausible-looking failure" from "verified progress."

---

## Who this is for

- You are interested in **autonomous / agentic research loops** and want a concrete,
  runnable reference implementation rather than a diagram.
- You care about **evaluation integrity** — held-out splits, anti-overfitting gates,
  provenance, and reproducibility — as much as about the agents themselves.
- You want to **swap in your own domain**: the search loop, the gate, and the
  assurance layer are domain-agnostic; the shipped example is Euclidean-TSP heuristics.

Everything runs **fully offline and deterministically by default.** LLM features
(hypothesis generation, coding workers, literature analysis, cross-model review) are
**opt-in** and reuse your local Claude Code login through the Claude Agent SDK — no
separate API key required for the default path.

## Quick start

```bash
uv sync                                            # dependencies (pyyaml, claude-agent-sdk)
uv run python orchestrator.py init                 # git init + baseline + protection manifest
uv run python orchestrator.py ground               # research-question certificate (literature evidence flow)
uv run python orchestrator.py run --generations 4  # parallel portfolio run (heuristic + lexical literature + halving + momentum)
uv run python orchestrator.py status               # campaign status (literature stats + momentum + admissions/reviews)
uv run python orchestrator.py report               # multi-seed test-split report → first run waits for approval (exit 3 + request_id)
uv run python orchestrator.py approve <request_id> # approve publication intent → report then seals multi-seed eval, claims, and report
uv run python orchestrator.py report --reviewer codex  # (after approval) include cross-model codex adversarial review (opt-in)
uv run python orchestrator.py verify-protection
uv run python tests/test_phase2.py                 # Phase 2 drill
uv run python tests/test_phase3.py                 # Phase 3 drill (literature grounding)
uv run python tests/test_phase4.py                 # Phase 4 drill (momentum / steering / halving / pairwise)
uv run python tests/test_phase5.py                 # Phase 5 drill (bootstrap / claims / report / gate / reviewer / families)
uv run python tests/test_phase6.py                 # Phase 6a drill (sandbox argv / masking / fail-closed, no Docker needed)
uv run python tests/test_phase6b.py                # Phase 6b drill (real literature fetch / extraction / snapshot, offline fake HTTP/LLM)
uv run python tests/test_phase6c.py                # Phase 6c drill (TSP feasibility / objective recompute / seed absence / blindness)
```

Add `--gate pairwise` to keep admission (the scalar-epsilon rule) as-is but let a
blind LLM judge panel pick the winner *among* candidates that already passed admission
(the default is a deterministic scalar):

```bash
uv run python orchestrator.py run --generations 4 --proposer claude --gate pairwise
```

The LLM hypothesis proposer, coding worker, and literature analyzer all run on the
Claude Agent SDK (reusing your local Claude Code login — no separate API key). Coder
hypotheses fire only under `--proposer claude`, and the LLM literature path only under
`--literature claude` (the default is fully offline):

```bash
uv run python orchestrator.py run --generations 4 --proposer claude --literature claude
```

## Mental model (the concepts you actually need)

| Concept | What it is |
|---|---|
| **Campaign** | One end-to-end run, namespaced by id (`c<timestamp>`). Provenance for every experiment lives on `hyp/<campaign>/rNNNN-*` branches + `experiments/ledger.jsonl`. |
| **Generation** | One round. The proposer emits **K diverse hypotheses** (default 4), each targeting a different bottleneck; they run **in parallel** in isolated git worktrees. |
| **Hypothesis certificate** | statement / mechanism / intervention / predicted_effect / **falsifier** (rejection condition) / minimal_test. |
| **dev / gate / test splits** | Three disjoint, differently-seeded splits. **dev** drives search; **gate** is a hidden hold-out that decides admission; **test** is touched exactly once, at `report`. |
| **Blind admission gate** | A candidate must beat the incumbent's *gate* score by a margin to win; the winner (only one per generation) is ff-merged to `main`. Prevents development-set overfitting. |
| **Verdict** | Deterministic classification of each candidate (see table below). |
| **Provenance** | `main` accretes only gate-passing experiments (ff-merge); everything else is preserved on `hyp/*` branches + the append-only ledger. |

### Two-stage evaluation (development / blind admission)

- The **dev split** is for search — all K candidates are scored against the incumbent
  at the start of the generation. Only the top `gate_top_k` improvers
  (`valid_positive`) advance.
- The **gate split** is a separate hidden hold-out. A candidate must beat the
  incumbent's gate score by `gate_min_relative_improvement` to become the winner, and
  only that one winner is ff-merged to `main`. This filters out candidates that
  improved marginally on dev but did not generalize (development-set overfitting).
- **Blind protocol**: gate scores exist only in `record_type=gate` ledger records and
  gate metric files. They never leak into insight, `best_primary` (always the dev
  score), proposer context, or experiment records. The incumbent gate score is cached
  per-commit; because evaluation is deterministic, this cache is exact, not
  approximate.
- The **test split** is used exactly **once**, at `report` at the end of the campaign.
  Re-running it requires `--force`, and each such use is recorded in the
  multiple-comparisons disclosure.

### Verdict classification

| verdict | meaning | handling |
|---|---|---|
| `valid_positive` | dev-relative improvement ≥ `min_relative_improvement` | gate candidate (only the winner is KEPT) |
| `valid_inconclusive` | change below threshold | reject, increment generation stagnation |
| `valid_negative` | metric regressed / NaN divergence / no_skill / crash / timeout | reject — **no repair, distilled as scientific evidence** |
| `invalid_implementation` | mechanical failure (patch failed / coder mechanical smoke failure / non-determinism) | round voided |
| `contract_violation` | touched a protected path / src symlink / oversized diff | reject + evidence preserved |

`valid_positive` only means "improved on dev"; adoption is conditional on passing the
blind gate. In a single generation only one gate winner is accepted; the other
dev-improvers stay `valid_positive` but with `decision=REJECT`.

Why divergence/crash/timeout is `valid_negative` (the attribution rule): the baseline
is proven runnable at `init`, so any runtime failure *after* a valid intervention is
attributed to the intervention. There is deliberately no post-evaluation "repair" —
this blocks false repair. Coder-hypothesis repair is allowed only for **mechanical
smoke-stage failures** (`nonzero_exit` / `missing_artifact` / `malformed_artifact`),
and only when a scorable model has not yet been produced. Timeout, divergence,
no_skill, and dev-stage failures are all kept as evidence.

## What the loop does

Each generation:

1. Verify the protection manifest (SHA-256 of every protected file).
2. Generate **one hypothesis certificate** — statement / mechanism / intervention /
   predicted_effect / **falsifier** / minimal_test. Deterministic heuristic by
   default; under `--proposer claude` the Claude Agent SDK generates it (tools fully
   disabled, JSON-schema-forced output, heuristic fallback on validation failure).
3. Create an isolated git worktree off the incumbent (branch
   `hyp/<campaign>/rNNNN-<param>` — the campaign namespace means no collision with
   prior-campaign branches even after `init --force`).
4. Substitute **exactly one parameter** in the HYPERPARAMS marker block of
   `src/train.py` (ast parse + round-trip verify; only mechanical failures are
   repaired, within a bounded retry count).
5. Commit **before** evaluation (commit = pure code change).
6. Check protected/editable globs (both **before and after** evaluation).
7. Run the **root** evaluator: smoke (2 epochs) → dev. The `evaluation/` copy inside
   the worktree is never executed.
8. Classify → **write-ahead** the verdict to the ledger → on improvement `--ff-only`
   merge, otherwise reject (the branch is preserved as provenance).
9. Distill the lesson into `insight_memory.json`, update state, check stop conditions.

## Phase 2 — parallel portfolio · blind gate · LLM coder

Each **generation** has the proposer emit **K diverse hypotheses (default 4)** at
once, each aimed at a different bottleneck, run **in parallel** in isolated worktrees
(concurrent.futures threads + subprocess evaluators; only the git-shared-state
mutations — worktree add/remove, branch deletion, merge — are serialized under a lock;
workers touch only their own worktree).

**LLM coding worker (executor="coder"):** some walls can't be crossed with
hyperparameters — e.g. the irreducible floor of a linear model. Crossing them requires
editing code. The proposer can emit coder hypotheses
(`portfolio.max_coder_hypotheses`, default 1), and a ClaudeCoder edits `src/**` inside
the worktree. Isolation is layered:

- A **PreToolUse guard hook** is the sole authorizer. cwd cannot cage the SDK's tools
  (absolute paths are allowed), so the hook resolves every tool call via realpath and
  rejects Read/Glob/Grep outside the worktree and Write/Edit outside `src/`. Bash and
  network tools are disabled entirely.
- `permission_mode="dontAsk"` + `allowed_tools=[]` make it **fail-closed**: if the
  hook errors or times out, the default is deny.
- **Root-fingerprint snapshot comparison** before and after the coder call (git status
  + protection manifest) aborts the campaign if an escape touches the root via an
  absolute path.
- Symlinks under `src/`, oversized diffs, and a determinism re-check (gate-entering
  candidates re-run dev twice and compare bitwise) are all checked.

Model-class expansion is handled **as data, without executing candidate code inside
the evaluator**: the evaluator only validates and applies the artifact's
`feature_spec` (a list of products of original feature indices, up to 32 terms, degree
≤ 3) during scoring.

### The mock ML task (Phase 2 illustration)

> **Note on the current domain:** the shipped default domain is now **Euclidean-TSP**
> heuristics (Phase 6c, contract v8). The synthetic-regression task below is the
> original Phase 2 illustration and is the clearest way to understand the coder
> mechanism; see [docs/WORKBOOK.md](docs/WORKBOOK.md) for the TSP domain.

Pure-Python synthetic regression (no external dependencies, fully deterministic):

- 8 features with heterogeneous scales `[1, 1, 5.0, 0.2, 1, 3.0, 1, 0.5]` (condition
  number ~625) — this makes `feature_scaling` actually matter.
- `y = 0.3 + w·x + 0.3·x0·x1 + N(0, 0.25)` — the interaction term is **not** learnable
  by a linear model, so there is an irreducible floor (~0.39) that hyperparameters
  cannot cross.
- Constant predictor RMSE ~1.68, baseline ~0.51–0.54, unscaled lr ≥ 0.08 diverges to
  NaN.
- **The coder's target**: extending `FEATURE_SPEC` in `src/train.py` with a product
  term like x0·x1 breaks through the floor. In a live demo the LLM coder added this
  term, reached dev RMSE ~0.25 (below the floor), and had that improvement — impossible
  with hyperparameters alone — accepted through the blind gate. This is the core Phase
  2 demonstration.

### What else Phase 2 closes

- **blind admission gate**: even something that looks good on dev is not adopted if it
  fails to generalize on the hidden gate split (development-set overfitting). The
  3-way split (dev/gate/test) uses entirely different seeds, all absent from the
  worktree.
- **gate-score blindness**: the gate value never enters proposer context, insight,
  `best_primary`, or experiment records. `distill_insight` never reads gate records,
  and an automated test enforces this invariant.
- **LLM coder isolation**: fail-closed PreToolUse guard hook + root-fingerprint
  snapshot comparison + src symlink / oversized-diff / determinism re-check +
  smoke-only mechanical repair.
- **test split is one-shot**: `report` increments the multiple-comparisons disclosure
  counter on re-run.
- **parallel safety**: only git-shared-state mutations are lock-serialized; worker
  threads write no state/ledger (everything after a barrier, on the main thread);
  gc.auto disabled.

## Phase 3 — literature grounding (claim-level evidence graph)

Hypotheses stand on literature evidence, novelty and contradictions are reported, and
every citation is auditable. `literature/` is a **separately controlled service**: it
does not import the orchestrator, reads no file other than the corpus (no access to
state / ledger / metrics — closed at the signature level), and writes nothing at
runtime.

- **Offline corpus** (`literature/corpus/tsp_corpus.json`, `tsp-heuristics-v1`): 13
  TSP-heuristics papers / 13 claims. Each claim has a locator (section, table, page),
  population, conditions, limitations, and structured tags. It includes contradiction
  pairs, negative results reachable only through citation traversal, a
  citation-laundering trap, and prompt-injection fixtures. This snapshot is a curated
  offline corpus (provenance empty); to refresh it from real literature, regenerate on
  a networked host via `ground --refresh` (OpenAlex/S2 adapters). The search backend
  sits behind a `Retriever` protocol, so it can be swapped for a real-API adapter.
- **Dual mode**: the default is deterministic lexical search (3 indexes + citation BFS
  + coverage stop — reproducible, testable). `--literature claude` delegates query
  decomposition, stance judgment, and novelty description to the LLM, but **search
  execution is always the deterministic backend**, and the LLM can only *downgrade* a
  deterministic "supports" (structurally blocking laundering). The LLM path is
  structured-output-only with `tools=[]`, so literature text has no surface on which
  to execute code or shell, and it falls back to lexical on failure so it never blocks
  a generation.
- **Hypothesis certificate extension**: `supporting_evidence_ids` (only
  whitelist-verified ids) + `nearest_prior_work`. Hypotheses carry **ids only** — claim
  prose never enters the blindness-scan surface of the ledger or insight. Novelty is
  categorical with no numeric score (`replication / regime_extension /
  contradiction_test / unexplored`).
- **Evidence memory is separate from the ledger**:
  `experiments/evidence/evidence.jsonl` (append-only, timestamped to distinguish crash
  retries) + per-generation snapshots
  `experiments/generations/gNNNN/evidence.json` (idempotent). It is computed and
  recorded only *before* any gate score exists (right before propose, on the main
  thread), so invariant 4 (blindness) is preserved both structurally and temporally.
- **Audit and cost**: `report` resolves every citation of an accepted hypothesis into
  (paper, claim, locator) (unresolvable = hard error) and sums the three-way
  proposal/coder/literature cost. The campaign LLM literature budget
  (`llm_max_campaign_budget_usd`) is enforced by summing evidence.jsonl and downgraded
  to lexical on overrun.

## Phase 4 — directional branch refinement

Makes search narrow **in the observed direction** rather than widening blindly. All
three parts structurally preserve the gate-blindness, false-repair, and
literature-closure invariants.

- **Gome search momentum + evidence steering** (offline, deterministic): at the start
  of each generation, only the experiment records in the ledger are folded
  (`extract_update_vectors` → `search_momentum_table`) into a direction score per
  `{param}:{move}` — accept +1.0 / gate-dropped dev-improver +0.4 / regression −1.0 /
  divergence·timeout −1.0 (+ boundary value recorded) / inconclusive −0.2, decayed at
  each generation boundary. It is **not stored in state**; like insight_memory it is
  recomputed from the ledger every time, so replay==live holds trivially under crash
  recovery, and since only the dev signal and accept/reject bits are inputs, the gate
  score structurally cannot enter. The heuristic proposer re-ranks candidates by this
  momentum (1st priority), literature stance (2nd; supports < none < contradicts, and
  refuting evidence only downgrades, never removes), and static priority (3rd), and
  adds an accelerated step after consecutive accepts plus a geometric bisection toward
  a divergence boundary. At least one of the K is reserved for a
  momentum-0 / literature-unsupported direction to prevent hypothesis collapse. With
  `refinement.enabled=false` the behavior is byte-identical to the Phase 3 proposer.
  Literature steering comes out of the engine's pure method
  `Grounding.move_guidance()` (categorical enum + evidence ids only, no prose or
  numbers), and `attach()` remains annotation-only.
- **Successive halving** (offline): all K branches of a generation run a cheap smoke
  rung, and only the top `max(min_keep, ceil(K·keep_fraction))` by smoke score advance
  to the dev rung. Elimination is the new verdict `pruned` — a **budget decision, not
  scientific evidence** — so it distills no insight and carries momentum weight 0, but
  registers the tested endpoint to prevent re-proposal (symmetric across
  _finish_generation and replay). The coder's mechanical-repair loop is **inside** the
  smoke stage, so there is nothing to repair at the halving cut — the false-repair
  boundary is structurally preserved.
- **SciNav pairwise gate** (LLM opt-in, `--gate pairwise`): admission is **always** the
  deterministic scalar-epsilon rule (the anti-overfitting guarantee), and only when two
  or more candidates pass admission does an anonymized blind judge panel (N=3, majority
  vote) pick the winner. The LLM can never relax admission. Judges see only the
  contract, the hypothesis certificate (ids only), the bounded code diff, and the
  **dev** metric — **the gate score is not even an input** (blindness holds by
  closure). Per-judge A/B labels are scrambled by sha256 parity to cancel position
  bias, verdicts are forced to a 4-way enum, and an anti-injection framing precedes any
  untrusted candidate material. Abstention, sub-majority, SDK failure, and campaign
  budget exhaustion all fall back to the deterministic scalar choice, and the
  deterministic counterpart `scalar_winner` is always recorded alongside in the gate
  record so divergence can be audited after the fact. The gate record is extended with
  `mode` / `scalar_winner` / `pairwise` (vote detail).

## Phase 5 — Assurance & reporting (claim-evidence ledger · deterministic report · cross-model reviewer · human gate)

Seals a campaign's results into **claims backed only by logged evidence**, has a
heterogeneous model family adversarially audit those claims, and requires human
approval before touching the untouched test split (the publication analog). All of it
is fully offline and deterministic (only the reviewer is opt-in and non-deterministic),
and the gate score enters no output.

- **Multi-seed finalist reproduction + paired bootstrap CI**: `heldout_config` goes to
  v3 with N seeds hidden in the test split (`assurance.finalist_seeds`, default 5); the
  evaluator scores per-seed datasets via `--seed-index` and emits per-example squared
  error only on test. `assurance/stats.py` **pairs baseline and incumbent per-example
  on the same dataset** (verifying fingerprint identity) and computes a confidence
  interval on the RMSE difference via a pooled paired bootstrap over the N×600 pool. The
  RNG seed is derived from campaign·commit and logged, reproducible from the ledger. If
  no candidate was ever adopted, incumbent==baseline, so it honestly reports effect 0,
  CI [0,0], status inconclusive.
- **claim-evidence ledger** (`experiments/claims.jsonl`): a derived artifact
  **fully regenerated** at report time from ledger, stats, and contract (5
  deterministic rules: main effect · campaign summary · adopted improvement · negative
  results · literature grounding). It is sealed into `final_report` via
  `claims_sha256`.
- **Deterministic report.md + SVG figures**: every number in the report is inserted
  only from claim/meta values, and a post-render digit scan hard-rejects any untraceable
  number (the number-via-claim invariant). The three figures are generated
  byte-deterministically from the immutable log as stdlib SVG and audited by sha256.
- **cross-model codex adversarial reviewer** (`--reviewer codex`, opt-in): an
  OpenAI-family `codex exec` audits each claim's numbers against the raw test data in a
  read-only sandbox (ARIS). The response is bound to the request via an echo_token
  planted in the prompt and re-verified; all failures are recorded as
  `status="unavailable"` and never block the report, and it never silently falls back
  to a Claude reviewer (preserving real heterogeneity — resolving the correlated bias of
  the Phase 4 pairwise judges). Results live only in `experiments/report/review/` and a
  `review` ledger record, and never enter report.md (keeping the number-via-claim scan
  consistent).
- **human approval gate**: the first `report` (and any `--force` re-run) writes an
  approval request to the ledger and emits **exit 3 + a request_id**. A human must
  approve intent (commit · dev numbers · seed plan · disclosures) via
  `approve <request_id>` to proceed. Approval state is derived from the ledger, not
  state, and the fingerprint includes the prior seal count so each `--force` forces
  re-approval and a stale approval is invalidated as the campaign advances.
- **momentum coder-family classification**: Phase 4's coarse `coder:none` momentum key
  is subdivided into families (`feature_spec_interaction`, etc.) that deterministically
  classify the coder diff. The family is stored in the experiment record so replay==live
  holds.

## Phase 6a — execution-isolation sandbox (real OS isolation · Blueprint Layer 5)

The one place untrusted candidate code is actually *executed* is `_run_train` in
`evaluation/evaluate.py` (scoring stays on the host trust path, and held-out seeds are
never passed there). Phase 6a wraps only that execution in OS isolation.

- **`Sandbox` protocol** (`sandbox/runner.py` — protected, stdlib, loaded by the
  evaluator via absolute path): `SubprocessSandbox` (current behavior ported,
  byte-identical under the `subprocess` default) + `ContainerSandbox` (`docker run`). It
  mirrors the literature `Retriever` shim pattern exactly.
- **Container hardening**: `--network none` (no network), `--read-only` (read-only
  rootfs), `--cap-drop ALL` + `--security-opt no-new-privileges`, non-root user,
  `--memory/--cpus/--pids-limit`, `--rm --init` (ephemeral PID namespace → no TOCTOU
  daemon persistence). The workspace is mounted `:ro`; only artifacts get a separate
  fresh host dir mounted rw (the worktree original stays immutable).
- **Seed/ledger masking**: whether the workspace is ROOT or a worktree,
  `heldout_config.json` is masked to an empty file and `experiments/` (gate scores) is
  masked with tmpfs → physically absent from the container FS.
- **fail-closed**: on missing daemon/image it never silently falls back to
  `subprocess`; it aborts with an actionable error (preflight at `run`/`init`/`report`
  entry). The image must be `docker pull`ed in advance (runs use `--network none`, so
  on-demand pull is impossible).
- **Contract v6 `sandbox` block**: `backend` (subprocess|container) · `image` (digest
  pin) · `memory_mb` · `cpus` · `pids_limit`. The backend is fixed per-campaign at init
  → baseline and candidates share a pipeline, so the Phase 5 paired bootstrap stays
  valid. The evaluator declares `SUPPORTED_SANDBOX_BACKENDS`, and init cross-checks it
  against the contract. `run_evaluator` echoes `metrics["sandbox"]` provenance against
  the requested backend (detects bypass / stale versions).
- **gVisor drop-in**: on Linux, adding a single `--runtime=runsc` line in
  `sandbox/runner.py` strengthens syscall isolation (mount / security model unchanged).

Opt-in example (after starting the Docker daemon and pulling the image):

```yaml
# research_contract.yaml
sandbox:
  backend: container
  image: "python:3.14-slim@sha256:<digest>"
  memory_mb: 512
  cpus: 1.0
  pids_limit: 128
```

```bash
docker pull python:3.14-slim@sha256:<digest>   # pulled in advance since runs use --network none
uv run python orchestrator.py init --force     # the baseline is trained in the container too
uv run python orchestrator.py run --generations 1
```

## Protection model (what Phase 1 closes)

- **Physical absence of held-out seeds**: `evaluation/heldout_config.json` is created
  at init and not tracked by git. A worktree checks out only tracked files, so the seed
  simply does not exist in a candidate workspace.
- **Only the root evaluator is authoritative** + the evaluator trusts nothing outside
  `evaluation/` (hardcoded budget — cross-checked once against the contract at init;
  imports dataset by absolute path).
- **nonce echo**: the orchestrator passes a fresh nonce each round, the evaluator
  echoes it in metrics (never passed to the training subprocess) → blocks forged
  metrics.
- **Isolated training subprocess**: env constructed from scratch (PATH,
  PYTHONHASHSEED=0), `-s -B` flags, its own session + process-group SIGKILL on timeout.
  **Phase 6a: this execution now goes through the `Sandbox` protocol and is promoted to
  OS isolation under `sandbox.backend: container` (see the Phase 6a section).**
- **Artifact validation**: symlink rejection, 1MB cap, finiteness check, train_rmse
  recomputation.
- **SHA-256 manifest** verified every round + protected files chmod 444 + worktree
  diff/status checked by glob before and after evaluation + the evaluator embeds its own
  hash in metrics for cross-check against the manifest.
- **Crash recovery**: write-ahead-ledger based — an interrupted round is treated as
  aborted (its round number is burned); an accepted-but-unmerged case is re-run.
  tested/stagnation/last_accepted are reconstructed wholesale from the ledger, so
  whenever the process dies the state cannot drift from the ledger. An accept that
  became un-mergeable is downgraded before recording; a merge failure after recording is
  offset by a correction record.
- **Single-instance lock** (flock): blocks a concurrent `run` from destroying an
  in-progress worktree. Moving a protected file via rename is also detected via diff
  `--no-renames`.

### Honest limitations (Phase 6b/6c work)

- **Real isolation is implemented in Phase 6a** — but enforced only when you opt into
  `sandbox.backend: container`. Under that backend the held-out seeds are masked and
  absent from the container FS, and TOCTOU (background daemon) / out-of-workspace file
  writes / network are OS-blocked by `--network none` / read-only rootfs / ephemeral PID
  namespace. **Under the default `subprocess`**, the training subprocess is at the
  policy level of protection, so the threats above remain (current behavior, no Docker
  needed). For a hardening guarantee, switch the contract to `container` + a
  digest-pinned image.
- The literature engine was mock-corpus-only — the real-API (OpenAlex/S2) adapters had
  only a `Retriever` shim and were unimplemented (Phase 6b, now done). The coder
  hypothesis's intervention-family classification is keyword-matching and conservative
  (give up on ambiguity → unexplored).
- **The pairwise judges are the same Claude family**, so correlated bias is not
  eliminated — a truly heterogeneous reviewer is closed by the Phase 5 cross-model codex
  reviewer (`--reviewer codex`). But codex review is an opt-in, advisory path dependent
  on a local `codex` login and account limits, so on failure it is merely recorded as
  review-absent and never blocks the report (being non-deterministic, it is not an input
  to any deterministic output).
- Successive halving pays off most at K=8 in the current scale, where branches are
  filled up to the number of parameters (heuristic one-per-param ≤6); at K=4 it mostly
  just saves dev-evaluation time.
- Real literature API (OpenAlex/S2) adapters (6b) and swapping in a real research domain
  (6c) are **now done** (see roadmap). gVisor/Firecracker backends are a one-line
  `--runtime=runsc` drop-in on Linux (follow-up). Containerizing the coder agent itself
  is also follow-up hardening.

## File structure

```
research_contract.yaml    # Layer 1 typed contract (v6: + sandbox block) — immutable, protected
orchestrator.py           # coordinator + gate + coder + literature shim + momentum/halving + approval gate + multi-seed report + sandbox wiring (protected)
sandbox/runner.py         # Phase 6a execution isolation (protected): Sandbox protocol / SubprocessSandbox / ContainerSandbox / preflight
assurance/                # Phase 5 pure package (protected): stats/claims/report_md/figures/svgfig/reviewer/gate/families
literature/engine.py      # literature engine: corpus/search/stance/novelty/move_guidance/LLM analyzer (protected)
literature/corpus/tsp_corpus.json  # 13 TSP-heuristics papers / 13 claims (protected, git-tracked)
src/train.py              # editable surface (HYPERPARAMS block + FEATURE_SPEC)
evaluation/evaluate.py    # protected evaluator → metrics.json (--split dev|gate|test, --seed-index, --sandbox-backend)
evaluation/dataset.py     # synthetic data (public train + dev/gate seeds + N separate test seeds)
evaluation/heldout_config.json  # created at init, untracked (schema v3: dev/gate seeds + N test seeds, absent from worktree)
protection/hashes.json    # SHA-256 manifest (22 files, git-tracked)
tests/test_phase2.py      # unit drills (guard hook / stagnation / blindness / feature_spec)
tests/test_phase3.py      # literature drills (determinism / blindness canary / laundering / injection / contract)
tests/test_phase4.py      # refinement drills (momentum / steering / halving / pruned / pairwise / contract)
tests/test_phase5.py      # assurance drills (bootstrap / claims / report digit-scan / gate / codex reviewer stub / families)
tests/test_phase6.py      # sandbox drills (subprocess regression / container argv·masking / fail-closed / provenance echo, no Docker)
experiments/              # runtime: state.json, ledger.jsonl, rounds/, generations/, evidence/,
                          #   claims.jsonl, report/(report.md · figures/ · review/) (gitignored)
insight_memory.json       # derived data reconstructable from the ledger (gitignored)
.worktrees/               # per-experiment isolation (gitignored)
```

Provenance convention: `main` accretes only gate-passing experiments via ff-merge.
Every other experiment is preserved on `hyp/<campaign>/rNNNN-*` branches +
`experiments/ledger.jsonl` — the gate decision is recorded separately as a
`record_type=gate` record (the score stays inside the ledger for blindness; the console
shows only PASS/FAIL).

## Modifying a protected file on purpose

When not running: `chmod u+w <file>` → edit → `uv run python orchestrator.py init
--force` (regenerates dev/gate/test seeds + manifest + re-baselines). `--force` empties
`experiments/`, so back up prior-campaign records first if you need them. Phase 6a moves
the contract schema to v6 (+ sandbox block) and heldout_config to v3 (N test seeds), so
you cannot continue from an older contract/state; start a new campaign with `init
--force`. The protection manifest covers 22 files (including assurance/ 9 + sandbox/ 2).

## Roadmap (against the blueprint)

Phase 1 (constrained keep/reject) + Phase 2 (portfolio · blind gate · LLM coder) +
Phase 3 (claim-level literature grounding · mock corpus) + Phase 4 (Gome search
momentum · evidence steering · successive halving · SciNav pairwise gate) + Phase 5
(assurance & reporting — multi-seed finalist reproduction + paired bootstrap CI,
claim-evidence ledger, deterministic report.md + SVG figures, cross-model codex
adversarial reviewer, human approval gate) + Phase 6a (execution-isolation sandbox —
`Sandbox` protocol, Docker `ContainerSandbox` isolating `train.py` execution at the OS
level: no network · read-only rootfs · seed/ledger masking · ephemeral PID namespace,
`subprocess` default / `container` opt-in, fail-closed) + Phase 6b (real literature API
— OpenAlex adapter in `literature/sources.py` + `ground --refresh` for fetch → LLM
extraction → frozen corpus snapshot; network only at refresh, campaigns stay
deterministic reading the frozen snapshot with lexical) + Phase 6c (domain swapped to
Euclidean-TSP heuristics — the evaluator generates instances from hidden seeds → passes
only coordinates to the solver → recomputes tour length) are all **done**.

### Refreshing the real literature corpus (`ground --refresh`) — networked-host runbook

The machinery is complete and offline-tested (handlers included,
`tests/test_phase6b.py`). **Only the real fetch** runs on a networked host. OpenAlex is
keyless (a `mailto` for the polite pool); the S2 key is **env-only** (`S2_API_KEY`, not
in the contract, not committed). The snapshot uses the network only at refresh;
campaigns read the frozen snapshot deterministically. Anti-laundering guard: an injected
abstract cannot produce `effect=improves` (the only support-granting stance) — if there
is no improvement cue or an injection marker is present, it is downgraded to
`conditional`.

Prerequisites (networked host): outbound HTTPS to
`api.openalex.org` / `api.semanticscholar.org`, and Claude SDK credentials if
`extractor: claude`.

```bash
export S2_API_KEY=...            # only for s2 fetch. env-only, not in contract / not committed.
chmod u+w literature/corpus/tsp_corpus.json
# options: --source openalex|s2|both|contract(default)  --extractor claude|deterministic
#          --max-papers N  --mailto you@example.com
uv run python orchestrator.py ground --refresh --source openalex --mailto you@example.com
git diff literature/corpus/tsp_corpus.json   # human review of the tag diff
# review focus: effect=improves (support-granting) claims, injection_flagged papers, dropped_claims_policy
# (optional) for a real-source campaign, set retriever in research_contract.yaml to openalex|s2 (after chmod).
#            leaving it lexical skips the provenance assertion — works with any snapshot.
uv run python orchestrator.py init --force   # re-hash the manifest + re-baseline (resets experiments/)
```

Limitations: the `claude` extractor is non-deterministic, so a real refresh produces a
new diff each time (only the offline path is reproducible); the `_urllib_get` seam and
real SDK call are finally verified only on a real network (outside offline test coverage
— only the handler's chmod gate, validate-before-overwrite, and REVIEW output are
offline-verified).

### Remaining follow-ups (honest limitations)

- `literature/corpus/tsp_corpus.json` is still a curated offline snapshot (provenance
  empty). Reflecting real literature is completed on a networked host per the runbook
  above: `ground --refresh` → human tag-diff review → `init --force`.
- evidence_steering (patcher) is now aligned: `orchestrator.rank()` / the explore filter
  convert raw params to families via `engine.PARAM_TO_FAMILY`, so literature steering
  actually works (verified by the real-corpus end-to-end drill in
  `tests/test_phase4.py`). One honest gap remains — the corpus tags a
  `neighborhood_operator` improvement as an `add_operator` move, but the editable surface
  (`segment_max`) only emits increase/decrease, so that claim matches no move (there is
  no editable surface corresponding to "add a move").
- gate/test run the solver on held-out instances, so trusted scoring is complete only on
  the `container` backend (subprocess can read the seed file by absolute path — dev/smoke
  only). Running gate/report under `subprocess` makes the orchestrator **always warn**,
  and `sandbox.require_container_for_trusted_splits: true` fail-closes it entirely. The
  report.md header also stamps the trust grade. See the `tests/test_phase6c.py` canary
  for the detailed threat model.

## Documentation

- **[docs/WORKBOOK.md](docs/WORKBOOK.md)** — the practical, hands-on usage guide (how to
  run it, read the results, adapt it to your own problem, and fix it when it breaks).
- **[docs/BLUEPRINT.md](docs/BLUEPRINT.md)** — the design rationale and layer spec (why
  it is built this way).
- **[docs/HANDOFF.md](docs/HANDOFF.md)** — a single-entry-point orientation to the
  internals.
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — how to set up, test, and contribute.
- **[SECURITY.md](SECURITY.md)** — the threat model and how to report a vulnerability.

Korean versions of each document are preserved alongside as `*.ko.md`.

## License

Licensed under the [Apache License 2.0](LICENSE).
