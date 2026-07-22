*The Korean original is preserved at [WORKBOOK.ko.md](WORKBOOK.ko.md).*

# AutoResearch Workbook — Practical Usage Guide (Everything in One Document)

The goal of this single document is to let you **understand, run, read the
results of, repurpose for your own problem, and fix when broken** the
AutoResearch framework. If the README and `docs/HANDOFF.md` are about "what was
built and why (design and invariants)," this document is about "so how do you
use it (practice and examples)."

Written against: contract schema v8 (`autoresearch-phase6c-tsp`, Euclidean-TSP
domain), with Phases 1 through 6c complete. Most of the console output and
numbers in the body are the results of **campaign `c20260718051332`, which was
actually just run in this repository** (§6).

---

## Table of Contents

- [0. 30-Second Summary and Mental Model](#0-30-second-summary-and-mental-model)
- [1. What AutoResearch Actually Does](#1-what-autoresearch-actually-does)
- [2. Installation and Prerequisites](#2-installation-and-prerequisites)
- [3. Five-Minute Quickstart (Copy-Paste)](#3-five-minute-quickstart-copy-paste)
- [4. Eight Concepts You Must Know](#4-eight-concepts-you-must-know)
- [5. Command Reference](#5-command-reference)
- [6. Worked Example — One Real TSP Campaign](#6-worked-example--one-real-tsp-campaign)
- [7. How to Read the Outputs](#7-how-to-read-the-outputs)
- [8. Advanced Recipes](#8-advanced-recipes)
- [9. Swapping in Your Own Research Domain](#9-swapping-in-your-own-research-domain)
- [10. Troubleshooting & FAQ](#10-troubleshooting--faq)
- [11. Health Check (Test Drills)](#11-health-check-test-drills)
- [12. Appendix — Reference Dictionary](#12-appendix--reference-dictionary)

---

## 0. 30-Second Summary and Mental Model

**One line:** AutoResearch runs an autonomous research loop — "form a hypothesis
→ change code/hyperparameters and run in an isolated environment → score with a
trustworthy evaluator → keep only what passes" — by standing up the **evaluator,
the contract, and provenance (evidence history) before the agent.**

**Core philosophy (one sentence):** *Without a trustworthy evaluator, no amount
of adding agents makes you any more scientific.* Every mechanism in this
repository exists to distinguish "plausible-looking failure" from "verified
progress."

The whole picture, in an analogy familiar to people doing AI/ML:

| ML world | AutoResearch |
|---|---|
| train / validation / test split | **dev / gate / test** three splits (generated from hidden seeds) |
| tuning hyperparameters on validation | keep/reject search on the **dev split** |
| test used exactly once, at paper-submission time | the **test split** is used exactly once, in `report` (requires human approval) |
| preventing validation overfitting | **blind gate** = re-confirming generalization on the hidden gate split |
| ablation study | even rejected experiments are all preserved as evidence in the ledger |
| NAS / population-based training | K hypotheses in parallel per **generation** → one winner adopted |
| successive halving (HPO) | cheap smoke rung → only the top ones go to the dev rung (§4) |
| preventing RLHF reward hacking | the evaluator ignores self-reported scores and **recomputes them directly** |

The whole pipeline flows in this order. The parenthesized text above each arrow
is "what newly comes into being at that step."

```
init ──(baseline·hidden seeds·protection manifest)──▶ ground ──(literature grounding certificate)──▶
run ──(N generations: hypothesis→run→gate→adopt)──▶ status ──(status dashboard)──▶
report ──(exit 3: demands human approval)──▶ approve ──(approve publication intent)──▶
report ──(multi-seed test evaluation + bootstrap CI + seal report.md)──▶ done
```

If this is your first time seeing this document, I recommend the following flow:
copy-paste §3 (Quickstart) as-is and run one campaign, then check what each line
means in §6 (Worked Example), and consult §5, §7, and §10 like a dictionary
when you need to.

### Key Terms First (If You Get Stuck Here, See Below)

Here I nail down in advance just the 7 words most likely to trip you up while
reading the body.

- **campaign** — a single "research project unit." One campaign is the entire
  experiment history from the moment you fix a baseline with `init` until you
  seal the results with `report`. It has a unique id (`c20260718051332` =
  creation time) and its own hidden seeds, baseline, and ledger. `init --force`
  discards the previous campaign and opens a new one. → Corresponds to a
  **sweep / run group** in ML (a bundle of experiments grouped under the same
  data split and the same baseline).
- **baseline** — the **starting-point performance** fixed at `init` time. It
  does not change for the entire campaign. Every improvement is measured
  relative to it. (Measured: dev tour 6,579,300.)
- **incumbent (current champion)** — the **best solution/commit** among
  everything adopted so far. New candidates are evaluated against the incumbent,
  and if a candidate wins it becomes the **new incumbent** and is merged into
  main. Initially incumbent == baseline. → The **best checkpoint / the current
  best in model selection** in ML. (Measured: final incumbent = the restarts=16
  commit `4796a8c`.)
- **candidate** — the run result of a hypothesis proposed within a generation.
  It challenges the incumbent. The relationship among these three is key:
  **baseline (fixed) ← incumbent (the champion that gets updated) ← candidate
  (the challenger)**.
- **restarts** — one of the TSP solver's hyperparameters. In iterated local
  search, the number of times to "perturb slightly when stuck in a local
  optimum and **restart** local search a few more times." The higher it is, the
  higher the probability of finding a better tour, but **compute grows
  linearly** (restarts=16 → roughly 16× the compute). → The same as **random
  restart best-of-N** in ML (in non-convex optimization, running from several
  initial points multiple times and picking the best).
- **seal** — when `report` finalizes the results using the untouched **test
  split**. The numbers, statistics, claims, and figures at this moment are
  **locked immutably** as "this is the official result" via a `final_report`
  record + `claims.jsonl` + a sha256 hash (seal is the literal term in the
  code). Just like using a test set once for a paper, redoing it requires
  re-approval via `--force`, and that reuse count is counted in the
  **multiple-testing disclosure**.
- **dev / gate / test** — three splits made from hidden seeds. dev = exploration
  (keep/reject), gate = generalization re-confirmation (blind), test = single-use
  for the final report. → Exactly **train / val / test**.

---

## 1. What AutoResearch Actually Does

### 1.1 The Problem This Repository Is Currently Solving (Euclidean-TSP)

The current domain is the **Euclidean TSP (Traveling Salesman Problem)**. It
matters that this is a combinatorial optimization problem, not an ML problem like
regression or classification.

- The evaluator generates **city-coordinate instances** (60 cities, integer grid
  coordinates) from hidden seeds.
- It hands the solver (`src/train.py`) **only the coordinates**, and the solver
  returns a **tour (a permutation of city visits)** for each instance.
- The evaluator verifies that the permutation is valid and **recomputes the tour
  length directly**. Even if the solver writes its own score into
  `reported_objectives`, it is **completely ignored** (score forgery is
  impossible). Distances use TSPLIB EUC_2D integer rounding, so they are
  deterministic and byte-stable.
- Goal: **minimize** `mean_tour_length` (the mean tour length).

The editable surface (`src/train.py`) has two kinds of knobs:

```python
# --- HYPERPARAMS-BEGIN (auto-patched; do not edit by hand) ---
HYPERPARAMS = {
    "use_nn_construction": True,   # whether to use a nearest-neighbor initial solution
    "max_iterations": 20000,       # number of local search iterations
    "restarts": 1,                 # number of iterated local search restarts
    "initial_temperature": 0.0,    # simulated annealing initial temperature (0 = pure hill climbing)
    "cooling_rate": 0.995,         # temperature decay rate
    "segment_max": 3,              # or-opt maximum segment length
    "perturbation_strength": 4,    # kick (perturbation) strength before a restart
}
# --- HYPERPARAMS-END ---

NEIGHBORHOOD = "two_opt"   # or "or_opt" — code-level knob (only the LLM coder can change it)
```

- **Deterministic patcher**: changes **exactly one value** in the `HYPERPARAMS`
  block above (e.g., `restarts: 1 → 2`) — fully offline, deterministic, and free.
- **LLM coding worker (coder)**: can edit anywhere under `src/**`. The "real
  research" surface that touches the **algorithm itself** — for example changing
  `NEIGHBORHOOD` from `two_opt → or_opt`, changing the acceptance rule, or adding
  tabu memory. It appears only with `--proposer claude`.

### 1.2 evaluator-first: Why the Evaluator Comes First

Just as a bad reward model breaks RLHF via reward hacking in ML, here too, if
the evaluator is compromised, everything else becomes meaningless. So this
repository isolates the evaluator as a **trusted path** and structurally
guarantees the following:

- **Only the root copy of the evaluator has authority.** The `evaluation/` copy
  inside a worktree (the isolated working tree used for experiments) is never
  used for scoring.
- **Physical absence of the hidden seeds**: the dev/gate/test seeds live in
  `evaluation/heldout_config.json`, and this file is **not tracked by git**.
  Since a worktree checks out only tracked files, the candidate workspace has no
  seeds at all. (The physical version of preventing train/val/test leakage.)
- **nonce echo**: a fresh nonce is passed to every evaluation, and the evaluator
  must echo it back verbatim into the metrics → blocks forged metrics.
- **Ignore self-reports**: the solver's `reported_objectives` is discarded and
  the evaluator recomputes.
- **SHA-256 manifest**: hash-checks the protected files (`orchestrator.py`,
  `evaluation/**`, `research_contract.yaml`, `literature/**`, `assurance/**`,
  `sandbox/**`, and so on — 23 files) every round.

### 1.3 What Is Editable and What Is Protected

| Path | Status | Description |
|---|---|---|
| `src/**` | ✏️ editable | solver code (patcher/coder touch only here) |
| `research_contract.yaml` | 🔒 protected (0o444) | the research contract. Immutable during a run |
| `orchestrator.py` | 🔒 protected | coordinator, gates, CLI |
| `evaluation/**` | 🔒 protected | evaluator and datasets (`heldout_config.json` is git-untracked) |
| `literature/**` | 🔒 protected | literature engine + corpus (evidence provenance) |
| `assurance/**` | 🔒 protected | statistics, claims, and report generation |
| `sandbox/**` | 🔒 protected | the execution isolation boundary |
| `protection/hashes.json` | 🔒 protected (git-tracked) | the SHA-256 manifest |
| `experiments/**` | runtime (gitignore) | state, ledger, rounds, reports |
| `.worktrees/**` | runtime (gitignore) | per-experiment isolated working trees |

> **"To intentionally modify a protected file"** → you must follow the §9.3
> procedure (`chmod u+w` → edit → `init --force`). If you just edit it, the next
> `run`/`report` will halt with a protection violation.

---

## 2. Installation and Prerequisites

Prerequisites: `uv` (the Python package and virtual-environment manager) and a
Python 3.13+ toolchain. (`.python-version` is pinned to 3.14.)

```bash
cd /Users/gyubin.son/workspace/dev/autoresearch
uv sync    # install dependencies: just two — pyyaml + claude-agent-sdk
```

- **Offline by default**: the heuristic proposer + lexical literature + scalar
  gate **use no SDK or network at all.** They run immediately without any
  account login.
- **LLM path** (opt-in): `--proposer claude`, `--literature claude`,
  `--gate pairwise`, and `--reviewer codex` reuse your local Claude Code login
  (no separate API key needed) or your local `codex` login. You may hit account
  usage limits, and if you do, they fall back to the deterministic path.
- **container sandbox** (opt-in): requires a Docker daemon + a pre-pulled pinned
  image (§8.5).

---

## 3. Five-Minute Quickstart (Copy-Paste)

If you know nothing and just want to run one campaign, copy-paste this as-is.
It's all offline and deterministic, so it's safe.

```bash
cd /Users/gyubin.son/workspace/dev/autoresearch
uv sync

# 1) initialize: git, baseline, hidden seeds, protection manifest (skip if already done)
uv run python orchestrator.py init

# 2) (optional) literature grounding certificate — what prior work the research question stands on
uv run python orchestrator.py ground

# 3) run the campaign: 3-generation parallel portfolio (heuristic, offline)
uv run python orchestrator.py run --generations 3

# 4) check status
uv run python orchestrator.py status

# 5) report → the first time, it demands human approval and emits exit 3 + a request_id
uv run python orchestrator.py report        # exit 3, request_id is printed

# 6) approve the publication intent with that request_id
uv run python orchestrator.py approve <request_id_from_above>

# 7) report again → this time: multi-seed test-split evaluation + bootstrap CI + seal report.md
uv run python orchestrator.py report

# view the results
open experiments/report/report.md   # or: cat experiments/report/report.md
```

> **exit 3 is not an error.** It's a signal that "human approval is required." In
> scripts, handle it by distinguishing `0 = seal complete / 1 = error / 2 =
> argument error / 3 = approval required`.

If you want to run the full LLM path (opt-in):

```bash
uv run python orchestrator.py run --generations 3 \
  --proposer claude --literature claude --gate pairwise
```

---

## 4. Eight Concepts You Must Know

To use the framework properly, you just need to firmly grasp these eight.

### 4.1 generation vs round (experiment)

- **round (= experiment)**: the smallest unit of running one hypothesis.
  Numbered as `rNNNN` (r0001, r0002…), and the repository budget is counted by
  `budgets.max_rounds` (default 60).
- **generation**: the unit that puts out **K hypotheses in parallel (contract
  `portfolio.parallel_branches`, default 8)** in one round and picks a single
  winner. The N in `run --generations N` is the number of **generations**. The
  heuristic proposer fills at most one hypothesis per parameter (about 6–7), so
  K=8 is a ceiling, not a quota.

> Analogy: a generation = one generation of population-based training (evaluate
> several candidates simultaneously, then select), a round = one individual
> within that generation.

### 4.2 verdict ≠ decision (the most confusing point)

- **verdict** is the "scientific judgment": what happened on the dev split.
- **decision** is "whether it was adopted": did it ultimately get merged into
  main (`accept`/`reject`).

**Even with `verdict=valid_positive` (improved on dev), the `decision` can be
`REJECT`** — because it looked good on dev but failed to clear the blind gate, or
another candidate in the same generation won. In a single generation, **only one
gate winner** is `accept`ed.

Full list of verdicts:

| verdict | Meaning | Scientific signal? |
|---|---|---|
| `valid_positive` | dev relative improvement ≥ `min_relative_improvement` (0.2%) | ✅ (gate candidate) |
| `valid_inconclusive` | change below threshold | ✅ (weak signal) |
| `valid_negative` | metric regression / divergence / no_skill / infeasible / timeout / crash | ✅ (disconfirming evidence) |
| `pruned` | successive halving cut (a budget decision) | ❌ (not science) |
| `invalid_implementation` | patch failure / coder mechanical failure / nondeterminism | ❌ (mechanical failure) |
| `contract_violation` | touched a protected path / src symlink / oversized diff | ❌ (protocol violation) |
| `ff_conflict` | it's the winner but main moved/got dirty just before the merge | ❌ (infrastructure) |
| `aborted` | evaluator infrastructure crash / interruption | ❌ (infrastructure) |

> **No-false-repair principle**: any runtime failure that occurs after a valid
> patch (divergence, timeout, no_skill, infeasible, dev-stage failure) is **not
> repaired** and is instead recorded as evidence as `valid_negative`. This
> structurally prevents manufacturing fake progress by "fixing it to look
> plausible." Repair is allowed only for **mechanical failures at the smoke
> stage** (nonzero_exit / missing_artifact / malformed_artifact), and only when a
> scorable answer has not yet been produced.

### 4.3 blind admission gate and blindness (generalization re-confirmation)

In a generation, the top `gate_top_k` (default 2) dev-improvement candidates
(`valid_positive`) are re-scored on the **gate split** (hidden seeds completely
different from dev). A candidate must beat the incumbent's (current champion's)
gate score by `gate_min_relative_improvement` (default 0.1%) to be **admitted**,
and among the admitted candidates only one winner is ff-merged into main.

> Why? It filters out candidates that improved marginally on dev but do not
> generalize (= validation overfitting). Think of dev = validation, gate =
> another held-out validation.

**blindness invariant (very important):** the gate score exists in only two
places — the `record_type=gate` ledger record and
`experiments/generations/gNNNN/gate/*.json`. The gate score **never enters
insight, `best_primary` (always the dev score), the proposer context, search
momentum, console output, or the report/claims.** The console prints only
PASS/FAIL and the winner's run_id, and hides the score as "scores withheld."
(Measured example: in g0003 in §6, the dev winner's score is 6351315.225, but the
gate score 6320049.425 exists only in the gate record.)

### 4.4 search momentum (Gome-style directional steering)

At the start of each generation, it folds only the experiment records in the
ledger to produce a directional score per `{param}:{move}` (e.g.,
`restarts:increase`). Because it is **recomputed from the ledger every time
rather than stored in state**, replay == live holds trivially even after
recovering from a crash.

Weights (the signal for whether a direction is working):

| Result | Weight |
|---|---|
| `valid_positive` + accept | **+1.0** |
| `valid_positive` (dropped at gate) | +0.4 |
| `valid_negative` (regression/divergence/timeout) | −1.0 (+ records a divergence boundary value) |
| `valid_inconclusive` | −0.2 |
| `pruned` / invalid / contract_violation | 0.0 |

It decays by `momentum_decay` (0.5) at each generation boundary. The heuristic
proposer re-ranks candidates with this momentum as the first priority, literature
stance as the second, and static priority as the third. When consecutive accepts
on a single (param, direction) accumulate to `accelerate_after` (2) times, it
uses an **acceleration step (a squared step)** once. In the §6 example,
`restarts` jumping 1→2→4→**16** is exactly this acceleration (4²). Also, at least
one of the K slots is forcibly reserved for a **momentum-0, literature-unsupported
direction** to prevent hypothesis collapse (digging in only one direction).

> Since the inputs are only the dev signal and the accept/reject bit, **the gate
> score structurally cannot enter** (blindness is preserved). If
> `refinement.enabled=false`, this steering turns off and the behavior becomes
> byte-identical to Phase 3.

### 4.5 successive halving (saving budget)

All K branches of a generation run the cheap **smoke rung** (short training, dev
split, 30 seconds), and only the top `max(min_keep, ceil(K·keep_fraction))` =
`max(2, ceil(8·0.5))=4` by smoke score advance to the expensive **dev rung**
(full training, 120 seconds). Those eliminated get verdict `pruned`.

> This is exactly successive halving from HPO. **Important:** pruned is **a
> budget decision, not science** — it distills no insight and its momentum weight
> is 0. Do not read "pruned = this hypothesis failed." That said, the tested
> endpoint is registered to prevent re-proposal.

### 4.6 literature grounding (a claim-level evidence graph)

It makes hypotheses stand on literature grounding. `literature/` is a
**separately governed service**: it does not import the orchestrator, reads no
file other than the corpus, and writes nothing at runtime (hermetic).

- **Offline corpus** (`literature/corpus/tsp_corpus.json`, `tsp-heuristics-v1`):
  13 TSP heuristics papers / 13 claims. Includes contradiction pairs,
  citation-laundering traps, and prompt-injection fixtures. (Right now it's a
  curated mock snapshot — provenance is empty. To update it with real
  literature, use `ground --refresh` in §8.6.)
- **Default lexical**: deterministic token-overlap search + citation BFS (1 hop)
  + coverage-based stopping. Fully offline.
- **`--literature claude`**: the LLM handles only query decomposition, stance
  judgments, and narration, but **search execution is always the deterministic
  backend**. Anti-laundering rule: **the LLM can only downgrade a deterministic
  "supports" and cannot newly grant one.**
- A hypothesis carries **only evidence ids** (`supporting_evidence_ids`,
  whitelist-validated). Claim prose never enters the blindness-scan surface.
  Novelty is category-only, with no numbers: `replication / regime_extension /
  contradiction_test / unexplored`.

### 4.7 human approval gate (the publication analog)

The untouched **test split** is like using a test set once when submitting a
paper. So the first `report` (and every `--force` re-run) demands human approval
**before computing any test numbers**:

1. `report` → writes an approval request to the ledger and prints **exit 3 +
   request_id**.
2. A human reviews the intent (commit, dev numbers, seed plan, disclosure).
3. `approve <request_id>` → records the approval.
4. `report` again → proceed.

Approval status is **derived from the ledger**, not from state. Since the
fingerprint (= incumbent + baseline commit + contract + evaluator sha + number of
prior seals) changes as the campaign advances, running more `run`s after approval
makes the **approval stale (invalid)** and requires re-approval.

> Internal operations like coder execution or raising the budget deliberately do
> not go through the approval gate — this is to prevent trivial approval fatigue
> from accumulating and causing you to rubber-stamp the genuinely important
> "irreversible decisions."

### 4.8 sandbox backend and trust grade

Only the point where candidate code is actually *executed* (`_run_train` in
`evaluation/evaluate.py`) is OS-isolated. Scoring stays on the host trusted path,
and the seeds do not pass into it.

- **`subprocess` (default)**: no OS isolation. Byte-identical to current
  behavior, no Docker needed. **Not trust-grade on gate/test** — the solver could
  read the hidden seed file by absolute path and overfit. That's why a large
  warning appears at every gate/report.
- **`container` (opt-in)**: via `docker run` — network blocked, read-only rootfs,
  dropped capabilities, non-root, resource limits, ephemeral PID namespace +
  **masks the hidden seed file and the ledger** (physically absent in the
  container FS). Only then are the gate/test scores **trust-grade**.

Policy: setting `sandbox.require_container_for_trusted_splits: true` blocks
gate/report under subprocess with a hard error (the default is warning only). The
trust grade is also stamped in the report.md header. For detailed setup, see §8.5.

> Honest limitation: **numbers produced under the default subprocess "look
> honest but are not trust-grade."** To assert results to others, reproduce them
> under container.

---

## 5. Command Reference

Common rules:
- Invocation form: `uv run python orchestrator.py <subcommand> [flags]`.
- `init`/`run`/`report`/`ground`/`approve` take a **single-instance lock**
  (flock, `.orchestrator.lock`) — if you run two at once, the second fails with
  `error: another orchestrator process is running ...`. `status`/
  `verify-protection` are safe anytime without a lock.
- exit codes: `0` = success, `1` = OrchestratorError (an error, `error: ...` on
  stderr), `2` = argument error, `3` = report approval required.

### 5.1 `init` — initialization and baseline

```bash
uv run python orchestrator.py init [--force]
```

What it does: prepares the git repository (if absent, `git init`, gc.auto=0) →
**cross-validates** the hardcoded constants of the contract and the evaluator
(budget/metric/split/seed caps/N_CITIES/sandbox backend — if they mismatch, halts
immediately with `... drift`) → generates the hidden dev/gate/test seeds
(`heldout_config.json`, schema v4, N=`finalist_seeds` test seeds) → writes the
protection manifest → initial commit → **baseline dev evaluation** → makes
protected files read-only (0o444).

`--force`: **deletes `experiments/` entirely** and regenerates seeds and manifest
+ re-baselines. Back up first if you need the previous campaign's records. After
changing the contract/evaluator/domain, you must start a new campaign with this.

Measured output (last two lines):
```
[init] baseline mean_tour_length (dev) = 6579300.425000 at f31e2eadfaae
[init] protected files set read-only; ready: `uv run python orchestrator.py run --generations N`
```

Note: if you `init` without `--force` when already initialized,
`already initialized (experiments/state.json exists); use --force to re-baseline`.

### 5.2 `ground` — literature grounding certificate (+ `--refresh` maintenance)

```bash
uv run python orchestrator.py ground [--literature lexical|claude] [--model NAME]
```

What it does: runs the literature evidence flow over the contract's `objective`
to produce a **research-question certificate**. Writes it to
`experiments/evidence/question_certificate.json`, appends a
`kind=question_grounding` bundle to evidence.jsonl, and prints the certificate
JSON to stdout.

`--refresh` is a completely different **maintenance** operation (regenerating the
corpus via a real API) — see §8.6.

### 5.3 `run` — run the campaign

```bash
uv run python orchestrator.py run \
  [--generations N] \                 # default 3
  [--proposer heuristic|claude] \     # default heuristic (claude also enables the coder)
  [--model NAME] \                    # override the claude proposer/coder model
  [--max-budget-usd X] \              # cap per claude proposer proposal, default 0.5
  [--gate scalar|pairwise] \          # default scalar
  [--literature lexical|claude]       # default lexical
```

What it does: runs `--generations` generations. Each generation: protection
verification → momentum + literature grounding → propose K hypotheses → parallel
execution in isolated worktrees (smoke rung → halving cut → dev rung) → blind
gate admission → (if pairwise) judge selection → ff-merge the winner → distill
insight.

**Important entry conditions:**
- If not initialized, `not initialized — run orchestrator.py init first`.
- If the main working tree has **uncommitted changes to tracked files**, it
  refuses (`... has uncommitted tracked changes ...; commit or restore them
  first`). Since a generation branches from HEAD, a dirty tree would desync the
  proposal from the actual run.

Stop conditions: `max_rounds` (reaching 60) / `max_generations` / `stagnation`
(4 consecutive winnerless generations) / `search_space_exhausted` (no proposals).

Principle behind the flags — **the contract decides "whether," the CLI decides
"how."** If `pairwise_gate.enabled=false` in the contract, `--gate pairwise` is
silently ignored with only a warning. `--literature claude` is likewise ignored
if `literature.enabled=false`.

Measured output (full in §6):
```
— generation g0001 —
  [r0001] restarts: 1 -> 2  mean_tour_length=6508432.5000  verdict=valid_positive  decision=ACCEPT
  ...
  [gate] candidates ['r0001'] -> winner r0001
...
generations executed: 3 (total 3; experiments 15); stop: requested generations done
mean_tour_length (dev): baseline 6579300.425000 -> best 6351315.225000 (+3.47% relative)
incumbent commit: 4796a8c70e16  stagnation: 0 generations
```

### 5.4 `status` — status dashboard (read-only)

```bash
uv run python orchestrator.py status
```

What it does: prints the contract, metric, experiment/generation counts,
baseline/best (dev), stagnation, the last 12 experiments, the last 5 gate
decisions (**scores hidden**), search momentum (ledger-derived), literature
statistics, and approval/review status. Takes no lock — safe even while `run` is
in progress.

### 5.5 `report` — final report (single-use test split)

```bash
uv run python orchestrator.py report [--force] [--reviewer none|codex] [--model NAME]
```

What it does (if approved): sandbox preflight → trust-policy warning/block →
write-ahead `report_attempt` record → evaluates baseline and incumbent on
**N=`finalist_seeds` (5) test seeds** → paired bootstrap CI → evidence audit (a
hard error on an uninterpretable citation) → writes `claims.jsonl` +
`figures/*.svg` + `report.md` + `report.json` → (opt-in) codex review → seals
`final_report` → prints the full report JSON to stdout.

- If not approved, **exit 3** (§4.7).
- If a sealed report already exists and there's no `--force`:
  `N final report(s) already exist — the test split is single-use. ...`. `--force`
  means a new intent (re-approval required) + increments the multiple-testing
  disclosure counter.
- `--reviewer codex` actually runs only if `reviewer.enabled: true` is also set.
  Even if it fails, it does not block the report and is recorded as
  `status="unavailable"` (no fallback to Claude).

### 5.6 `approve` — approve/deny the publication intent

```bash
uv run python orchestrator.py approve <request_id> [--deny] [--reason "text"]
```

What it does: appends an approve/deny decision to the ledger. You can give just a
**unique prefix** for the `request_id` (e.g., `approve 23ee75` is OK). Deny with
`--deny`; later `approve`-ing the same id can reverse the denial. Fingerprint
freshness is not checked here — it is checked at `report` time (which is how
stale approvals get filtered out).

### 5.7 `verify-protection` — integrity check

```bash
uv run python orchestrator.py verify-protection
```

`OK — 23 protected files match the manifest` (exit 0) or, per violation,
`VIOLATION: <file>` (exit 1). No lock.

---

## 6. Worked Example — One Real TSP Campaign

Below is the full process of campaign `c20260718051332`, which was **actually
just run** in this repository. All numbers and output are measured.

### Step 0 — Confirm the starting point

```bash
$ uv run python orchestrator.py status
contract:   autoresearch-phase6c-tsp
objective:  Minimize the mean held-out Euclidean-TSP tour length produced by the solver in src/train.py across h
metric:     mean_tour_length (minimize, min rel improvement 0.2%; gate epsilon 0.10%)
experiments: 0 / 60  generations: 0
baseline:   6579300.425000 (dev)
best:       6579300.425000 (dev) at f31e2eadfaae
stagnation: 0 / 4 generations

approval (report intent 3285689c7b88): none

$ uv run python orchestrator.py verify-protection
OK — 23 protected files match the manifest
```

How to read it: 0 experiments so far, incumbent == baseline (6,579,300). The
metric is minimize, the dev improvement threshold is 0.2%, and the gate threshold
is 0.10%.

### Step 1 — Literature grounding

```bash
$ uv run python orchestrator.py ground
```

The output (excerpt) is the research-question certificate. Just the essentials:
```json
{
  "mode": "lexical",
  "evidence_counts_by_stance": {"supports": 7, "adjacent": 4, "contradicts": 1},
  "contradictions": [{"topic": "acceptance_criterion",
                      "evidence_ids": ["ev_cl-0501", "ev_cl-0601"]}],
  "coverage": {"stopped_because": "coverage_stable", "queries_run": 3,
               "topics_uncovered": ["initial_temperature"]}
}
```

How to read it: it found support 7 / contradict 1 / adjacent 4 pieces of evidence
in the corpus, and detected a **contradiction pair** on the `acceptance_criterion`
topic (one paper claims improvement, another claims regression). Coverage
stabilized and it stopped after 3 queries.

### Step 2 — Run the 3-generation campaign

```bash
$ uv run python orchestrator.py run --generations 3
```

First, the trust warning appears 3 times on stderr (gate split, subprocess
backend):
```
[warn] gate split runs under the 'subprocess' sandbox backend, which has NO
filesystem isolation: a candidate solver can read the held-out seed file by
absolute path and overfit the hidden instances, so its gate score is not
trust-grade. Set sandbox.backend: container for a trustworthy gate result.
```

Then the per-generation progress:

```
— generation g0001 —
  [r0001] restarts: 1 -> 2               mean_tour_length=6508432.5000        verdict=valid_positive       decision=ACCEPT
  [r0002] use_nn_construction: True -> False  mean_tour_length=smoke 18318703.1000  verdict=pruned (smoke_rank_below_cutoff)  decision=REJECT
  [r0003] max_iterations: 20000 -> 50000  mean_tour_length=6573762.8750        verdict=valid_inconclusive   decision=REJECT
  [r0004] initial_temperature: 0.0 -> 0.5  mean_tour_length=smoke 7371016.5000   verdict=pruned (smoke_rank_below_cutoff)  decision=REJECT
  [r0005] cooling_rate: 0.995 -> 0.99     mean_tour_length=6579300.4250        verdict=valid_inconclusive   decision=REJECT
  [r0006] segment_max: 3 -> 4             mean_tour_length=6579300.4250        verdict=valid_inconclusive   decision=REJECT
  [r0007] perturbation_strength: 4 -> 8   mean_tour_length=smoke 7324408.0250   verdict=pruned (smoke_rank_below_cutoff)  decision=REJECT
  [gate] candidates ['r0001'] -> winner r0001
— generation g0002 —
  [r0008] restarts: 2 -> 4                mean_tour_length=6455142.9000        verdict=valid_positive       decision=ACCEPT
  [r0009] max_iterations: 20000 -> 8000   mean_tour_length=6575426.8500        verdict=valid_negative (metric_regression)  decision=REJECT
  [r0010] initial_temperature: 0.0 -> 1.0  mean_tour_length=smoke 7371016.5000   verdict=pruned (smoke_rank_below_cutoff)  decision=REJECT
  [r0011] cooling_rate: 0.995 -> 0.999    mean_tour_length=6508432.5000        verdict=valid_inconclusive   decision=REJECT
  [r0012] segment_max: 3 -> 2             mean_tour_length=smoke 7324408.0250   verdict=pruned (smoke_rank_below_cutoff)  decision=REJECT
  [r0013] perturbation_strength: 4 -> 2   mean_tour_length=smoke 7324408.0250   verdict=pruned (smoke_rank_below_cutoff)  decision=REJECT
  [gate] candidates ['r0008'] -> winner r0008
— generation g0003 —
  [r0014] restarts: 4 -> 16               mean_tour_length=6351315.2250        verdict=valid_positive       decision=ACCEPT
  [r0015] initial_temperature: 0.0 -> 2.0  mean_tour_length=6462573.6250        verdict=valid_inconclusive   decision=REJECT
  [gate] candidates ['r0014'] -> winner r0014

generations executed: 3 (total 3; experiments 15); stop: requested generations done
mean_tour_length (dev): baseline 6579300.425000 -> best 6351315.225000 (+3.47% relative)
incumbent commit: 4796a8c70e16  stagnation: 0 generations
```

**The science story that actually unfolded in this one campaign:**

1. **Halving cheaply cut half of them.** In g0001, `use_nn_construction=False`,
   `initial_temperature=0.5`, and `perturbation_strength=8` had poor smoke-rung
   scores, so they couldn't advance to the dev rung and were `pruned` (shown as
   `mean_tour_length=smoke ...`). This means "cut here for budget reasons," not
   "this hypothesis is wrong."

2. **restarts was the only thing that worked consistently.** In g0001, only
   `restarts 1→2` improved dev meaningfully (6,579,300 → 6,508,432) and passed
   the gate. The others were inconclusive.

3. **momentum pushed the direction.** In g0002, `restarts 2→4` won again, and in
   g0003 momentum pushed `restarts:increase` strongly and used an **acceleration
   step** to jump to `restarts 4→16` (=4²), winning again (see the momentum table
   below the list).

4. **valid_negative = genuine disconfirmation.** In g0002,
   `max_iterations 20000→8000` worsened the metric (`metric_regression`) and
   became valid_negative. This remains as evidence that "reducing iterations
   makes it worse," and is reflected in momentum as
   `max_iterations:decrease: -0.50`.

5. **Honest interpretation:** the improvement this campaign found is essentially
   "doing more restarts (= iterated local search) shortens the tour." That's
   actually true, but the price is **16× the compute**. It's a legitimate win
   under the current goal of minimizing only `mean_tour_length`, but if you want
   to see "quality per compute budget," you have to put something like
   `solve_seconds` into the objective function (the lesson of §9). The framework
   does not hide this trade-off; it surfaces it in report.md's Limitations and as
   a secondary metric.

If you check the momentum from point 3 above via `status`, it prints like this —
`restarts:increase` is at the top with +1.75, so you can see that consecutive
accepts made it confident in this direction:

```text
search momentum (ledger-derived, dev signals only):
  restarts:increase: score +1.75  last=accepted
  max_iterations:decrease: score -0.50  last=valid_negative
  initial_temperature:increase: score -0.20  last=valid_inconclusive
  ...
```

### Step 3 — Report attempt (human approval gate)

```bash
$ uv run python orchestrator.py report
approval required before the test split is touched.
  request_id : 23ee75978e39
  intent     : incumbent 4796a8c70e16 vs baseline f31e2eadfaae, 5 test seed(s)
  review it, then: uv run python orchestrator.py approve 23ee75978e39
$ echo $?
3
```

exit 3. No test number has been computed yet. Only an `approval_request` was
written to the ledger.

### Step 4 — Approve

```bash
$ uv run python orchestrator.py approve 23ee75978e39
approve recorded for request_id 23ee75978e39
```

### Step 5 — Seal the report

```bash
$ uv run python orchestrator.py report
[warn] test split runs under the 'subprocess' sandbox backend ... not trust-grade ...
{ ... full final_report JSON ... }
```

This step reproduces baseline and incumbent on each of 5 test seeds (160
instances each), so it takes a while (hundreds of seconds). When done, the
artifacts pile up in `experiments/report/`.

### Step 6 — Read the results (`report.md`)

Measured excerpt of `experiments/report/report.md`:

```markdown
# AutoResearch Campaign Report — autoresearch-phase6c-tsp

Campaign c20260718051332 · baseline f31e2eadfaae · incumbent 4796a8c70e16 · 2026-07-18

Execution isolation: subprocess — NOT trust-grade: no filesystem isolation, so the
solver could read the held-out seed. Re-run under sandbox.backend: container to
trust these numbers.

## Headline result
**Status: verified.** Pooled mean test tour length 6554278.24 → 6303584.31; effect
250693.93 (3.82% relative), 95% CI [237076.58, 264382.16] across 5 hidden test
seed(s) × 800 instances. [claim_0001]

## Per-seed reproduction
| seed | baseline tour length | incumbent tour length | delta |
|------|----------------------|-----------------------|-------|
| s0 | 6585181.16 | 6313810.64 | 271370.52 |
| s1 | 6545216.19 | 6302670.83 | 242545.36 |
| s2 | 6571730.65 | 6331909.84 | 239820.81 |
| s3 | 6536335.89 | 6282253.64 | 254082.25 |
| s4 | 6532927.31 | 6287276.59 | 245650.73 |
Seed consistency (fraction improving): 100.00%.

## Admitted interventions
- restarts 1 -> 2 improved dev ... generation 1 and passed the blind admission gate. [claim_0003]
- restarts 2 -> 4 improved dev ... generation 2 and passed the blind admission gate. [claim_0004]
- restarts 4 -> 16 improved dev ... generation 3 and passed the blind admission gate. [claim_0005]

## Negative results
- Intervening on max_iterations did not help: 1 valid negative result(s) (failure
  classes: metric_regression). ... [claim_0006]

## Literature grounding
- Accepted change r0001/r0008/r0014 is grounded in 1 evidence record(s) from prior
  work (tsp:2018.0009). [claim_0007..0009]
```

**The crux of interpretation — what `Status: verified` means:**
- On 5 **completely untouched test seeds**, the mean tour versus baseline dropped
  from 6,554,278 → 6,303,584 (effect 250,694, 3.82%).
- The 95% bootstrap CI is **[237076, 264382], which does not include 0** → the
  improvement is statistically certain → `verified`. (If the CI straddles 0,
  `inconclusive`; if the upper bound < 0, `refuted`; if even one test seed is
  unclean, `unsupported`.)
- **seed consistency 100%** — all 5 seeds improved. Not a coincidence.
- Crucially, **the +3.47% on dev was reproduced as +3.82% on the untouched
  test.** Thanks to the blind gate filtering out dev-overfitting, this is
  empirical proof that the dev improvement genuinely generalized.

> In report.md, **every number is inserted only from a claim**, and after
> rendering a digit-scan hard-rejects any untraceable number (the
> number-via-claim invariant). That is, it structurally guarantees that "any
> number in this document can be traced to some claim in
> `experiments/claims.jsonl`."

---

## 7. How to Read the Outputs

### 7.1 Map of the `experiments/` tree (after one campaign)

```
experiments/
├── state.json                     # resumable campaign pointer (current state)
├── ledger.jsonl                   # ★ source of truth (append-only, 9 record_types)
├── baseline/metrics_dev.json      # init baseline evaluation
├── rounds/rNNNN/
│   ├── hypothesis.json            # hypothesis certificate (recorded before the ledger write)
│   ├── metrics_smoke.json         # smoke rung evaluation
│   └── metrics_dev.json           # dev rung evaluation
├── generations/gNNNN/
│   ├── steering.json              # momentum, move_guidance, batch (pure audit)
│   ├── evidence.json              # generation literature-grounding snapshot
│   └── gate/                      # ★ the only place gate scores exist
│       ├── incumbent.json
│       ├── <run_id>.json          # candidate gate-split score
│       └── <run_id>_dev_recheck.json  # determinism recheck
├── evidence/
│   ├── evidence.jsonl             # evidence memory (separate from the ledger, append-only)
│   └── question_certificate.json  # `ground` certificate
├── claims.jsonl                   # fully regenerated at report time (5 claim types)
└── report/
    ├── report.md                  # ★ the human-readable conclusion
    ├── report.json                # copy of the final_report record (machine-readable)
    ├── {baseline,incumbent}_test_sK.json  # per-seed test evaluation
    ├── figures/*.svg              # dev_trajectory / test_paired_rmse / verdict_mix
    └── review/<request_id>/       # codex review (only when opted in)
```

Alongside the root: `insight_memory.json` (reconstructible from the ledger),
`artifacts/solution.json` (a temporary artifact overwritten every evaluation —
**not a per-round archive**), `.worktrees/`, `evaluation/heldout_config.json`
(hidden seeds, git-untracked), `protection/hashes.json` (the manifest,
git-tracked).

> **All of this is gitignored.** Provenance survives only via the `hyp/*` git
> branches + `experiments/ledger.jsonl`. `init --force` deletes `experiments/`,
> so back up first if needed.

### 7.2 Ledger record_type dictionary

`experiments/ledger.jsonl` is append-only JSONL. What this campaign produced:
`baseline`×1, `experiment`×15, `gate`×3, `approval_request`×1,
`approval_decision`×1, `report_attempt`×1 (and after sealing, `final_report`×1).

| record_type | When | Key fields |
|---|---|---|
| `baseline` | once at init | commit, primary, metrics_path |
| `experiment` | per candidate | hypothesis (certificate), verdict, decision, primary, smoke_primary, best_primary_before/after, executor, failure_class, coder_family |
| `gate` | per generation (recorded first) | candidates, results{run_id:score}, incumbent_gate, winner, scalar_winner, mode, pairwise, reason, selection_rule |
| `correction` | when a merge fails after adoption | corrects (the invalidated run_id), reason |
| `approval_request` | report exit 3 | request_id, fingerprint (5 keys), payload (dev numbers, seed plan, disclosure) |
| `approval_decision` | approve | request_id, decision (approve/deny), reason |
| `report_attempt` | just before writing test | request_id, fingerprint (write-ahead marker) |
| `review` | on codex review | status, overall, review_path, review_sha256 |
| `final_report` | seal (last of all) | test/ci/dev/costs/multiple_testing_disclosure/claims_sha256/… (identical to report.json) |

### 7.3 Answering three questions

**Q1. "Did the candidate win?"**
- The `last N gate decisions` line in `status`, or `[gate] ... -> winner rNNNN`
  in the `run` console.
- To be sure, the generation's `gate` record: the `winner` field + `reason`
  (`beat the incumbent on the blind gate split`). The admission scores are only
  here in `results`.
- Or whether `best_commit`/`best_primary` in `state.json` moved.

**Q2. "What's the conclusion?"**
- For humans: `## Headline result` in `experiments/report/report.md` → `Status:`
  + effect + 95% CI.
- For machines: `primary_status` + `test.effect_abs` + `ci.abs` in
  `experiments/report/report.json`. Only `verified` (CI lower bound > 0) is a
  genuine improvement. `unsupported` does not mean "failure" but "one of the test
  seeds was unclean, so the bootstrap was skipped."

**Q3. "On what evidence?"**
- The `hypothesis.supporting_evidence_ids` of the adopted experiment record →
  looking those ids up in `experiments/evidence/evidence.jsonl` yields
  `{canonical_paper_id, claim, stance, locator, ...}`. (Measured: r0014 cites
  `ev_cl-0901` → paper `tsp:2018.0009` §3.1 "multi-start restarts lower expected
  best route length".)
- At report time, an **evidence audit** runs, and if an adopted hypothesis cites
  uninterpretable evidence, the report hard-fails with `evidence audit failed:
  ...`.

### 7.4 Dissecting the hypothesis certificate

The certificate of the measured r0014 (the final winner):

```json
{
  "id": "h_r0014_restarts",
  "statement": "Changing restarts from 4 to 16 will minimize mean_tour_length by at least 0.2% relative to the incumbent.",
  "mechanism": "Independent multi-start restarts reduce the variance of the best tour found and lower the expected result, at a cost linear in the restart count.",
  "intervention": {"param": "restarts", "from": 4, "to": 16, "kind": "restarts_up"},
  "predicted_effect": "mean_tour_length improves from 6455142.9000 by >= 0.2% ...",
  "falsifier": "mean_tour_length fails to improve by >= 0.2% (or becomes degenerate) in the deterministic dev evaluation — the single decisive test",
  "minimal_test": "one smoke + one dev evaluation on the patched worktree",
  "supporting_evidence_ids": ["ev_cl-0901"],
  "nearest_prior_work": ["tsp:2015.0006", "tsp:2018.0009"],
  "proposer": "heuristic", "executor": "patcher"
}
```

This is the essence of the framework's "scientific-ness": every hypothesis has a
**statement / mechanism / intervention / predicted_effect / falsifier (rejection
condition) / minimal_test**. It enforces Popperian falsifiability.

---

## 8. Advanced Recipes

### 8.1 LLM proposer + coding worker (`--proposer claude`)

```bash
uv run python orchestrator.py run --generations 3 --proposer claude --max-budget-usd 0.5
```

What changes:
- The Claude Agent SDK takes over hypothesis generation (tools fully disabled,
  JSON schema enforced, heuristic fallback on validation failure). On
  SDK/validation failure, `[warn] claude proposer failed (...); falling back to
  heuristic`.
- **The coder turns on** (`portfolio.max_coder_hypotheses`, default 1). The coder
  edits `src/**` to cross walls that hyperparameters can't — for example changing
  `NEIGHBORHOOD` from `two_opt→or_opt` or adding tabu.

**Coder isolation (multi-layer):**
- The **PreToolUse guard hook** is the sole authorizer. Since cwd can't confine
  SDK tools (absolute paths are allowed), the hook resolves every tool call via
  realpath and rejects Read/Glob/Grep outside the worktree and Write/Edit outside
  `src/`. **Bash and network tools are disabled entirely.**
- `permission_mode="dontAsk"` + `allowed_tools=[]` make it **fail-closed** (if
  the hook errors or times out, the default is deny).
- Before and after the coder call, it **compares snapshots of the root
  fingerprint (git status + protection manifest)** → if it detects an escape that
  touches the root via absolute paths, it halts the campaign with `root working
  tree mutated during coder round ...`.
- This is a **separate layer from the execution isolation (container)** of §8.5.
  Coder isolation guards the "editing moment," the container guards the
  "execution moment."

> Note: the LLM path may hit account usage limits. If it does, crash recovery
> cleans up the interrupted generation, so it's safe, and you can resume after
> the limit resets.

### 8.2 LLM literature analysis (`--literature claude`)

```bash
uv run python orchestrator.py run --generations 3 --proposer claude --literature claude
```

The LLM does (1) additional search query decomposition, (2) per-evidence stance
judgments + narration. **Search execution is always the deterministic lexical
backend.** Anti-laundering: the LLM can only downgrade a deterministic "supports"
and cannot newly grant one (grant attempts are counted as
`coverage.llm_supports_coerced`). On failure, `[literature] LLM analyst failed
(...); falling back to lexical grounding`.

### 8.3 pairwise gate (SciNav-style blind judging)

```bash
uv run python orchestrator.py run --generations 3 --proposer claude --gate pairwise
```

**Admission is always the deterministic scalar epsilon rule in every mode** (an
anti-overfitting guarantee). Only when 2 or more candidates pass admission does
an anonymized blind judge panel (N=3, majority vote) pick the winner. The judges
see only the contract, the hypothesis certificate (ids only), the bounded code
diff, and the **dev** metric — **they don't even receive the gate score as
input.** A/B labels are shuffled by sha256 parity to cancel position bias, the
verdict is a 4-way enum choice, and abstentions, sub-majority, SDK failure, and
budget exhaustion all fall back to the deterministic scalar, always co-recording
`scalar_winner` (divergence audit).

> Console: `[gate] candidates [...] -> winner rNNNN [pairwise: agreed with scalar]`
> (or `overrode scalar` / `scalar fallback`).

### 8.4 cross-model codex adversarial reviewer (ARIS)

A genuinely heterogeneous model (the OpenAI family) audits each claim's numbers
against the raw test data. This closes the correlated bias that remains because
the pairwise judges are all the same Claude family.

```bash
# 1) in the contract, set reviewer.enabled: true (after chmod, §9.3), and have codex login ready
# 2) opt-in run (this re-issues a sealed report, so --force + re-approval are required)
uv run python orchestrator.py report --force --reviewer codex
uv run python orchestrator.py approve <new request_id>
uv run python orchestrator.py report --force --reviewer codex
```

- Calls `codex exec` in a **read-only sandbox**. Binds request and response via
  an `echo_token` planted in the prompt and re-verifies (on mismatch, an
  `echo_mismatch` warning).
- **advisory**: failures are all recorded as `status="unavailable"` (codes:
  `codex_not_found`, `codex_not_authenticated`, `timeout`, `schema_violation`,
  etc.) but do not block the report, and it **does not silently fall back to a
  Claude reviewer** (preserving the heterogeneity purpose).
- Results go only to `experiments/report/review/` and the `review` ledger record.
  They do not enter report.md (number-via-claim scan consistency).

### 8.5 Making it trust-grade with the container sandbox (step by step)

The default subprocess is not trust-grade on gate/test. If you want real
isolation:

```bash
# 1) start the Docker daemon (colima start or Docker Desktop)
colima start

# 2) pre-pull the image by digest — runs are --network none, so on-demand pull is impossible!
docker pull python:3.14-slim@sha256:<digest>

# 3) unlock the protected contract, then edit the sandbox block
chmod u+w research_contract.yaml
```

The sandbox block in `research_contract.yaml`:
```yaml
sandbox:
  backend: container
  image: "python:3.14-slim@sha256:<digest>"   # must be a digest pin (reproducibility)
  memory_mb: 512
  cpus: 1.0
  pids_limit: 128
  require_container_for_trusted_splits: true   # hard-blocks gate/report under subprocess
```

```bash
# 4) re-hash the manifest + re-baseline (baseline is also trained in the container)
uv run python orchestrator.py init --force

# 5) run — each candidate solver runs inside docker run
uv run python orchestrator.py run --generations 1
```

The actual docker argv (key flags):
```
docker run --rm --init --network none --read-only
  --tmpfs /tmp:rw,size=64m,noexec,nosuid,nodev
  --user 65534:65534 --cap-drop ALL --security-opt no-new-privileges
  --memory 512m --memory-swap 512m --cpus 1.0 --pids-limit 128
  -v <workspace>:/w:ro
  -v <fresh_artifacts>:/w/artifacts:rw               # the only writable surface
  -v <empty_mask>:/w/evaluation/heldout_config.json:ro   # masks the hidden seed
  --tmpfs /w/experiments:rw,size=1m,...              # masks the gate-score ledger
  -v <instances>:/w/instances.json:ro               # coordinates only
  -e PYTHONHASHSEED=0 -e AUTORESEARCH_INSTANCES=/w/instances.json
  python:3.14-slim@sha256:<digest> python -s -B src/train.py
```

**Fail-closed discipline:** if the daemon/image is missing, it **never silently
falls back to subprocess** and instead halts with an actionable error. For
example:
- `Docker daemon not reachable — start it (colima start / Docker Desktop), or set sandbox.backend: subprocess in the contract`
- `pinned sandbox image absent — run `docker pull ...` first (runs are --network none, so the image cannot be pulled on demand)`

**Provenance echo verification:** the evaluator echo-checks
`metrics["sandbox"].backend` against the requested backend. On mismatch,
`ProtectionViolation: sandbox backend echo mismatch ...` (detecting old versions
or bypass). On success, the report.md header is stamped `trust-grade`.

> On Linux, adding a single `--runtime=runsc` line to `sandbox/runner.py`
> promotes to gVisor isolation (the mount/security model unchanged). The backend
> is fixed per campaign, so you can't change it midway.

### 8.6 Refreshing the real-literature corpus (`ground --refresh`)

The current corpus is a curated offline mock. To update it with real literature
(OpenAlex/S2), run this maintenance operation **on a networked host**. (The
campaign runs deterministically over a frozen snapshot, so the network is used
only during refresh.)

```bash
export S2_API_KEY=...            # only for s2 fetch. env-only, never in the contract/commits
chmod u+w literature/corpus/tsp_corpus.json
uv run python orchestrator.py ground --refresh \
  --source openalex --extractor claude --max-papers 60 --mailto you@example.com

git diff literature/corpus/tsp_corpus.json   # human reviews the tag diff
uv run python orchestrator.py init --force   # re-hash + re-baseline
```

- Flow: fetch → dedup (DOI > arXiv > title) → LLM (or deterministic) claim
  extraction → write to `.refresh.tmp` and re-validate → if valid, os.replace (if
  invalid, keep the old snapshot).
- **Anti-laundering guard:** to keep an injected abstract from producing
  `effect=improves` (the only support-granting stance), it downgrades to
  `conditional` if there's no improvement cue or if there's an injection marker.
  `DeterministicExtractor` never emits `improves`.
- The **REVIEW BEFORE FREEZE** block that is always printed after refresh shows
  the number of support-granting claims, injection-flagged papers, and
  policy-dropped claims — **this human review is the freeze gate.**
- Honest limitation: the `claude` extractor is nondeterministic, so a real
  refresh produces a new diff each time. `_urllib_get` and real SDK calls are
  finally verified only on a real network (outside the offline tests).

---

## 9. Swapping in Your Own Research Domain

The framework's real value is not "solving TSP" but being **a scaffold onto which
you can put any research problem on top of these safeguards**. Swapping the
domain means co-redesigning the following four.

### 9.1 What to change (4 surfaces)

| File | Role | When changing |
|---|---|---|
| `src/train.py` | the editable solver/model | new algorithm + `HYPERPARAMS` marker block + (optional) code knobs |
| `evaluation/dataset.py` | instance/data generation | new data distribution + `SPLIT_SIZES` + `load_train`/`load_split`/`fingerprint` |
| `evaluation/evaluate.py` | **the trusted evaluator** | new metric recomputation + hardcoded constants (budget/metric/split/N) + failure_class |
| `research_contract.yaml` | the contract | `objective`, `primary_metric`, budgets, portfolio, etc. |
| `literature/corpus/*.json` | (optional) literature | new domain claim corpus (or `ground --refresh`) |

### 9.2 Design rules you must follow (break them and the guarantees collapse)

1. **The evaluator must not trust self-reports.** Verify the solver's outputs and
   **recompute the metric directly** (like recomputing tour length in TSP). This
   is the crux of preventing reward hacking.
2. **Hardcode the evaluator constants and cross-validate them against the
   contract.** The evaluator does not trust anything outside `evaluation/`.
   `init` cross-validates the contract and evaluator constants
   (budget/metric/split/N) once and fail-fasts on drift.
3. **Hidden seeds only in the evaluator.** Keep the dev/gate/test seeds in
   `heldout_config.json` (git-untracked) and pass instances to the solver
   carrying **only opaque ids (i0, i1…)**. Do not build a structure where the
   seed integers pass into the sandbox (this blocks regenerating other splits).
4. **Put what you actually want into the objective function.** The lesson of §6:
   if you minimize only `mean_tour_length`, even a "win by spending more compute"
   becomes legitimate. If compute budget matters, reflect something like
   `solve_seconds` in the objective. (What the evaluator measures is the
   "direction of the science.")
5. **Set `min_relative_improvement` / `gate_min_relative_improvement` larger than
   the noise.** Since a deterministic evaluation would adopt even a 1e-9 wobble
   under a `>` rule, use a relative epsilon.

### 9.3 Procedure for intentionally modifying a protected file

When not running:
```bash
chmod u+w <file>                                 # unlock read-only
# ... edit ...
uv run python orchestrator.py init --force       # regenerate seeds and manifest + re-baseline
```

`--force` empties `experiments/`, so back up first if you need the previous
campaign's records. If the contract schema or the heldout_config schema changes,
you can't continue from the previous state and must start a new campaign.

> Tip: for a large domain swap, the convention in this project is to rsync-clone
> into a scratchpad, drill it end-to-end there, and reflect it into the real
> directory once verified.

---

## 10. Troubleshooting & FAQ

### 10.1 Error messages → what to do

| Symptom / message | Cause | What to do |
|---|---|---|
| `exit 3` (report) | human approval required (not an error) | `approve <request_id>`, then re-run |
| `approval pending for request_id ...` | awaiting approval | `approve` that id |
| `report intent ... was denied` | denied via `--deny` | reverse it by `approve`-ing the same id |
| `N final report(s) already exist — test split is single-use` | already sealed | `report --force` (re-approval required, multiple-testing disclosure increments) |
| approved but exit 3 again | ran more `run`s, so the **approval is stale** | re-approve the new request_id |
| `main working tree has uncommitted tracked changes` | main is dirty | commit/restore the tracked changes (untracked ones don't matter) |
| `already initialized ...; use --force` | already initialized | to continue, just `run`; for a new campaign, `init --force` (records are deleted) |
| `another orchestrator process is running (lock: ...)` | concurrent execution | kill the other process; if it's a dead process, check `.orchestrator.lock` |
| `VIOLATION: <file>` / `protected files modified: ...` | tampered protected file | revert it, or modify it properly via the §9.3 procedure |
| `... drift: contract ... vs evaluator ...` | contract and evaluator constants mismatch | align the contract ↔ evaluator values and `init --force` |
| `evidence audit failed: <run> cites <id> ...` | an adopted hypothesis cites uninterpretable evidence | check corpus/grounding consistency |
| `[warn] ... subprocess ... NOT trust-grade` | gate/test under subprocess | ignore (current behavior) or use §8.5 container |
| `sandbox.backend 'container' ... image absent` | image not pulled | `docker pull <digest>` first |
| `Docker daemon not reachable` | daemon is off | `colima start`/Docker Desktop, or set backend to subprocess |
| `sandbox backend echo mismatch` | evaluator bypass/old version | check evaluator integrity (`verify-protection`) |
| `literature ... is not writable (0o444)` | refresh but the corpus is read-only | `chmod u+w <corpus>`, re-run, then `init --force` |

### 10.2 Frequently asked

**Q. I ran `run --generations 3` but got 15 experiments?**
generation ≠ round. Up to K=8 hypotheses run per generation (the heuristic does
one per parameter, so ~6–7), and experiments accumulate. 3 generations × ~5–7 = 15.

**Q. I want to see the gate scores in the console/status.**
You can't (blindness). Open `experiments/generations/gNNNN/gate/*.json` or the
`results` of the `gate` ledger record directly. This is intentional design to
prevent transcripts from being pasted into the LLM context.

**Q. I see a lot of `pruned` — is that failure?**
No. It's the budget cut of successive halving (not science). It means "not
top-ranked at the smoke rung, so no dev-rung budget was spent," not "this
direction is wrong."

**Q. It's `valid_positive` — why was it REJECTed?**
The dev improvement is real, but it either didn't clear the blind gate or another
candidate in the same generation won. Only one accept per generation (§4.2).

**Q. `primary_status: unsupported` in `status`?**
It means one of the test seeds was unclean, so the bootstrap was skipped — not
"improvement failed." Check the failure_class in
`experiments/report/*_test_s*.json`.

**Q. I hit an account limit during the LLM path.**
Crash recovery cleans up the interrupted generation, so it's safe. Resume after
the limit resets. Literature falls back to lexical and pairwise to the
deterministic scalar, so the generation isn't blocked.

**Q. May I assert the results to others?**
subprocess-backend numbers are "honest but not trust-grade." To assert them,
reproduce under the §8.5 container so the report.md header is stamped
`trust-grade`.

---

## 11. Health Check (Test Drills)

To check that the framework is healthy, run 7 drills. **All are fully offline** —
no Docker, network, SDK, or real codex needed (all replaced with fake seams).
Each file prints `[ok  ] <name>` / `[FAIL] <name> — <detail>` per line, and exits
0 if everything passes.

```bash
uv run python tests/test_phase2.py   # blindness / coder-guard fail-closed / gate correctness / recovery
uv run python tests/test_phase3.py   # literature grounding (determinism / canary / laundering / injection)
uv run python tests/test_phase4.py   # momentum fold / steering / halving / pairwise
uv run python tests/test_phase5.py   # bootstrap / claims / report digit-scan / approval gate / codex stub
uv run python tests/test_phase6.py   # sandbox docker argv, masking, fail-closed (no Docker needed)
uv run python tests/test_phase6b.py  # real-literature fetch, extraction, snapshot (fake HTTP/LLM)
uv run python tests/test_phase6c.py  # TSP feasibility, recomputation, seed absence, blindness
```

For a CI smoke gate, just require "all 7 exit 0." Note: you must run each file
directly, not via pytest, and run it **from the repository root** (since it reads
the real contract and corpus, a different CWD or an edited corpus will FAIL the
count drills).

---

## 12. Appendix — Reference Dictionary

### 12.1 Contract fields (current values)

`research_contract.yaml` (schema v8). Parentheses show the current value.

```
primary_metric.name/direction/min_relative_improvement   (mean_tour_length / minimize / 0.002)
budgets.smoke_train_timeout_s / dev_train_timeout_s       (30 / 120)
budgets.max_rounds / repair_attempts                      (60 / 2)
portfolio.parallel_branches                               (8)   # K per generation
portfolio.gate_top_k                                      (2)   # number of dev-improvement candidates sent to the gate
portfolio.gate_min_relative_improvement                  (0.001) # gate admission epsilon
portfolio.max_coder_hypotheses                           (1)   # 0 turns off the coder
portfolio.max_generations                                (null)
portfolio.coder_max_turns / coder_max_budget_usd         (25 / 1.5)
portfolio.halving.enabled/keep_fraction/min_keep         (true / 0.5 / 2)
stop_conditions.stagnation_generations                   (4)
refinement.enabled/momentum_decay/exploit_fraction       (true / 0.5 / 0.75)
refinement.accelerate_after/evidence_steering            (2 / true)
pairwise_gate.enabled/judges/judge_model                 (true / 3 / claude-haiku-4-5)
pairwise_gate.judge_max_budget_usd                       (0.4)
literature.enabled/retriever/corpus_path                 (true / lexical / literature/corpus/tsp_corpus.json)
literature.max_evidence_per_generation/_per_hypothesis   (12 / 4)
literature.max_queries/stabilization_window/citation_hops (6 / 2 / 1)
literature.llm_max_budget_usd/_campaign_budget_usd       (0.5 / null)
literature.refresh.{sources,max_papers,extractor,...}    (openalex / 60 / claude / ...)
assurance.finalist_seeds/bootstrap_resamples/confidence  (5 / 10000 / 0.95)
reviewer.enabled/backend/timeout_s                       (false / codex / 300)
human_gate.enabled/require_approval_for                  (true / [first_report, force_report])
sandbox.backend/image/memory_mb/cpus/pids_limit          (subprocess / null / 512 / 1.0 / 128)
sandbox.require_container_for_trusted_splits             (false)
```

Examples of contract-validation failures (hard error on load): `finalist_seeds`
outside [1,16], `bootstrap_resamples < 100`, `confidence_level` not
0.90/0.95/0.99, `momentum_decay` outside (0,1), `judges` even or >5,
`evidence_steering=true` while `literature.enabled=false`, and so on.

### 12.2 failure_class dictionary

Attached by the evaluator: `invalid_workspace`, `evaluator_error`, `timeout`,
`nonzero_exit`, `missing_artifact`, `malformed_solution`, `infeasible_solution`
(not a valid tour — a scientific negative), `no_skill` (worse than an
identity-order tour — degenerate). Attached by the orchestrator:
`metric_regression` (classify), `patch_failed`, `oversized_diff`,
`coder_unavailable`, `coder_error`, `nondeterministic`, `smoke_rank_below_cutoff`
(pruned), symlink/protected/editable violations.

### 12.3 Domain constants (TSP)

`N_CITIES=60`, `GRID=1_000_000`, `TRAIN_SEED=20260401` (public),
`N_TRAIN_INSTANCES=40`, `SPLIT_SIZES={dev:40, gate:40, test:160}`,
`SOLVER_SEED=1337`, distance = TSPLIB EUC_2D integer rounding. Test uses a pool of
`finalist_seeds` (5) × 160 = 800 instances for the paired bootstrap.

### 12.4 Glossary (engineering terms)

- **fail-closed**: taking "deny" as the default when the judgment is ambiguous or
  the checker errors/times out. (The opposite is fail-open = pass when ambiguous.)
- **write-ahead**: writing "I will do this" to the ledger first, before the
  actual work. Even if it dies midway, recovery knows what was in progress. (The
  same concept as a DB's WAL.)
- **ff-merge (fast-forward merge)**: a merge that just moves the branch pointer
  forward without creating a fork. Only gate-passing experiments pile up on
  `main` this way.
- **flock (file lock)**: locking a file so the same operation can't run twice at
  once.
- **worktree**: a git feature that checks out different commits of the same
  repository into separate folders simultaneously. Each experiment runs in
  isolation in its own worktree.
- **tmpfs**: an in-memory temporary filesystem. When masking the ledger in the
  container, it "overlays an empty tmpfs" to hide the original.
- **provenance echo**: a technique that makes the requested values
  (backend/nonce/split) get echoed back verbatim into the result to detect
  forgery, bypass, and old versions.
- **blindness (here)**: the invariant that structurally blocks the gate score
  from leaking into the search signal, proposer, insight, or report.
- **incumbent**: the current champion (best). A candidate must beat it to become
  the new incumbent.

---

### What to do next (recommendations)

1. **One more campaign, a little differently:** run
   `run --generations 4 --proposer claude` and watch whether the LLM coder
   attempts an "algorithm-level" improvement by changing `NEIGHBORHOOD` (an
   empirical demonstration of crossing the hyperparameter wall with code).
2. **Reproduce at trust-grade:** turn on the container backend per §8.5,
   reproduce the same campaign, and confirm the report.md header changes to
   `trust-grade`.
3. **Port to your own problem:** use §9 to put a small domain on top (e.g.,
   another combinatorial optimization or a simple regression). Directly
   experiencing how what you put into the objective function changes the results
   (the §6 lesson) is the fastest way to understand this framework.

For details not found in this document, dig in this order: `README.md` (usage
summary) → `docs/HANDOFF.md` (the 14 invariants) → `docs/BLUEPRINT.md` (design
rationale) → the code.
