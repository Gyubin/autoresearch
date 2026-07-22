# AutoResearch — Session Handoff (read this first, after Phase 6b/6c)

*The Korean original is preserved at [HANDOFF.ko.md](HANDOFF.ko.md).*

This document is the **single entry point** for a fresh session picking up this
project. Read it in order: this file → `docs/BLUEPRINT.md` (design rationale +
layer specs) → `README.md` (usage) → the code you need.

Written: 2026-07-18 (updated to reflect Phase 6b/6c). Phases 1–5 + 6a + 6b + 6c
complete. **The domain has been swapped from a mock synthetic regression to a
Euclidean-TSP heuristic** (contract v8).

---

## 0. 30-second orientation

- **What**: An autonomous research loop built on the 2026 SOTA blueprint
  (`docs/BLUEPRINT.md`). The core philosophy is to stand up a trustworthy
  evaluator, contract, and provenance layer *before* the agent swarm.
- **Where**: `/Users/gyubin.son/workspace/dev/autoresearch` (git repo, branch main).
- **Stack**: Python 3.14 (pinned via `.python-version`), a uv virtualenv, and
  dependencies limited to pyyaml + claude-agent-sdk only. **Orchestration is
  implemented with the Claude Agent Python SDK, not LangGraph** (a decision
  confirmed by the user — the blueprint's LangGraph recommendation is ignored).
- **Done**: Phase 1 (constrained keep/reject loop) + Phase 2 (parallel portfolio
  + blind admission gate + LLM coding worker) + Phase 3 (claim-level literature
  grounding — mock-corpus-only, dual mode) + Phase 4 (directed branch refinement
  — Gome search momentum + evidence steering + successive halving + SciNav
  pairwise gate) + Phase 5 (assurance + reporting — multi-seed finalist
  reproduction + paired bootstrap CI, claim-evidence ledger, deterministic
  report.md + SVG figures, cross-model codex reviewer, human approval gate,
  momentum coder-family classification) + Phase 6a (execution isolation — a
  `Sandbox` protocol in the `sandbox/` package, with a Docker `ContainerSandbox`
  that runs `train.py` under OS isolation: network cut off, read-only rootfs,
  seed/ledger masking, ephemeral PID namespace. `subprocess` is the default
  (identical to current behavior), `container` is opt-in, fail-closed).
  Phase 6b (real literature API — an OpenAlex adapter in `literature/sources.py`
  + `ground --refresh` for fetch → LLM extraction → frozen corpus snapshot.
  The network is touched only during refresh; the campaign stays deterministic
  on the frozen snapshot via lexical retrieval. The S2 key is env-only, and an
  anti-laundering guard prevents an injected abstract from manufacturing support)
  + Phase 6c (domain swapped to a Euclidean-TSP heuristic — the evaluator
  generates an instance from a hidden seed → passes only the coordinates to the
  solver → recomputes the tour length. The seed never crosses into the sandbox
  (opaque id), and self-reported objectives are ignored).
- **Done (follow-ups)**: (A) evidence_steering param→family alignment —
  `orchestrator.rank()` / the explore filter convert the raw param to a family via
  `engine.PARAM_TO_FAMILY` so literature steering actually works (`tests/test_phase4.py`
  drills this end-to-end against the real corpus). Remaining gap: the corpus's
  `neighborhood_operator/add_operator` claims have no corresponding move on the edit
  surface (segment_max=increase/decrease), so they don't match (an honest non-goal).
  (C) container trust policy binding — if the gate/test runs on a non-isolated backend
  the orchestrator warns continuously, and with `sandbox.require_container_for_trusted_splits: true`
  it fails closed; the report.md header stamps the trust grade. The "we warn" comment
  in `evaluate.py` becomes true.
- **Not done (operational)**: the current corpus (`tsp_corpus.json`) is a curated
  offline snapshot (provenance is empty). Reflecting real literature (Part C) means
  running `ground --refresh` on a networked host → human-reviewing the tag diff →
  `init --force` (the handler is verified offline by `tests/test_phase6b.py`; only
  the live-network path is uncovered). See the README runbook "refreshing the real
  literature corpus."
- **Next (when running)**: `uv run python orchestrator.py run --generations N` (or
  the full LLM path `--proposer claude --literature claude --gate pairwise`).

## 1. Try running it now

```bash
cd /Users/gyubin.son/workspace/dev/autoresearch
uv sync
uv run python orchestrator.py status              # current campaign state (incl. literature stats + approvals/reviews)
uv run python orchestrator.py verify-protection   # protected-file integrity (20 files)
uv run python tests/test_phase2.py                # Phase 2 unit drills
uv run python tests/test_phase3.py                # Phase 3 literature drills
uv run python tests/test_phase4.py                # Phase 4 refinement drills (momentum/steering/halving/pairwise)
uv run python tests/test_phase5.py                # Phase 5 drills (bootstrap/claims/report/gate/reviewer/families)
uv run python orchestrator.py ground              # research-question certificate (literature evidence flow)
uv run python orchestrator.py run --generations 2 # parallel generation run (heuristic + lexical + halving + momentum, no SDK needed)
uv run python orchestrator.py run --generations 1 --proposer claude --literature claude --gate pairwise  # full LLM path
uv run python orchestrator.py report              # (approval required) multi-seed report on the test split → exit 3 + request_id
uv run python orchestrator.py approve <request_id># approve publication intent → subsequent report proceeds
uv run python orchestrator.py report --reviewer codex  # after approval, include cross-model codex review (opt-in, requires reviewer.enabled)
```

**Phase 5 report flow**: `report` requires human approval before it touches the
untouched test split (a publication analog). The first `report` (and any `--force`
re-run) writes an approval request to the ledger and prints **exit 3 + request_id**;
only after a human approves the intent (commit, dev numbers, seed plan, disclosures)
via `approve <request_id>` does the multi-seed test evaluation → paired bootstrap CI →
`experiments/claims.jsonl` → `experiments/report/report.md` + `figures/*.svg` →
(opt-in) codex review → `final_report` seal happen. Approval is derived from the
ledger rather than from state (fingerprint = incumbent/baseline commit +
contract/evaluator sha + prior seal count), and it goes stale — requiring
re-approval — as the campaign advances.

The unit of execution is not a round but a **generation**. In one generation K
hypotheses (contract `portfolio.parallel_branches`, which is 8 in Phase 4) run in
parallel. Coder hypotheses only appear when `--proposer claude` (the heuristic is
fully offline and deterministic). For literature grounding, the contract
(`literature.enabled`) decides *whether* and `--literature {lexical,claude}`
decides *how* — the default lexical is fully offline. **Phase 4**: search
momentum, evidence steering, and successive halving run when the contract turns
them on (`refinement.enabled`, `portfolio.halving.enabled`) and are all offline
and deterministic. For the pairwise gate, the contract (`pairwise_gate.enabled`)
decides *whether* and `--gate {scalar,pairwise}` decides *how* — the default
scalar is fully offline, and admission is a deterministic scalar rule in either
mode.

## 2. File map

| File | Role | protected? |
|---|---|---|
| `research_contract.yaml` | typed contract (schema v6: + sandbox block) | ✅ immutable |
| `orchestrator.py` | coordinator + gate + coder + literature seam + momentum/halving + approval gate + multi-seed report + sandbox wiring (preflight/echo) + CLI | ✅ |
| `sandbox/` | Phase 6a execution isolation (`runner.py`): `Sandbox` protocol, `SubprocessSandbox` (current), `ContainerSandbox` (docker), `build_sandbox`, `preflight`. stdlib, absolute-path load, no runtime file IO | ✅ (§3 closure) |
| `assurance/` | Phase 5 pure package: `stats.py` (paired bootstrap), `claims.py` (claim ledger), `report_md.py` + `figures.py` + `svgfig.py` (deterministic report/figures), `reviewer.py` (codex adapter), `gate.py` (approval derivation), `families.py` (coder families) | ✅ (§3-13 closure) |
| `literature/engine.py` | literature engine: corpus validation, LexicalRetriever, EvidenceEngine, move_of/move_guidance, ClaudeLiteratureAnalyst, FallbackAnalyst (~1000 lines, stdlib) | ✅ |
| `literature/corpus/tsp_corpus.json` | 13 TSP heuristic papers / 13 claims (incl. contradiction pairs, laundering traps, injection fixtures) | ✅ (git-tracked) |
| `evaluation/evaluate.py` | protected evaluator → metrics.json (`--split dev|gate|test`) | ✅ |
| `evaluation/dataset.py` | synthetic data, `load_split`, `SPLIT_SIZES` | ✅ |
| `evaluation/heldout_config.json` | hidden dev/gate/test seeds (created by init, **git-untracked**) | — |
| `src/train.py` | editable surface: HYPERPARAMS block + FEATURE_SPEC | edit target |
| `protection/hashes.json` | protected-file SHA-256 manifest (22 files: + 2 from sandbox/) | ✅ (git-tracked) |
| `tests/test_phase2.py` | unit drills (guard hooks/stagnation/blindness/feature_spec) | — |
| `tests/test_phase3.py` | literature drills (determinism/blindness canary/laundering/injection/contract v5) | — |
| `tests/test_phase4.py` | refinement drills (momentum fold/steering/halving/pruned/pairwise/contract v5) | — |
| `tests/test_phase5.py` | assurance drills (bootstrap/claims/report digit-scan/gate state machine/codex reviewer stub/families/record inertness) | — |
| `tests/test_phase6.py` | sandbox drills (subprocess regression/container argv hardening/seed·ledger masking/fail-closed preflight/timeout teardown/contract v6/no-drift/provenance echo — no Docker needed) | — |
| `experiments/` | runtime: state.json, ledger.jsonl, rounds/, generations/, evidence/, **claims.jsonl**, **report/(report.md·report.json·{role}_test_s{k}.json·figures/·review/)** | gitignored |
| `insight_memory.json` | derived lessons reconstructible from the ledger | gitignored |
| `.worktrees/` | per-experiment isolated worktrees | gitignored |
| `docs/BLUEPRINT.md` | original research document (Phase 5 design spec) | — |
| `docs/archive/` | ledger/rounds archives of closed campaigns | — |

Things to find in `orchestrator.py`: `load_contract` (contract v6 parsing —
top-level key whitelist + halving/refinement/pairwise_gate/assurance/reviewer/human_gate/sandbox
blocks), `run_evaluator` (passes sandbox CLI args + validates the backend echo),
`_sandbox_preflight` (fail-closed, entered from `run`/`init`/`report`),
`_load_evaluator_declarations` (cross-checks sandbox backends),
`cmd_report` (approval gate → multi-seed test → stats → claims → report/figures →
review → seal), `cmd_approve`, `_reviewer_raw_test`, `extract_update_vectors` /
`search_momentum_table` / `_momentum_weight` (Phase 4a — ledger-derived search
momentum, not persisted to state), `run_generation` (the generation loop —
momentum → grounding → propose → attach → 2-stage halving pool → gate → persist),
`_experiment_smoke_stage` / `_experiment_dev_stage` / `_apply_halving` /
`_prune_record` (Phase 4b), `_run_gate` (admission — deterministic scalar epsilon)
+ `_select_gate_winner` / `_scalar_gate_winner` / `PairwiseJudge` /
`_judge_campaign_spend` / `_candidate_diff` (Phase 4c — selection/blind
judge/budget), `_finish_generation` (ledger/merge/state), `ClaudeCoder` +
`_make_worktree_guard` (coder isolation), `distill_insight` (blindness invariant +
pruned→None), `replay_ledger_fields` (state reconstruction for recovery), `recover`
(crash recovery), `_build_literature` / `_build_judge` / `_BudgetGuardedLiterature`
(service construction / campaign budget), `_sdk_structured_query` (shared SDK call —
shared by proposer/judge), `cmd_report` (test split + evidence audit + 4-way cost
summation). Things to find in `literature/engine.py`: `load_corpus` (validation /
content policy), `move_of` (the shared move vocabulary — imported by orchestrator),
`EvidenceEngine.ground/attach` (10-step flow / single authoritative evidence writer —
attach is still annotation-only), `Grounding.move_guidance` (Phase 4 steering —
categorical, id-only, pure method), `_hyp_stance` (anti-laundering rule),
`_coder_family` (coder-hypothesis family classification — gives up when ambiguous),
`ClaudeLiteratureAnalyst` (LLM path, only allowed to downgrade supports).

## 3. Invariants that must be preserved (breaking them collapses the Phase 1–4 guarantees)

1. **Contract immutability**: `orchestrator.py` only reads `research_contract.yaml`.
   The baseline is recorded in `experiments/state.json`. Changing the
   contract/evaluator/dataset makes init fail-fast on schema drift, and you must
   start a new campaign with `init --force`.
2. **Only the root copy of the evaluator is authoritative**. The workspace's
   evaluation/ copy is never used for scoring. The evaluator trusts nothing outside
   evaluation/ (it hardcodes budgets, metrics, and splits, and cross-checks against
   the contract at init).
3. **Physical absence of held-out seeds**: the 3 seeds (dev/gate/test) are
   git-untracked and therefore do not exist in the worktree.
4. **gate blindness**: gate scores live only in `record_type=gate` ledger records +
   the gate metrics file. They must never leak into insight, `best_primary` (always
   the dev score), proposer context, experiment records, or **search momentum,
   steering.json, move_guidance, or the pairwise judge packet**. `distill_insight`
   does not read gate records. `tests/test_phase2.py` (literal scan),
   `tests/test_phase3.py` (canary 0.424242), and `tests/test_phase4.py`
   (momentum/judge canary) check this — don't break it when adding new fields in
   Phase 5.
5. **No false repair**: a runtime failure after a valid intervention (divergence /
   timeout / no_skill / dev-stage failure) is all scientific evidence =
   valid_negative and must not be repaired. Coder repair applies only to mechanical
   failures at the smoke stage (nonzero_exit / missing_artifact / malformed_artifact).
   **Phase 4b**: the coder repair loop lives inside `_experiment_smoke_stage`, and the
   halving cut happens on the main thread after the smoke stage ends, so a halving
   elimination (`pruned`) is structurally never subject to repair.
6. **Coder isolation**: `permission_mode="dontAsk"` + `allowed_tools=[]` + the
   PreToolUse guard hook is the sole permitter (fail-closed). Before and after the
   coder call, a `_root_fingerprint` snapshot comparison detects root escape → on
   detection, abort the campaign. Bash and network tools are disabled.
7. **Write-ahead ordering**: per generation, the gate record first → K experiment
   records → ff-merge → on failure a correction record. `replay_ledger_fields` groups
   by generation to reconstruct stagnation (a flat reconstruction would pollute the
   winning generation with a stagnation of K-1).
8. **Parallel safety**: only git shared-state mutations (worktree add/remove/prune,
   branch -D, merge) are serialized via `Git._mutation_lock`. Worker threads only work
   inside their own worktree and do not write to state/ledger (all of that happens on
   the main thread after the barrier). gc.auto=0.
9. **Literature engine closure (Phase 3)**: `literature/` does not import
   orchestrator, reads no file other than the corpus (the `ground()` signature cannot
   accept state/ledger/metrics), and **writes nothing at runtime** (it's a protected
   path, so a runtime cache would make the root fingerprint falsely detect coder
   escape and abort the campaign). Evidence is written to
   `experiments/evidence/evidence.jsonl`, **separate** from the ledger, and only at
   the point before the gate runs. Hypotheses carry only evidence **ids** (claim prose
   must not enter the blindness-scan surface). The LLM literature verdict can only
   downgrade a deterministic "supports." `tests/test_phase3.py` drills these
   invariants (incl. the gate canary). **Phase 4**: `Grounding.move_guidance()` is a
   pure method under the same closure (categorical enum + evidence ids only, no prose,
   no numbers).
10. **Search momentum is derived and not persisted (Phase 4a)**: momentum is not
    stored in state and is recomputed every generation from the ledger's experiment
    records (`extract_update_vectors` → `search_momentum_table`). Its input closure is
    identical to `replay_ledger_fields` (experiment records only, ignoring
    gate/correction/evidence), so gate scores structurally cannot enter and replay==live
    is self-evident. **Resist the temptation to persist a new field to state** —
    keeping it derived makes crash recovery free. It uses only the dev signal and the
    accept/reject bit (§3-4). steering.json is a pure audit artifact with no code that
    reads it.
11. **pruned is a budget decision, not science (Phase 4b)**: the halving-elimination
    verdict `pruned` is None in `distill_insight` and gets a search-momentum weight of
    0. But it does register the tested endpoint — you must keep the symmetry that
    **both** `_finish_generation` and `replay_ledger_fields` exclude only aborted, so
    that replay==live holds (an asymmetry would cause re-proposal thrashing). The smoke
    proxy score is recorded separately as `smoke_primary` (dev-split shortened
    training, harmlessly visible to the proposer) and `primary` (dev) is None.
12. **pairwise is selection-only, admission is deterministic (Phase 4c)**: admission
    (gate-split epsilon) is a deterministic scalar rule in either mode and the LLM
    cannot relax it. The pairwise judge only picks the winner among candidates that
    passed admission (`_select_gate_winner` always returns from the admitted set only).
    The judge's input closure is contract, hypothesis certificate (ids only), a bounded
    diff, and **dev** metrics — the gate score, `state["gate"]`, gate metrics, and
    `gate_record["results"]` never enter. The verdict is forced into a 4-way enum, with
    anti-injection framing in front of untrusted candidate material. Abstention,
    failure, and budget exhaustion fall back to the deterministic scalar and always
    include `scalar_winner` alongside (divergence audit). The gate record's
    `pairwise.cost_usd` records the **per-generation delta** (the judge instance lives
    across generations, so `total_cost_usd` is an accumulator — recording the
    cumulative value would make `_judge_campaign_spend` re-sum the prefix and
    double-count quadratically, and the ledger would be self-contradictory on restart).
13. **assurance closure (Phase 5)**: `assurance/` imports only stdlib, does not import
    orchestrator/literature/yaml, and **does no file IO** (stricter than literature —
    data in, strings/dicts out). The sole exception is `reviewer.run_review`, which
    calls `codex exec` via an injectable runner and touches only its own scratch
    workdir (`experiments/report/review/<request_id>/`) — it must never access the repo
    root or a protected dir. All read/write is done by the orchestrator's `cmd_report`
    and all of it goes only under `experiments/**` (a runtime write to a protected path
    would make the root fingerprint falsely detect coder escape). There is no wall
    clock or ambient randomness (bootstrap seeds are derived from contract/commit and
    logged, timestamps are passed as arguments), so claims, report.md, and the SVGs are
    byte-deterministic and audited via sha256. **Gate scores enter nowhere in the Phase
    5 artifacts** (claims, report, and the reviewer packet all carry only dev+test
    numbers). Every number in report.md is inserted only from a claim/meta value and
    `scan_untraced_digits` checks after rendering. The codex review is **advisory** — a
    failure is merely recorded as status "unavailable" and does not block the pipeline,
    and it does not silently fall back to a Claude reviewer (preserving the heterogeneity
    purpose). `tests/test_phase5.py` drills these invariants.
14. **Sandbox closure, fail-closed, seed absence (Phase 6a)**: `sandbox/` is a trusted
    path — stdlib only, does not import orchestrator, and its only runtime file IO is
    creating the candidate's scratch (artifacts/log). The evaluator loads this module
    **by absolute path** (not via sys.path) so a workspace copy cannot shadow the
    isolation code. The `container` backend masks the seeds/ledger so that in any
    workspace (including ROOT) the seed is absent from the container FS, and scoring
    stays on the host so the seed never enters the sandbox. When the daemon/image is
    absent it **never silently falls back to `subprocess`** but aborts with an
    actionable error (the same discipline as the reviewer). The backend is fixed at init
    → baseline and candidates share the same pipeline. `run_evaluator` validates the
    backend echo. `tests/test_phase6.py` drills this.

## 4. Mock task (means of demonstration)

Pure-Python synthetic regression. `y = 0.3 + w·x + 0.3·x0·x1 + N(0,0.25)`, with 8
heterogeneous-scale features. **The x0·x1 interaction is something a linear model
can't capture, creating a floor (~0.39) that hyperparameters alone can't beat.** If
the coder adds a product term ([0,1], etc.) to `FEATURE_SPEC` in `src/train.py`, it
breaks through the floor (demonstration: the coder reaches dev RMSE ~0.25, passes the
gate, then merges). The evaluator **validates and scores** the artifact's
`feature_spec` (a list of original-index products, up to 32 terms, degree 3) **as
data** — it does not run the candidate code inside the evaluator.

This task is not real science but a **surrogate problem for demonstrating the
pipeline**. Phase 5 can be tested either by swapping in a real research domain or by
layering the assurance stack on top of this task.

## 5. Phase entry points (see blueprint §2, §8)

### Phase 3 — Literature grounding (Layer 2) ✅ done
A separately-governed `literature/` service (offline mock corpus + a `Retriever`
seam), dual mode (deterministic lexical default / `--literature claude` opt-in —
tools=[] structured-output only, only allowed to downgrade supports), hypothesis
certificates `supporting_evidence_ids` / `nearest_prior_work` (ids only, whitelisted),
categorical novelty, contradiction reporting, coverage stopping, a `ground`
certificate, an evidence memory (separate from the ledger), a report evidence audit +
a campaign literature budget. What remains: real API adapters (OpenAlex/S2) behind the
`Retriever` protocol are unimplemented.

### Phase 4 — Directed branch refinement (Layer 4 deepened) ✅ done
- **4a search momentum + evidence steering**: `extract_update_vectors` /
  `search_momentum_table` (ledger-derived, not persisted, §3-10), a 3-tier heuristic
  ordering (momentum > literature stance > static) + value progression (accelerating
  steps / geometric bisection at the divergence boundary) + forced explore slots
  (defense against hypothesis collapse). The deferred evidence steering is inserted via
  `Grounding.move_guidance()` (pure, categorical, id-only) and `attach()` stays
  annotation-only. With `refinement.enabled=false` it's byte-identical to Phase 3. The
  ClaudeProposer prompt gets momentum/guidance sections + a soft explore retry.
- **4b successive halving**: `run_experiment` is split into smoke/dev stages, with a
  2-stage generation pool (everyone smokes → `_apply_halving` rank cut → survivors run
  dev). The elimination verdict is `pruned` (§3-11). K=8 + `halving.enabled` is the
  default contract.
- **4c SciNav pairwise gate**: `_run_gate` = admission (deterministic scalar) +
  `_select_gate_winner` (selection). `PairwiseJudge` (N=3 blind majority vote, sha256
  label swap, enum verdict, anti-injection), `--gate pairwise` opt-in, scalar fallback +
  `scalar_winner` alongside, a campaign budget guard (§3-12).
- Adversarial review: due to the session usage limit only 1 of the auto 4 lenses ran to
  completion (robustness-recovery, which found 1 pairwise cost-accumulation bug → fixed,
  regression drill added). The remaining lenses were shored up with inline review + a
  re-run.

### Phase 5 — assurance + reporting (Layers 8, 9) ✅ done
- **Multi-seed finalist reproduction**: heldout_config schema v3 (N test seeds,
  `finalist_seeds`=5), `evaluate.py --seed-index` (test-only) + `per_example_sq_errors`,
  `run_evaluator` seed_index echo, `init` generates 2+N seeds + `MAX_TEST_SEEDS` (=16,
  cross-checked against `MAX_FINALIST_SEEDS`).
- **paired bootstrap CI** (`assurance/stats.py`, stdlib): pairs baseline/incumbent
  squared errors per-example within each seed's dataset (asserts fingerprint identity),
  pooled resampling over an N×600 pool, deterministic log seeds. A hierarchical
  bootstrap is rejected (seeds are not clusters). incumbent==baseline is
  aliasing → effect 0 / CI[0,0] / inconclusive.
- **claim-evidence ledger** (`assurance/claims.py`): fully regenerated at report time,
  5 deterministic rules (primary_effect · campaign_summary · admitted_improvement ·
  negative_result · literature_grounding), a `claims_sha256` seal in `final_report`.
- **deterministic report.md + SVG** (`assurance/report_md.py` + `figures.py` +
  `svgfig.py`): numbers are inserted only from claim/meta and forced via a digit-scan
  after rendering, 3 kinds of figures with sha256.
- **cross-model codex reviewer** (`assurance/reviewer.py`): a `codex exec` subprocess
  (read-only sandbox / stdin packet / `--output-schema` / echo_token re-verification),
  an ARIS-style claim audit, advisory (failure=unavailable, no Claude fallback).
  `--reviewer codex` opt-in + `reviewer.enabled`.
- **human gate** (`assurance/gate.py` + `approve`): approval for publication (first
  report) / `--force`, ledger-derived state, exit 3.
- **momentum coder families** (`assurance/families.py`): deterministic diff
  classification, stores `coder_family` in the experiment record (replay==live).

### Phase 6a — Execution-isolation sandbox (Layer 5 deepened) ✅ done
- **The isolation boundary is the single `_run_train`**: the only place untrusted
  candidate code actually *runs* is `_run_train` in `evaluation/evaluate.py` (scoring
  stays on the host trusted path and the seed never crosses there). So isolating just
  this execution closes off seed reads, network, and TOCTOU.
- **The `Sandbox` protocol** (`sandbox/runner.py`, protected / stdlib / absolute-path
  load): `SubprocessSandbox` (a port of current behavior, byte-identical default with
  `backend: subprocess`) + `ContainerSandbox` (`docker run` — `--network none` /
  `--read-only` / `--cap-drop ALL` / `--security-opt no-new-privileges` / non-root user
  / `--memory/--cpus/--pids-limit` / `--rm --init`). It mirrors the `Retriever` seam
  pattern from `literature` exactly.
- **Seed/ledger masking**: whether the workspace is ROOT or a worktree,
  `heldout_config.json` is masked as an empty file and `experiments/` (the gate scores)
  as tmpfs → physically absent on the container FS. artifacts is mounted rw only via a
  separate fresh host dir (the worktree original stays immutable).
- **fail-closed**: when the daemon/image is absent it never silently falls back to
  `subprocess` but aborts with an actionable error (`sandbox.preflight`, the preflight
  entered from `run`/`init`/`report`). The same discipline as the reviewer not falling
  back to Claude.
- **contract v6**: a `sandbox` block (backend/image/memory_mb/cpus/pids_limit),
  `schema_version 6`, `sandbox/**` in `protected_globs`. The backend is fixed
  campaign-wide at init → baseline and candidates share the same pipeline, so the Phase
  5 paired bootstrap stays valid. The evaluator declares `SUPPORTED_SANDBOX_BACKENDS`
  and init cross-checks it against the contract (same as budgets/metric).
- **provenance echo**: `metrics["sandbox"]={backend,image,isolated}`, and
  `run_evaluator` validates that the echo matches the requested backend (like the
  nonce/split echo) → detects bypass or a stale build.
- **gVisor follow-up drop-in**: on Linux, adding a single `--runtime=runsc` line to
  `sandbox/runner.py` strengthens syscall isolation (the mount/security model stays
  unchanged). Firecracker / containerizing the coder itself is a follow-up. Drill:
  `tests/test_phase6.py` (no Docker needed, a fake runner for argv/masking/preflight).

### Phase 6b/6c — Real literature + real domain (Layer 2 deepened) ← next
- Implement real literature API (OpenAlex/S2) adapters behind the `Retriever` protocol +
  a fetch-once snapshot cache (staying deterministic) + `ground --refresh`.
- Swap the mock synthetic regression for a real research domain (redesign the evaluator,
  contract, and literature corpus).

## 6. Development workflow conventions (what was upheld in this project)

- **Design first**: before a big change, get a design proposal from the Plan subagent
  and correct it (concrete questions on git pitfalls, state atomicity, the SDK spec,
  etc.). Measure/investigate numbers and APIs before writing code.
- **Adversarial verification**: after implementation, do a multi-lens review
  (correctness / evaluator bypass / schema consistency / robustness) → rebuttal
  verification per finding → fix only confirmed defects. In Phase 2 only the recovery
  lens ran to completion (0 confirmed), the rest were cut off by the usage limit — I
  substituted direct drills for verification. **From Phase 3+, run this review to
  completion.**
- **Test outside the real directory**: rsync-clone into the scratchpad and run the E2E
  drills there; once verified, apply to the real directory.
- **Procedure for modifying a protected file**: `chmod u+w <file>` → modify →
  `init --force` (regenerates seeds/manifest + re-baselines, experiments/ is reset).

## 7. Known limitations / cautions (see blueprint §7, README)

- **Seed reads, network, and TOCTOU are now blocked at the OS level under
  `sandbox.backend: container`** (Phase 6a): the container masks `heldout_config.json`
  (physical absence) and blocks daemon survival / writes outside the workspace via
  `--network none`, a read-only rootfs, and an ephemeral PID namespace. **But the default
  is `subprocess`** — under this backend the policy-level limitations below remain as-is
  (preserving current behavior, no Docker needed). If you want the hardening guarantee,
  opt into `container` in the contract and pin the image by digest.
- (`subprocess` backend) held-out seeds are readable by the local user from the root, and
  while the coder can't read them (via the guard hook) the training subprocess is at the
  level of policy protection. TOCTOU (background daemon), file writes outside the
  workspace (detected but not prevented), and network cannot be blocked without an OS
  sandbox (`container`). The codex reviewer's read-only sandbox is a separate layer.
- ClaudeCoder and the pairwise judge can hit the **account usage limit** (SDK error).
  Crash recovery cleans up an interrupted generation so it's safe, but resume after the
  limit resets. The literature LLM path falls back to lexical and pairwise to the
  deterministic scalar so a generation isn't blocked (recorded via `mode` /
  `fallback_reason`). **Note**: the Phase 4 adversarial review itself hit this limit, so
  only some of the auto 4 lenses ran to completion — account for the limit when running
  an LLM multi-agent review.
- The literature engine is mock-corpus-only (the real API adapter is just a `Retriever`
  seam). The intervention-family classification of coder hypotheses is keyword matching —
  when ambiguous it gives up classifying (unexplored). The pairwise judge is the same
  Claude family, so a correlation bias remains (a truly heterogeneous reviewer is Phase
  5).
- Phase 6b/6c unimplemented: the real literature API adapter (currently just a
  `Retriever` seam), the real-research-domain task swap. The codex reviewer depends on a
  local `codex` login / account limit (opt-in / advisory), and on failure it's merely
  recorded as review-absent and does not block the report.
- **Phase 5 assurance inherits §7's seed trust boundary — this boundary can now be closed
  with `sandbox.backend: container` (Phase 6a)**. Under the `subprocess` backend the
  training subprocess has no OS sandbox, so an adversarial coder could traverse outside
  the worktree in `train.py` to read the hidden seeds (gate/test) or write out-of-band to
  the gitignored `experiments/` (including the ledger). Under the `container` backend the
  seeds/ledger are masked and network / writes outside the worktree are blocked at the OS
  level, so this path is closed. Either way, the assurances of multi-seed test
  reproduction, paired bootstrap, the claim ledger, and the approval gate hold for an
  **honest candidate + a deterministic pipeline**, and the headline `primary_effect` comes
  only from the live bootstrap, not the ledger, so it can't be overturned by forging the
  ledger. Enforcement against an adversarial coder holds under `container` (the adversarial
  review's evaluator-bypass lens). A report_attempt that crashed before sealing is counted
  conservatively as `crashed_report_attempts` in the multiple-testing disclosure.

## 8. New-session start prompt (example)

> "In the AutoResearch system in this directory (`~/workspace/dev/autoresearch`), I want
> to continue implementing Phase 6b (real literature API OpenAlex/S2 adapters — behind
> `Retriever` + a fetch-once snapshot cache). First read `docs/HANDOFF.md` and
> `docs/BLUEPRINT.md`, grasp the current state (complete up to the 6a execution-isolation
> sandbox), and then propose a design first."

Binding decisions are recorded in memory (`autoresearch-project-decisions`) and
auto-recalled, but the two docs above are the source of the actual spec.
