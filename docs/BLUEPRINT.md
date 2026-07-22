# Autoresearch in 2026: SOTA Methods, Architecture, and Implementation Blueprint

*The Korean original is preserved at [BLUEPRINT.ko.md](BLUEPRINT.ko.md).*

> Source research document (provided by the user, assessment date 2026-07-16). This is
> the design rationale for the system and the reference specification for the Phase 3–5
> implementation. **This file is the only copy of the blueprint that persists outside the
> conversation** — new sessions read the architecture from here.

## Executive assessment

There is no single "SOTA autoresearch system." Each system optimizes a different part of
the research lifecycle:

- **Arbor, Gome, ERA, SciNav, AlphaEvolve, R&D-Agent** — focused on search over
  executable code / empirical artifacts.
- **AI Scientist v2, AutoResearchClaw, EvoScientist, ARIS** — longer end-to-end
  research workflows.
- **Co-Scientist** — hypothesis generation and scientific deliberation.
- **Robin** — integrates hypothesis generation + experiment planning + biological data analysis.
- **PaperQA2, Ai2's scientific search systems** — citation-grounded literature synthesis.

The strongest architecture is a synthesis of the best ideas from these:

> **Arbor-style hypothesis-tree refinement (portfolio level) + Gome-style directional
> diagnostic updates (inside a branch) + SciNav-style pairwise ranking (when no trustworthy
> scalar metric exists) + Karpathy/ERA-style invariant evaluation + EvoScientist-style
> success/failure memory + ARIS-style claim-to-evidence auditing.**

High performance does not come from putting several chat agents into a group conversation.
It comes from:

1. A trustworthy evaluator.
2. Explicit search over multiple hypotheses.
3. Separation of strategy, implementation, and evaluation.
4. Isolated and reproducible execution.
5. Persistent but compressed experiment memory.
6. blind validation and anti-overfitting controls.
7. A report generator that only makes claims backed by logged evidence.

Benchmarks show that end-to-end autonomy is still unstable (ResearchClawBench strongest
agent 21.5/50, AutoResearchBench deep discovery 9.39%, PaperBench strongest 21.0%).
Therefore the right deployment model for 2026 is **bounded autonomous experimentation with
strategic human gates**, not unbounded autonomous publication.

---

## 1. SOTA framework summary

### 1.1 Arbor — Hypothesis-Tree Refinement (HTR)
A long-lived coordinator + short-lived executors + a persistent HTR structure. Each tree
node bundles hypothesis / artifact (Git branch) / observations / validation status /
distilled lessons / parent and children. The executor runs an experiment in an isolated
Git worktree and then terminates. The key ideas: **persistent search state (failed branches
are preserved as evidence too), short-lived workers (prevent unbounded context growth),
artifact lineage, insight back-propagation, separation of the development evaluator from the
admission evaluator.** MLE-Bench Lite 86.36% Any Medal (GPT-5.5, 2026-06 preprint).
Ablation: removing the tree/insight degrades performance substantially → structured memory
and search are the source of the gains.
→ **Use HTR as the top-level control structure.**

### 1.2 Gome — "Reasoning as Gradient"
The better the base model is at diagnosing failures, the more inefficient exhaustive tree
expansion becomes. It maps optimization concepts: gradient (a structured diagnosis of what
should change), momentum (remembering the direction of successful updates), distributed
optimization (multiple independent reasoning processes that implement related updates),
learning rate (the magnitude of the code change). full MLE-Bench 35.1% any-medal (12h, V100).
The key idea: after an experiment, extract a structured "update vector" (observed failure →
root cause → directional update → evidence) to guide several related implementations.
→ **Macro is HTR; inside a promising branch, Gome-style directional updates.**

### 1.3 ERA — Empirical Research Assistance (Nature)
The LLM rewrites empirical software, generates several candidates, and uses tree search to
decide which candidate to explore further. It combines methods from the literature to
generate new executable solutions. Beat published results on 8 of 9, with the best case
+14%. Core principle: **make the composition of scientific artifacts, evaluators, and
candidates explicit** (e.g. combining A's preprocessing + B's objective function + C's
optimizer + a new regularizer). A strong template for scorable computational science.

### 1.4 SciNav (ICLR 2026)
When no trustworthy single scalar metric exists: tree search + **pairwise relative judgment**
+ top-K branch selection. "Given the research contract, evidence, code diff, and artifacts,
which candidate is more scientifically sound? A/B/indistinguishable/both invalid" — more
reliable than an unconstrained 1–10 score.
→ **When there is no deterministic evaluator, pairwise ranking + hard validity checks +
multiple blind judges.**

### 1.5 AI Scientist v2
Removes reliance on human templates. Uses progressive agentic tree search to go from
idea → implementation → experiment → analysis/visualization → manuscript → automated review.
Improves figures with VLM feedback. Submitted 3 autonomous papers to an ICLR workshop, 1 of
which exceeded the average acceptance bar (workshop acceptance rate 70%). Lesson: **removing
human structure increases generality but decreases reliability.** Use it as a reference for
workflow coverage and document generation, but replace the weak parts with stricter evidence
and evaluation services.

### 1.6 AutoResearchClaw
Structured multi-agent debate + self-healing executor + **Pivot/Refine decisions** +
verifiable results/citation reporting + cross-run memory of failures. 7 human-intervention
modes (including confidence-based pause). +54.7% over AI Scientist v2 on ARC-Bench (its own
benchmark, 2026-05 preprint). The key idea: **Refine (the hypothesis is still plausible →
fix the implementation/design) vs Pivot (the evidence refutes the hypothesis → a
scientifically different branch).** Runtime error → repair. A valid negative result →
scientific revision/pivot (not debugging).

### 1.7 EvoScientist / ARIS
- **EvoScientist**: **separates** idea memory (promising directions, rejected ideas,
  novelty/feasibility lessons) from experiment memory (preprocessing, model, training,
  debugging, evaluation strategies) → prevents coding tricks from being mistaken for
  scientific evidence.
- **ARIS**: the executor and the reviewer are **different model families**. The assurance
  layer = integrity verification + result-to-claim mapping + auditing manuscript claims
  against raw evidence and the claim ledger + mathematical checks + visual inspection of the
  rendered paper. Reusable skills + a research-wiki memory.
→ **Adopt EvoScientist's memory separation + ARIS's claim ledger.**

### 1.8 Co-Scientist (Nature)
Continuously generates, critiques, ranks, and evolves hypotheses. Asynchronous execution +
a tournament to concentrate test-time compute on promising directions. Roles: Generation,
Reflection, Ranking, Evolution, Proximity, Meta-review. A scientist-in-the-loop collaborator.
→ **Borrow the tournament + meta-review for the ideation front-end, and connect it to the
independent execution system.**

### 1.9 Robin
Integrates literature-search + data-analysis agents. An example of a successful scientific
system that includes an external experimental loop (wet-lab). Domain-specific — do not treat
it as a general coding-agent framework.

### 1.10 R&D-Agent / AIDE / Karpathy / AlphaEvolve
- **R&D-Agent** (MS): separates the Research Agent (ideas, diagnosis) from the Development
  Agent (implementation, execution, runtime errors). full MLE-Bench 30.22%.
- **AIDE**: treats candidates as a code-solution tree, iterating draft/debug/improve.
  **One meaningful change at a time, with a summary of the branch state** (do not keep an
  unbounded conversation).
- **Karpathy autoresearch**: only `train.py` is editable, `prepare.py` and the evaluation
  are read-only, a fixed 5-minute budget, a single metric `val_bpb`, improvements are kept
  and regressions are reverted. **The clearest demonstration of evaluator-first
  autoresearch.** ← the direct basis for Phase 1 of this project.
- **AlphaEvolve**: LLM mutation + an automated evaluator + evolutionary selection.

---

## 2. Recommended architecture — 9 layers

```
research goal → [1]research contract → [2]literature/evidence engine → research-question certificate
→ [3]hypothesis portfolio → [4]search manager/coordinator
→ multiple short-lived coding executors → [6]visible development evaluator
→ [7]experiment ledger → insight/failure distillation → (feedback to the coordinator)
→ [6]blind admission gate → on approval, best artifact/main
→ [6]clean reproduction/seeds/ablation → final untouched evaluation → [8]claim-evidence ledger
→ report → [9]adversarial review + human approval
```

### Layer 1 — Research contract (implemented: research_contract.yaml)
A typed, versioned contract. objective, primary_metric{name,direction,minimum_effect},
secondary_metrics, baseline, editable_globs, protected_globs, budgets,
validation{search/admission/final split}, stop_conditions. Read-only mount inside the
executor.

### Layer 2 — Evidence/literature engine (Phase 3 target, not yet implemented)
Not simply retrieving 10 similar abstracts, but building an **evidence graph**:
1. Decompose the research question into concepts, mechanisms, methods, datasets, and outcome terms
2. Generate lexical/semantic/citation/author queries
3. Search across multiple indexes
4. Canonicalize with stable IDs such as DOI/PubMed/arXiv
5. Search full text where permitted
6. Extract claim-level evidence with page/section/table/figure locators
7. Trace references and citing papers
8. Identify supporting/contradicting/adjacent work
9. Report novelty based on the nearest prior claim (not an unfounded score)
10. Stop when evidence coverage stabilizes, not at an arbitrary number of searches

evidence record schema:
```json
{
  "evidence_id": "ev_0142",
  "canonical_paper_id": "doi:10.1234/example",
  "claim": "Method A improves minority-class recall under label imbalance.",
  "stance": "supports",
  "locator": {"section": "4.2", "table": "Table 3", "pages": [7, 8]},
  "population_or_dataset": "Dataset X",
  "conditions": "imbalance ratio >= 10:1",
  "limitations": ["single-site dataset", "no calibration analysis"]
}
```
Stack: PaperQA2 (local/curated full-text) + Semantic Scholar/OpenAlex/Crossref/
arXiv/PubMed/Europe PMC (discovery) + Ai2 Asta/ScholarQA + pgvector/Qdrant/Vespa/
OpenSearch (vector) + graph/relation edge tables (citation and claim relations).
**Isolate literature text so it cannot directly execute code or shell.**

### Layer 3 — Hypothesis certificate (implemented: Hypothesis certificate)
A falsifiable hypothesis:
```json
{
  "hypothesis_id": "h_008",
  "statement": "...", "mechanism": "...", "intervention": "...",
  "predicted_observations": {"macro_f1": "no decrease > 0.002", "ece": "-10%"},
  "falsifier": "ECE does not improve across >= 3 seeds.",
  "minimal_decisive_test": "...",
  "supporting_evidence_ids": ["ev_0142"], "nearest_prior_work": ["doi:..."],
  "risk": "low", "estimated_cost": 0.3
}
```

### Layer 4 — Hybrid search manager (portfolio implemented in Phase 2)
- Macro: maintain several scientifically distinct branches (no six micro-variants of the same idea).
- Branch-local: Gome-style directional updates (observation → root cause → direction → bounded change), momentum.
- Non-scalar: pairwise top-K, Pareto frontier.
- acquisition: `a(h) = E[ΔM] + β·uncertainty + γ·novelty − λ·cost − ρ·risk`.
- Initial policy: 8–16 initial hypotheses, 4–8 parallel branches, cheap proxy evaluations
  first, successive halving, 60–80% exploit / 20–40% explore, 2–3 repairs, 3–5 finalist seeds.

### Layer 5 — Coding executor (patcher + ClaudeCoder implemented in Phase 2; execution OS isolation in Phase 6a)
Short-lived, isolated (worktree+container), one hypothesis, cannot modify the evaluator or
hidden data, cannot merge directly to main, returns a structured result. The context packet:
the contract, the hypothesis, the repo map, the current baseline, literature evidence, a few
distilled lessons, allowed paths/commands, the admission criteria. **Must NOT receive**:
hidden gate data, final test results, secrets, the full raw conversation, unrelated
experiments, authority to change the goal.

**Phase 6a execution isolation**: wrap the point where untrusted candidate code is actually
*executed* (`_run_train` in `evaluate.py` — scoring stays on the host trust path) in a
`Sandbox` protocol. `subprocess` (identical to current behavior, default) +
`ContainerSandbox` (`docker run` — network blocked, read-only rootfs, dropped caps, non-root,
resource limits, ephemeral PID namespace, held-out seed/ledger masking). If the Docker
daemon/image is absent, fail-closed rather than silently falling back. On Linux, gVisor as a
drop-in via `--runtime=runsc`; Firecracker and containerizing the coder itself are follow-ups.
(Contract v6 `sandbox` block.)

### Layer 6 — Evaluation service (core; 3-stage implementation in Phase 1+2)
- **Development evaluator**: visible to the coordinator/executor, detailed diagnostics, for search.
- **blind admission gate**: a separate service, does not expose hidden labels, full logs, or
  per-example feedback; only selects whether generalization is better than the incumbent.
- **Final untouched evaluator**: only after the final candidate is frozen.
- A 3-way split is safer than repeatedly using a single hidden set.
- Protections: read-only mount, protected-file hashes (before/after), diff allowlist, egress
  blocked by default, fixed CPU/GPU/RAM/wall-clock, pinned image + lockfile, data
  hashes/lineage, structured metrics.json (not prose extraction), NaN/leakage/duplicate/
  degenerate checks, clean reproduction from the committed artifact. Do not use an LLM judge
  as the primary evaluator when a deterministic test is possible.

### Layer 7 — Experiment/insight memory (ledger+insight implemented in Phase 1+2)
| Memory | Contents |
|---|---|
| Evidence | papers, claims, contradictions, citations |
| Hypothesis | hypothesis, parent, status, falsifier |
| Experiment | commit, environment, metrics, seeds, artifacts |
| Insight | distilled reusable lessons from successes/failures |
Do not use raw chat transcripts as long-term memory. A valid failed experiment is evidence too.

### Layer 8 — claim-evidence ledger (Phase 5 target, not yet implemented)
Built before writing the report:
```json
{
  "claim_id": "claim_031",
  "text": "The proposed method improves macro-F1 by 1.2 percentage points.",
  "status": "verified",
  "supporting_runs": ["run_220","run_221","run_222"],
  "baseline_runs": ["run_012","run_013","run_014"],
  "effect_size": 0.012, "confidence_interval": [0.007, 0.017],
  "statistical_test": "paired bootstrap",
  "supporting_literature": ["ev_0142"], "limitations": ["single dataset family"]
}
```
The report generator can insert numbers only by referencing this ledger. Tables and figures
are generated directly from immutable logged artifacts.

### Layer 9 — human gates (Phase 5 target, not yet implemented)
Human approval for changing the research question, purchasing/collecting data, starting
expensive compute, wet-lab/robotics, biomedical/chemical/safety-sensitive experiments,
public release, novelty/clinical claims, and final manuscript submission. Intervening at
high-leverage points is better than either fully autonomous or fully micro-managed operation
(AutoResearchClaw).

---

## 3. Coding agent loop (implemented in Phase 1+2)

```
SELECT hypothesis → create an isolated worktree from the current incumbent → PLAN one coherent intervention
→ PATCH → VERIFY the protected/dependency policy → LINT/TYPE/UNIT/SMOKE
→ REPAIR execution failures (bounded) → RUN the development experiment on a fixed budget → PARSE structured metrics
→ CLASSIFY (invalid implementation / valid negative / valid inconclusive / valid positive) → COMMIT the artifact + provenance
→ DISTILL the insight into the hypothesis tree → send the top candidates to the blind admission gate → MERGE only on approval
```
**Repairable**: syntax, imports, shapes, config, runtime exceptions, artifact paths.
**Not repairable**: a refuted hypothesis, a failed statistical result, absence of novelty, a
metric regression from a correctly executed intervention → these are returned to the
coordinator. **Atomic intervention**: one coherent intervention per experiment (do not
change architecture, optimizer, augmentation, loss, batch, lr, calibration simultaneously).

---

## 4. Practical stack
- **Orchestration**: (the document recommends LangGraph) — **this project is implemented
  with the Claude Agent SDK for Python** (user decision). Uses the SDK runtime instead of
  LangGraph.
- **Coding backend**: OpenHands SDK (production worker) / mini-swe-agent (a minimal, auditable
  worker) — this project uses ClaudeCoder (Agent SDK).
- **Literature**: PaperQA2 + Semantic Scholar/OpenAlex/Crossref/arXiv/PubMed + a claim-level
  evidence graph.
- **State/artifacts**: PostgreSQL + pgvector/Qdrant + S3/MinIO + Git + MLflow
  (this project implements it with local files/JSONL/git in Phase 1–2; the stack above for
  production scale-up).
- **Execution infrastructure**: Docker/gVisor/Kata/Firecracker + K8s/Slurm/Ray + Git worktrees
  + OCI digest + uv/Poetry/Conda/Nix lock + DVC/lakeFS. **Egress blocked by default;
  literature search goes through a separate controlled service.**

---

## 6. Evaluation strategy (self-metrics instrumented in Phase 5)
| Capability | Benchmark |
|---|---|
| Literature discovery | AutoResearchBench |
| Literature understanding | AstaBench |
| Scientific Python generation | ScienceAgentBench |
| ML engineering | full MLE-Bench |
| Paper reproduction | PaperBench |
| end-to-end rediscovery | ResearchClawBench |
| Own domain | a private hidden benchmark based on past projects |
Metrics to record: valid-experiment rate, code-execution rate, improvement rate,
blind-gate acceptance rate, final-test improvement, reproduction rate, citation
precision/recall, claim-evidence consistency, cost/compute per accepted improvement,
number of human interventions, number of distinct hypotheses explored, fraction of
correctly preserved negative results, evaluator/data-policy violation rate.

---

## 7. Common failure modes (what the design defends against)
- **Evaluator hacking**: raising the score without improving the artifact. → hidden
  evaluator, protected hashes, separate process, invariant checks, resource accounting,
  manual adversarial testing.
- **Dev-set overfitting**: hundreds of experiments against the same dev data. → blind gate,
  untouched final set, periodic refresh, nested validation, experiment-count-aware
  statistical correction.
- **Hypothesis collapse**: all branches are micro-variants of the same approach. →
  diversity-aware selection, mechanism-level clustering, a novelty term, a minimum allocation
  to exploration branches.
- **Context contamination**: contradictory plans accumulate in a long-lived coding agent. →
  short-lived executors, compressed branch context.
- **False repair**: mistaking a valid negative result for a coding failure and fixing it
  until it turns positive. → distinguish execution validity from the scientific result,
  freeze the meaning of the hypothesis and evaluator within an experiment.
- **Citation laundering**: papers that are related but do not support the claim. →
  claim-level evidence extraction, precise locators, contradiction search, independent
  citation verification.
- **Manuscript-first**: a compelling narrative before trustworthy results. → do not generate
  a manuscript before the experiment and claim-ledger coverage is satisfied.
- **Multi-agent consensus without truth**: multiple agents reinforcing the same unfounded
  assumption. → independent evidence, deterministic evaluation, heterogeneous reviewer
  models, adversarial roles, blind candidate labels.

---

## 8. Implementation path (Phase definitions — the roadmap for this project)
- **Phase 1 (done)**: constrained-execution autoresearch. 1 repository, 1 immutable
  evaluator, 1 scalar metric, 1 coding worker, worktree, structured logging, a fixed budget.
  Karpathy-style keep/reject first.
- **Phase 2 (done)**: hypothesis portfolio. Typed certificates, 4–8 parallel branches, a
  coordinator, failure classification, insight distillation, blind admission evaluation. →
  an Arbor-type loop.
- **Phase 3**: literature grounding. PaperQA2 or an equivalent search service, canonical
  paper IDs, claim-level evidence, citation-graph traversal, novelty/contradiction reporting.
  **Prevent literature text from executing code or shell.**
- **Phase 4**: directional branch refinement. Gome-style diagnostic updates, momentum memory,
  SciNav-style pairwise selection (qualitative artifacts), cost-aware scheduling +
  successive halving.
- **Phase 5**: scientific assurance + reporting. Multiple seeds + confidence intervals,
  baseline + ablation, claim-evidence ledger, deterministic figure generation, cross-model
  adversarial reviewer, human approval for novelty/publication.
- **Phase 6a**: execution isolation (deepening Layer 5). OS-isolate the running of the
  candidate `train.py` with a `Sandbox` protocol + a Docker `ContainerSandbox` (network
  blocked, read-only, seed/ledger masking, ephemeral PID). `subprocess` default, `container`
  opt-in, fail-closed. gVisor as a `--runtime=runsc` drop-in. (Next: 6b real literature API
  OpenAlex/S2 + snapshot cache, 6c swap in a real research domain.)

## Final recommendation (original text)
```
Orchestrator: LangGraph + PostgreSQL checkpointing   ← this project: Claude Agent SDK
Research policy: Arbor HTR + Gome branch-local updates + SciNav pairwise
Coding workers: OpenHands SDK / mini-swe-agent        ← this project: ClaudeCoder(SDK)
Literature: PaperQA2 + Semantic Scholar/OpenAlex/... + claim-level evidence graph
Execution: Git worktrees + Docker/gVisor/Firecracker + K8s/Slurm/Ray
Tracking: MLflow + PostgreSQL + S3/MinIO + dataset/OCI hashes
Assurance: immutable dev evaluator + blind admission gate + untouched final test
           + claim-evidence ledger + independent reviewer model
Human control: approval gate for scope changes, high compute, new data, safety-sensitive, novelty, publication
```

> **The most important rule: build the evaluator, the provenance model, and the experiment
> contract before the agent swarm.** Without a trustworthy way to distinguish valid progress
> from plausible failure, adding more agents and test-time compute only makes things more
> expensive and more persuasive, not more scientific.
