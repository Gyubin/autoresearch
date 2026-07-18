"""Phase 3 drills: literature grounding — corpus integrity, determinism,
blindness of the evidence path, anti-laundering, injection confinement,
coverage stop, fallback semantics, contract v3.

Run from the repo root:  uv run python tests/test_phase3.py
Self-contained checks (no pytest), same conventions as tests/test_phase2.py.
Everything here is offline: no SDK call is ever made (LLM paths are stubbed).
"""

from __future__ import annotations

import asyncio
import copy
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import orchestrator as orch  # noqa: E402
from literature.engine import (  # noqa: E402
    ANTI_INJECTION_SENTENCE,
    ClaudeLiteratureAnalyst,
    CorpusError,
    EvidenceEngine,
    FallbackAnalyst,
    load_corpus,
)

FAILS: list[str] = []
CORPUS_PATH = ROOT / "literature" / "corpus" / "tsp_corpus.json"
INJECTION_MARKER = "SYSTEM NOTE TO AUTOMATED REVIEWERS"
GATE_CANARY = "0.424242"

# TSP HYPERPARAMS (the patcher surface for the Phase 6c domain).
HP = {"use_nn_construction": True, "max_iterations": 20000, "restarts": 1,
      "initial_temperature": 0.0, "cooling_rate": 0.995, "segment_max": 3,
      "perturbation_strength": 4}


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "ok  " if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def make_engine(**overrides) -> EvidenceEngine:
    kwargs = dict(max_evidence_per_generation=12,
                  max_evidence_per_hypothesis=4, max_queries=20,
                  stabilization_window=2, citation_hops=1)
    kwargs.update(overrides)
    return EvidenceEngine(load_corpus(CORPUS_PATH), **kwargs)


def ground(engine: EvidenceEngine, **overrides):
    kwargs = dict(objective="Minimize the mean held-out Euclidean-TSP tour "
                            "length produced by src/train.py.",
                  hyperparams=dict(HP),
                  insights=[], best_primary_dev=0.38, tested={})
    kwargs.update(overrides)
    return engine.ground(**kwargs)


def mk_hyp(hid: str, param: str | None, frm=None, to=None, statement="",
           executor: str = "patcher", brief: str = "",
           cited: list[str] | None = None) -> orch.Hypothesis:
    return orch.Hypothesis(
        id=hid, round=1, statement=statement or f"change {param}",
        mechanism="test mechanism", predicted_effect="improves",
        falsifier="no improvement", minimal_test="one dev eval",
        proposer="heuristic", executor=executor,
        implementation_brief=brief,
        intervention=({"param": param, "from": frm, "to": to,
                       "kind": param or executor} if param
                      else {"param": None, "from": None, "to": None,
                            "kind": "coder"}),
        supporting_evidence_ids=list(cited or []),
    )


# ---------------------------------------------------------------------------
# T1 — corpus integrity + corruption fail-fast
# ---------------------------------------------------------------------------

def test_corpus_integrity(tmp: Path) -> None:
    corpus = load_corpus(CORPUS_PATH)
    check("corpus: 13 papers / 13 claims",
          len(corpus.papers) == 13 and len(corpus.claims) == 13,
          f"{len(corpus.papers)}/{len(corpus.claims)}")
    rel = [r for r in corpus.claims["cl_0601"].get("relations") or []
           if r.get("type") == "contradicts" and r.get("to_claim") == "cl_0501"]
    check("corpus: acceptance contradiction pair annotated", bool(rel))
    check("corpus: injection fixture flagged",
          corpus.claims["cl_1301"].get("fixture") == "prompt_injection"
          and INJECTION_MARKER in corpus.claims["cl_1301"]["claim"])
    traversal_paper = corpus.papers["tsp:2021.0012"]
    searchable = (traversal_paper["title"] + traversal_paper["abstract"]
                  + " ".join(s.get("text", "") for s in traversal_paper["sections"])
                  + " ".join(traversal_paper["concepts"])
                  + corpus.claims["cl_1201"]["claim"]).lower()
    check("corpus: traversal-only paper avoids lexical tokens",
          "tour" not in searchable and "local" not in searchable
          and "2-opt" not in searchable)

    raw = json.loads(CORPUS_PATH.read_text())

    def expect_error(name: str, mutate) -> None:
        bad = copy.deepcopy(raw)
        mutate(bad)
        path = tmp / f"{name}.json"
        path.write_text(json.dumps(bad))
        try:
            load_corpus(path)
            check(f"corpus: {name} rejected", False, "no CorpusError raised")
        except CorpusError:
            check(f"corpus: {name} rejected", True)

    truncated = tmp / "truncated.json"
    truncated.write_text(CORPUS_PATH.read_text()[:200])
    try:
        load_corpus(truncated)
        check("corpus: truncated JSON rejected", False)
    except CorpusError:
        check("corpus: truncated JSON rejected", True)

    expect_error("duplicate claim_id",
                 lambda c: c["claims"].append(copy.deepcopy(c["claims"][0])))
    expect_error("dangling reference",
                 lambda c: c["papers"][0]["references"].append("mock:9999.9999"))
    expect_error("metric-like decimal in claim text",
                 lambda c: c["claims"][0].__setitem__(
                     "claim", "improves heldout rmse to 0.4242 exactly"))
    expect_error("metric-like decimal in population field",
                 lambda c: c["claims"][0].__setitem__(
                     "population_or_dataset", "regression with noise 0.4242"))

    def add_slug_twin(c):
        twin = copy.deepcopy(c["claims"][0])
        twin["claim_id"] = c["claims"][0]["claim_id"].replace("_", ".")
        c["claims"].append(twin)
    expect_error("evidence-id slug collision", add_slug_twin)


# ---------------------------------------------------------------------------
# T2 — deterministic reproducibility
# ---------------------------------------------------------------------------

def test_deterministic_reproducibility() -> None:
    a = ground(make_engine()).to_bundle()
    b = ground(make_engine()).to_bundle()
    check("determinism: byte-identical grounding bundles",
          json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True))
    hyp_a = mk_hyp("h_x", "max_iterations", 20000, 50000)
    hyp_b = mk_hyp("h_x", "max_iterations", 20000, 50000)
    ga, gb = ground(make_engine()), ground(make_engine())
    make_engine().attach([hyp_a], ga)
    make_engine().attach([hyp_b], gb)
    check("determinism: attach output identical",
          json.dumps(hyp_a.to_dict(), sort_keys=True)
          == json.dumps(hyp_b.to_dict(), sort_keys=True)
          and json.dumps(ga.novelty, sort_keys=True)
          == json.dumps(gb.novelty, sort_keys=True))


# ---------------------------------------------------------------------------
# T3 — evidence records are inert to insight distillation and replay
# ---------------------------------------------------------------------------

def test_evidence_record_inert() -> None:
    evidence_record = {"record_type": "evidence", "generation": 1,
                       "cost_usd": 0.42, "evidence": [{"evidence_id": "ev_x",
                                                       "score": 0.48}]}
    check("inert: distill_insight(evidence) is None",
          orch.distill_insight(evidence_record) is None)

    def exp(run_id, gen, param, to, decision, verdict="valid_positive"):
        return {"record_type": "experiment", "run_id": run_id,
                "generation": gen, "verdict": verdict, "decision": decision,
                "hypothesis": {"intervention": {"param": param, "from": 0,
                                                "to": to}}}
    base = [exp("r0001", 1, "lr", 1, "accept"),
            exp("r0002", 1, "epochs", 2, "reject"),
            exp("r0003", 2, "l2", 3, "reject", "valid_inconclusive")]
    with_evidence = [evidence_record, base[0], base[1],
                     {"record_type": "evidence", "generation": 2}, base[2]]
    s1: dict = {}
    s2: dict = {}
    orch.replay_ledger_fields(s1, base)
    orch.replay_ledger_fields(s2, with_evidence)
    check("inert: replay unchanged by interleaved evidence records", s1 == s2,
          f"{s1} != {s2}")


# ---------------------------------------------------------------------------
# T4 — hypotheses carry ids only; blindness literal-scan survives
# ---------------------------------------------------------------------------

def test_hypothesis_ids_only() -> None:
    engine = make_engine()
    grounding = ground(engine)
    claim_sentinel = "Metropolis"  # distinctive corpus claim prose
    hyps = [mk_hyp("h_r0001_use_nn_construction", "use_nn_construction", False, True),
            mk_hyp("h_r0002_coder", None,
                   statement="swap the local-search neighborhood operator",
                   executor="coder",
                   brief="change NEIGHBORHOOD from two_opt to or_opt exchange")]
    engine.attach(hyps, grounding)
    for hyp in hyps:
        text = json.dumps(hyp.to_dict())
        check(f"ids-only: {hyp.id} carries no claim prose / gate literals",
              claim_sentinel not in text and "0.48" not in text
              and "0.37" not in text and INJECTION_MARKER not in text)
    check("ids-only: construction hypothesis cites the supporting claim",
          "ev_cl-0101" in hyps[0].supporting_evidence_ids)
    check("ids-only: coder hypothesis cites the neighborhood claim",
          "ev_cl-0301" in hyps[1].supporting_evidence_ids)

    record = {"record_type": "experiment", "run_id": "r0001", "generation": 1,
              "verdict": "valid_positive", "decision": "reject",
              "best_primary_before": 0.50, "primary": 0.39,
              "hypothesis": hyps[0].to_dict()}
    ins = orch.distill_insight(record)
    check("ids-only: insight literal-scan clean",
          ins is not None and "0.48" not in json.dumps(ins)
          and "0.37" not in json.dumps(ins)
          and claim_sentinel not in json.dumps(ins))


# ---------------------------------------------------------------------------
# T5 — gate canary: no output surface can contain a gate score
# ---------------------------------------------------------------------------

def test_gate_canary() -> None:
    # A gate score exists in the process (as it would in state.json), but the
    # engine's input closure cannot receive it: prove the canary never
    # reaches any literature output surface.
    _fake_state = {"gate": {"incumbent_scores": {"c0ffee": 0.424242}}}
    engine = make_engine()
    grounding = ground(engine)
    hyps = [mk_hyp("h_r0001_temp", "initial_temperature", 0.0, 0.5)]
    engine.attach(hyps, grounding)
    surfaces = {
        "bundle": json.dumps(grounding.to_bundle()),
        "proposer_view": json.dumps(grounding.proposer_view()),
        "hypothesis": json.dumps(hyps[0].to_dict()),
        "coder_packet": json.dumps(grounding.for_hypothesis(hyps[0])),
        "certificate": json.dumps(engine.question_certificate(grounding)),
        # Phase 4: the structured steering surface is proposer-visible too.
        "move_guidance": json.dumps(grounding.move_guidance()),
    }
    contract = orch.load_contract()
    ctx = orch.ProposalContext(
        contract=contract, round_index=1,
        current_hyperparams={"max_iterations": 20000, "initial_temperature": 0.0},
        best_primary=0.38, tested={}, last_accepted=None, insights=[],
        evidence=grounding.proposer_view())
    surfaces["proposer_prompt"] = orch.ClaudeProposer()._prompt(ctx, 4, 1, None)
    coder = orch.ClaudeCoder(contract)
    surfaces["coder_prompt"] = coder._initial_prompt(
        hyps[0], {"insights": [], "evidence": grounding.for_hypothesis(hyps[0])})
    for name, text in surfaces.items():
        check(f"canary: gate literal absent from {name}",
              GATE_CANARY not in text)
    assert _fake_state  # the canary object stays alive through the drill


# ---------------------------------------------------------------------------
# T6 — citation whitelist (proposer validation + attach re-validation)
# ---------------------------------------------------------------------------

def test_citation_whitelist() -> None:
    contract = orch.load_contract()
    engine = make_engine()
    grounding = ground(engine)
    ctx = orch.ProposalContext(
        contract=contract, round_index=1,
        current_hyperparams=dict(HP, use_nn_construction=False),
        best_primary=0.38, tested={}, last_accepted=None, insights=[],
        evidence=grounding.proposer_view())
    raw = {"statement": "enable nearest-neighbor construction for a better start",
           "mechanism": "construction basin", "executor": "patcher",
           "param": "use_nn_construction", "new_value": True,
           "implementation_brief": None, "predicted_effect": "shorter tour",
           "falsifier": "no change",
           "supporting_evidence_ids": ["ev_cl-0101", "ev_FAKE_0.48"]}
    validated = orch.ClaudeProposer()._validate_item(raw, ctx, set(), 0, 1)
    check("whitelist: hallucinated id dropped, real id kept",
          not isinstance(validated, str)
          and validated.supporting_evidence_ids == ["ev_cl-0101"],
          str(validated))

    hyp = mk_hyp("h_r0009_restarts", "restarts", 1, 2, cited=["ev_BOGUS"])
    engine.attach([hyp], grounding)
    report = grounding.novelty["per_hypothesis"]["h_r0009_restarts"]
    check("whitelist: attach strips unknown ids",
          "ev_BOGUS" not in hyp.supporting_evidence_ids
          and any(e["evidence_id"] == "ev_BOGUS"
                  and e["reason"] == "unknown_id"
                  for e in report["laundering_filtered"]))

    schema = orch.ClaudeProposer()._schema({"lr": 0.005}, 4, True,
                                           ["ev_a", "ev_b"])
    items = schema["properties"]["hypotheses"]["items"]
    cited = items["properties"]["supporting_evidence_ids"]
    check("whitelist: schema enum pins the id namespace",
          cited["items"].get("enum") == ["ev_a", "ev_b"])
    empty = orch.ClaudeProposer()._schema({"lr": 0.005}, 4, True, [])
    cited_empty = (empty["properties"]["hypotheses"]["items"]
                   ["properties"]["supporting_evidence_ids"])
    check("whitelist: empty pack yields NO enum (SDK strict-mode guard)",
          "enum" not in cited_empty["items"])


# ---------------------------------------------------------------------------
# T7 — anti-laundering: adjacent claims can never become "supports"
# ---------------------------------------------------------------------------

def test_anti_laundering() -> None:
    engine = make_engine()
    grounding = ground(engine)
    # cl_1201 is an off-model-class (general_combinatorial) neighborhood claim,
    # reached only via citation traversal: same family as a segment_max
    # hypothesis but NOT on the euclidean_tsp task -> adjacent, not supporting.
    trap = grounding.record("ev_cl-1201")
    check("laundering: trap claim retrieved but adjacent",
          trap is not None and trap["stance"] == "adjacent")

    hyp = mk_hyp("h_r0001_segment_max", "segment_max", 3, 4,
                 cited=["ev_cl-1201"])
    engine.attach([hyp], grounding)
    report = grounding.novelty["per_hypothesis"]["h_r0001_segment_max"]
    check("laundering: forged citation stripped with stance reason",
          "ev_cl-1201" not in hyp.supporting_evidence_ids
          and any(e["evidence_id"] == "ev_cl-1201"
                  and e["reason"].startswith("stance_")
                  for e in report["laundering_filtered"]))

    # A cross-model-class pair is not a genuine contradiction: the off-task
    # claim must never be paired against an on-model-class result.
    check("laundering: no cross-model-class contradiction surfaced",
          not any("ev_cl-1201" in c["evidence_ids"]
                  for c in grounding.contradictions))

    # Coder-family matching: multiword keywords need contiguity, and the
    # (batch_size vs lr) tie the old subset matcher lost is now resolved to
    # the correct family.
    coder_hyp = mk_hyp(
        "h_r0031_coder", None, executor="coder",
        statement="Relocate short segments during local search",
        brief="use a larger or-opt segment relocation neighborhood operator")
    fam, move = engine._coder_family(
        *engine._hypothesis_family(coder_hyp)[2:])
    check("coder-family: contiguous phrase matching picks neighborhood_operator",
          fam == "neighborhood_operator", f"got {fam}/{move}")

    # LLM stance verdicts may only downgrade: a forged "supports" for the
    # trap claim must be coerced away (deterministic rule is authoritative).
    analyst = ClaudeLiteratureAnalyst(engine)
    calls = {"n": 0}

    def fake_query(prompt: str, schema: dict) -> dict:
        calls["n"] += 1
        if "decompose" in prompt.lower() or "queries" in schema.get(
                "properties", {}):
            return {"queries": []}
        return {"judgments": [
            {"evidence_id": "ev_cl-1201", "stance": "supports"},   # grant (trap)
            {"evidence_id": "ev_cl-0101", "stance": "adjacent"},   # downgrade
            {"evidence_id": "ev_cl-0601", "stance": "adjacent"},   # sideways
        ], "narrative": "forged narrative"}
    analyst._query = fake_query  # type: ignore[method-assign]
    llm_grounding = ground(analyst)
    trap_llm = llm_grounding.record("ev_cl-1201")
    check("laundering: LLM 'supports' coerced (never granted)",
          trap_llm is not None and trap_llm["stance"] == "adjacent"
          and llm_grounding.coverage.get("llm_supports_coerced", 0) >= 1
          and llm_grounding.mode == "claude" and calls["n"] >= 2)
    downgraded = llm_grounding.record("ev_cl-0101")
    check("laundering: LLM may downgrade a supports",
          downgraded is not None and downgraded["stance"] == "adjacent")
    sideways = llm_grounding.record("ev_cl-0601")
    check("laundering: non-supports relabels ignored (certificate "
          "consistency)",
          sideways is not None and sideways["stance"] == "contradicts")


# ---------------------------------------------------------------------------
# T8 — novelty is categorical (no numeric novelty score anywhere)
# ---------------------------------------------------------------------------

def test_novelty_categorical() -> None:
    engine = make_engine()
    grounding = ground(engine)
    hyps = [
        mk_hyp("h_construction", "use_nn_construction", False, True),
        mk_hyp("h_restarts", "restarts", 1, 2),
        mk_hyp("h_unknown", None, executor="coder",
               statement="emit verbose diagnostic logging",
               brief="write extra debug values to a side channel during the run"),
    ]
    engine.attach(hyps, grounding)
    per = grounding.novelty["per_hypothesis"]
    check("novelty: nn-construction (matches enable/improves) -> replication",
          per["h_construction"]["novelty_category"] == "replication",
          per["h_construction"]["novelty_category"])
    check("novelty: unmatched coder brief -> unexplored",
          per["h_unknown"]["novelty_category"] == "unexplored",
          per["h_unknown"]["novelty_category"])
    check("novelty: all categories in the fixed enum",
          all(rep["novelty_category"] in
              ("replication", "regime_extension", "contradiction_test",
               "unexplored") for rep in per.values()))

    floats: list = []

    def scan(node) -> None:
        if isinstance(node, float):
            floats.append(node)
        elif isinstance(node, dict):
            for v in node.values():
                scan(v)
        elif isinstance(node, list):
            for v in node:
                scan(v)
    scan(grounding.novelty)
    check("novelty: no float anywhere in the report (categorical only)",
          not floats, str(floats))
    # A restarts-increase hypothesis meets the on-family restart-strategy claim.
    restart_relations = {c["relation"]
                         for c in per["h_restarts"]["nearest_prior_claims"]}
    check("novelty: restarts nearest priors are on-family",
          per["h_restarts"]["novelty_category"] in ("replication",
                                                    "regime_extension")
          and bool(restart_relations))


# ---------------------------------------------------------------------------
# T9 — citation traversal does real work
# ---------------------------------------------------------------------------

def test_citation_traversal() -> None:
    # "restart / multi-start" hits the multi-start paper (tsp:2018.0009)
    # lexically; the balanced-partition-tree paper it CITES (tsp:2021.0012)
    # avoids every query token by construction, so it is reachable only by hop.
    query = [{"topic": "restart_strategy",
              "query": "restart variance multi start"}]

    def ids_with_hops(hops: int) -> set[str]:
        engine = make_engine(citation_hops=hops,
                             max_evidence_per_generation=32)
        g = engine.ground_with_queries(
            query, objective="o", hyperparams={}, insights=[],
            best_primary_dev=None, tested={})
        return g.valid_ids()
    with_hops = ids_with_hops(1)
    without_hops = ids_with_hops(0)
    check("traversal: hop-1 surfaces the citation-only paper's claim",
          "ev_cl-1201" in with_hops)
    check("traversal: hops=0 cannot reach it lexically",
          "ev_cl-1201" not in without_hops, str(sorted(without_hops)))


# ---------------------------------------------------------------------------
# T10 — coverage stop fires before the query cap
# ---------------------------------------------------------------------------

def test_coverage_stop() -> None:
    grounding = ground(make_engine(max_queries=20))
    cov = grounding.coverage
    check("coverage: stabilization stop before the cap",
          cov["stopped_because"] == "coverage_stable"
          and cov["queries_run"] < 20,
          json.dumps(cov))


# ---------------------------------------------------------------------------
# T11 — injection confinement
# ---------------------------------------------------------------------------

def test_injection_confinement(tmp: Path) -> None:
    engine = make_engine()
    grounding = ground(engine)
    moves = [("use_nn_construction", False, True),
             ("max_iterations", 20000, 50000),
             ("restarts", 1, 2), ("initial_temperature", 0.0, 0.5),
             ("segment_max", 3, 4), ("perturbation_strength", 4, 8)]
    hyps = [mk_hyp(f"h_{p}", p, f, t) for p, f, t in moves]
    hyps.append(mk_hyp("h_coder", None, executor="coder",
                       statement="swap to an or-opt neighborhood operator",
                       brief="change NEIGHBORHOOD to or-opt segment relocation"))
    engine.attach(hyps, grounding)
    check("injection: fixture never in supporting_evidence_ids",
          all("ev_cl-1301" not in h.supporting_evidence_ids for h in hyps))
    check("injection: marker never in any coder packet",
          all(INJECTION_MARKER
              not in json.dumps(grounding.for_hypothesis(h)) for h in hyps))

    contract = orch.load_contract()
    ctx = orch.ProposalContext(
        contract=contract, round_index=1,
        current_hyperparams={p: f for p, f, _ in moves},
        best_primary=0.38, tested={}, last_accepted=None, insights=[],
        evidence=grounding.proposer_view())
    prompt = orch.ClaudeProposer()._prompt(ctx, 4, 1, None)
    check("injection: proposer prompt carries the anti-injection framing",
          ANTI_INJECTION_SENTENCE in prompt)
    if INJECTION_MARKER in prompt:
        check("injection: marker only after the framing sentence",
              prompt.index(ANTI_INJECTION_SENTENCE)
              < prompt.index(INJECTION_MARKER))

    # The tool calls the injected text asks for are denied by the coder
    # guard regardless (fail-closed) — reuse the Phase 2 harness.
    wt = tmp / "wt"
    (wt / "src").mkdir(parents=True)
    denials: list[dict] = []
    guard = orch._make_worktree_guard(wt, denials)

    def decide(tool: str, field: str, value: str) -> str:
        out = asyncio.run(guard({"tool_name": tool,
                                 "tool_input": {field: value}}, "tid", None))
        return out["hookSpecificOutput"]["permissionDecision"]
    check("injection: guard denies reading the held-out config",
          decide("Read", "file_path",
                 str(ROOT / "evaluation" / "heldout_config.json")) == "deny")
    check("injection: guard denies writing a new evaluator",
          decide("Write", "file_path",
                 str(ROOT / "evaluation" / "evaluate.py")) == "deny")
    check("injection: guard denies Bash outright",
          decide("Bash", "command", "cat evaluation/heldout_config.json")
          == "deny")


# ---------------------------------------------------------------------------
# T12 — empty retrieval, disabled parity, missing corpus
# ---------------------------------------------------------------------------

def test_empty_and_disabled(tmp: Path) -> None:
    engine = make_engine()
    g = engine.ground_with_queries(
        [{"topic": "generic", "query": "zzz qqq unrelated nonsense"}],
        objective="o", hyperparams={}, insights=[], best_primary_dev=None,
        tested={})
    hyp = mk_hyp("h_x", "max_iterations", 20000, 50000)
    engine.attach([hyp], g)
    check("empty: zero-hit grounding is a legal state",
          g.evidence == [] and hyp.supporting_evidence_ids == [])

    contract = orch.load_contract()
    ctx = orch.ProposalContext(
        contract=contract, round_index=1, current_hyperparams={"lr": 0.005},
        best_primary=0.38, tested={}, last_accepted=None, insights=[],
        evidence=[])
    prompt = orch.ClaudeProposer()._prompt(ctx, 4, 1, None)
    check("disabled: no evidence section when the pack is empty",
          ANTI_INJECTION_SENTENCE not in prompt)
    coder_prompt = orch.ClaudeCoder(contract)._initial_prompt(
        hyp, {"insights": [], "evidence": []})
    check("disabled: coder prompt clean without evidence",
          ANTI_INJECTION_SENTENCE not in coder_prompt)

    class Cfg:
        corpus_path = str(tmp / "missing_corpus.json")
        max_evidence_per_generation = 12
        max_evidence_per_hypothesis = 4
        max_queries = 6
        stabilization_window = 2
        citation_hops = 1
        llm_max_budget_usd = 0.5
    from literature.engine import build_engine
    try:
        build_engine(Cfg(), "lexical")
        check("missing corpus: fail-fast", False, "no CorpusError")
    except CorpusError:
        check("missing corpus: fail-fast", True)

    class RelCfg(Cfg):
        corpus_path = "literature/corpus/tsp_corpus.json"
    try:
        build_engine(RelCfg(), "lexical")
        check("relative corpus_path without root: fail-fast (no CWD "
              "resolution)", False, "no CorpusError")
    except CorpusError:
        check("relative corpus_path without root: fail-fast (no CWD "
              "resolution)", True)
    engine_rooted = build_engine(RelCfg(), "lexical", corpus_root=ROOT)
    check("relative corpus_path resolves against corpus_root",
          len(engine_rooted.corpus.claims) == 13)


# ---------------------------------------------------------------------------
# T13 — LLM fallback + campaign budget exhaustion
# ---------------------------------------------------------------------------

def test_llm_fallback_and_budget(tmp: Path) -> None:
    engine = make_engine()

    class Boom:
        def ground(self, **_kwargs):
            raise RuntimeError("usage limit reached")
    fallback = FallbackAnalyst(Boom(), engine)  # type: ignore[arg-type]
    g = ground(fallback)
    check("fallback: analyst failure degrades to lexical",
          g.mode == "claude+fallback" and len(g.evidence) > 0)

    log = tmp / "evidence.jsonl"
    log.write_text(json.dumps({"cost_usd": 5.0}) + "\n")
    original = orch.EVIDENCE_LOG_PATH
    orch.EVIDENCE_LOG_PATH = log
    try:
        guarded = orch._BudgetGuardedLiterature(Boom(), engine, 1.0)
        g2 = ground(guarded)
        check("budget: campaign cap degrades to lexical (tagged)",
              g2.mode == "claude+budget_exhausted")
        guarded_ok = orch._BudgetGuardedLiterature(
            FallbackAnalyst(Boom(), engine), engine, 100.0)
        g3 = ground(guarded_ok)
        check("budget: under cap the primary path is used",
              g3.mode == "claude+fallback")
    finally:
        orch.EVIDENCE_LOG_PATH = original


# ---------------------------------------------------------------------------
# T14 — contract v3 validation
# ---------------------------------------------------------------------------

def test_contract_literature(tmp: Path) -> None:
    contract = orch.load_contract()
    lit = contract.literature
    check("contract: literature block parsed",
          contract.schema_version == 8 and lit.enabled
          and lit.retriever == "lexical"
          and lit.corpus_path == "literature/corpus/tsp_corpus.json"
          and lit.llm_max_campaign_budget_usd is None)

    text = orch.CONTRACT_PATH.read_text()

    def expect_reject(name: str, mutated: str) -> None:
        path = tmp / (name.replace("/", "_").replace(" ", "_") + ".yaml")
        path.write_text(mutated)
        try:
            orch.load_contract(path)
            check(f"contract: {name} rejected", False, "no ContractError")
        except orch.ContractError:
            check(f"contract: {name} rejected", True)

    expect_reject("schema_version 2",
                  text.replace("schema_version: 8", "schema_version: 2"))
    expect_reject("corpus outside literature/",
                  text.replace(
                      'corpus_path: "literature/corpus/tsp_corpus.json"',
                      'corpus_path: "evaluation/heldout_config.json"'))
    expect_reject("corpus path traversal",
                  text.replace(
                      'corpus_path: "literature/corpus/tsp_corpus.json"',
                      'corpus_path: "literature/../evaluation/'
                      'heldout_config.json"'))
    expect_reject("unknown retriever",
                  text.replace("retriever: lexical", "retriever: pubmed"))
    ok_path = tmp / "no_retriever.yaml"
    ok_path.write_text(text.replace("  retriever: lexical\n", ""))
    check("contract: retriever defaults to lexical",
          orch.load_contract(ok_path).literature.retriever == "lexical")


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for sub in ("t1", "t11", "t12", "t13", "t14"):
            (tmp / sub).mkdir()
        test_corpus_integrity(tmp / "t1")
        test_deterministic_reproducibility()
        test_evidence_record_inert()
        test_hypothesis_ids_only()
        test_gate_canary()
        test_citation_whitelist()
        test_anti_laundering()
        test_novelty_categorical()
        test_citation_traversal()
        test_coverage_stop()
        test_injection_confinement(tmp / "t11")
        test_empty_and_disabled(tmp / "t12")
        test_llm_fallback_and_budget(tmp / "t13")
        test_contract_literature(tmp / "t14")

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): {FAILS}")
        return 1
    print("all Phase 3 unit drills passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
