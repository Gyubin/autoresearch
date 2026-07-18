"""Claim-evidence ledger builder (Phase 5, Blueprint Layer 8).

Every numeric assertion the report makes must trace to a claim here; the report
generator inserts numbers ONLY by claim reference. Claims are a DERIVED
artifact: build_claims is a pure function of (ledger, bootstrap result, evidence
index, disclosure, costs, contract metadata), so experiments/claims.jsonl is
rebuilt whole-file at report time (like insight_memory.json) and is byte-stable
given its inputs. Gate scores never enter — claims use dev + test numbers only.

Claim kinds, always emitted in this fixed order (so claim ids are stable):
  1. primary_effect     — incumbent vs baseline on the hidden test seeds (+CI)
  2. campaign_summary    — disclosure + cost arithmetic
  3. admitted_improvement — one per accepted, uncorrected experiment (dev only)
  4. negative_result     — one per parameter with a valid_negative experiment
  5. literature_grounding — one per accepted experiment citing evidence
"""

from __future__ import annotations

import json

CLAIMS_SCHEMA_VERSION = 1
_ROUND = 6  # decimals for the display `values` map (byte-stable across machines)


def _r(x):
    return round(x, _ROUND) if isinstance(x, (int, float)) else x


def _corrected(records: list[dict]) -> set[str]:
    return {r["corrects"] for r in records
            if r.get("record_type") == "correction" and r.get("corrects")}


def _claim(kind: str, text: str, status: str, **fields) -> dict:
    claim = {
        "schema_version": CLAIMS_SCHEMA_VERSION,
        "claim_id": "",  # assigned in order at the end
        "kind": kind,
        "text": text,
        "status": status,
        "supporting_runs": [],
        "baseline_runs": [],
        "effect_size": None,
        "confidence_interval": None,
        "statistical_test": None,
        "supporting_literature": [],
        "limitations": [],
        "values": {},
        "provenance": {},
    }
    claim.update(fields)
    return claim


def _statistical_test_text(boot) -> str:
    return (f"paired-example bootstrap, pooled across {boot.n_seeds} seed(s), "
            f"B={boot.resamples}, {int(round(boot.confidence * 100))}% "
            f"percentile CI, seed={boot.bootstrap_seed}")


def _primary_effect(boot, meta: dict) -> dict:
    aliased = bool(meta.get("incumbent_runs_aliased"))
    per_seed = [{
        "seed_index": s.seed_index,
        "rmse_baseline": _r(s.rmse_baseline),
        "rmse_incumbent": _r(s.rmse_incumbent),
        "delta": _r(s.delta),
        "failure_baseline": s.failure_baseline,
        "failure_incumbent": s.failure_incumbent,
    } for s in boot.per_seed]

    if not boot.clean:
        status = "unsupported"
    elif boot.ci_abs[0] > 0:
        status = "verified"
    elif boot.ci_abs[1] < 0:
        status = "refuted"
    else:
        status = "inconclusive"

    limitations = ["single synthetic Euclidean-TSP instance family (uniform "
                   "random points)"]
    if aliased:
        limitations.append("incumbent == baseline: no candidate was admitted, "
                           "so the reported effect is exactly zero")
    if not boot.clean:
        limitations.append("at least one test-seed run scored unclean; the "
                           "bootstrap was skipped (no seed dropping)")

    values = {
        "n_seeds": boot.n_seeds,
        "n_examples": boot.n_examples,
        "resamples": boot.resamples,
        "confidence_pct": int(round(boot.confidence * 100)),
        "bootstrap_seed": boot.bootstrap_seed,
        "effect_abs": _r(boot.effect_abs),
        "effect_rel_pct": _r(boot.effect_rel * 100) if boot.effect_rel is not None else None,
        "ci_lo": _r(boot.ci_abs[0]) if boot.ci_abs else None,
        "ci_hi": _r(boot.ci_abs[1]) if boot.ci_abs else None,
        "rmse_baseline_pooled": _r(boot.rmse_baseline_pooled),
        "rmse_incumbent_pooled": _r(boot.rmse_incumbent_pooled),
        "seed_consistency_pct": (_r(boot.seed_consistency * 100)
                                 if boot.seed_consistency is not None else None),
        "per_seed": per_seed,
    }
    text = (f"On {boot.n_seeds} hidden test seed(s), the admitted incumbent "
            f"changes held-out {meta['metric_name']} versus the campaign "
            f"baseline; the effect and its confidence interval are reported "
            f"from a paired-example bootstrap.")
    return _claim(
        "primary_effect", text, status,
        supporting_runs=[f"report-incumbent-s{k}" for k in range(boot.n_seeds)],
        baseline_runs=[f"report-baseline-s{k}" for k in range(boot.n_seeds)],
        effect_size=boot.effect_abs,
        confidence_interval=list(boot.ci_abs) if boot.ci_abs else None,
        statistical_test=_statistical_test_text(boot),
        limitations=limitations, values=values,
        provenance={"metrics_files": [
            f"experiments/report/{role}_test_s{k}.json"
            for role in ("baseline", "incumbent")
            for k in range(boot.n_seeds)]})


def _campaign_summary(disclosure: dict, costs: dict, meta: dict) -> dict:
    # Only genuine NUMBERS belong in the values map (bool is an int subclass in
    # Python — exclude it so a flag like incumbent_runs_aliased does not render
    # as a numeral in the report's numbers list).
    def _num(v) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    values = {f"disclosure_{k}": v for k, v in disclosure.items() if _num(v)}
    values.update({f"cost_{k}": _r(v) for k, v in costs.items() if _num(v)})
    text = ("Campaign accounting: experiment count, gate evaluations, "
            "generations, admitted count, test-split accesses, and total LLM "
            "spend, all derived from the immutable ledger.")
    return _claim("campaign_summary", text, "verified", values=values,
                  provenance={"contract_id": meta.get("contract_id"),
                              "campaign_id": meta.get("campaign_id")})


def _admitted_improvements(records: list[dict], corrected: set[str],
                           direction: str) -> list[dict]:
    out = []
    for r in records:
        if r.get("record_type") != "experiment":
            continue
        if r.get("decision") != "accept" or r.get("run_id") in corrected:
            continue
        hyp = r.get("hypothesis") or {}
        iv = hyp.get("intervention") or {}
        before, after = r.get("best_primary_before"), r.get("primary")
        rel_pct = None
        if isinstance(before, (int, float)) and isinstance(after, (int, float)) and before:
            rel = (before - after) / abs(before)
            if direction == "maximize":
                rel = -rel
            rel_pct = _r(rel * 100)
        if iv.get("param") is not None:
            what = f"{iv['param']} {iv['from']!r} -> {iv['to']!r}"
        else:
            what = f"coder change ({(hyp.get('statement') or '')[:60]})"
        text = (f"{what} improved dev mean tour length from {before} to {after} "
                f"in generation {r.get('generation')} and passed the blind "
                f"admission gate.")
        out.append(_claim(
            "admitted_improvement", text, "verified",
            supporting_runs=[r.get("run_id")],
            values={"dev_before": _r(before), "dev_after": _r(after),
                    "dev_rel_pct": rel_pct, "generation": r.get("generation")},
            provenance={"ledger_run_ids": [r.get("run_id")],
                        "coder_family": r.get("coder_family")}))
    return out


def _negative_results(records: list[dict], corrected: set[str]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for r in records:
        if r.get("record_type") != "experiment" or r.get("run_id") in corrected:
            continue
        if r.get("verdict") != "valid_negative":
            continue
        iv = (r.get("hypothesis") or {}).get("intervention") or {}
        param = iv.get("param")
        key = str(param) if param is not None else "coder"
        groups.setdefault(key, []).append(r)
    out = []
    for key in sorted(groups):
        runs = groups[key]
        fcs = sorted({str(r.get("failure_class")) for r in runs})
        text = (f"Intervening on {key} did not help: {len(runs)} valid negative "
                f"result(s) (failure classes: {', '.join(fcs)}). A validly "
                f"executed negative result is evidence of what does not work.")
        out.append(_claim(
            "negative_result", text, "verified",
            supporting_runs=[r.get("run_id") for r in runs],
            values={"n_runs": len(runs)},
            provenance={"param": key, "failure_classes": fcs}))
    return out


def _literature_groundings(records: list[dict], corrected: set[str],
                           evidence_index: dict) -> list[dict]:
    out = []
    for r in records:
        if r.get("record_type") != "experiment" or r.get("run_id") in corrected:
            continue
        if r.get("decision") != "accept":
            continue
        ids = (r.get("hypothesis") or {}).get("supporting_evidence_ids") or []
        if not ids:
            continue
        resolved = all(eid in evidence_index for eid in ids)
        papers = sorted({str((evidence_index.get(eid) or {}).get(
            "canonical_paper_id")) for eid in ids if eid in evidence_index})
        text = (f"Accepted change {r.get('run_id')} is grounded in "
                f"{len(ids)} evidence record(s) from prior work "
                f"({', '.join(papers) or 'unresolved'}).")
        out.append(_claim(
            "literature_grounding", text,
            "verified" if resolved else "unsupported",
            supporting_runs=[r.get("run_id")],
            supporting_literature=list(ids),
            provenance={"papers": papers}))
    return out


def build_claims(records: list[dict], boot, evidence_index: dict,
                 disclosure: dict, costs: dict, meta: dict) -> list[dict]:
    """Deterministic claim list (fixed order => stable claim ids)."""
    corrected = _corrected(records)
    direction = meta.get("metric_direction", "minimize")
    claims: list[dict] = [
        _primary_effect(boot, meta),
        _campaign_summary(disclosure, costs, meta),
    ]
    claims += _admitted_improvements(records, corrected, direction)
    claims += _negative_results(records, corrected)
    claims += _literature_groundings(records, corrected, evidence_index)
    for n, claim in enumerate(claims, start=1):
        claim["claim_id"] = f"claim_{n:04d}"
    return claims


def claims_jsonl_payload(claims: list[dict]) -> str:
    """Canonical one-claim-per-line JSONL (sorted keys => byte-stable)."""
    return "".join(json.dumps(c, sort_keys=True) + "\n" for c in claims)
