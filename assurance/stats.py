"""Paired-example bootstrap confidence intervals (Phase 5, Blueprint Layer 8).

Pure stdlib (no numpy/scipy). The finalist is reproduced on N hidden test
seeds; on each seed the baseline and incumbent are scored on the SAME dataset,
so their per-example squared errors are paired. We pool the N x 600 paired
examples and bootstrap over the pooled pairs (joint resampling of indices),
which captures test-sampling variability — the only randomness here, since
training is deterministic. Seeds are i.i.d. draws from one generator, not
statistical clusters, so a hierarchical (resample-seeds-then-examples)
bootstrap would only add small-N noise; the per-seed deltas are reported
separately as the "not an artifact of one seed" evidence.

Determinism: the bootstrap RNG seed is derived from the campaign/commit ids
(derive_bootstrap_seed) and logged, so a CI is reproducible from the immutable
ledger + metrics files. No wall clock, no ambient randomness.
"""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from dataclasses import dataclass

_RMSE_TOLERANCE = 1e-9  # per-seed recompute vs the evaluator's heldout_rmse


class StatsError(Exception):
    """Raised on structurally invalid inputs (e.g. broken pairing)."""


@dataclass(frozen=True)
class SeedStats:
    """Per-seed summary. delta = rmse_baseline - rmse_incumbent (>0 improves
    for a minimized metric). None fields mark an unclean run on that seed."""
    seed_index: int
    rmse_baseline: float | None
    rmse_incumbent: float | None
    delta: float | None
    failure_baseline: str | None
    failure_incumbent: str | None
    fingerprint: str | None


@dataclass(frozen=True)
class BootstrapResult:
    clean: bool
    n_seeds: int
    n_examples: int
    effect_abs: float | None
    effect_rel: float | None
    ci_abs: tuple[float, float] | None
    ci_rel: tuple[float, float] | None
    rmse_baseline_pooled: float | None
    rmse_incumbent_pooled: float | None
    resamples: int
    confidence: float
    bootstrap_seed: int
    seed_consistency: float | None
    per_seed: tuple[SeedStats, ...]


def derive_bootstrap_seed(campaign_id: str, baseline_commit: str,
                          incumbent_commit: str) -> int:
    """Deterministic, magic-constant-free RNG seed logged with the CI."""
    payload = f"{campaign_id}|{baseline_commit}|{incumbent_commit}|paired-bootstrap"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8],
                          "big")


def _rmse(errors: list[float]) -> float:
    return math.sqrt(sum(errors) / len(errors))


def _clean_errors(metrics: dict) -> list[float] | None:
    """Per-example squared errors iff the run scored cleanly, else None."""
    if (metrics.get("primary_metric") or {}).get("value") is None:
        return None
    errs = (metrics.get("metrics") or {}).get("per_example_sq_errors")
    if not isinstance(errs, list) or not errs:
        return None
    if not all(isinstance(e, (int, float)) and math.isfinite(e) for e in errs):
        return None
    return [float(e) for e in errs]


def extract_paired_errors(
    baseline_metrics: list[dict], incumbent_metrics: list[dict]
) -> tuple[list[list[float]], list[list[float]], list[SeedStats]]:
    """Pair per-example errors across N seeds.

    Returns (errs_b, errs_inc, seed_stats). errs_* contain one 600-length list
    per CLEAN seed pair, in seed order; when every pair is clean, that is all N
    seeds and the bootstrap may run. A fingerprint mismatch on a clean pair is
    a hard StatsError (the two candidates were scored on different data — the
    pairing would be invalid). Unclean seeds contribute a SeedStats with Nones
    and no errors; the caller treats "any unclean" as bootstrap-skipped (no
    seed dropping, which would be selection bias).
    """
    if len(baseline_metrics) != len(incumbent_metrics):
        raise StatsError(
            f"seed count mismatch: {len(baseline_metrics)} baseline vs "
            f"{len(incumbent_metrics)} incumbent metrics")
    errs_b: list[list[float]] = []
    errs_inc: list[list[float]] = []
    seed_stats: list[SeedStats] = []
    for i, (mb, mi) in enumerate(zip(baseline_metrics, incumbent_metrics)):
        eb = _clean_errors(mb)
        ei = _clean_errors(mi)
        fb = mb.get("failure_class")
        fi = mi.get("failure_class")
        fp_b = (mb.get("dataset") or {}).get("heldout_fingerprint")
        fp_i = (mi.get("dataset") or {}).get("heldout_fingerprint")
        if eb is None or ei is None:
            seed_stats.append(SeedStats(
                seed_index=i, rmse_baseline=None, rmse_incumbent=None,
                delta=None, failure_baseline=fb, failure_incumbent=fi,
                fingerprint=fp_b if fp_b == fp_i else None))
            continue
        if fp_b != fp_i:
            raise StatsError(
                f"seed_index {i}: baseline/incumbent fingerprints differ "
                f"({fp_b} vs {fp_i}) — pairing invalid (data drift)")
        rb, ri = _rmse(eb), _rmse(ei)
        # Self-consistency / tamper check: the scalar heldout_rmse and the
        # per_example_sq_errors come from the SAME evaluator run, so this only
        # catches a metrics file whose scalar was hand-edited to disagree with
        # its own error array. The real integrity guarantees are the nonce,
        # evaluator self-hash, and protection manifest (in run_evaluator); the
        # heldout_fingerprint equality above is what catches data drift.
        for name, errs, m in (("baseline", eb, mb), ("incumbent", ei, mi)):
            reported = (m.get("metrics") or {}).get("heldout_rmse")
            recomputed = _rmse(errs)
            if reported is not None and abs(reported - recomputed) > _RMSE_TOLERANCE:
                raise StatsError(
                    f"seed_index {i} {name}: per-example RMSE {recomputed} "
                    f"disagrees with evaluator heldout_rmse {reported}")
        errs_b.append(eb)
        errs_inc.append(ei)
        seed_stats.append(SeedStats(
            seed_index=i, rmse_baseline=rb, rmse_incumbent=ri, delta=rb - ri,
            failure_baseline=None, failure_incumbent=None, fingerprint=fp_b))
    return errs_b, errs_inc, seed_stats


def _percentile_ci(data: list[float], confidence: float) -> tuple[float, float]:
    """Two-sided percentile CI via stdlib inclusive quantiles."""
    alpha = (1.0 - confidence) / 2.0
    n = round(1.0 / alpha)  # 0.95 -> 40, 0.90 -> 20, 0.99 -> 200
    qs = statistics.quantiles(data, n=n, method="inclusive")
    return qs[0], qs[n - 2]


def paired_bootstrap(
    errs_b: list[list[float]], errs_inc: list[list[float]],
    seed_stats: list[SeedStats], *, resamples: int, seed: int,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Pooled paired-example bootstrap of the RMSE difference.

    Clean only when every seed pair is clean (len(errs_b) == len(seed_stats));
    otherwise no bootstrap runs (effect/CI None) — the evaluator stays the sole
    authority on skill, and dropping unclean seeds would be selection bias.
    """
    n_seeds = len(seed_stats)
    clean = n_seeds > 0 and len(errs_b) == n_seeds and len(errs_inc) == n_seeds
    if not clean:
        return BootstrapResult(
            clean=False, n_seeds=n_seeds, n_examples=0,
            effect_abs=None, effect_rel=None, ci_abs=None, ci_rel=None,
            rmse_baseline_pooled=None, rmse_incumbent_pooled=None,
            resamples=resamples, confidence=confidence, bootstrap_seed=seed,
            seed_consistency=None, per_seed=tuple(seed_stats))

    pooled_b = [e for seed_errs in errs_b for e in seed_errs]
    pooled_i = [e for seed_errs in errs_inc for e in seed_errs]
    m = len(pooled_b)
    rmse_b_pooled = _rmse(pooled_b)
    rmse_i_pooled = _rmse(pooled_i)
    effect_abs = rmse_b_pooled - rmse_i_pooled
    effect_rel = effect_abs / rmse_b_pooled if rmse_b_pooled > 0 else 0.0

    rng = random.Random(seed)
    randrange = rng.randrange
    deltas_abs: list[float] = []
    deltas_rel: list[float] = []
    for _ in range(resamples):
        sb = 0.0
        si = 0.0
        for _ in range(m):
            j = randrange(m)  # ONE index — resample pairs jointly
            sb += pooled_b[j]
            si += pooled_i[j]
        rb = math.sqrt(sb / m)
        ri = math.sqrt(si / m)
        deltas_abs.append(rb - ri)
        deltas_rel.append((rb - ri) / rb if rb > 0 else 0.0)

    ci_abs = _percentile_ci(deltas_abs, confidence)
    ci_rel = _percentile_ci(deltas_rel, confidence)
    improving = sum(1 for s in seed_stats if s.delta is not None and s.delta > 0)
    seed_consistency = improving / n_seeds

    return BootstrapResult(
        clean=True, n_seeds=n_seeds, n_examples=m,
        effect_abs=effect_abs, effect_rel=effect_rel,
        ci_abs=ci_abs, ci_rel=ci_rel,
        rmse_baseline_pooled=rmse_b_pooled,
        rmse_incumbent_pooled=rmse_i_pooled,
        resamples=resamples, confidence=confidence, bootstrap_seed=seed,
        seed_consistency=seed_consistency, per_seed=tuple(seed_stats))
