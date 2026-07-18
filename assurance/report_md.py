"""Deterministic Markdown report whose every numeral traces to a claim.

The report is built from a document builder with two kinds of segment:
  * lit(text)  — connective prose. Asserted DIGIT-FREE at build time (this is
                 the "digit-free template" guarantee, enforced structurally).
  * val(text)  — a value taken from claims.jsonl or the structural meta map.
                 Its character span is recorded.

After rendering, scan_untraced_digits verifies that every digit run in the
document lies inside a recorded val() span. So a number can only appear in the
report if it came from a claim (its text or its `values`) or a whitelisted meta
token (commit ids, the report date, figure filenames) — never from a
hand-written literal. Any violation aborts the report before it is written.
"""

from __future__ import annotations

import re

_DIGIT = re.compile(r"[0-9]")
# A numeric run must START and END with a digit, so trailing punctuation that
# belongs to adjacent literal prose (a comma separator, a sentence period) is
# not swallowed into the number and mis-flagged as crossing a val() span.
_DIGIT_RUN = re.compile(r"[0-9](?:[0-9.,eE+\-]*[0-9])?")


class ReportError(Exception):
    pass


class _Doc:
    def __init__(self) -> None:
        self._parts: list[str] = []
        self._spans: list[tuple[int, int]] = []
        self._pos = 0

    def lit(self, s: str) -> "_Doc":
        if _DIGIT.search(s):
            raise ReportError(f"literal report text contains a digit: {s!r}")
        self._parts.append(s)
        self._pos += len(s)
        return self

    def val(self, s: str) -> "_Doc":
        s = str(s)
        start = self._pos
        self._parts.append(s)
        self._pos += len(s)
        self._spans.append((start, self._pos))
        return self

    def num(self, value, fmt: str | None = None) -> "_Doc":
        """A claim/meta numeric value (or 'n/a' when absent)."""
        if value is None:
            return self.lit("n/a")
        return self.val(format(value, fmt) if fmt else str(value))

    def text(self) -> str:
        return "".join(self._parts)

    @property
    def spans(self) -> list[tuple[int, int]]:
        return self._spans


def scan_untraced_digits(doc: str, spans: list[tuple[int, int]]) -> list[str]:
    """Digit runs in `doc` NOT fully inside a recorded val() span."""
    untraced = []
    for m in _DIGIT_RUN.finditer(doc):
        s, e = m.start(), m.end()
        if not any(a <= s and e <= b for a, b in spans):
            untraced.append(m.group())
    return untraced


def _by_kind(claims: list[dict], kind: str) -> list[dict]:
    return [c for c in claims if c.get("kind") == kind]


def _one(claims: list[dict], kind: str) -> dict | None:
    found = _by_kind(claims, kind)
    return found[0] if found else None


def render_report(claims: list[dict], meta: dict) -> str:
    """Render report.md; raises ReportError on an untraced digit or a missing
    required claim."""
    d = _Doc()
    primary = _one(claims, "primary_effect")
    summary = _one(claims, "campaign_summary")
    if primary is None or summary is None:
        raise ReportError("report requires primary_effect + campaign_summary claims")
    pv = primary["values"]
    sv = summary["values"]

    # --- header ---
    d.lit("# AutoResearch Campaign Report — ").val(meta["contract_id"]).lit("\n\n")
    d.lit("Campaign ").val(meta["campaign_id"])
    d.lit(" · baseline ").val(meta["baseline_commit_short"])
    d.lit(" · incumbent ").val(meta["incumbent_commit_short"])
    d.lit(" · ").val(meta["report_date"]).lit("\n\n")

    # --- headline ---
    d.lit("## Headline result\n\n")
    d.val(primary["text"]).lit("\n\n")
    d.lit("**Status: ").val(primary["status"]).lit(".** ")
    d.lit("Pooled mean test tour length ").num(pv.get("rmse_baseline_pooled"), ".2f")
    d.lit(" → ").num(pv.get("rmse_incumbent_pooled"), ".2f")
    d.lit("; effect ").num(pv.get("effect_abs"), ".2f")
    d.lit(" (").num(pv.get("effect_rel_pct"), ".2f").lit("% relative), ")
    d.num(pv.get("confidence_pct"), "d").lit("% CI [")
    d.num(pv.get("ci_lo"), ".2f").lit(", ").num(pv.get("ci_hi"), ".2f").lit("]")
    d.lit(" across ").num(pv.get("n_seeds"), "d").lit(" hidden test seed(s) × ")
    d.num(pv.get("n_examples"), "d").lit(" instances. [").val(primary["claim_id"]).lit("]\n\n")
    d.lit("Statistical test: ").val(primary.get("statistical_test") or "n/a").lit(".\n\n")

    # --- per-seed table ---
    d.lit("## Per-seed reproduction\n\n")
    d.lit("| seed | baseline tour length | incumbent tour length | delta |\n")
    d.lit("|------|----------------------|-----------------------|-------|\n")
    for row in pv.get("per_seed") or []:
        d.lit("| s").num(row.get("seed_index"), "d").lit(" | ")
        d.num(row.get("rmse_baseline"), ".2f").lit(" | ")
        d.num(row.get("rmse_incumbent"), ".2f").lit(" | ")
        d.num(row.get("delta"), ".2f").lit(" |\n")
    d.lit("\nSeed consistency (fraction improving): ")
    d.num(pv.get("seed_consistency_pct"), ".2f").lit("%.\n\n")
    d.lit("![per-seed paired tour length](figures/").val(meta["fig_paired"]).lit(")\n\n")

    # --- search trajectory ---
    d.lit("## Search trajectory\n\n")
    d.lit("![dev trajectory](figures/").val(meta["fig_trajectory"]).lit(")\n\n")
    d.lit("![verdict composition](figures/").val(meta["fig_verdicts"]).lit(")\n\n")

    # --- admitted interventions ---
    d.lit("## Admitted interventions\n\n")
    admitted = _by_kind(claims, "admitted_improvement")
    if not admitted:
        d.lit("No candidate was admitted during this campaign.\n\n")
    for c in admitted:
        d.lit("- ").val(c["text"]).lit(" [").val(c["claim_id"]).lit("]\n")
    if admitted:
        d.lit("\n")

    # --- negative results ---
    d.lit("## Negative results\n\n")
    negatives = _by_kind(claims, "negative_result")
    if not negatives:
        d.lit("No valid negative results were recorded.\n\n")
    for c in negatives:
        d.lit("- ").val(c["text"]).lit(" [").val(c["claim_id"]).lit("]\n")
    if negatives:
        d.lit("\n")

    # --- literature grounding ---
    d.lit("## Literature grounding\n\n")
    lit_claims = _by_kind(claims, "literature_grounding")
    if not lit_claims:
        d.lit("No accepted change cited literature evidence.\n\n")
    for c in lit_claims:
        d.lit("- ").val(c["text"]).lit(" [").val(c["claim_id"]).lit("]\n")
    if lit_claims:
        d.lit("\n")

    # --- costs and disclosure ---
    d.lit("## Costs and multiple-testing disclosure\n\n")
    d.val(summary["text"]).lit(" [").val(summary["claim_id"]).lit("]\n\n")
    for key in sorted(sv):
        d.lit("- ").lit(key.replace("_", " ")).lit(": ").num(sv[key]).lit("\n")
    d.lit("\n")

    # --- limitations ---
    d.lit("## Limitations\n\n")
    for lim in primary.get("limitations") or []:
        d.lit("- ").val(lim).lit("\n")
    d.lit("\nThis report was generated deterministically; every number above "
          "traces to a claim in experiments/claims.jsonl.\n")

    doc = d.text()
    untraced = scan_untraced_digits(doc, d.spans)
    if untraced:
        raise ReportError(
            f"report has {len(untraced)} untraced numeral(s): {untraced[:8]}")
    return doc
