"""AutoResearch Phase 5 assurance package (Blueprint Layers 8 & 9).

Scientific assurance + reporting: multi-seed finalist reproduction, paired
bootstrap confidence intervals, the claim-evidence ledger, deterministic
report/figure rendering, the cross-model (codex) adversarial reviewer, the
human approval gate, and coder-family classification for search momentum.

Closure rules (protected trusted path, like literature/ but stricter — see
docs/HANDOFF.md invariant 13):

1. Imports stdlib only. Never imports orchestrator, literature, or yaml.
   Import direction is strictly orchestrator -> assurance.
2. Data in, strings/dicts out. NO file IO anywhere in this package, with a
   single deliberate exception: ``reviewer.run_review`` invokes the external
   ``codex exec`` CLI through an injectable runner and touches only its own
   scratch workdir under experiments/report/review/. Every other read and
   write is performed by the orchestrator, always under experiments/**.
3. No wall clock, no ambient randomness. Bootstrap RNG seeds are derived from
   campaign/commit ids and logged; timestamps are passed in. This makes the
   report and figures byte-deterministic and drillable by hash.
4. Gate blindness holds: nothing here receives gate scores. Claims, the
   report, and the reviewer packet use dev + test numbers only.

Submodules (stats, claims, report_md, figures, svgfig, reviewer, gate,
families) are imported explicitly by the orchestrator as needed.
"""
