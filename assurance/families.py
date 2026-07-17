"""Deterministic coder-family classification for search momentum (Phase 5).

Phase 4 folded every LLM-coder hypothesis under the single momentum key
"coder:none" (no single parameter direction). This module bins a coder change
by its bounded diff into a small set of intervention families, so momentum can
steer at the family level (e.g. "adding feature interactions keeps paying off")
without an LLM.

Deterministic and pure: classification is a function of (hypothesis, diff)
only. The result is stored on the experiment record at write time (like
verdict/decision), so both the live momentum recompute and crash-recovery
replay read the same stored field — no diff is re-derived at replay time and
replay == live holds. Ambiguity (zero signals, or more than one — a non-atomic
change) yields "none", preserving the Phase 4 fallback exactly.
"""

from __future__ import annotations

import re

FAMILIES = (
    "feature_spec_interaction",  # FEATURE_SPEC gains a multi-index product term
    "feature_spec_unary",        # FEATURE_SPEC gains a single-index/transform term
    "hyperparam_code",           # a HYPERPARAMS value edited via code
    "training_loop",             # train() logic changed (gradient/loss/schedule)
    "none",                      # patcher, or ambiguous/non-atomic coder change
)

_MULTI_INDEX = re.compile(r"\[\s*\d+\s*(?:,\s*\d+\s*)+\]")  # e.g. [0, 1]
_UNARY_TERM = re.compile(r"\[\s*\d+\s*\]")                  # e.g. [3]
_HP_KEY = re.compile(
    r"""^\s*["'](lr|epochs|momentum|l2|batch_size|feature_scaling)["']\s*:""")
_LOOP_TOKENS = (
    "def train", "for epoch", "grad", "loss", "schedule", "learning_rate",
    " += ", "weight update",
)


def _changed_lines(diff: str) -> list[str]:
    """Added/removed content lines of a unified diff (headers stripped)."""
    out = []
    for line in diff.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith(("+", "-")):
            out.append(line[1:])
    return out


def _added_lines(diff: str) -> list[str]:
    return [line[1:] for line in diff.splitlines()
            if line.startswith("+") and not line.startswith("+++")]


def _signals(diff: str) -> set[str]:
    changed = _changed_lines(diff)
    added = _added_lines(diff)
    sig: set[str] = set()

    touches_feature_spec = any("FEATURE_SPEC" in ln for ln in changed)
    if touches_feature_spec and any(_MULTI_INDEX.search(ln) for ln in added):
        sig.add("feature_spec_interaction")
    elif touches_feature_spec and any(_UNARY_TERM.search(ln) for ln in added):
        sig.add("feature_spec_unary")

    if any(_HP_KEY.search(ln) for ln in changed):
        sig.add("hyperparam_code")

    if any(tok in ln for ln in changed for tok in _LOOP_TOKENS):
        sig.add("training_loop")

    return sig


def classify(hypothesis: dict, diff: str) -> str:
    """Family of a coder change; "none" for a patcher or an ambiguous diff.

    Exactly one signal is required: zero signals (nothing recognizable) or two+
    (a non-atomic change spanning categories) both fall back to "none", which
    keeps the momentum key at "coder:none" — the atomic-intervention principle.
    """
    if (hypothesis.get("executor") or "") != "coder":
        return "none"  # patcher folds under "{param}:{move}", not a coder family
    sig = _signals(diff or "")
    return next(iter(sig)) if len(sig) == 1 else "none"
