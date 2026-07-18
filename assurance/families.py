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

FAMILIES = (
    "neighborhood_operator",   # NEIGHBORHOOD swapped, or a move operator changed
    "construction_change",     # initial-tour construction changed
    "acceptance_criterion",    # SA / acceptance / cooling rule changed
    "perturbation",            # perturbation / kick between restarts changed
    "search_loop",             # the solve / local-search loop structure changed
    "none",                    # patcher, or ambiguous/non-atomic coder change
)

_NEIGHBORHOOD_TOKENS = ("NEIGHBORHOOD", "two_opt", "or_opt", "three_opt",
                        "2-opt", "or-opt", "_two_opt_move", "_or_opt_move")
_CONSTRUCTION_TOKENS = ("nearest_neighbor", "greedy_edge", "greedy",
                        "construct", "use_nn_construction")
_ACCEPT_TOKENS = ("metropolis", "acceptance", "_accepts", "temperature",
                  "cooling", "anneal")
_PERTURB_TOKENS = ("double_bridge", "perturb", "_perturb", "kick")
_LOOP_TOKENS = ("def solve", "def local_search", "def solve_instance",
                "tabu", "while ")


def _changed_lines(diff: str) -> list[str]:
    """Added/removed content lines of a unified diff (headers stripped)."""
    out = []
    for line in diff.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith(("+", "-")):
            out.append(line[1:])
    return out


def _signals(diff: str) -> set[str]:
    changed = _changed_lines(diff)
    sig: set[str] = set()
    if any(tok in ln for ln in changed for tok in _NEIGHBORHOOD_TOKENS):
        sig.add("neighborhood_operator")
    if any(tok in ln for ln in changed for tok in _CONSTRUCTION_TOKENS):
        sig.add("construction_change")
    if any(tok in ln for ln in changed for tok in _ACCEPT_TOKENS):
        sig.add("acceptance_criterion")
    if any(tok in ln for ln in changed for tok in _PERTURB_TOKENS):
        sig.add("perturbation")
    if any(tok in ln for ln in changed for tok in _LOOP_TOKENS):
        sig.add("search_loop")
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
