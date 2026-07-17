"""Human approval gate for the single-use test split (Phase 5, Layer 9).

The test split is a publication analog: the first `report` (and every
`report --force` re-run) requires a human to approve the INTENT — commits, dev
numbers, seed plan, disclosure-so-far — BEFORE any test number is computed.
This is a pre-spend control, not a post-hoc rubber stamp.

Approval state is DERIVED from the ledger, never persisted in state.json
(mirrors the momentum "derive don't persist" philosophy, so crash recovery and
replay stay free). A request is identified by a deterministic hash of its
fingerprint: the same intent yields the same request_id (duplicate requests are
idempotent), and any campaign advance changes the fingerprint — and thus the
id — so a stale approval is invalid by construction. These functions are pure
(no wall clock: `now` is passed in; no file IO: records come from the caller).
"""

from __future__ import annotations

import hashlib
import json

# Fields that identify a report INTENT. prior_sealed_reports counts previously
# sealed final_report records, so each `--force` re-run is a distinct intent
# (a fresh approval), while a crash mid-report — which seals nothing — leaves
# the fingerprint unchanged and lets a rerun resume on the same approval.
FINGERPRINT_KEYS = ("incumbent_commit", "baseline_commit",
                    "contract_sha256", "evaluator_sha256",
                    "prior_sealed_reports")


def request_id_for(fingerprint: dict) -> str:
    canonical = {k: fingerprint[k] for k in FINGERPRINT_KEYS}
    payload = json.dumps(canonical, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def make_request(campaign_id: str, fingerprint: dict, payload: dict,
                 now: str) -> dict:
    return {
        "record_type": "approval_request",
        "timestamp_utc": now,
        "campaign_id": campaign_id,
        "request_id": request_id_for(fingerprint),
        "fingerprint": {k: fingerprint[k] for k in FINGERPRINT_KEYS},
        "payload": payload,
    }


def make_decision(campaign_id: str, request_id: str, decision: str,
                  reason: str | None, now: str) -> dict:
    if decision not in ("approve", "deny"):
        raise ValueError(f"decision must be approve|deny, got {decision!r}")
    return {
        "record_type": "approval_decision",
        "timestamp_utc": now,
        "campaign_id": campaign_id,
        "request_id": request_id,
        "decision": decision,
        "reason": reason,
    }


def latest_request(records: list[dict]) -> dict | None:
    reqs = [r for r in records if r.get("record_type") == "approval_request"]
    return reqs[-1] if reqs else None


def find_request(records: list[dict], request_id: str) -> dict | None:
    """Exact match, else a unique prefix match (convenience for the CLI)."""
    reqs = [r for r in records if r.get("record_type") == "approval_request"]
    for r in reqs:
        if r.get("request_id") == request_id:
            return r
    matches = [r for r in reqs
               if str(r.get("request_id", "")).startswith(request_id)]
    return matches[0] if len(matches) == 1 else None


def decision_for(records: list[dict], request_id: str) -> dict | None:
    """The LATEST decision for a request (a later `approve` supersedes an
    earlier `deny` for the same intent, giving an explicit escape hatch)."""
    decs = [r for r in records
            if r.get("record_type") == "approval_decision"
            and r.get("request_id") == request_id]
    return decs[-1] if decs else None


def consumed_request_ids(records: list[dict]) -> set[str]:
    """Request ids consumed by a sealed report (final_report references it)."""
    out: set[str] = set()
    for r in records:
        if r.get("record_type") == "final_report":
            rid = r.get("request_id")
            if rid:
                out.add(str(rid))
    return out


def is_pending(records: list[dict]) -> bool:
    req = latest_request(records)
    return req is not None and decision_for(records, req["request_id"]) is None


def approval_status(records: list[dict], fingerprint: dict) -> str:
    """Status of the current intent's approval.

    Returns one of: none, pending, approved_fresh, approved_stale,
    approved_consumed, denied.
    """
    cur = request_id_for(fingerprint)
    consumed = consumed_request_ids(records)
    dec = decision_for(records, cur)
    if dec is not None:
        if dec["decision"] == "deny":
            return "denied"
        return "approved_consumed" if cur in consumed else "approved_fresh"
    req = latest_request(records)
    if req is None:
        return "none"
    rid = req.get("request_id")
    if rid == cur:
        return "pending"
    prior = decision_for(records, str(rid)) if rid else None
    if (prior and prior["decision"] == "approve" and rid not in consumed):
        # A human approved an OLDER intent; the campaign has since advanced.
        return "approved_stale"
    return "none"
