"""Consented fan-out of MF Engine reads across family members (design: docs/design/family-coordinator.md).

Every rule here is enforced in code before any network call, never in the prompt:
  * the asker's own live data is always allowed (as before)
  * another member's live data needs that member's opt-in (`share_live_mf_with_family` in F00x.access.json),
    a linked MF Engine login, and the tool inside the member's consent scope
  * each call runs under that member's own MF Engine identity (one identity per request)
  * other members' payloads are redacted (PAN, phone, email, address, account numbers; folios masked)
  * every call made on someone else's behalf is appended to an audit log, keyed by the x-request-id sent to MF Engine
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from . import mf_engine

log = logging.getLogger("familyos.member_gate")

# Tool → consent scope. Identity/KYC tools and tools that already aggregate a family inside MF Engine
# (getFamilyPortfolioSummary, getHouseholdTaxHarvestPlan) are deliberately absent.
FAMILY_SAFE: dict[str, str] = {
    "getPortfolioSummary": "portfolio",
    "getPortfolioAllocationAnalysis": "portfolio",
    "getCapitalGainsSummary": "tax",
    "getHarvestingOpportunities": "tax",
    "getLiquidityLadder": "liquidity",
    "getSipBook": "sips",
    "getSipSchedule": "sips",
    "getPendingActions": "sips",
}
SCOPES = sorted(set(FAMILY_SAFE.values()))

AUDIT_DIR = Path(__file__).resolve().parent.parent / "data" / "audit"
CALL_TIMEOUT_S = 75  # above mf_engine's own 60s heavy-tool timeout

Status = Literal["self", "ok", "no_identity", "no_consent", "out_of_scope", "unknown_member"]
REASONS = {
    "no_identity": "has not linked an MF Engine login",
    "no_consent": "has not shared live mutual-fund data with the family",
    "out_of_scope": "has not shared this kind of mutual-fund data",
    "unknown_member": "is not a member of this family",
}


@dataclass
class Delegate:
    member_id: str
    status: Status

    @property
    def allowed(self) -> bool:
        return self.status in ("self", "ok")


class MemberGate:
    def __init__(self, ctx, asker: Optional[str]):
        self.ctx, self.asker = ctx, asker

    def status(self, member_id: str, tool: str) -> Status:
        if member_id not in self.ctx.members:
            return "unknown_member"
        if mf_engine.identity(self.ctx.family_id, member_id) is None:
            return "no_identity"
        if member_id == self.asker:
            return "self"
        pol = self.ctx.access.get(member_id, {})
        if not pol.get("share_live_mf_with_family"):
            return "no_consent"
        scope = (pol.get("live_mf_consent") or {}).get("scope")
        if scope and FAMILY_SAFE.get(tool) not in scope:
            return "out_of_scope"
        return "ok"

    def resolve(self, member_ids: Optional[list[str]], tool: str) -> list[Delegate]:
        ids = list(dict.fromkeys(member_ids)) if member_ids else list(self.ctx.members)
        return [Delegate(m, self.status(m, tool)) for m in ids]

    def any_delegate_possible(self) -> bool:
        """True when fan-out could reach anyone besides the asker (so the tool is worth offering)."""
        linked = set(mf_engine.linked_members(self.ctx.family_id))
        return any(m != self.asker and self.ctx.access.get(m, {}).get("share_live_mf_with_family") for m in linked)


# ── redaction ────────────────────────────────────────────────────────────────
_DROP = {"pan", "phone", "mobile", "phone_number", "email", "email_address", "address", "dob", "date_of_birth",
         "account_number", "bank_account_number", "access_token", "refresh_token", "aadhaar"}
_FOLIO = re.compile(r"folio", re.I)


def redact(obj: Any) -> Any:
    """Strip identifiers from another member's payload; folio numbers keep only their last 4 characters."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            kl = str(k).lower()
            if kl in _DROP:
                continue
            if _FOLIO.search(kl) and isinstance(v, (str, int)):
                s = str(v)
                out[k] = "•" * max(0, len(s) - 4) + s[-4:]
            else:
                out[k] = redact(v)
        return out
    if isinstance(obj, list):
        return [redact(x) for x in obj]
    return obj


# ── audit ────────────────────────────────────────────────────────────────────
def audit(family_id: str, asker: Optional[str], on_behalf_of: str, tool: str, args: dict, request_id: str, status: str):
    try:
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        line = {"ts": datetime.now(timezone.utc).isoformat(), "family_id": family_id, "asker": asker,
                "on_behalf_of": on_behalf_of, "tool": tool,
                "args_hash": uuid.uuid5(uuid.NAMESPACE_OID, json.dumps(args, sort_keys=True, default=str)).hex[:12],
                "request_id": request_id, "status": status}
        with (AUDIT_DIR / f"{family_id}.live_mf.jsonl").open("a") as fh:
            fh.write(json.dumps(line) + "\n")
    except OSError as e:  # never fail an answer because the audit disk is unavailable; but say so loudly
        log.error("live MF audit write failed: %s", e)


# ── fan-out ──────────────────────────────────────────────────────────────────
def fan_out(gate: MemberGate, tool: str, args: dict, member_ids: Optional[list[str]]) -> dict:
    """Run one FAMILY_SAFE tool for each allowed member in parallel. Refusals and errors are data, not exceptions."""
    if tool not in FAMILY_SAFE:
        raise ValueError(f"{tool} is not available for family fan-out; allowed: {sorted(FAMILY_SAFE)}")
    fid, asker = gate.ctx.family_id, gate.asker
    delegates = gate.resolve(member_ids, tool)
    results: dict[str, dict] = {}
    calls: list[dict] = []

    def one(d: Delegate):
        rid = f"familyos-{uuid.uuid4().hex[:12]}"
        try:
            data = mf_engine.call(tool, args, fid, d.member_id, request_id=rid)
            status, payload = "ok", data if d.status == "self" else redact(data)
        except mf_engine.MFEngineError as e:
            status, payload = "error", {"code": e.code, "message": e.message}
        except Exception as e:  # network, timeout, MCP transport
            status, payload = "error", {"code": "UNAVAILABLE", "message": str(e)[:200]}
        if d.status != "self":
            audit(fid, asker, d.member_id, tool, args, rid, status)
        return d, rid, status, payload

    allowed = [d for d in delegates if d.allowed]
    for d in delegates:
        if not d.allowed:
            results[d.member_id] = {"status": d.status, "reason": REASONS[d.status]}
    if allowed:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(allowed))) as ex:
            futs = {ex.submit(one, d): d for d in allowed}
            for fut, d in futs.items():
                try:
                    d, rid, status, payload = fut.result(timeout=CALL_TIMEOUT_S)
                except concurrent.futures.TimeoutError:
                    rid, status, payload = None, "error", {"code": "TIMEOUT", "message": "MF Engine did not respond in time"}
                entry = {"status": status, "gate": d.status}
                if status == "ok":
                    entry.update(source_id=f"MF:{d.member_id}:{tool}:{rid}", data=payload)
                else:
                    entry["error"] = payload
                results[d.member_id] = entry
                calls.append({"member_id": d.member_id, "gate": d.status, "status": status, "request_id": rid,
                              "source_id": entry.get("source_id")})

    withheld = [m for m, r in results.items() if r["status"] in REASONS]
    return {"tool": tool, "members": results, "withheld": withheld, "_calls": calls}
