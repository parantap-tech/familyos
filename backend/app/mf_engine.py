"""MF Engine investor MCP (`/mcp/investor`) — live mutual-fund tools for the FamilyOS agent.

Follows the MF Engine integration guide:
  * Streamable HTTP, Bearer MCP token + the *asking* investor's identity headers on every call
  * one identity per request — a member's tools run only with that member's own login
  * tool results are a JSON string in content[0].text; `{"error": true, ...}` payloads are errors, not data
  * read-only: the two mutation tools are never exposed and `x-mcp-take-action` is never sent
  * tools/list is cached per process; heavy analytics tools get a 60s timeout

Configuration (never logged):
  MF_ENGINE_MCP_URL        the /mcp/investor endpoint issued by MF Engine
  MF_ENGINE_MCP_TOKEN      Bearer token issued by MF Engine (or "mcp_token" in the identities file)
  MF_ENGINE_IDENTITIES     path to JSON, default backend/.secrets/mf_engine_identities.json:
      {"mcp_token": "...",
       "members": {"F002:M203": {"access_token": "...", "refresh_token": "..."},
                   "F001:M003": {"agentic_user_id": "..."}}}
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from .mcp_source import _run  # shared asyncio/exception-group handling

log = logging.getLogger("familyos.mf_engine")

DEFAULT_URL = ""  # set MF_ENGINE_MCP_URL to the endpoint issued by MF Engine
SECRETS = Path(__file__).resolve().parent.parent / ".secrets" / "mf_engine_identities.json"

# Read-only tools the agent may use. Mutations (submitRiskProfile, stageBucketPurchaseCart) are deliberately absent.
ALLOWLIST = [
    "getFamilyPortfolioSummary", "getPortfolioHealthScore", "getPortfolioSummary", "getPortfolioXray",
    "getPortfolioAllocationAnalysis", "comparePortfolioFunds", "getCapitalGainsSummary", "getHarvestingOpportunities",
    "getSmartRedemptionPlan", "getTaxAwareRedemptionPlan", "getSipBook", "getSipSchedule", "getSipProjection",
    "getPortfolioStressTest", "getRiskProfileStatus", "getPendingActions", "calculateGoalPlan",
    "calculateRetirementPlan", "calculateEmergencyFund", "getCasImportHistory", "searchFunds", "getFundDetail",
    "getPortfolioFactorExposure", "getHouseholdTaxHarvestPlan", "getLiquidityLadder",
    # Goal planner (financial_plan): read-only views of the asker's own plan.
    "get_financial_plan", "explain_goal", "preview_financial_plan",
]
# Writes are never exposed, even though MF Engine only applies them with x-mcp-take-action (never sent).
MUTATIONS = {"submitRiskProfile", "stageBucketPurchaseCart",
             "add_goal", "update_goal", "remove_goal", "update_financial_plan_section", "compute_financial_plan"}
HEAVY = {"getPortfolioXray", "getPortfolioFactorExposure", "getPortfolioStressTest", "comparePortfolioFunds", "getSmartRedemptionPlan",
         "getHouseholdTaxHarvestPlan", "getLiquidityLadder"}
PREFIX = "mf_"

_tools_cache: dict[str, Any] = {"at": 0.0, "tools": None, "error": None}


class MFEngineError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code, self.message = code, message


def _secrets() -> dict:
    p = Path(os.environ.get("MF_ENGINE_IDENTITIES", SECRETS))
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def url() -> str:
    return os.environ.get("MF_ENGINE_MCP_URL", DEFAULT_URL)


def token() -> Optional[str]:
    return os.environ.get("MF_ENGINE_MCP_TOKEN") or _secrets().get("mcp_token")


def identity(family_id: str, member_id: Optional[str]) -> Optional[dict]:
    if not member_id:
        return None
    return _secrets().get("members", {}).get(f"{family_id}:{member_id}")


def configured() -> bool:
    return bool(token()) and bool(url()) and os.environ.get("FAMILYOS_MF_ENGINE", "1") != "0"


def linked_members(family_id: str) -> list[str]:
    return [k.split(":", 1)[1] for k in _secrets().get("members", {}) if k.startswith(f"{family_id}:")]


def _headers(ident: Optional[dict], conversation_id: Optional[str], request_id: Optional[str] = None) -> dict:
    h = {"Authorization": f"Bearer {token()}", "x-request-id": request_id or f"familyos-{uuid.uuid4().hex[:12]}"}
    if conversation_id:
        h["mcp-session-id"] = conversation_id
    if ident:
        if ident.get("access_token"):
            h["access-token"] = ident["access_token"]
            if ident.get("refresh_token"):
                h["refresh-token"] = ident["refresh_token"]
        elif ident.get("agentic_user_id"):
            h["x-agentic-user-id"] = ident["agentic_user_id"]
    return h  # NOTE: x-mcp-take-action is never set — FamilyOS is read-only.


async def _with_session(ident, fn, timeout: float, conversation_id=None, request_id=None):
    import httpx2
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    async with httpx2.AsyncClient(headers=_headers(ident, conversation_id, request_id), timeout=timeout) as http:
        async with streamable_http_client(url(), http_client=http) as (read, write):
            async with ClientSession(read, write) as s:
                await s.initialize()
                return await fn(s)


def tool_defs(force: bool = False) -> list[dict]:
    """Allowlisted MF Engine tools as Anthropic tool definitions (cached for the process)."""
    if not configured():
        return []
    if _tools_cache["tools"] is not None and not force:
        return _tools_cache["tools"]
    if not force and _tools_cache["error"] and time.time() - _tools_cache["at"] < 60:
        return []

    async def go(s):
        return (await s.list_tools()).tools

    try:
        tools = _run(_with_session(None, go, 20))
    except Exception as e:
        _tools_cache.update(at=time.time(), error=str(e)[:300], tools=None)
        log.warning("MF Engine tools/list failed: %s", e)
        return []
    defs = []
    for t in tools:
        if t.name not in ALLOWLIST or t.name in MUTATIONS:
            continue
        schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {"type": "object", "properties": {}}
        defs.append({"name": PREFIX + t.name, "input_schema": schema,
                     "description": "[MF Engine · live mutual-fund data for the asking investor] " + (t.description or t.name)[:900]})
    _tools_cache.update(at=time.time(), tools=defs, error=None)
    return defs


def call(tool: str, args: dict, family_id: str, member_id: Optional[str], conversation_id: Optional[str] = None,
         request_id: Optional[str] = None) -> Any:
    """Call one MF Engine tool as `member_id` (`request_id` becomes x-request-id, for audit correlation). Returns the parsed JSON payload; raises MFEngineError on error payloads."""
    name = tool.removeprefix(PREFIX)
    if name in MUTATIONS or name not in ALLOWLIST:
        raise MFEngineError("NOT_ALLOWED", f"{name} is not available to FamilyOS (read-only allowlist)")
    ident = identity(family_id, member_id)
    if ident is None:
        raise MFEngineError("NO_IDENTITY", "This family member has not linked an MF Engine login, so their live "
                                           "mutual-fund data is unavailable. Use the FamilyOS records instead.")

    async def go(s):
        return await s.call_tool(name, args)

    res = _run(_with_session(ident, go, 60 if name in HEAVY else 30, conversation_id, request_id))
    payload: Any = None
    for b in res.content or []:
        if getattr(b, "type", None) == "text":
            try:
                payload = json.loads(b.text)
            except json.JSONDecodeError:
                payload = {"text": b.text}
            break
    if payload is None:
        payload = getattr(res, "structured_content", None)
    if getattr(res, "is_error", False):
        raise MFEngineError("TOOL_ERROR", json.dumps(payload)[:300] if payload else "tool error")
    if isinstance(payload, dict) and payload.get("error") is True:
        raise MFEngineError(str(payload.get("code", "ERROR")), str(payload.get("message", ""))[:300])
    return payload


def status(family_id: Optional[str] = None) -> dict:
    out = {"configured": configured(), "endpoint": url(), "tools_available": len(_tools_cache["tools"] or []),
           "tools_error": _tools_cache["error"]}
    if family_id:
        out["linked_members"] = linked_members(family_id)
    return out
