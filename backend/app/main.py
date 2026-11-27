from __future__ import annotations

from typing import Optional

import json
import logging

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError

from . import agent, agentic, finance, graph_store, llm, mcp_source, mf_engine
from .context_engine import AccessDenied, store
from .contract import FamilyFinancialData, RecordBatch

log = logging.getLogger("familyos.api")

app = FastAPI(title="FamilyOS Context Engine", version="1.0")

# Requests relayed by Cloudflare (the public demo tunnel) carry cf-* headers; Next's /api rewrite forwards them.
_EXTERNAL_HEADERS = ("cf-connecting-ip", "cf-ray", "cdn-loop")


def is_external(request) -> bool:
    h = request.headers
    if any(k in h for k in _EXTERNAL_HEADERS):
        return True
    host = (h.get("x-forwarded-host") or h.get("host") or "").split(":")[0]
    return host not in ("localhost", "127.0.0.1", "")


@app.middleware("http")
async def private_family_gate(request, call_next):
    """Real-person (private) families are served to this machine only — never through the public tunnel."""
    parts = request.url.path.split("/")
    if len(parts) > 3 and parts[1:3] == ["api", "family"] and store.is_private(parts[3]) and is_external(request):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": "family not found"}, status_code=404)
    return await call_next(request)


@app.on_event("startup")
def _warm_graph():
    """Build every family's context and mirror it to Neo4j in the background (best-effort)."""
    import threading

    def run():
        for f in store.families():
            try:
                store.get(f["family_id"])
            except Exception as e:
                log.warning("could not build %s: %s", f["family_id"], e)
    threading.Thread(target=run, daemon=True).start()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _ctx(family_id: str):
    try:
        return store.get(family_id)
    except KeyError:
        raise HTTPException(404, f"family {family_id} not found")
    except ValidationError as e:
        raise HTTPException(502, f"financial data engine payload violates Contract A: {e}")


def _viewer(ctx, member_id: Optional[str]):
    if member_id is not None and member_id not in ctx.members:
        raise HTTPException(403, f"{member_id} is not a member of {ctx.family_id}")
    return member_id


@app.get("/api/health")
def health(family_id: Optional[str] = Query(None)):
    on = llm.enabled()
    return {"ok": True, "llm": on, "model": llm.MODEL if on else None, "mode": "agentic" if on else "deterministic",
            "speed": agentic.SPEED if on else None, "effort": agentic.EFFORT if on else None,
            "neo4j": graph_store.status(family_id), "data_source": store.source,
            "mf_engine": mf_engine.status(family_id)}


@app.get("/api/families")
def families(request: Request):
    """Demo family index: id, name, city, summary, default viewer and suggested questions."""
    out = []
    external = is_external(request)
    for f in store.families():
        if external and store.is_private(f["family_id"]):
            continue
        try:
            ctx = store.get(f["family_id"])
            out.append({**f, "member_count": len(ctx.members), "record_count": len(ctx.facts)})
        except Exception as e:
            out.append({**f, "error": str(e)[:200]})
    return out


@app.post("/api/graph/sync-all")
def graph_sync_all():
    res = {}
    for f in store.families():
        res[f["family_id"]] = graph_store.sync(store.get(f["family_id"])) is not None
    return res


@app.get("/api/family/{family_id}/context")
def context(family_id: str, member_id: Optional[str] = Query(None)):
    """Contract B — enriched family context, filtered to what member_id may see."""
    ctx = _ctx(family_id)
    return ctx.to_contract_b(_viewer(ctx, member_id))


@app.get("/api/family/{family_id}/overview")
def overview(family_id: str, member_id: Optional[str] = Query(None)):
    ctx = _ctx(family_id)
    viewer = _viewer(ctx, member_id)
    cur = ctx.current_facts(viewer)
    s = finance.snapshot(ctx, cur)
    _, hidden = ctx.visible_facts(viewer)
    members = []
    for m in ctx.members.values():
        mine = [f for f in cur if f.record.member_id == m.member_id]
        members.append({
            **m.model_dump(mode="json"),
            "monthly_income": str(s.income_by_member[m.member_id]),
            "holdings": [{"record_id": f.record.record_id, "type": f.record.type, "category": f.record.category,
                          "label": ctx.entities[f.entity_id].name, "amount": f.record.amount,
                          "frequency": f.record.frequency, "emi": f.record.attributes.get("emi")}
                         for f in mine if f.record.type not in ("transaction",)],
        })
    return {"family_id": family_id, "summary": s.summary(ctx), "members": members,
            "goals": ctx.goals, "withheld_record_count": hidden}


@app.get("/api/family/{family_id}/graph")
def graph(family_id: str, member_id: Optional[str] = Query(None)):
    ctx = _ctx(family_id)
    return ctx.graph(_viewer(ctx, member_id))


@app.get("/api/family/{family_id}/timeline")
def timeline(family_id: str, member_id: Optional[str] = Query(None)):
    ctx = _ctx(family_id)
    return ctx.to_contract_b(_viewer(ctx, member_id))["memory_timeline"]


class AskRequest(BaseModel):
    question: str
    member_id: Optional[str] = None
    history: list[dict] = []


@app.post("/api/family/{family_id}/ask")
def ask(family_id: str, req: AskRequest):
    """Contract C."""
    ctx = _ctx(family_id)
    _viewer(ctx, req.member_id)
    try:
        if llm.enabled():
            try:
                return agentic.ask(family_id, req.question, req.member_id, req.history)
            except (PermissionError, AccessDenied):
                raise
            except Exception as e:
                log.exception("agentic path failed; using deterministic fallback")
                r = agent.ask(family_id, req.question, req.member_id, req.history)
                return {**r, "mode": "deterministic_fallback", "fallback_reason": str(e)[:200]}
        return {**agent.ask(family_id, req.question, req.member_id, req.history), "mode": "deterministic"}
    except (PermissionError, AccessDenied) as e:
        raise HTTPException(403, str(e))


@app.post("/api/family/{family_id}/ask/stream")
def ask_stream(family_id: str, req: AskRequest):
    """Contract C as server-sent events: `trace` events while the agent works, then one `answer` event."""
    ctx = _ctx(family_id)
    _viewer(ctx, req.member_id)

    def gen():
        sse = lambda ev: f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
        if llm.enabled():
            try:
                for ev in agentic.stream(family_id, req.question, req.member_id, req.history):
                    yield sse(ev)
                return
            except Exception as e:
                log.exception("agentic stream failed; using deterministic fallback")
                yield sse({"type": "trace", "node": "Fallback", "detail": f"LLM unavailable ({str(e)[:80]}) — deterministic pipeline", "ms": 0, "kind": "verify"})
                mode = "deterministic_fallback"
        else:
            mode = "deterministic"
        r = agent.ask(family_id, req.question, req.member_id, req.history)
        for t in r["trace"]:
            yield sse({"type": "trace", **t, "kind": "step"})
        yield sse({"type": "answer", "data": {**r, "mode": mode}})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/family/{family_id}/records")
def ingest(family_id: str, batch: RecordBatch):
    """Push endpoint for the Financial Data Engine: new records -> temporal memory update."""
    if batch.family_id != family_id:
        raise HTTPException(400, "family_id mismatch")
    _ctx(family_id)
    try:
        return store.ingest(batch)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/integrations/mcp")
def mcp_info():
    """What the configured Financial Data Engine MCP server exposes."""
    info = mcp_source.describe()
    if not info["configured"]:
        return info
    try:
        info["tools"] = mcp_source.list_tools()
        info["families"] = mcp_source.list_families()
        info["reachable"] = True
    except Exception as e:
        info.update(reachable=False, error=str(e)[:300])
    return info


@app.post("/api/integrations/mcp/test")
def mcp_test(family_id: str):
    """Fetch one family over MCP, map it to Contract A and validate it — without loading it into the context engine."""
    if not mcp_source.configured():
        raise HTTPException(400, "No MCP server configured")
    try:
        raw = mcp_source.raw_family(family_id)
    except Exception as e:
        return {"stage": "fetch", "ok": False, "error": str(e)[:400]}
    try:
        payload = mcp_source.to_contract_a(raw, family_id)
    except Exception as e:
        return {"stage": "mapping", "ok": False, "error": str(e)[:400],
                "raw_preview": json.dumps(raw, default=str)[:1500]}
    report = validate_contract(payload)
    return {"stage": "validation", "ok": report["valid"], **report}


@app.get("/api/integrations/mf-engine")
def mf_engine_info(family_id: Optional[str] = None):
    """MF Engine investor MCP: connection, allowlisted tools visible to the agent, linked members."""
    out = mf_engine.status(family_id)
    if out["configured"]:
        out["agent_tools"] = [t["name"] for t in mf_engine.tool_defs(force=True)]
        out["tools_error"] = mf_engine.status()["tools_error"]
    return out


class LiveConsentRequest(BaseModel):
    viewer_id: str                     # who is making the change; must be the member themselves
    granted: bool
    scope: Optional[list[str]] = None  # subset of member_gate.SCOPES; None = every family-safe tool


@app.get("/api/family/{family_id}/consent")
def live_consent_status(family_id: str):
    """Who shares live MF Engine data with the family, and who has linked an MF Engine login at all."""
    from . import member_gate
    ctx = _ctx(family_id)
    linked = set(mf_engine.linked_members(family_id))
    return {"scopes": member_gate.SCOPES, "members": [
        {"member_id": m, "name": ctx.members[m].name, "linked": m in linked,
         "share_live_mf_with_family": bool(ctx.access.get(m, {}).get("share_live_mf_with_family")),
         "live_mf_consent": ctx.access.get(m, {}).get("live_mf_consent")} for m in ctx.members]}


@app.post("/api/family/{family_id}/consent/{member_id}")
def set_live_consent(family_id: str, member_id: str, req: LiveConsentRequest):
    """Only the member can grant or revoke sharing of their own live mutual-fund data."""
    from . import member_gate
    ctx = _ctx(family_id)
    _viewer(ctx, member_id)
    if req.viewer_id != member_id:
        raise HTTPException(403, "only the member themselves can change sharing of their live mutual-fund data")
    bad = sorted(set(req.scope or []) - set(member_gate.SCOPES))
    if bad:
        raise HTTPException(422, f"unknown scope(s) {bad}; valid: {member_gate.SCOPES}")
    return store.set_live_consent(family_id, member_id, req.granted, req.scope)


@app.post("/api/contract/validate")
def validate_contract(payload: dict):
    """Dry run for the Financial Data Engine: validate a Contract A payload (full snapshot or record batch)
    without storing anything. Returns every error with its JSON path."""
    model = RecordBatch if "members" not in payload else FamilyFinancialData
    try:
        obj = model.model_validate(payload)
    except ValidationError as e:
        return {"valid": False, "kind": model.__name__,
                "errors": [{"path": ".".join(str(p) for p in err["loc"]), "error": err["msg"]} for err in e.errors()]}
    warnings = []
    for r in obj.financial_records:
        if r.type == "loan" and "emi" not in r.attributes:
            warnings.append(f"{r.record_id}: loan without attributes.emi — cash flow will flag it as missing")
        if r.type in ("bank_balance", "investment", "loan", "asset") and r.as_of is None:
            warnings.append(f"{r.record_id}: {r.type} without as_of — valid_from used for ordering")
        if r.entity is None and r.type in ("bank_balance", "investment", "loan"):
            warnings.append(f"{r.record_id}: no entity — repeated statements cannot be linked to the same account")
    return {"valid": True, "kind": model.__name__, "records": len(obj.financial_records), "warnings": warnings}


@app.post("/api/family/{family_id}/reset")
def reset(family_id: str):
    """Demo helper: drop pushed records and remembered goals, and re-pull from the Financial Data Engine."""
    store.reset(family_id)
    _ctx(family_id)  # rebuild + re-sync the Neo4j mirror
    return {"ok": True}


@app.post("/api/family/{family_id}/graph/sync")
def graph_sync(family_id: str):
    """Force a re-sync of the Neo4j mirror (e.g. after starting Neo4j)."""
    st = graph_store.sync(_ctx(family_id))
    if st is None:
        raise HTTPException(503, graph_store.status().get("error") or "Neo4j unavailable")
    return st
