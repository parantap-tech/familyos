import json
import os
import shutil
from decimal import Decimal
from pathlib import Path

import pytest

os.environ["FAMILYOS_DISABLE_LLM"] = "1"
os.environ["FAMILYOS_NEO4J"] = "0"

from fastapi.testclient import TestClient  # noqa: E402

from app import context_engine, finance  # noqa: E402
from app.contract import FamilyFinancialData  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    for f in ("F001.financial.json", "F001.access.json"):
        shutil.copy(DATA / f, tmp_path / f)
    monkeypatch.setattr(context_engine.store, "dir", tmp_path)
    context_engine.store._cache.clear()
    from app.main import app
    yield TestClient(app)
    context_engine.store._cache.clear()


def payload():
    return json.loads((DATA / "F001.financial.json").read_text())


# ---------------- Contract A
def test_mock_conforms_to_contract():
    FamilyFinancialData.model_validate(payload())


@pytest.mark.parametrize("mutate,err", [
    (lambda p: p["financial_records"][0].update(amount=250000.5), "string"),
    (lambda p: p["financial_records"][0].update(amount="2.5L"), "decimal"),
    (lambda p: p["financial_records"][0].update(member_id="M999"), "unknown member"),
    (lambda p: p["financial_records"][1].update(record_id="R001"), "duplicate"),
    (lambda p: p["financial_records"][0].pop("source"), "source"),
])
def test_contract_rejects_bad_payloads(mutate, err):
    p = payload()
    mutate(p)
    with pytest.raises(Exception) as e:
        FamilyFinancialData.model_validate(p)
    assert err.lower() in str(e.value).lower()


def test_missing_amount_is_unknown_not_zero(client):
    ctx = context_engine.store.get("F001")
    ctx.facts["R012"].record.amount = None
    s = finance.snapshot(ctx, ctx.current_facts(None))
    assert s.income_by_member["M002"] == 0  # excluded from sum...
    assert any("R012" in m for m in s.missing)  # ...but flagged, never silently zero


# ---------------- temporal memory
def test_supersession_and_history(client):
    ctx = context_engine.store.get("F001")
    assert ctx.facts["R001"].status == "superseded" and ctx.facts["R001"].superseded_by == "R002"
    assert ctx.facts["R003"].superseded_by == "R004"
    assert ctx.facts["R009"].status == ctx.facts["R010"].status == "event"  # transactions append
    assert ctx.facts["R011"].status == "ended"


def test_entity_resolution_merges_name_variants(client):
    ctx = context_engine.store.get("F001")
    assert ctx.facts["R003"].entity_id == ctx.facts["R004"].entity_id == ctx.facts["R009"].entity_id


def test_ingest_job_change_supersedes_salary(client):
    rec = {**payload()["financial_records"][1], "record_id": "R100", "amount": "300000",
           "valid_from": "2026-09-01", "description": "Salary — Orbit Systems",
           "entity": {"entity_id": "EMP-ORBIT", "entity_type": "employer", "name": "Orbit Systems"},
           "source": {"document_id": "D100", "filename": "rajesh_offer_orbit.pdf"}}
    r = client.post("/api/family/F001/records", json={"schema_version": "1.0", "family_id": "F001", "financial_records": [rec]})
    assert r.status_code == 200, r.text
    assert r.json()["memory_changes"][0]["superseded_record_id"] == "R002"
    ctx = context_engine.store.get("F001")
    assert ctx.facts["R002"].status == "superseded" and ctx.facts["R001"].status == "superseded"
    assert client.post("/api/family/F001/records", json={"schema_version": "1.0", "family_id": "F001", "financial_records": [rec]}).status_code == 422


# ---------------- access control
def test_private_record_hidden_from_other_members(client):
    as_father = client.get("/api/family/F001/context", params={"member_id": "M001"}).json()
    as_son = client.get("/api/family/F001/context", params={"member_id": "M003"}).json()
    assert "R018" not in {f["record_id"] for f in as_father["financial_context"]}
    assert as_father["withheld_record_count"] == 1
    assert "R018" in {f["record_id"] for f in as_son["financial_context"]}
    assert client.get("/api/family/F001/context", params={"member_id": "X9"}).status_code == 403


# ---------------- calculations
def test_emi_formula():
    assert round(finance.emi(Decimal(12_000_000), Decimal("8.5"), Decimal(20))) == 104139


# ---------------- Contract C end to end
def test_demo_questions(client):
    ask = lambda q, m="M003": client.post("/api/family/F001/ask", json={"question": q, "member_id": m}).json()
    a = ask("Can our family afford to buy a ₹1.5 crore house next year?")
    assert a["intent"] == "affordability"
    assert {"R002", "R012", "R015", "R005"} <= {s["record_id"] for s in a["supporting_records"]}
    for k in ("answer", "key_findings", "supporting_records", "calculations", "assumptions", "confidence"):
        assert k in a
    b = ask("What happens if father retires next year?")
    assert b["intent"] == "income_change" and b["plan"]["scenario_member_id"] == "M001"
    assert any("remembered the goal" in i for i in b["context_insights"])
    c = ask("Which family member contributes the most to our current household income?")
    assert c["key_findings"][0].startswith("Arjun") and "47.6%" in c["key_findings"][0]
    d = ask("Can our family afford to buy a ₹1.5 crore house next year?", "M001")
    assert "R018" not in {s["record_id"] for s in d["supporting_records"]} and d["withheld_record_count"] == 1


# ---------------- LLM narrative guard
def test_grounding_guard_rejects_invented_numbers():
    from app.agent import _grounded
    blob = "income ₹6.3 lakh/month; EMI ₹1,04,139; ratio 22.9%; upfront ₹40.5 lakh"
    assert _grounded("Your family earns ₹6.3 lakh a month and the EMI is about ₹1,04,139 (22.9%).", blob)[0]
    ok, bad = _grounded("You could afford an EMI of ₹1.5 lakh and save ₹12 lakh.", blob)
    assert not ok and bad


# ---------------- agentic toolbox (no API calls)
def test_toolbox_enforces_access_and_scenarios(client):
    from app.agentic import ToolBox
    ctx = context_engine.store.get("F001")
    father, son = ToolBox(ctx, "M001"), ToolBox(ctx, "M003")
    ids = lambda box: {r["record_id"] for r in box.get_financial_records(include_history=True)["records"]}
    assert "R018" not in ids(father) and "R018" in ids(son)
    assert father.get_family_profile()["records_private_to_other_members"] == 1
    base = son.household_cashflow()
    retired = son.household_cashflow(income_overrides=[{"member_id": "M001", "monthly_income": "0"}])
    assert Decimal(base["monthly_income"]) == 630000 and Decimal(retired["monthly_income"]) == 380000
    ev = son.evaluate_property_purchase(price="15000000", horizon_months=12)
    assert ev["verdict"] == "affordable" and ev["new_emi"] == "104139"
    with pytest.raises(ValueError):
        son.household_cashflow(income_overrides=[{"member_id": "M999", "monthly_income": "0"}])


def test_calculator_is_safe_and_exact():
    from app.agentic import safe_eval
    assert safe_eval("(2,50,000 + 80000) * 12") == Decimal("3960000")
    assert safe_eval("round(0.1 + 0.2, 2)") == Decimal("0.30")
    for bad in ("__import__('os')", "open('x')", "2 ** 100000", "a + 1"):
        with pytest.raises(Exception):
            safe_eval(bad)


def test_agent_grounding_check():
    from app.agentic import ungrounded
    assert ungrounded("EMI ₹1.04 lakh, income ₹6.3L, ratio 22.9%", "104139 630000 22.9") == []
    assert ungrounded("you could save ₹9 lakh", "630000") == ["900000"]


def test_graph_tool_rejects_writes():
    from app import graph_store
    for q in ("MATCH (n) DETACH DELETE n", "CREATE (x:Member)", "MATCH (n) SET n.a = 1", "LOAD CSV FROM 'x' AS r RETURN r"):
        with pytest.raises(ValueError):
            graph_store.read_query(q, "F001")


# ---------------- demo families F002–F004
@pytest.fixture()
def all_families(tmp_path, monkeypatch):
    for f in DATA.glob("F00*.*.json"):
        if f.name.split(".")[1] in ("financial", "access"):
            shutil.copy(f, tmp_path / f.name)
    shutil.copy(DATA / "families.json", tmp_path / "families.json")
    monkeypatch.setattr(context_engine.store, "dir", tmp_path)
    context_engine.store._cache.clear()
    yield context_engine.store
    context_engine.store._cache.clear()


def test_all_families_validate_and_build(all_families):
    ids = [f["family_id"] for f in all_families.families()]
    assert ids == ["F001", "F002", "F003", "F004"]
    for fid in ids:
        ctx = all_families.get(fid)
        s = finance.snapshot(ctx, ctx.current_facts(None))
        assert s.monthly_income > 0 and s.surplus is not None


def test_joint_family_links_privacy_and_locked_assets(all_families):
    ctx = all_families.get("F002")
    rels = {(r["from"], r["type"], r["to"]) for r in ctx.relationships}
    assert ("M201", "grandparent_of", "M205") in rels and ("M203", "sibling_of", "M207") in rels
    assert ("M205", "covered_by", "POL-FLOATER-M203") in rels
    assert ("LOAN-CAR-M203", "secured_against", "VEH-M203") in rels
    assert ("LOAN-CAR-M203", "secured_against", "PROP-WHITEFIELD") not in rels  # no blanket loan→property links
    vis = {f.record.record_id for f in ctx.visible_facts("M203")[0]}
    assert {"R250", "R251", "R252", "R240"}.isdisjoint(vis)   # Karthik shares nothing; Meera's FD is private
    assert "R206" in vis                                        # …but the co-owned ancestral house is visible
    s = finance.snapshot(ctx, ctx.current_facts("M203"))
    locked_ids = {i for i, _ in s.locked}
    assert {"R202", "R216", "R217", "R220", "R238"} <= locked_ids
    assert {x["currency"] for x in s.foreign} == {"CAD"}        # not silently mixed into INR


def test_unknown_amount_and_job_change(all_families):
    ctx = all_families.get("F002")
    s = finance.snapshot(ctx, ctx.current_facts(None))
    assert any("R251" in m for m in s.missing)                  # founder equity: unknown, not zero
    kr = all_families.get("F004")
    assert kr.facts["R407"].superseded_by == "R408"            # new employer's salary supersedes old one
    assert kr.facts["R414"].status == "ended"                  # employer cover ended after job change


# ---------------- MCP integration (mock server over stdio)
def test_mcp_mapping_passthrough_and_custom_error():
    from app.mcp_mapping import to_contract_a
    p = payload()
    assert to_contract_a(p, "F001") is p
    no_version = {k: v for k, v in p.items() if k != "schema_version"}
    assert to_contract_a(no_version, "F001")["schema_version"] == "1.0"
    with pytest.raises(ValueError, match="map_custom"):
        to_contract_a({"customer": {}, "accounts": []}, "F001")


def test_mcp_stdio_roundtrip(monkeypatch):
    import sys
    from app import mcp_source
    from app.contract import FamilyFinancialData
    server = Path(__file__).resolve().parent.parent / "tools" / "mock_financial_mcp.py"
    monkeypatch.setenv("FINANCIAL_ENGINE_MCP_COMMAND", f"{sys.executable} {server}")
    assert {t["name"] for t in mcp_source.list_tools()} >= {"get_family_financial_data", "list_families"}
    data = FamilyFinancialData.model_validate(mcp_source.fetch_family("F003"))
    assert len(data.members) == 4 and len(data.financial_records) == 25
    with pytest.raises(RuntimeError):
        mcp_source.fetch_family("F999")


def test_mf_engine_guards(monkeypatch, tmp_path):
    from app import mf_engine as m
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps({"mcp_token": "t", "members": {"F002:M203": {"access_token": "a"}}}))
    monkeypatch.setenv("MF_ENGINE_IDENTITIES", str(ids))
    assert m.linked_members("F002") == ["M203"]
    assert "x-mcp-take-action" not in m._headers(m.identity("F002", "M203"), None)
    assert not (set(m.ALLOWLIST) & m.MUTATIONS)
    with pytest.raises(m.MFEngineError, match="NOT_ALLOWED"):
        m.call("mf_submitRiskProfile", {}, "F002", "M203")
    with pytest.raises(m.MFEngineError, match="NO_IDENTITY"):      # unlinked member: refused before any network call
        m.call("mf_getPortfolioHealthScore", {}, "F002", "M204")


def test_cas_same_isin_in_two_folios_are_separate_holdings():
    from app.cas_nsdl import CasParse, Holding, to_contract_a
    from datetime import date
    cas = CasParse(as_of=date(2026, 9, 30), holder_name="Test",
                   holdings=[Holding("mf_folio", "INF879O01027", "Flexi Cap", Decimal(10), Decimal(1), Decimal(100)),
                             Holding("mf_folio", "INF879O01027", "Flexi Cap", Decimal(5), Decimal(1), Decimal(50))])
    p = to_contract_a(cas, "P999", "P999-M1", "x.pdf")
    assert len({r["entity"]["entity_id"] for r in p["financial_records"]}) == 2
    assert "PAN" not in json.dumps(p) and all(k in ("member_id", "name", "relationship") for k in p["members"][0])


# ---------------- family coordinator phase 1: consented MF Engine fan-out (no network)
@pytest.fixture()
def live_family(client, tmp_path, monkeypatch):
    """F001 with all three members linked to MF Engine; mf_engine.call recorded instead of hitting the network."""
    from app import member_gate, mf_engine
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps({"mcp_token": "t", "members": {f"F001:{m}": {"access_token": f"tok-{m}"} for m in ("M001", "M002", "M003")}}))
    monkeypatch.setenv("MF_ENGINE_IDENTITIES", str(ids))
    monkeypatch.setenv("MF_ENGINE_MCP_URL", "http://mf-engine.invalid/mcp/investor")
    monkeypatch.setattr(member_gate, "AUDIT_DIR", tmp_path / "audit")
    monkeypatch.setattr(mf_engine, "tool_defs", lambda force=False: [])
    calls = []

    def fake_call(tool, args, family_id, member_id, conversation_id=None, request_id=None):
        calls.append((member_id, tool, request_id))
        return {"pan": "ABCDE1234F", "email": "x@example.com", "folio_number": "1234567890",
                "holdings": [{"scheme_name": "Alpha Flexi Cap", "folio": "98765/43", "current_value": 100000}]}
    monkeypatch.setattr(mf_engine, "call", fake_call)
    return client, calls, tmp_path


def _consent(client, member, granted=True, scope=None, viewer=None):
    return client.post(f"/api/family/F001/consent/{member}", json={"viewer_id": viewer or member, "granted": granted, "scope": scope})


def test_fan_out_calls_only_consenting_members_and_redacts(live_family):
    from app.agentic import ToolBox
    client, calls, tmp = live_family
    assert _consent(client, "M001").status_code == 200          # Rajesh opts in; Priya does not
    box = ToolBox(context_engine.store.get("F001"), "M003")     # Arjun asks
    out = box.run("mf_family", {"tool": "getPortfolioSummary"})
    assert {c[0] for c in calls} == {"M001", "M003"}             # Priya's login is never used
    assert out["withheld"] == ["M002"] and out["members"]["M002"]["status"] == "no_consent"
    rajesh, arjun = out["members"]["M001"]["data"], out["members"]["M003"]["data"]
    assert "pan" not in rajesh and "email" not in rajesh and rajesh["folio_number"].endswith("7890") and "123456" not in rajesh["folio_number"]
    assert rajesh["holdings"][0]["folio"] == "••••5/43"
    assert arjun["pan"] == "ABCDE1234F"                          # the asker's own payload is untouched
    assert out["members"]["M001"]["source_id"].startswith("MF:M001:getPortfolioSummary:familyos-")
    audit = [json.loads(l) for l in (tmp / "audit" / "F001.live_mf.jsonl").read_text().splitlines()]
    assert [(a["asker"], a["on_behalf_of"]) for a in audit] == [("M003", "M001")]
    assert audit[0]["request_id"] == next(c[2] for c in calls if c[0] == "M001")
    assert box.withheld_live == {"M002": "has not shared live mutual-fund data with the family"}
    assert {c["member_id"] for c in box.live_calls} == {"M001", "M003"}


def test_consent_is_self_service_scoped_and_revocable(live_family):
    from app.agentic import ToolBox
    client, calls, _ = live_family
    assert _consent(client, "M001", viewer="M003").status_code == 403   # Arjun cannot opt his father in
    assert _consent(client, "M001", scope=["bogus"]).status_code == 422
    assert _consent(client, "M001", scope=["liquidity"]).status_code == 200
    box = ToolBox(context_engine.store.get("F001"), "M003")
    out = box.run("mf_family", {"tool": "getCapitalGainsSummary", "member_ids": ["M001"]})
    assert out["members"]["M001"]["status"] == "out_of_scope" and not calls
    box.run("mf_family", {"tool": "getLiquidityLadder", "args": {"target_amount": 3000000}, "member_ids": ["M001"]})
    assert calls[-1][:2] == ("M001", "getLiquidityLadder")
    assert _consent(client, "M001", granted=False).status_code == 200
    n = len(calls)
    ToolBox(context_engine.store.get("F001"), "M003").run("mf_family", {"tool": "getLiquidityLadder", "member_ids": ["M001"]})
    assert len(calls) == n                                        # revoked: no call on the next request
    status = client.get("/api/family/F001/consent").json()
    assert {m["member_id"]: m["share_live_mf_with_family"] for m in status["members"]}["M001"] is False


def test_mf_family_tool_offered_only_when_someone_else_shares(live_family):
    from app.agentic import ToolBox
    client, _, _ = live_family
    names = lambda: {t["name"] for t in ToolBox(context_engine.store.get("F001"), "M003").tool_defs}
    assert "mf_family" not in names()
    _consent(client, "M002")
    assert "mf_family" in names() and list(names())  # offered once Priya shares
    with pytest.raises(ValueError, match="not available for family fan-out"):
        ToolBox(context_engine.store.get("F001"), "M003").run("mf_family", {"tool": "getInvestorProfile"})


def test_verify_requires_withheld_members_to_be_named(live_family):
    from app.agentic import ToolBox, consent_check, withheld_check
    client, _, _ = live_family
    _consent(client, "M001")
    box = ToolBox(context_engine.store.get("F001"), "M003")
    box.run("mf_family", {"tool": "getPortfolioSummary"})
    claim = "Our family can raise 30 lakh from mutual funds by March."
    assert not withheld_check(box, claim, [])["ok"]
    assert withheld_check(box, claim, ["Priya has not shared live mutual-fund data"])["ok"]
    assert withheld_check(box, "Rajesh can raise 10 lakh.", [])["ok"]          # not a family-wide claim
    assert consent_check(box)["ok"]
    _consent(client, "M001", granted=False)                                     # revoked mid-answer
    assert not consent_check(box)["ok"]
