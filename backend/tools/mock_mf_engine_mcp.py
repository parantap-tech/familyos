"""Stand-in for MF Engine's /mcp/investor server, for testing FamilyOS plumbing only.

Mimics the documented behaviour: every tool needs an investor identity header (access-token or x-agentic-user-id),
returns a JSON *string* in content[0].text, and reports failures as {"error": true, "code", "message"}.
Numbers are synthetic but internally consistent: every tool derives its output from the same lot-level holdings,
so tax buckets, ELSS lock-in, exit loads and FIFO behave the way FamilyOS should expect from the real tools.
The real maths lives in MF Engine (src/mcp/investor_support/investor_support_household.service.ts).

    .venv/bin/python tools/mock_mf_engine_mcp.py 9300      → http://127.0.0.1:9300/mcp
    tools/seed_live_demo.py writes the matching identities file and consent states.
"""
import json
import sys
from datetime import date, timedelta

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.context import Context

server = MCPServer("mf-investor-mcp-mock")
TODAY = date(2026, 11, 26)
EXEMPTION, LTCG, STCG, SLAB = 125_000, 0.125, 0.20, 0.30


def lot(days, invested, value, pre2023=False):
    return {"days": days, "invested": invested, "value": value, "pre2023": pre2023}


def fund(scheme_id, name, kind, folio, lots, exit_load=None):
    """kind: equity | elss | debt. exit_load: (pct, months)."""
    return {"scheme_id": scheme_id, "scheme": name, "kind": kind, "folio": folio, "lots": lots, "exit_load": exit_load}


# ── personas, keyed by the identity value FamilyOS sends ─────────────────────
P = {
    # F001 Mehta: Rajesh is the MF Engine family head; Arjun asks; Priya is linked but does not share by default.
    "demo-rajesh": {"name": "Rajesh Mehta", "pan": "ABCPM1234K", "email": "rajesh@example.com", "family": "mehta", "head": True,
                    "realised_ltcg_fy": 40_000,
                    "funds": [fund("SCH-101", "Flexi Cap Fund – Direct Growth", "equity", "4455667/12",
                                   [lot(1900, 600_000, 1_350_000), lot(200, 300_000, 330_000)]),
                              fund("SCH-401", "Liquid Fund – Direct Growth", "debt", "8899001/03", [lot(400, 1_200_000, 1_260_000)]),
                              fund("SCH-150", "Thematic Innovation Fund", "equity", "3344556/78",
                                   [lot(500, 400_000, 330_000), lot(90, 200_000, 185_000)], exit_load=(1.0, 12))],
                    "sips": [{"scheme_id": "SCH-101", "amount": 25_000, "day": 5, "status": "ACTIVE"}]},
    "demo-priya": {"name": "Priya Mehta", "pan": "CDEPM9012M", "email": "priya@example.com", "family": "mehta",
                   "realised_ltcg_fy": 0,
                   "funds": [fund("SCH-330", "Corporate Bond Fund", "debt", "7788990/21",
                                  [lot(1500, 500_000, 640_000, pre2023=True), lot(300, 200_000, 214_000)]),
                             fund("SCH-205", "Midcap Fund – Direct Growth", "equity", "1122335/56",
                                  [lot(120, 300_000, 342_000)], exit_load=(1.0, 12))],
                   "sips": [{"scheme_id": "SCH-205", "amount": 10_000, "day": 10, "status": "ACTIVE", "mandate": "EXPIRED"}]},
    "demo-arjun": {"name": "Arjun Mehta", "pan": "BCDPM5678L", "email": "arjun@example.com", "family": "mehta",
                   "realised_ltcg_fy": 0,
                   "funds": [fund("SCH-205", "Midcap Fund – Direct Growth", "equity", "1122334/55",
                                  [lot(330, 520_000, 700_000)], exit_load=(1.0, 12)),
                             fund("SCH-512", "ELSS Tax Saver – Direct Growth", "elss", "5566778/90",
                                  [lot(1300, 150_000, 210_000), lot(600, 170_000, 240_000)])],
                   "sips": [{"scheme_id": "SCH-512", "amount": 12_500, "day": 1, "status": "ACTIVE"},
                            {"scheme_id": "SCH-205", "amount": 5_000, "day": 15, "status": "CANCELLED"}]},
    # F002 Iyer: consent matrix (see seed_live_demo.py).
    "demo-suresh": {"name": "Suresh Iyer", "pan": "DEFPI3456N", "email": "suresh@example.com", "family": "iyer", "head": True,
                    "realised_ltcg_fy": 0,
                    "funds": [fund("SCH-101", "Flexi Cap Fund – Direct Growth", "equity", "2233445/67", [lot(1400, 1_330_000, 1_850_000)]),
                              fund("SCH-205", "Midcap Fund – Direct Growth", "equity", "2233446/68", [lot(900, 840_000, 1_250_000)]),
                              fund("SCH-318", "Short Duration Debt Fund", "debt", "2233447/69", [lot(200, 980_000, 1_100_000)])],
                    "sips": [{"scheme_id": "SCH-101", "amount": 20_000, "day": 7, "status": "ACTIVE"}]},
    "demo-meera": {"name": "Meera Iyer", "pan": "EFGPI7890P", "email": "meera@example.com", "family": "iyer", "realised_ltcg_fy": 125_000,
                   "funds": [fund("SCH-101", "Flexi Cap Fund – Direct Growth", "equity", "6677889/01", [lot(800, 400_000, 560_000)])],
                   "sips": []},
    "demo-venkatesh": {"name": "Venkatesh Iyer", "pan": "FGHPI1122Q", "email": "venkatesh@example.com", "family": "iyer", "realised_ltcg_fy": 0,
                       "funds": [fund("SCH-318", "Short Duration Debt Fund", "debt", "9900112/34", [lot(2000, 800_000, 1_040_000, pre2023=True)])],
                       "sips": []},
    "demo-kamala": {"name": "Kamala Iyer", "pan": "GHIPI3344R", "email": "kamala@example.com", "family": "iyer", "realised_ltcg_fy": 0,
                    "funds": [fund("SCH-401", "Liquid Fund – Direct Growth", "debt", "1213141/51", [lot(700, 300_000, 321_000)])], "sips": []},
    "demo-empty": {"name": "Rohan Iyer", "family": "iyer", "realised_ltcg_fy": 0, "funds": [], "sips": []},
}
ERRORS = {"demo-expired": {"error": True, "code": "UNAUTHORIZED", "message": "MF Engine access token has expired; re-link the account"}}


def _who(ctx: Context):
    h = {k.lower(): v for k, v in (ctx.headers or {}).items()}
    if h.get("x-mcp-take-action"):
        return None, json.dumps({"error": True, "code": "UNEXPECTED_HEADER", "message": "take-action header received"})
    ident = h.get("access-token") or h.get("x-agentic-user-id")
    if ident in ERRORS:
        return None, json.dumps(ERRORS[ident])
    if not ident or ident not in P:
        return None, json.dumps({"error": True, "code": "FORBIDDEN", "message": "Investor context required"})
    return P[ident], None


# ── lot maths (synthetic but consistent) ─────────────────────────────────────
def lots(f, as_of_days=0):
    out = []
    for l in f["lots"]:
        days = l["days"] + as_of_days
        gain = l["value"] - l["invested"]
        if f["kind"] in ("equity", "elss"):
            bucket = "EQUITY_LTCG" if days >= 365 else "EQUITY_STCG"
        else:
            bucket = "DEBT_LTCG" if l["pre2023"] and days > 730 else "DEBT_SLAB"
        locked = f["kind"] == "elss" and days < 1095
        load = f["exit_load"][0] if f["exit_load"] and days < f["exit_load"][1] * 30.4 else 0.0
        out.append({**l, "days": days, "gain": gain, "bucket": bucket, "locked": locked, "exit_load_pct": load,
                    "purchase_date": (TODAY - timedelta(days=l["days"])).isoformat()})
    return sorted(out, key=lambda x: -x["days"])  # FIFO: oldest first


def rate(bucket):
    return {"EQUITY_STCG": STCG, "EQUITY_LTCG": LTCG, "DEBT_LTCG": LTCG}.get(bucket, SLAB)


def current_value(p):
    return sum(l["value"] for f in p["funds"] for l in f["lots"])


def invested(p):
    return sum(l["invested"] for f in p["funds"] for l in f["lots"])


def harvest_plan(p):
    remaining = max(0, EXEMPTION - p["realised_ltcg_fy"])
    left = remaining
    book, losses = [], []
    for f in sorted(p["funds"], key=lambda f: -sum(max(0, l["value"] - l["invested"]) for l in f["lots"])):
        lead = []
        for l in lots(f):
            if l["locked"] or l["bucket"] != "EQUITY_LTCG":
                break
            lead.append(l)
        g = sum(l["gain"] for l in lead)
        if g > 0 and left > 0:
            take = min(left, g)
            left -= take
            book.append({"scheme_id": f["scheme_id"], "scheme_name": f["scheme"], "ltcg_gain_to_book": take,
                         "approx_redeem_value": round(sum(l["value"] for l in lead) * take / g)})
        st = -sum(l["gain"] for l in lots(f) if l["gain"] < 0 and not l["locked"] and "LTCG" not in l["bucket"])
        lt = -sum(l["gain"] for l in lots(f) if l["gain"] < 0 and not l["locked"] and "LTCG" in l["bucket"])
        if st + lt >= 5_000:
            losses.append({"scheme_id": f["scheme_id"], "scheme_name": f["scheme"], "short_term_loss": st, "long_term_loss": lt})
    booked = remaining - left
    return {"realized_ltcg_this_fy": p["realised_ltcg_fy"], "ltcg_exemption_remaining": remaining,
            "gain_harvest": book, "gain_to_book_total": booked, "tax_saved_vs_future_ltcg": round(booked * LTCG),
            "loss_harvest": losses}


def ok(d):
    return json.dumps({"as_of": TODAY.isoformat(), **d})


# ── tools ────────────────────────────────────────────────────────────────────
@server.tool(description="Portfolio summary for the asking investor.")
def getPortfolioSummary(ctx: Context) -> str:
    p, err = _who(ctx)
    if err:
        return err
    return ok({"investor": p["name"], **({"pan": p["pan"], "email": p["email"]} if "pan" in p else {}),
               "invested": invested(p), "current_value": current_value(p),
               "holdings": [{"scheme_id": f["scheme_id"], "scheme": f["scheme"], "folio": f["folio"],
                             "value": sum(l["value"] for l in f["lots"]), "invested": sum(l["invested"] for l in f["lots"])}
                            for f in p["funds"]]})


@server.tool(description="Allocation by asset class.")
def getPortfolioAllocationAnalysis(ctx: Context) -> str:
    p, err = _who(ctx)
    if err:
        return err
    by = {}
    for f in p["funds"]:
        k = "debt" if f["kind"] == "debt" else "equity"
        by[k] = by.get(k, 0) + sum(l["value"] for l in f["lots"])
    total = sum(by.values()) or 1
    return ok({"asset_class": [{"asset_class": k, "value": v, "pct": round(v * 100 / total, 1)} for k, v in by.items()]})


@server.tool(description="STCG/LTCG summary for the financial year.")
def getCapitalGainsSummary(ctx: Context, financial_year: str | None = None) -> str:
    p, err = _who(ctx)
    if err:
        return err
    taxable = max(0, p["realised_ltcg_fy"] - EXEMPTION)
    return ok({"financial_year": financial_year or "2026-27",
               "summary": {"equity": {"ltcg": {"gain": p["realised_ltcg_fy"], "exemption": EXEMPTION, "taxable_gain": taxable,
                                               "estimated_tax": round(taxable * LTCG)}}}})


@server.tool(description="Holdings with unrealised losses above a threshold.")
def getHarvestingOpportunities(ctx: Context, min_loss_amount: float = 5000) -> str:
    p, err = _who(ctx)
    if err:
        return err
    c = [{"scheme_id": f["scheme_id"], "scheme_name": f["scheme"], "unrealised_loss": -l["gain"], "days_held": l["days"]}
         for f in p["funds"] for l in lots(f) if -l["gain"] >= min_loss_amount]
    return ok({"min_loss_threshold": min_loss_amount, "total_candidates": len(c), "candidates": c})


@server.tool(description="Cash raisable by a date, cheapest first, after exit load and estimated tax.")
def getLiquidityLadder(ctx: Context, target_amount: float | None = None, needed_by: str | None = None, slab_rate_pct: float | None = None) -> str:
    p, err = _who(ctx)
    if err:
        return err
    ahead = max(0, (date.fromisoformat(needed_by) - TODAY).days) if needed_by else 0
    exempt = max(0, EXEMPTION - p["realised_ltcg_fy"])
    schemes, locked_value, waits = [], 0, []
    for f in p["funds"]:
        tr = []
        for l in lots(f, ahead):
            if l["locked"]:
                locked_value += l["value"]
                waits.append({"scheme_name": f["scheme"], "value": l["value"], "reasons": ["ELSS still locked"]})
                continue
            load = l["value"] * l["exit_load_pct"] / 100
            tr.append({**l, "load": load})
            if l["exit_load_pct"]:
                waits.append({"scheme_name": f["scheme"], "value": l["value"], "reasons": [f"exit load {l['exit_load_pct']}% still applies"]})
            if l["bucket"] == "EQUITY_STCG" and l["gain"] > 0 and 365 - l["days"] <= 60:
                waits.append({"scheme_name": f["scheme"], "value": l["value"], "reasons": [f"turns LTCG in {365 - l['days']} days"]})
        if tr:
            cost = sum(t["load"] + max(0, t["gain"]) * rate(t["bucket"]) for t in tr) / sum(t["value"] for t in tr)
            schemes.append((cost, f, tr))
    order, net_total = [], 0.0
    for _, f, tr in sorted(schemes, key=lambda s: s[0]):
        gross = load = tax = 0.0
        full = True
        for t in tr:
            ex = min(exempt, max(0, t["gain"])) if t["bucket"] == "EQUITY_LTCG" else 0
            t_tax = max(0, max(0, t["gain"]) - ex) * rate(t["bucket"])
            t_net = t["value"] - t["load"] - t_tax
            if target_amount is not None and net_total >= target_amount:
                full = False
                continue
            frac = 1.0
            if target_amount is not None and net_total + t_net > target_amount:
                frac, full = (target_amount - net_total) / t_net, False
            exempt -= ex * frac
            gross, load, tax, net_total = gross + t["value"] * frac, load + t["load"] * frac, tax + t_tax * frac, net_total + t_net * frac
        if gross:
            order.append({"scheme_id": f["scheme_id"], "scheme_name": f["scheme"], "folio": f["folio"], "redeem_gross": round(gross),
                          "exit_load": round(load), "estimated_tax": round(tax), "net_proceeds": round(gross - load - tax), "full_exit": full})
    return ok({"target_amount": target_amount, "target_met": None if target_amount is None else round(net_total) >= round(target_amount),
               "shortfall": None if target_amount is None else round(max(0, target_amount - net_total)),
               "totals": {"planned_net": round(net_total), "locked_elss_value": locked_value},
               "redemption_order": order, "cheaper_if_you_wait": waits})


@server.tool(description="Household LTCG harvest plan; only the MF Engine family head sees every member.")
def getHouseholdTaxHarvestPlan(ctx: Context, min_loss_amount: float | None = None) -> str:
    p, err = _who(ctx)
    if err:
        return err
    head = bool(p.get("head"))
    members = [q for q in P.values() if q.get("family") == p["family"]] if head else [p]
    plans = [{"name": q["name"], "is_family_head": bool(q.get("head")), "has_portfolio": bool(q["funds"]),
              **(harvest_plan(q) if q["funds"] else {})} for q in members]
    return ok({"viewer_is_family_head": head,
               **({} if head else {"visibility_note": "Only the family head can see other members' holdings. Showing your own plan only."}),
               "household": {"members_considered": len(plans),
                             "ltcg_gain_to_book_total": sum(x.get("gain_to_book_total", 0) for x in plans),
                             "estimated_tax_saved": sum(x.get("tax_saved_vs_future_ltcg", 0) for x in plans)},
               "members": plans})


@server.tool(description="Family roll-up of linked members (head only sees everyone).")
def getFamilyPortfolioSummary(ctx: Context) -> str:
    p, err = _who(ctx)
    if err:
        return err
    members = [q for q in P.values() if q.get("family") == p["family"]] if p.get("head") else [p]
    return ok({"viewer_is_family_head": bool(p.get("head")), "family_total_value": sum(current_value(q) for q in members),
               "members": [{"name": q["name"], "current_value": current_value(q)} for q in members]})


@server.tool(description="SIP book.")
def getSipBook(ctx: Context) -> str:
    p, err = _who(ctx)
    return err or ok({"sips": [{k: v for k, v in s.items() if k != "mandate"} for s in p["sips"]],
                      "monthly_total": sum(s["amount"] for s in p["sips"] if s["status"] == "ACTIVE")})


@server.tool(description="Upcoming SIP instalments.")
def getSipSchedule(ctx: Context, days: int = 30) -> str:
    p, err = _who(ctx)
    if err:
        return err

    def nxt(d):
        this = TODAY.replace(day=d)
        return (this if TODAY.day < d else (TODAY.replace(day=1) + timedelta(days=32)).replace(day=d)).isoformat()
    return ok({"upcoming": [{"scheme_id": s["scheme_id"], "amount": s["amount"], "date": nxt(s["day"])}
                            for s in p["sips"] if s["status"] == "ACTIVE"]})


@server.tool(description="Things the investor must act on (expired mandates, failed instalments).")
def getPendingActions(ctx: Context) -> str:
    p, err = _who(ctx)
    if err:
        return err
    acts = [{"type": "MANDATE_EXPIRED", "scheme_id": s["scheme_id"], "message": "Next SIP instalment will fail: renew the mandate"}
            for s in p["sips"] if s.get("mandate") == "EXPIRED"]
    return ok({"pending": acts, "count": len(acts)})


@server.tool(description="Portfolio health: 0-100 score, grade, top 3 actions.")
def getPortfolioHealthScore(ctx: Context) -> str:
    p, err = _who(ctx)
    return err or ok({"score": 72, "grade": "B", "top_actions": ["Midcap allocation above target", "No emergency debt bucket"]})


@server.tool(description="Tax-aware plan to raise a target amount by redeeming funds.")
def getSmartRedemptionPlan(ctx: Context, target_amount: float, goal: str = "tax", source: str = "all") -> str:
    return getLiquidityLadder(ctx, target_amount=target_amount)


@server.tool(description="MUTATION: saves risk profile (must never be exposed by FamilyOS).")
def submitRiskProfile(ctx: Context, answers: dict) -> str:
    return json.dumps({"dry_run": True})


if __name__ == "__main__":
    server.run(transport="streamable-http", host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 9300)
