"""FamilyOS agentic reasoning — Claude is the brain, the context engine + finance module are its tools.

LangGraph loop:
    prepare ─▶ agent ⇄ tools
                 │
                 ▼ (final_answer)
               verify ──(ungrounded numbers → error back to Claude)──▶ agent
                 │
                 ▼
               respond  (Contract C)

Guarantees that do NOT depend on the model behaving:
  * access control is enforced inside every tool (Claude never sees records the asker may not see)
  * every number is produced by deterministic tools; `verify` rejects answers quoting numbers no tool produced
  * provenance: calculations and records cited come from tool outputs, not from the model's text
"""
from __future__ import annotations

import ast
import json
import logging
import operator
import os
import re
import time
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Iterator, Optional, TypedDict

from langgraph.graph import END, StateGraph

from . import finance, graph_store, llm, mf_engine
from .context_engine import TODAY, FamilyContext, Fact, store
from .money import CRORE, LAKH, inr, inr_short, pct, q

log = logging.getLogger("familyos.agentic")
MAX_TURNS = 12
MAX_REPAIRS = 2
EFFORT = os.environ.get("FAMILYOS_EFFORT", "low")
SPEED = os.environ.get("FAMILYOS_SPEED", "fast")  # Opus 5.5 fast mode: up to 2.5x output speed for live demos

# ===================================================================== tool definitions
MEMBER_IDS = {"type": "array", "items": {"type": "string"}}
INCOME_OVERRIDES = {
    "type": "array",
    "description": "Scenario: replace a member's total monthly income (e.g. '0' if they retire or lose their job).",
    "items": {"type": "object", "additionalProperties": False, "required": ["member_id", "monthly_income"],
              "properties": {"member_id": {"type": "string"}, "monthly_income": {"type": "string", "description": "INR per month as a decimal string"}}},
}

TOOLS = [
    {
        "name": "get_family_profile",
        "description": "Family members (ids, names, roles, ages), relationships (spouse_of, parent_of, owns, owes, nominee_of …), "
                       "remembered family goals and how many records are private to the asker. Call this first.",
        "input_schema": {"type": "object", "additionalProperties": False, "properties": {}},
    },
    {
        "name": "get_financial_records",
        "description": "Financial records visible to the asker, with provenance (source document) and temporal status "
                       "(current / superseded / ended / event). Use include_history=true to see how values changed over time.",
        "input_schema": {"type": "object", "additionalProperties": False, "properties": {
            "member_ids": {**MEMBER_IDS, "description": "Filter by owner or co-owner; omit for the whole family."},
            "types": {"type": "array", "items": {"type": "string", "enum": ["income", "expense", "bank_balance", "investment", "insurance", "loan", "asset", "liability", "transaction", "tax"]}},
            "include_history": {"type": "boolean", "description": "Include superseded/ended records and transactions."},
        }},
    },
    {
        "name": "household_cashflow",
        "description": "Deterministic household snapshot: income by member, EMIs, expenses, monthly surplus, liquid assets, "
                       "property, liabilities, net worth, income share per member. Supports what-if scenarios.",
        "input_schema": {"type": "object", "additionalProperties": False, "properties": {
            "income_overrides": INCOME_OVERRIDES,
            "exclude_record_ids": {"type": "array", "items": {"type": "string"}, "description": "Scenario: drop these records (e.g. a loan that is closed, an asset that is sold)."},
            "additional_monthly_expense": {"type": "string", "description": "Scenario: extra INR/month of spending."},
            "scenario_label": {"type": "string"},
        }},
    },
    {
        "name": "evaluate_property_purchase",
        "description": "Deterministic affordability check for buying a property: upfront cash (down payment + stamp duty/costs) vs "
                       "funds available at purchase, new EMI, EMI-to-income ratio, surplus after EMI and a verdict. Can be run "
                       "under an income scenario. Defaults: 20% down, 7% costs, 8.5% for 20 yrs, 10% investment haircut, "
                       "50% of surplus saved, 6-month emergency reserve.",
        "input_schema": {"type": "object", "additionalProperties": False, "required": ["price", "horizon_months"], "properties": {
            "price": {"type": "string", "description": "INR, e.g. '15000000' for 1.5 crore"},
            "horizon_months": {"type": "integer", "description": "Months until purchase"},
            "down_payment_pct": {"type": "string"}, "loan_rate_pct": {"type": "string"}, "loan_tenure_years": {"type": "string"},
            "income_overrides": INCOME_OVERRIDES,
        }},
    },
    {
        "name": "portfolio_breakdown",
        "description": "Deterministic investment portfolio analysis over ALL visible holdings: total value, split by asset class "
                       "(listed equity, mutual funds, bonds, gold…), by fund house / issuer, top holdings with % of portfolio, "
                       "and unrealised gains/losses where the statement provides cost. Use this for any portfolio total, share, "
                       "concentration or gain/loss question instead of adding holdings yourself.",
        "input_schema": {"type": "object", "additionalProperties": False, "properties": {
            "member_ids": {**MEMBER_IDS, "description": "Omit for everyone visible to the asker."},
            "top_n": {"type": "integer", "description": "How many top holdings / gainers / losers to list (default 10)."},
        }},
    },
    {
        "name": "calculate",
        "description": "Exact decimal arithmetic for any other number you need (+ - * / ** parentheses, min, max, round). "
                       "You must use this instead of doing maths yourself. Give a short label.",
        "input_schema": {"type": "object", "additionalProperties": False, "required": ["label", "expression"], "properties": {
            "label": {"type": "string"}, "expression": {"type": "string", "description": "e.g. '(250000 + 80000) * 12'"},
        }},
    },
    {
        "name": "get_memory_timeline",
        "description": "Chronological financial events: new facts, updates that superseded earlier facts (salary hikes, new balances) and transactions.",
        "input_schema": {"type": "object", "additionalProperties": False, "properties": {"member_id": {"type": "string"}}},
    },
    {
        "name": "member_dependency_graph",
        "description": "Multi-hop traversal in the Neo4j family graph: everything connected to one member — family links, what they "
                       "earn from / own / owe / are insured by, who co-owns or co-owes each item, nominees, and what loans are secured "
                       "against. Use it to see who else is affected when this member's situation changes.",
        "input_schema": {"type": "object", "additionalProperties": False, "required": ["member_id"], "properties": {"member_id": {"type": "string"}}},
    },
    {
        "name": "query_family_graph",
        "description": "Run a READ-ONLY Cypher query on the Neo4j family graph for relationship questions the other tools don't answer "
                       "(paths between members and assets, shared liabilities, history chains). Use $family_id as a parameter. "
                       "Use the finance tools, not Cypher, for money arithmetic.\n" + graph_store.SCHEMA_DOC,
        "input_schema": {"type": "object", "additionalProperties": False, "required": ["cypher", "purpose"], "properties": {
            "cypher": {"type": "string"}, "purpose": {"type": "string", "description": "One line: what you want to find out."},
        }},
    },
    {
        "name": "remember_goal",
        "description": "Persist a family financial goal stated in conversation (e.g. buying a home) so future questions can re-evaluate it.",
        "input_schema": {"type": "object", "additionalProperties": False, "required": ["kind", "label", "amount", "target_date"], "properties": {
            "kind": {"type": "string", "description": "e.g. property_purchase, education, retirement, vehicle"},
            "label": {"type": "string", "description": "e.g. 'Buy a ₹1.5 crore home'"},
            "amount": {"type": "string"}, "target_date": {"type": "string", "description": "ISO date"},
        }},
    },
    {
        "name": "final_answer",
        "description": "Submit the answer to the family. Call exactly once, after you have gathered evidence with the other tools. "
                       "Every number must come from a tool result.",
        "strict": True,
        "input_schema": {"type": "object", "additionalProperties": False,
                         "required": ["answer", "key_findings", "context_insights", "assumptions", "missing_information", "cited_record_ids", "confidence"],
                         "properties": {
                             "answer": {"type": "string", "description": "The direct answer in at most 4 sentences / 90 words, warm and plain, addressed to the asker. Lead with the verdict."},
                             "key_findings": {"type": "array", "items": {"type": "string"}, "description": "3–4 findings, each under 25 words, with numbers."},
                             "context_insights": {"type": "array", "items": {"type": "string"}, "description": "2–4 cross-member insights, each under 30 words: dependencies, who carries what, life stage, history, risks."},
                             "assumptions": {"type": "array", "items": {"type": "string"}},
                             "missing_information": {"type": "array", "items": {"type": "string"}},
                             "cited_record_ids": {"type": "array", "items": {"type": "string"}},
                             "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                         }},
    },
]

SYSTEM = """You are FamilyOS, an AI that understands a family as one interconnected financial system (India; amounts in INR, \
use lakh/crore naturally). You answer questions from one family member, reasoning across ALL members whose data they may see.

How to work:
- Investigate with tools before answering. Start with get_family_profile, then gather what the question needs. Call independent tools in parallel.
- The family graph lives in Neo4j: use member_dependency_graph to trace who is affected by a member's change (co-borrowers, nominees, secured assets), and query_family_graph for other relationship questions.
- Every number in your answer must come from a tool result. Never do arithmetic in your head — use calculate, household_cashflow or evaluate_property_purchase.
- For portfolio totals, shares, concentration or gains use portfolio_breakdown — never assemble a total by adding a hand-picked list of holdings.
- Think like a family financial planner: who earns, who owns, who owes, life stage (ages, retirement), how one member's change affects others, \
recent changes in history, concentration risk, insurance/nominee coverage, and remembered goals.
- For what-if questions, compare the baseline against the scenario (run both). If the family has a remembered goal, re-evaluate it under the scenario.
- When the family states a new goal (e.g. buying a house), call remember_goal.
- Cite record ids you relied on. Clearly separate verified facts (records), calculated values (tools) and assumptions.
- Missing data is unknown, never zero — say what is missing. Never speculate about records you cannot see; if some records are \
private to other members, say the picture is partial.
- Locked investments (EPF, PPF, NPS, SCSS, Sukanya Samriddhi, ESOPs) are not available for near-term goals; amounts in other \
currencies (CAD, USD) are not converted — mention them in their own currency. Check insurance coverage (who is covered by which \
policy, co-pay, waiting periods), loan moratoriums (future EMIs), and AIS vs ITR consistency when relevant.
- Do not give definitive lending, tax or investment recommendations; frame results as estimates under stated assumptions.
- Tone: these are real families. On protection, illness or loss, be calm and gentle — frame it as "if X could no longer earn" \
or "protection for the family", never graphic or alarming wording, and focus on what would keep the family secure.
- Tools prefixed mf_ are LIVE mutual-fund data from MF Engine for the asker only (holdings, XIRR, capital gains, \
tax-aware redemption plans, SIPs, stress tests). Prefer them for mutual-fund questions; their numbers are verified facts. \
Use mf_getLiquidityLadder (target_amount, needed_by) when a goal needs cash from mutual funds, e.g. a down payment; \
use mf_getHouseholdTaxHarvestPlan for family tax-harvesting questions (it covers every member only when the asker is the \
MF Engine family head; otherwise relay its visibility_note). \
Resolve scheme ids from earlier results, never invent them. They cannot place orders: point to the MF Engine app for that. \
When you use them, end the answer with: "Mutual fund investments are subject to market risks." \
If an mf_ tool returns an error, continue with FamilyOS records and say live data was unavailable.
- Be efficient: gather what you need in one or two rounds of parallel tool calls, then answer. Keep the answer short — detail lives in findings.
- Finish by calling final_answer exactly once."""


# ===================================================================== tool execution
class ToolBox:
    def __init__(self, ctx: FamilyContext, asker: Optional[str]):
        self.ctx, self.asker = ctx, asker
        self.visible, self.hidden = ctx.visible_facts(asker)   # access control applied once, here
        self.current = [f for f in self.visible if f.status == "current"]
        self.calcs: list[dict] = []
        self.records_touched: set[str] = set()
        self.assumptions: list[str] = []
        self.live_calls: list[dict] = []   # MF Engine calls (provenance for live data)
        # Live MF Engine tools only when the asker has linked their own MF Engine login (one identity per request).
        self.mf_tools = mf_engine.tool_defs() if mf_engine.identity(ctx.family_id, asker) else []
        self.tool_defs = TOOLS[:-1] + self.mf_tools + TOOLS[-1:]   # keep final_answer last

    def _n(self, mid: str) -> str:
        return self.ctx.members[mid].name

    def _check_members(self, ids):
        bad = [m for m in ids or [] if m not in self.ctx.members]
        if bad:
            raise ValueError(f"unknown member_id(s) {bad}; valid: {list(self.ctx.members)}")

    def _overrides(self, items) -> dict[str, Decimal]:
        out = {}
        for o in items or []:
            self._check_members([o["member_id"]])
            out[o["member_id"]] = Decimal(o["monthly_income"])
        return out

    def _log(self, calcs: list[finance.Calc], scenario: str | None = None):
        for c in calcs:
            d = c.to_dict()
            if scenario:
                d["label"] = f"[{scenario}] {d['label']}"
            self.calcs.append(d)
            self.records_touched.update(c.record_ids)

    # --- tools
    def get_family_profile(self) -> dict:
        b = self.ctx.to_contract_b(self.asker)
        ages = {}
        for m in self.ctx.members.values():
            if m.date_of_birth:
                ages[m.member_id] = TODAY.year - m.date_of_birth.year - ((TODAY.month, TODAY.day) < (m.date_of_birth.month, m.date_of_birth.day))
        names = {m.member_id: m.name for m in self.ctx.members.values()} | {e["entity_id"]: e["name"] for e in b["entities"]}
        out = {
            "today": TODAY.isoformat(), "asker": self.asker and {"member_id": self.asker, "name": self._n(self.asker)},
            "members": [{**m, "age": ages.get(m["member_id"])} for m in b["members"]],
            "remembered_goals": self.ctx.goals,
            "records_private_to_other_members": self.hidden,
        }
        if graph_store.status().get("connected"):
            out["relationships"] = ("Stored in the Neo4j family graph — use member_dependency_graph(member_id) or "
                                    "query_family_graph to traverse who owns, owes, co-borrows, insures or depends on what.")
        else:  # graph down: fall back to the in-memory relationship list
            out["relationships"] = [f"{names.get(r['from'], r['from'])} —{r['type']}→ {names.get(r['to'], r['to'])}"
                                    for r in b["relationships"] if r["type"] != "has_member"]
        return out

    def get_financial_records(self, member_ids=None, types=None, include_history=False) -> dict:
        self._check_members(member_ids)
        pool = self.visible if include_history else self.current
        rows = []
        for f in pool:
            r = f.record
            if member_ids and not ({r.member_id, *r.co_owner_ids} & set(member_ids)):
                continue
            if types and r.type not in types:
                continue
            rows.append({
                "record_id": r.record_id, "owner": self._n(r.member_id),
                "co_owners": [self._n(c) for c in r.co_owner_ids], "type": r.type, "category": r.category,
                "description": r.description, "amount": r.amount, "amount_readable": inr_short(r.value),
                "frequency": r.frequency, "attributes": r.attributes, "entity": self.ctx.entities[f.entity_id].name,
                "status": f.status, "effective_from": str(f.effective_from), "effective_until": str(f.effective_until) if f.effective_until else None,
                "superseded_by": f.superseded_by, "supersedes": f.supersedes,
                "source_document": r.source.filename, "confidence": r.confidence,
            })
            self.records_touched.add(r.record_id)
        return {"records": rows, "count": len(rows), "records_private_to_other_members": self.hidden}

    def _snapshot(self, income_overrides=None, exclude_record_ids=None, additional_monthly_expense=None):
        facts = [f for f in self.current if f.record.record_id not in set(exclude_record_ids or [])]
        s = finance.snapshot(self.ctx, facts, self._overrides(income_overrides))
        if additional_monthly_expense:
            extra = Decimal(additional_monthly_expense)
            s.monthly_outflow = (s.monthly_outflow or Decimal(0)) + extra
            s.surplus = s.monthly_income - s.monthly_outflow
            s.expenses = s.expenses + [("SCENARIO", extra)]
            s.calcs.append(finance.Calc("Surplus with additional expense", f"{inr(s.monthly_income)} − {inr(s.monthly_outflow)}", inr(s.surplus), s.surplus))
        return s

    def household_cashflow(self, income_overrides=None, exclude_record_ids=None, additional_monthly_expense=None, scenario_label=None) -> dict:
        s = self._snapshot(income_overrides, exclude_record_ids, additional_monthly_expense)
        label = scenario_label or ("scenario" if (income_overrides or exclude_record_ids or additional_monthly_expense) else None)
        self._log(s.calcs, label)
        summ = s.summary(self.ctx)
        shares = [{"member": r["name"], "monthly_income": str(r["monthly_income"]), "readable": inr_short(r["monthly_income"]),
                   "share_pct": str(q(r["share_pct"], "0.1"))} for r in finance.contributions(self.ctx, s)]
        for r in finance.contributions(self.ctx, s):
            self._log([finance.Calc(f"{r['name'].split()[0]}'s share of household income", f"{inr(r['monthly_income'])} ÷ {inr(s.monthly_income)}",
                                    pct(r["share_pct"]), r["share_pct"], r["record_ids"])], label)
        return {"scenario": label or "baseline (latest known values)", **summ,
                "readable": {k: inr_short(Decimal(v)) for k, v in summ.items() if isinstance(v, str) and _is_num(v)},
                "income_shares": shares,
                "income_records": {self._n(m): ids for m, ids in s.income_records.items()},
                "note": "Income amounts are as reported on source documents (gross vs net not verified)."}

    def evaluate_property_purchase(self, price, horizon_months, down_payment_pct=None, loan_rate_pct=None, loan_tenure_years=None, income_overrides=None) -> dict:
        s = self._snapshot(income_overrides)
        params = {k: Decimal(v) for k, v in {"down_payment_pct": down_payment_pct, "loan_rate_pct": loan_rate_pct, "loan_tenure_years": loan_tenure_years}.items() if v}
        res = finance.affordability(s, Decimal(price), int(horizon_months), params)
        label = "scenario" if income_overrides else None
        self._log(s.calcs + res["calcs"], f"{label} · {inr_short(Decimal(price))} home" if label else f"{inr_short(Decimal(price))} home")
        for a in finance.assumption_text(res["params"], int(horizon_months)):
            if a not in self.assumptions:
                self.assumptions.append(a)
        g = lambda k: str(q(res[k], "1")) if res[k] is not None else None
        return {
            "verdict": res["verdict"], "price": g("price"), "upfront_cash_needed": g("upfront"),
            "funds_available_at_purchase": g("corpus"), "funding_gap": g("gap"), "loan_amount": g("loan"),
            "new_emi": g("new_emi"), "total_emi_after_purchase": g("total_emi"),
            "emi_to_income_pct": str(q(res["emi_ratio"], "0.1")) if res["emi_ratio"] is not None else None,
            "monthly_surplus_after_new_emi": g("post_surplus"), "household_monthly_income": str(s.monthly_income),
            "readable": {k: inr_short(res[v]) for k, v in [("price", "price"), ("upfront", "upfront"), ("available", "corpus"),
                                                           ("gap", "gap"), ("loan", "loan"), ("new_emi", "new_emi"), ("surplus_after", "post_surplus")]},
            "assumptions": finance.assumption_text(res["params"], int(horizon_months)),
            "verdict_rule": "affordable if no funding gap, EMI ≤ 40% of income and positive surplus; stretch if gap ≤ 15% of upfront and EMI ≤ 50%",
        }

    def portfolio_breakdown(self, member_ids=None, top_n=10) -> dict:
        self._check_members(member_ids)
        rows = [f for f in self.current if f.record.type == "investment" and f.record.value is not None
                and f.record.currency == finance.BASE_CURRENCY
                and (not member_ids or {f.record.member_id, *f.record.co_owner_ids} & set(member_ids))]
        total = sum((f.record.value for f in rows), Decimal(0))
        if not rows or total == 0:
            return {"total": "0", "note": "No INR investment holdings visible."}
        share = lambda v: str(q(v / total * 100, "0.1"))

        def house(f):
            n = (f.record.description or "").strip()
            first = n.split()[0].strip("#-").upper() if n else "UNKNOWN"
            return {"PARAG": "PPFAS", "GOVT": "GOVT OF INDIA (SGB)"}.get(first, first)

        def group(key):
            g: dict[str, Decimal] = {}
            for f in rows:
                g[key(f)] = g.get(key(f), Decimal(0)) + f.record.value
            return [{"group": k, "value": str(q(v, "0.01")), "readable": inr_short(v), "share_pct": share(v)}
                    for k, v in sorted(g.items(), key=lambda kv: -kv[1])]

        top = sorted(rows, key=lambda f: -f.record.value)[:top_n]
        with_gain = [f for f in rows if f.record.attr_decimal("unrealised_gain") is not None]
        gains = sorted(with_gain, key=lambda f: -f.record.attr_decimal("unrealised_gain"))
        cost = sum((f.record.attr_decimal("cost_basis") or Decimal(0) for f in with_gain), Decimal(0))
        unreal = sum((f.record.attr_decimal("unrealised_gain") for f in with_gain), Decimal(0))
        item = lambda f: {"record_id": f.record.record_id, "name": f.record.description, "value": str(f.record.value),
                          "readable": inr_short(f.record.value), "share_pct": share(f.record.value),
                          "unrealised_gain": f.record.attributes.get("unrealised_gain"), "category": f.record.category}
        self._log([finance.Calc("Total investment portfolio", f"sum of {len(rows)} holdings", inr(total), total, [f.record.record_id for f in rows])])
        if with_gain:
            self._log([finance.Calc("Unrealised gain on holdings with cost basis", f"{len(with_gain)} MF folios: value − cost",
                                    inr(unreal), unreal, [f.record.record_id for f in with_gain])])
        for f in top[:5]:
            self._log([finance.Calc(f"{(f.record.description or '')[:40]} — share of portfolio",
                                    f"{inr(f.record.value)} ÷ {inr(total)}", f"{share(f.record.value)}%", f.record.value / total * 100,
                                    [f.record.record_id])])
        return {"holdings_count": len(rows), "total": str(q(total, "0.01")), "readable_total": inr_short(total),
                "by_asset_class": group(lambda f: f.record.category), "by_fund_house_or_issuer": group(house)[:12],
                "top_holdings": [item(f) for f in top],
                "unrealised": {"holdings_with_cost": len(with_gain), "cost": str(q(cost, "0.01")), "gain": str(q(unreal, "0.01")),
                               "gain_pct_on_cost": str(q(unreal / cost * 100, "0.1")) if cost else None,
                               "top_gainers": [item(f) for f in gains[:min(5, top_n)]],
                               "top_losers": [item(f) for f in gains[::-1][:min(5, top_n)] if f.record.attr_decimal("unrealised_gain") < 0]},
                "as_of": max(str(f.effective_from) for f in rows)}

    def calculate(self, label, expression) -> dict:
        v = safe_eval(expression)
        self._log([finance.Calc(label, expression, _fmt_any(v), v)])
        return {"label": label, "expression": expression, "result": str(q(v, "0.01")), "readable": _fmt_any(v)}

    def get_memory_timeline(self, member_id=None) -> dict:
        tl = self.ctx.to_contract_b(self.asker)["memory_timeline"]
        if member_id:
            tl = [e for e in tl if e["member_id"] == member_id]
        return {"events": tl[:40]}

    def member_dependency_graph(self, member_id) -> dict:
        self._check_members([member_id])
        out = graph_store.member_dependencies(self.ctx.family_id, member_id)
        out["source"] = "neo4j"
        return out

    def query_family_graph(self, cypher, purpose) -> dict:
        out = graph_store.read_query(cypher, self.ctx.family_id)
        out["source"] = "neo4j"
        return out

    def remember_goal(self, kind, label, amount, target_date) -> dict:
        Decimal(amount)
        date.fromisoformat(target_date)
        goal = {"kind": kind, "label": label, "amount": amount, "target_date": target_date,
                "set_by": self.asker, "created_at": TODAY.isoformat(), "source": "conversation"}
        store.remember_goal(self.ctx.family_id, goal)
        return {"saved": goal}

    def run(self, name: str, args: dict) -> dict:
        if name.startswith(mf_engine.PREFIX):
            t0 = time.perf_counter()
            out = mf_engine.call(name, args, self.ctx.family_id, self.asker)
            self.live_calls.append({"tool": name.removeprefix(mf_engine.PREFIX), "args": args,
                                    "ms": round((time.perf_counter() - t0) * 1000), "source": "MF Engine (live)"})
            return {"source": "MF Engine live data for " + self._n(self.asker), "data": out,
                    "disclaimer": "Mutual fund investments are subject to market risks. Read all scheme related documents carefully."}
        fn = getattr(self, name, None)
        if fn is None or name.startswith("_") or name == "run":
            raise ValueError(f"unknown tool {name}")
        return fn(**args)


def _is_num(v: str) -> bool:
    try:
        Decimal(v)
        return True
    except InvalidOperation:
        return False


def _fmt_any(v: Decimal) -> str:
    return inr_short(v) if abs(v) >= 1000 else str(q(v, "0.01")).rstrip("0").rstrip(".")


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}


def safe_eval(expr: str) -> Decimal:
    expr = re.sub(r"(?<=\d),(?=\d)", "", expr.replace("₹", ""))  # strip digit grouping (1,04,139), keep arg commas
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return Decimal(str(n.value))
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and (b != b.to_integral_value() or abs(b) > 600):
                raise ValueError("only integer exponents ≤ 600")
            return _OPS[type(n.op)](a, int(b) if isinstance(n.op, ast.Pow) else b)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return -ev(n.operand) if isinstance(n.op, ast.USub) else ev(n.operand)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("min", "max", "round", "abs"):
            args = [ev(a) for a in n.args]
            if n.func.id == "round":
                return round(args[0], int(args[1]) if len(args) > 1 else 0)
            return {"min": min, "max": max, "abs": abs}[n.func.id](*args)
        raise ValueError(f"unsupported expression element: {ast.dump(n)[:60]}")
    if len(expr) > 300:
        raise ValueError("expression too long")
    return ev(ast.parse(expr, mode="eval"))


# ===================================================================== grounding check
_NUM = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(crore|cr\b|lakh|lac|l\b|k\b|%)?", re.I)
_UNIT = {"crore": CRORE, "cr": CRORE, "lakh": LAKH, "lac": LAKH, "l": LAKH, "k": Decimal(1000)}


def numbers(text: str) -> list[Decimal]:
    out = []
    for m in _NUM.finditer(text):
        try:
            v = Decimal(m.group(1).replace(",", ""))
        except InvalidOperation:
            continue
        out.append(v * _UNIT.get((m.group(2) or "").lower(), Decimal(1)))
    return out


def ungrounded(claim_text: str, evidence_text: str) -> list[str]:
    allowed = numbers(evidence_text)
    bad = []
    for v in numbers(claim_text):
        if v <= 31 or v in (2025, 2026, 2027, 2028, 2029, 2030) or (v == v.to_integral_value() and 1900 <= v <= 2100):
            continue  # small counts, months, years
        if not any(abs(v - a) <= max(abs(a) * Decimal("0.02"), Decimal("0.5")) for a in allowed):
            bad.append(str(v))
    return sorted(set(bad))


# ===================================================================== LangGraph
class AState(TypedDict, total=False):
    family_id: str
    question: str
    member_id: Optional[str]
    history: list[dict]
    messages: list
    turns: int
    repairs: int
    pending: list          # tool_use blocks to execute
    final: Optional[dict]
    final_block_id: Optional[str]
    evidence: str          # concatenated tool outputs for grounding
    checks: list[dict]
    trace: list[dict]
    response: dict


def _box(config) -> ToolBox:
    """Per-request toolbox (holds the access-filtered view) travels in the run config, not in graph state."""
    return config["configurable"]["box"]


def _t(state: AState, node: str, t0: float, detail: str, kind: str = "step") -> list[dict]:
    return state.get("trace", []) + [{"node": node, "ms": round((time.perf_counter() - t0) * 1000), "detail": detail, "kind": kind}]


def prepare(state: AState, config) -> AState:
    t0 = time.perf_counter()
    box = _box(config)
    ctx, asker = box.ctx, box.asker
    msgs: list = []
    for h in (state.get("history") or [])[-6:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            if msgs and msgs[-1]["role"] == h["role"]:
                continue
            msgs.append({"role": h["role"], "content": h["content"][:2000]})
    if msgs and msgs[0]["role"] != "user":
        msgs = msgs[1:]
    if msgs and msgs[-1]["role"] == "user":
        msgs = msgs[:-1]
    who = f"{ctx.members[asker].name} ({ctx.members[asker].relationship}, {asker})" if asker else "a family administrator"
    msgs.append({"role": "user", "content": f"[Asked by {who} on {TODAY.isoformat()}]\n{state['question']}"})
    return {"messages": msgs, "turns": 0, "repairs": 0, "evidence": state["question"], "checks": [],
            "trace": _t(state, "Context loaded", t0, f"family {ctx.family_id}; access policy applied for {who}")}


def agent(state: AState, config) -> AState:
    t0 = time.perf_counter()
    msgs = state["messages"]
    kw = {"speed": SPEED} if SPEED == "fast" else {}
    try:
        resp = llm._create(model=llm.MODEL, max_tokens=16000, system=SYSTEM, tools=_box(config).tool_defs, messages=msgs,
                           output_config={"effort": EFFORT}, **kw)
    except Exception as e:  # fast-mode rate limit / unavailability -> standard speed
        if not kw:
            raise
        log.warning("fast mode unavailable (%s); retrying at standard speed", e)
        resp = llm._create(model=llm.MODEL, max_tokens=16000, system=SYSTEM, tools=_box(config).tool_defs, messages=msgs,
                           output_config={"effort": EFFORT})
    if resp.stop_reason == "refusal":
        raise RuntimeError("model refused")
    msgs.append({"role": "assistant", "content": resp.content})
    uses = [b for b in resp.content if b.type == "tool_use"]
    final = next((b for b in uses if b.name == "final_answer"), None)
    names = ", ".join(b.name for b in uses) or "(no tool call)"
    upd: AState = {"messages": msgs, "turns": state["turns"] + 1, "pending": [b for b in uses if b.name != "final_answer"],
                   "trace": _t(state, "Claude reasoning", t0, f"decided: {names}", "llm")}
    if final:
        upd["final"], upd["final_block_id"] = dict(final.input), final.id
    elif not uses:
        msgs.append({"role": "user", "content": "Please submit your answer by calling final_answer."})
    return upd


def tools(state: AState, config) -> AState:
    box = _box(config)
    trace = list(state.get("trace", []))
    results, evidence = [], [state["evidence"]]
    for b in state["pending"]:
        t0 = time.perf_counter()
        try:
            out = box.run(b.name, dict(b.input))
            content, err = json.dumps(out, ensure_ascii=False, default=str), False
        except Exception as e:
            content, err = f"Error: {e}", True
        evidence.append(content)
        results.append({"type": "tool_result", "tool_use_id": b.id, "content": content, **({"is_error": True} if err else {})})
        args = ", ".join(f"{k}={_short(v)}" for k, v in dict(b.input).items())
        trace.append({"node": f"tool · {b.name}", "ms": round((time.perf_counter() - t0) * 1000),
                      "detail": (args or "—") + (" → error" if err else ""), "kind": "tool"})
    if state.get("final"):   # final_answer arrived alongside other calls: answer it too so the transcript stays valid
        results.append({"type": "tool_result", "tool_use_id": state["final_block_id"],
                        "content": "Received, but other tool calls were made in the same turn. Review their results and call final_answer again.",
                        "is_error": True})
    state["messages"].append({"role": "user", "content": results})
    return {"messages": state["messages"], "pending": [], "evidence": "\n".join(evidence), "trace": trace, "final": None}


def _short(v: Any) -> str:
    s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
    return s if len(s) <= 60 else s[:57] + "…"


def verify(state: AState, config) -> AState:
    t0 = time.perf_counter()
    f = state["final"]
    box = _box(config)
    claims = " ".join([f["answer"], *f["key_findings"], *f["context_insights"]])
    bad = ungrounded(claims, state["evidence"])
    unknown_ids = [r for r in f["cited_record_ids"] if r not in {x.record.record_id for x in box.visible}]
    checks = [
        {"check": "Every number traces to a tool result", "ok": not bad, "detail": ", ".join(bad)},
        {"check": "Cited records exist and are visible to asker", "ok": not unknown_ids, "detail": ", ".join(unknown_ids)},
        {"check": "Numbers come from deterministic tools or live MF Engine data",
         "ok": bool(box.calcs) or bool(box.live_calls) or not numbers(claims),
         "detail": f"{len(box.calcs)} calcs, {len(box.live_calls)} live MF Engine calls"},
    ]
    ok = all(c["ok"] for c in checks)
    upd: AState = {"checks": checks, "trace": _t(state, "Verification", t0,
                   f"{sum(c['ok'] for c in checks)}/{len(checks)} checks passed" + (f"; ungrounded: {', '.join(bad)}" if bad else ""), "verify")}
    if not ok and state["repairs"] < MAX_REPAIRS:
        problems = []
        if bad:
            problems.append(f"These numbers do not appear in any tool result: {', '.join(bad)}. Compute them with calculate (or remove them).")
        if unknown_ids:
            problems.append(f"These record ids are unknown or not visible: {', '.join(unknown_ids)}.")
        state["messages"].append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": state["final_block_id"],
                                                               "content": "Verification failed. " + " ".join(problems), "is_error": True}]})
        upd.update(repairs=state["repairs"] + 1, final=None, messages=state["messages"])
    return upd


def respond(state: AState, config) -> AState:
    t0 = time.perf_counter()
    box = _box(config)
    f = state["final"]
    ctx = box.ctx
    by_id = {x.record.record_id: x for x in box.visible}
    cited = [r for r in f["cited_record_ids"] if r in by_id]
    calc_ids = sorted({r for c in box.calcs for r in c["record_ids"] if r in by_id})
    ids = list(dict.fromkeys(cited + calc_ids))
    supporting = []
    for rid in ids:
        x = by_id[rid]
        r = x.record
        supporting.append({"record_id": rid, "member": ctx.members[r.member_id].name, "type": r.type, "category": r.category,
                           "description": r.description, "amount": r.amount, "currency": r.currency, "frequency": r.frequency,
                           "status": x.status, "valid_from": r.valid_from.isoformat(), "confidence": r.confidence,
                           "source": r.source.model_dump()})
    assumptions = list(dict.fromkeys(f["assumptions"] + box.assumptions))
    tool_calls = [t for t in state["trace"] if t.get("kind") == "tool"]
    grounded = all(c["ok"] for c in state["checks"])
    trace = _t(state, "Response assembled", t0, f"{len(supporting)} evidence records, {len(box.calcs)} calculations")
    resp = {
        "answer": f["answer"],
        "key_findings": f["key_findings"],
        "context_insights": f["context_insights"],
        "supporting_records": supporting,
        "calculations": list({(re.sub(r"^\[[^\]]*\] ", "", c["label"]), c["formula"], c["result"]): c for c in box.calcs}.values()),
        "assumptions": assumptions,
        "missing_information": f["missing_information"],
        "confidence": f["confidence"] if grounded else "low",
        "verification": {"checks": state["checks"], "narrator": "claude-agent", "narrative_grounded": grounded,
                         "ungrounded_numbers_rejected": [], "low_confidence_record_ids": [r for r in ids if by_id[r].record.confidence < 0.9],
                         "self_corrections": state["repairs"]},
        "intent": "agentic",
        "mode": "agentic",
        "model": llm.MODEL,
        "plan": {"tool_calls": len(tool_calls), "llm_turns": state["turns"]},
        "withheld_record_count": box.hidden,
        "live_sources": box.live_calls,
        "remembered_goals": store.get(ctx.family_id).goals,
        "trace": trace,
    }
    return {"response": resp, "trace": trace}


def _after_agent(state: AState) -> str:
    if state.get("final") and not state.get("pending"):
        return "verify"
    if state["turns"] >= MAX_TURNS:
        raise RuntimeError("agent exceeded turn budget")
    return "tools" if state.get("pending") else "agent"


def _after_verify(state: AState) -> str:
    return "respond" if state.get("final") else "agent"


def build():
    g = StateGraph(AState)
    for n, fn in [("prepare", prepare), ("agent", agent), ("tools", tools), ("verify", verify), ("respond", respond)]:
        g.add_node(n, fn)
    g.set_entry_point("prepare")
    g.add_edge("prepare", "agent")
    g.add_conditional_edges("agent", _after_agent, {"tools": "tools", "verify": "verify", "agent": "agent"})
    g.add_edge("tools", "agent")
    g.add_conditional_edges("verify", _after_verify, {"respond": "respond", "agent": "agent"})
    g.add_edge("respond", END)
    return g.compile()


GRAPH = build()


def stream(family_id: str, question: str, member_id: Optional[str], history: list[dict] | None = None) -> Iterator[dict]:
    """Yields {'type':'trace', ...} events as the agent works, then {'type':'answer', 'data': Contract C}."""
    ctx = store.get(family_id)
    if member_id is not None and member_id not in ctx.members:
        raise PermissionError(f"{member_id} is not a member of {ctx.family_id}")
    init = {"family_id": family_id, "question": question, "member_id": member_id, "history": history or [], "trace": []}
    cfg = {"recursion_limit": 60, "configurable": {"box": ToolBox(ctx, member_id)}}
    seen = 0
    for upd in GRAPH.stream(init, stream_mode="values", config=cfg):
        tr = upd.get("trace", [])
        for t in tr[seen:]:
            yield {"type": "trace", **t}
        seen = len(tr)
        if upd.get("response"):
            yield {"type": "answer", "data": upd["response"]}


def ask(family_id: str, question: str, member_id: Optional[str], history: list[dict] | None = None) -> dict:
    for ev in stream(family_id, question, member_id, history):
        if ev["type"] == "answer":
            return ev["data"]
    raise RuntimeError("agent finished without an answer")
