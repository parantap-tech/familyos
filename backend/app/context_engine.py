"""Family Context Engine.

Consumes Contract A payloads and builds an enriched, temporal, access-aware model of the family
(Contract B). Responsibilities:
  * family identity + relationship graph (spouse_of / parent_of / child_of)
  * entity resolution (same bank account / employer / portfolio across documents)
  * temporal memory (supersession for state facts, append-only for transactions)
  * access control (member consent policies)
  * persistent family goals remembered from conversations
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

from .contract import FamilyFinancialData, FinancialRecord, Member, RecordBatch
from .money import inr, inr_short, pct, q

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TODAY = date.fromisoformat(os.environ.get("FAMILYOS_TODAY", "2026-11-22"))

# Transactions are events: always appended, never superseding anything.
EVENT_TYPES = {"transaction"}
# A person holds one current salary/pension regardless of employer (a new employer's salary replaces the
# old one). Other income streams (rent, interest, fees) are tracked per source entity.
CATEGORY_KEYED_INCOME = {"salary", "pension", "family_pension"}

PARENT_ROLES = {"father", "mother"}
CHILD_ROLES = {"son", "daughter"}


# ---------------------------------------------------------------- entity resolution
_SUFFIXES = re.compile(r"\b(pvt|private|ltd|limited|llp|inc|co|bank|technologies|tech)\b\.?", re.I)


def normalise_name(name: str) -> str:
    n = _SUFFIXES.sub(" ", name.lower())
    return re.sub(r"[^a-z0-9•]+", " ", n).strip()


@dataclass
class Entity:
    entity_id: str
    entity_type: str
    name: str
    institution: Optional[str]
    owner_ids: list[str]
    aliases: set[str] = field(default_factory=set)
    record_ids: list[str] = field(default_factory=list)


@dataclass
class Fact:
    """A financial record enriched with temporal state."""
    record: FinancialRecord
    entity_id: str
    status: str = "current"  # current | superseded | ended | event
    effective_from: date | None = None
    effective_until: date | None = None
    superseded_by: str | None = None
    supersedes: str | None = None

    def to_dict(self) -> dict:
        r = self.record
        return {
            **r.model_dump(mode="json"),
            "entity_id": self.entity_id,
            "status": self.status,
            "effective_from": self.effective_from.isoformat() if self.effective_from else None,
            "effective_until": self.effective_until.isoformat() if self.effective_until else None,
            "superseded_by": self.superseded_by,
            "supersedes": self.supersedes,
        }


class AccessDenied(Exception):
    pass


class FamilyContext:
    def __init__(self, data: FamilyFinancialData, access: dict, goals: list[dict]):
        self.family_id = data.family_id
        self.members: dict[str, Member] = {m.member_id: m for m in data.members}
        self.access = access.get("policies", {})
        self.goals = goals
        self.entities: dict[str, Entity] = {}
        self.facts: dict[str, Fact] = {}
        self.relationships: list[dict] = []
        self.timeline: list[dict] = []
        self._build(data.financial_records)

    # ------------------------------------------------------------ build
    def _build(self, records: list[FinancialRecord]) -> None:
        self._build_family_relationships()
        for r in records:
            self.facts[r.record_id] = Fact(record=r, entity_id=self._resolve_entity(r))
        self._apply_temporal_rules()
        self._build_financial_relationships()
        self._build_timeline()

    def _build_family_relationships(self) -> None:
        ms = list(self.members.values())
        for m in ms:
            self.relationships.append(rel("FAMILY:" + self.family_id, "has_member", m.member_id))
        if any(m.spouse_id or m.parent_ids or m.other_relations for m in ms):
            self._explicit_relationships(ms)
            return
        parents = [m for m in ms if m.relationship in PARENT_ROLES]
        children = [m for m in ms if m.relationship in CHILD_ROLES]
        if len(parents) == 2:
            self.relationships.append(rel(parents[0].member_id, "spouse_of", parents[1].member_id))
        for p in parents:
            for c in children:
                self.relationships.append(rel(p.member_id, "parent_of", c.member_id))

    def _explicit_relationships(self, ms) -> None:
        seen: set[tuple] = set()

        def add(a, t, b):
            key = (t, *sorted((a, b))) if t in ("spouse_of", "sibling_of") else (t, a, b)
            if key not in seen:
                seen.add(key)
                self.relationships.append(rel(a, t, b))

        parents_of = {m.member_id: set(m.parent_ids) for m in ms}
        for m in ms:
            if m.spouse_id:
                add(m.member_id, "spouse_of", m.spouse_id)
            for p in m.parent_ids:
                add(p, "parent_of", m.member_id)
            for x in m.other_relations:
                add(m.member_id, x.type, x.member_id)
        for a in ms:  # derived: siblings share a parent; grandparents are parents of parents
            for b in ms:
                if a.member_id < b.member_id and parents_of[a.member_id] & parents_of[b.member_id]:
                    add(a.member_id, "sibling_of", b.member_id)
            for p in parents_of[a.member_id]:
                for gp in parents_of.get(p, ()):
                    add(gp, "grandparent_of", a.member_id)

    def _resolve_entity(self, r: FinancialRecord) -> str:
        """Map a record to a canonical entity. Uses the extractor's entity_id when present,
        otherwise matches on (owner, type, normalised name); else mints a synthetic entity."""
        e = r.entity
        owners = [r.member_id, *r.co_owner_ids]
        if e is None:
            eid = f"{r.type.upper()}-{r.category.upper()}-{'-'.join(owners)}"
            etype, name, inst = r.type, (r.description or r.category).split(" (")[0], None
        else:
            etype, name, inst = e.entity_type, e.name, e.institution
            eid = e.entity_id
            if eid is None:
                key = normalise_name(e.name)
                match = next((x for x in self.entities.values()
                              if x.entity_type == etype and r.member_id in x.owner_ids
                              and key in {normalise_name(a) for a in x.aliases}), None)
                eid = match.entity_id if match else f"ENT-{etype.upper()}-{key.replace(' ', '-').upper()}"
        ent = self.entities.get(eid)
        if ent is None:
            ent = self.entities[eid] = Entity(eid, etype, name, inst, owners)
        ent.aliases.add(name)
        if inst:
            ent.aliases.add(inst)
        ent.record_ids.append(r.record_id)
        return eid

    def _fact_key(self, f: Fact) -> tuple:
        r = f.record
        if r.type == "income" and r.category in CATEGORY_KEYED_INCOME:
            return (r.member_id, r.type, r.category)
        return (r.member_id, r.type, r.category, f.entity_id)

    def _apply_temporal_rules(self) -> None:
        groups: dict[tuple, list[Fact]] = {}
        for f in self.facts.values():
            r = f.record
            f.effective_from = r.as_of or r.valid_from
            f.effective_until = r.valid_until
            if r.type in EVENT_TYPES:
                f.status = "event"
                continue
            groups.setdefault(self._fact_key(f), []).append(f)
        # Explicit hints from the extractor win.
        for f in self.facts.values():
            if f.record.supersedes and f.record.supersedes in self.facts:
                self._link(self.facts[f.record.supersedes], f)
        for facts in groups.values():
            facts.sort(key=lambda x: (x.effective_from, x.record.record_id))
            for older, newer in zip(facts, facts[1:]):
                if older.superseded_by is None:
                    self._link(older, newer)
        for f in self.facts.values():
            if f.status == "current" and f.effective_until and f.effective_until < TODAY:
                f.status = "ended"

    @staticmethod
    def _link(older: Fact, newer: Fact) -> None:
        older.status = "superseded"
        older.superseded_by = newer.record.record_id
        newer.supersedes = older.record.record_id
        if older.effective_until is None:
            older.effective_until = newer.effective_from

    def _build_financial_relationships(self) -> None:
        verbs = {
            "employer": "earns_from", "pension_source": "earns_from", "business": "runs",
            "bank_account": "owns", "investment_portfolio": "owns", "deposit": "owns", "retirement_account": "owns",
            "property": "owns", "vehicle": "owns", "precious_metal": "owns",
            "loan_account": "owes", "credit_card": "owes", "liability": "owes",
            "insurance_policy": "insured_by", "income": "earns", "expense": "spends_on",
        }
        for ent in self.entities.values():
            verb = verbs.get(ent.entity_type, "has")
            for i, owner in enumerate(ent.owner_ids):
                self.relationships.append(rel(owner, verb if i == 0 else f"co_{verb}", ent.entity_id))
        secured: set[str] = set()
        for f in self.current_facts(None):
            a = f.record.attributes
            if a.get("nominee_member_id") in self.members:
                self.relationships.append(rel(a["nominee_member_id"], "nominee_of", f.entity_id))
            for mid in filter(None, (x.strip() for x in a.get("insured_member_ids", "").split(","))):
                if mid in self.members and mid not in self.entities[f.entity_id].owner_ids:
                    self.relationships.append(rel(mid, "covered_by", f.entity_id))
            for key, verb in (("paid_by_member_id", "pays_for"), ("guardian_member_id", "guardian_of"),
                              ("beneficiary_member_id", "supports")):
                if a.get(key) in self.members:
                    target = f.entity_id if verb != "supports" else a[key]
                    src = a[key] if verb != "supports" else f.record.member_id
                    self.relationships.append(rel(src, verb, target))
            tgt = a.get("secured_against_entity_id")
            if tgt in self.entities:
                self.relationships.append(rel(f.entity_id, "secured_against", tgt))
                secured.add(f.entity_id)
        # Fallback: a home loan without an explicit link is secured against a property its owners hold.
        for l in self.entities.values():
            if l.entity_type != "loan_account" or l.entity_id in secured:
                continue
            cats = {self.facts[r].record.category for r in l.record_ids}
            if cats & {"home_loan", "loan_against_property"}:
                for p in self.entities.values():
                    if p.entity_type == "property" and set(l.owner_ids) & set(p.owner_ids):
                        self.relationships.append(rel(l.entity_id, "secured_against", p.entity_id))

    def _build_timeline(self) -> None:
        for f in sorted(self.facts.values(), key=lambda x: (x.effective_from, x.record.record_id)):
            r = f.record
            who = self.members[r.member_id].name.split()[0]
            ent = self.entities[f.entity_id]
            if f.status == "event":
                kind, text = "transaction", f"{who}: {r.description} {inr_short(r.value)}"
            elif f.supersedes:
                prev = self.facts[f.supersedes].record
                delta = (r.value - prev.value) if (r.value is not None and prev.value is not None) else None
                change = pct(q(delta / prev.value * 100, "0.1")) if delta is not None and prev.value else ""
                arrow = "↑" if (delta or 0) > 0 else "↓"
                kind = "update"
                text = (f"{who}: {ent.name} {r.category.replace('_', ' ')} "
                        f"{inr_short(prev.value)} → {inr_short(r.value)} ({arrow}{change})")
            else:
                kind, text = "new_fact", f"{who}: {r.description} — {inr_short(r.value)}{_freq(r)}"
            self.timeline.append({
                "date": f.effective_from.isoformat(), "member_id": r.member_id, "kind": kind,
                "type": r.type, "text": text, "record_id": r.record_id,
                "superseded_record_id": f.supersedes, "source": r.source.filename,
            })
        self.timeline.sort(key=lambda e: e["date"], reverse=True)

    # ------------------------------------------------------------ access control
    def can_view(self, viewer_id: Optional[str], f: Fact) -> bool:
        if viewer_id is None:  # system / aggregate builder
            return True
        if viewer_id not in self.members:
            raise AccessDenied(f"{viewer_id} is not a member of family {self.family_id}")
        r = f.record
        if viewer_id == r.member_id or viewer_id in r.co_owner_ids:
            return True
        pol = self.access.get(r.member_id, {"share_with_family": False})
        return bool(pol.get("share_with_family")) and r.record_id not in pol.get("private_record_ids", [])

    def visible_facts(self, viewer_id: Optional[str]) -> tuple[list[Fact], int]:
        vis, hidden = [], 0
        for f in self.facts.values():
            if self.can_view(viewer_id, f):
                vis.append(f)
            else:
                hidden += 1
        return vis, hidden

    def current_facts(self, viewer_id: Optional[str]) -> list[Fact]:
        vis, _ = self.visible_facts(viewer_id)
        return [f for f in vis if f.status == "current"]

    # ------------------------------------------------------------ Contract B
    def to_contract_b(self, viewer_id: Optional[str]) -> dict:
        vis, hidden = self.visible_facts(viewer_id)
        vis_ids = {f.record.record_id for f in vis}
        ents = [e for e in self.entities.values() if set(e.record_ids) & vis_ids]
        ent_ids = {e.entity_id for e in ents}
        node_ok = lambda n: n in self.members or n in ent_ids or n.startswith("FAMILY:")
        return {
            "context_version": "1.0",
            "family_id": self.family_id,
            "as_of": TODAY.isoformat(),
            "viewer_member_id": viewer_id,
            "members": [m.model_dump(mode="json") for m in self.members.values()],
            "entities": [self._entity_dict(e, vis_ids) for e in ents],
            "relationships": [r for r in self.relationships if node_ok(r["from"]) and node_ok(r["to"])],
            "financial_context": [f.to_dict() for f in vis],
            "memory_timeline": [e for e in self.timeline if e["record_id"] in vis_ids],
            "goals": self.goals,
            "withheld_record_count": hidden,
        }

    def _entity_dict(self, e: Entity, vis_ids: set[str]) -> dict:
        cur = [self.facts[r] for r in e.record_ids if r in vis_ids and self.facts[r].status == "current"]
        latest = cur[-1].record if cur else None
        return {
            "entity_id": e.entity_id, "entity_type": e.entity_type, "name": e.name,
            "institution": e.institution, "owner_ids": e.owner_ids,
            "aliases": sorted(e.aliases), "record_ids": [r for r in e.record_ids if r in vis_ids],
            "current_value": latest.amount if latest else None,
            "current_record_id": latest.record_id if latest else None,
            "record_type": latest.type if latest else None,
        }

    def graph(self, viewer_id: Optional[str]) -> dict:
        b = self.to_contract_b(viewer_id)
        nodes = [{"id": "FAMILY:" + self.family_id, "kind": "family", "label": f"{self.members[next(iter(self.members))].name.split()[-1]} family"}]
        for m in b["members"]:
            nodes.append({"id": m["member_id"], "kind": "member", "label": m["name"], "role": m["relationship"]})
        for e in b["entities"]:
            if e["current_value"] is None:  # only entities with a current fact (no ended/transaction-only)
                continue
            nodes.append({"id": e["entity_id"], "kind": e["entity_type"], "label": e["name"],
                          "value": e["current_value"], "record_type": e["record_type"],
                          "record_ids": e["record_ids"]})
        ids = {n["id"] for n in nodes}
        edges = [r for r in b["relationships"] if r["from"] in ids and r["to"] in ids]
        return {"nodes": nodes, "edges": edges}


def rel(a: str, kind: str, b: str) -> dict:
    return {"from": a, "type": kind, "to": b}


def _freq(r: FinancialRecord) -> str:
    return {"monthly": "/month", "annual": "/year", "quarterly": "/quarter"}.get(r.frequency or "", "")


# ==================================================================== store / adapters
class ContextStore:
    """Persistent family context. Base data comes from the Financial Data Engine (HTTP if
    FINANCIAL_ENGINE_URL is set, else the contract-conformant mock file); incremental pushes
    and remembered goals are persisted as JSON alongside."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.dir = data_dir
        self._cache: dict[str, FamilyContext] = {}

    def _p(self, family_id: str, kind: str) -> Path:
        return self.dir / f"{family_id}.{kind}.json"

    def _load_json(self, path: Path, default):
        return json.loads(path.read_text()) if path.exists() else default

    source: dict = {"kind": "mock_file", "detail": None}

    def fetch_base(self, family_id: str) -> FamilyFinancialData:
        """Pull from the live Financial Data Engine when configured; fall back to the contract-conformant
        mock file if it is unreachable or sends an invalid payload (so a demo never dies on integration)."""
        from . import mcp_source
        if mcp_source.configured():
            try:
                data = FamilyFinancialData.model_validate(mcp_source.fetch_family(family_id))
                ContextStore.source = {"kind": "financial_data_engine_mcp", "detail": mcp_source.describe()["endpoint"]}
                return data
            except Exception as e:
                ContextStore.source = {"kind": "mock_file_fallback", "detail": f"MCP fetch failed: {str(e)[:200]}"}
                if not self._p(family_id, "financial").exists():
                    raise KeyError(family_id) from e
                return FamilyFinancialData.model_validate_json(self._p(family_id, "financial").read_text())
        url = os.environ.get("FINANCIAL_ENGINE_URL")
        if url:
            endpoint = f"{url.rstrip('/')}/families/{family_id}/financial-data"
            try:
                with urllib.request.urlopen(endpoint, timeout=5) as r:
                    data = FamilyFinancialData.model_validate_json(r.read())
                ContextStore.source = {"kind": "financial_data_engine", "detail": endpoint}
                return data
            except Exception as e:
                ContextStore.source = {"kind": "mock_file_fallback", "detail": f"{endpoint} failed: {str(e)[:160]}"}
        else:
            ContextStore.source = {"kind": "mock_file", "detail": None}
        path = self._p(family_id, "financial")
        if not path.exists():
            raise KeyError(family_id)
        return FamilyFinancialData.model_validate_json(path.read_text())

    def get(self, family_id: str) -> FamilyContext:
        if family_id not in self._cache:
            base = self.fetch_base(family_id)
            pushed = self._load_json(self._p(family_id, "ingested"), [])
            known = {r.record_id for r in base.financial_records}
            extra = [FinancialRecord.model_validate(r) for r in pushed if r["record_id"] not in known]
            data = base.model_copy(update={"financial_records": base.financial_records + extra})
            FamilyFinancialData.model_validate(data.model_dump())  # re-check integrity
            self._cache[family_id] = FamilyContext(
                data,
                self._load_json(self._p(family_id, "access"), {"policies": {}}),
                self._load_json(self._p(family_id, "goals"), []),
            )
            self._mirror(family_id)
        return self._cache[family_id]

    def _mirror(self, family_id: str) -> None:
        """Best-effort sync to the Neo4j mirror; never blocks or fails the request path."""
        try:
            from . import graph_store
            graph_store.sync(self._cache[family_id])
        except Exception:
            pass

    def ingest(self, batch: RecordBatch) -> dict:
        """Accept new records; return what changed in the family's memory."""
        before = self.get(batch.family_id)
        before_timeline = {e["record_id"] for e in before.timeline}
        path = self._p(batch.family_id, "ingested")
        stored = self._load_json(path, [])
        incoming = {r.record_id: r.model_dump(mode="json") for r in batch.financial_records}
        stored = [r for r in stored if r["record_id"] not in incoming] + list(incoming.values())
        for rid in incoming:
            if rid in before.facts:
                raise ValueError(f"record_id {rid} already exists; records are immutable — send a new record")
        unknown = {r.member_id for r in batch.financial_records} - set(before.members)
        if unknown:
            raise ValueError(f"unknown member(s): {sorted(unknown)}")
        path.write_text(json.dumps(stored, indent=2))
        self._cache.pop(batch.family_id, None)
        after = self.get(batch.family_id)
        changes = [e for e in after.timeline if e["record_id"] not in before_timeline]
        return {"accepted": list(incoming), "memory_changes": changes,
                "ingested_at": datetime.now(timezone.utc).isoformat()}

    def is_private(self, family_id: str) -> bool:
        return family_id.startswith("P")

    def families(self) -> list[dict]:
        idx = self._load_json(self.dir / "families.json", []) + self._load_json(self.dir / "private_families.json", [])
        have = {p.name.split(".")[0] for p in self.dir.glob("*.financial.json")}
        from . import mcp_source
        if mcp_source.configured():
            remote = mcp_source.list_families() or []
            have |= {f["family_id"] for f in remote if f.get("family_id")}
            known = {f["family_id"] for f in idx}
            idx = idx + [{"family_id": f["family_id"], "name": f.get("name") or f"{f['family_id']} (live)", "city": f.get("city"),
                          "summary": "Live data from the Financial Data Engine (MCP).", "questions": []}
                         for f in remote if f.get("family_id") and f["family_id"] not in known]
        out = [f for f in idx if f["family_id"] in have]
        out += [{"family_id": f, "name": f, "questions": []} for f in sorted(have - {x["family_id"] for x in out})]
        return out

    def remember_goal(self, family_id: str, goal: dict) -> None:
        ctx = self.get(family_id)
        ctx.goals = [g for g in ctx.goals if g["kind"] != goal["kind"]] + [goal]
        self._p(family_id, "goals").write_text(json.dumps(ctx.goals, indent=2))
        self._mirror(family_id)

    def set_live_consent(self, family_id: str, member_id: str, granted: bool, scope: Optional[list[str]] = None) -> dict:
        """A member opts in or out of sharing their live MF Engine data for family questions.
        Takes effect on the next request; callers must check that the requester *is* `member_id`."""
        ctx = self.get(family_id)
        if member_id not in ctx.members:
            raise ValueError(f"{member_id} is not a member of {family_id}")
        path = self._p(family_id, "access")
        doc = self._load_json(path, {"family_id": family_id, "policies": {}})
        pol = doc.setdefault("policies", {}).setdefault(member_id, {"share_with_family": False, "private_record_ids": []})
        pol["share_live_mf_with_family"] = bool(granted)
        if granted:
            pol["live_mf_consent"] = {"granted_at": TODAY.isoformat(), "granted_by": member_id, "scope": scope or None}
        else:
            pol.pop("live_mf_consent", None)
        path.write_text(json.dumps(doc, indent=2))
        ctx.access[member_id] = pol   # live context sees it immediately; no rebuild needed
        return {"member_id": member_id, **{k: pol.get(k) for k in ("share_live_mf_with_family", "live_mf_consent")}}

    def reset(self, family_id: str) -> None:
        for kind in ("ingested", "goals"):
            self._p(family_id, kind).unlink(missing_ok=True)
        self._cache.pop(family_id, None)


store = ContextStore()
