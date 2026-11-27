# Design: family coordinator with consented member fan-out

Status: draft · Owner: FamilyOS backend · Touches: `backend/app/agentic.py`, `backend/app/mf_engine.py`,
`backend/app/finance.py`, `backend/data/F00x.access.json`, the Members tab in the frontend

## 1. Problem

The agentic loop (`agentic.py`) answers family questions, but live mutual-fund data comes from only one person:

```python
# ToolBox.__init__
self.mf_tools = mf_engine.tool_defs() if mf_engine.identity(ctx.family_id, asker) else []
# ToolBox.run → mf_engine.call(name, args, self.ctx.family_id, self.asker)
```

So for "Can we raise ₹30 lakh for the down payment by March?", Claude sees the asker's live holdings, plus static
Contract A records for everyone else, which may be stale or missing. The answer is family-wide in wording only.

MF Engine's own family scoping (`getFamilyPortfolioSummary`, `getHouseholdTaxHarvestPlan`) helps only when the
asker is the MF Engine family head, and the MF Engine family may not match the FamilyOS family.

## 2. Goals and non-goals

**Goals**
- Answer family questions with live MF data from every member who has **opted in**. Each call runs under that
  member's own MF Engine identity.
- Keep every existing guarantee: consent enforced in code (not by the model), no LLM arithmetic, provenance on
  every number, and a `verify` step that rejects ungrounded claims.
- Show which members contributed live data and which were withheld, the same way the 🔒 private-record notice works today.

**Non-goals**
- Any write or cart action. FamilyOS stays read-only and never sends `x-mcp-take-action`.
- Free-form LLM agents for each member (deferred to phase 3, see §9).

## 3. Key decision: the coordinator is the existing agent; member workers are deterministic

Two shapes were considered:

| | A. Deterministic fan-out tool (chosen) | B. One LLM sub-agent per member (LangGraph `Send`) |
|---|---|---|
| How | The agent calls `mf_family(tool, args, member_ids)`. Code runs the tool once per consenting member, in parallel. | The coordinator writes a brief for each member, and each member gets its own Claude loop and tool set. |
| Numbers | Raw MF Engine payloads, grounded as they are today | Sub-agent summaries: a second layer of text to ground |
| Consent surface | One choke point in code | N prompts that must each stay in scope |
| Cost and latency | 1 LLM loop + parallel HTTP calls | N+1 LLM loops |
| Fits today's questions | Yes: "same tool for everyone, then combine" covers liquidity, harvesting, SIPs and gains | Only needed for open-ended per-member analysis |

Every number must come from deterministic tools anyway, so per-member LLMs would add cost and risk without
adding facts. B stays as a later option (§9) for questions like "review each person's portfolio in their own words".

## 4. Architecture

```
prepare ─▶ agent ⇄ tools ─────────────────────────────▶ verify ─▶ respond
              │       │                                  (+ consent audit check)
              │       ├─ existing FamilyOS tools
              │       ├─ mf_<tool>          (asker only, unchanged)
              │       ├─ mf_family ─────────┐
              │       └─ family_liquidity_plan (deterministic merge, finance.py)
              │                             ▼
              │                 MemberGate.resolve(member_ids)
              │                   ├─ linked?    (mf_engine.identity)
              │                   ├─ consented? (access.json: share_live_mf_with_family)
              │                   └─ tool in FAMILY_SAFE?
              │                             ▼
              │                 ThreadPool: mf_engine.call(tool, args, family_id, member_id)
              │                   └─ redact PII ─▶ tag provenance ─▶ audit log
```

There are no new graph nodes. The fan-out is a tool, so the agent ⇄ tools loop, tracing, the turn budget and the
verify repair loop all work unchanged.

## 5. Consent model

Extend `F00x.access.json`. The new field is **opt-in, default false**, separate from `share_with_family`, because live
brokerage data is richer than the household records a member has already shared:

```json
"M002": {
  "share_with_family": true,
  "private_record_ids": [],
  "share_live_mf_with_family": true,
  "live_mf_consent": {"granted_at": "2026-11-26", "granted_by": "M002", "scope": ["portfolio", "tax", "liquidity"]}
}
```

Rules (enforced in `MemberGate`, never in the prompt):
1. The asker's own data: always allowed, as today.
2. Another member's data: requires `share_live_mf_with_family` **and** a linked MF Engine identity **and** the tool
   in that member's `scope`.
3. Only the member themselves can grant or revoke consent: `POST /api/family/{id}/consent` checks
   `viewer_id == member_id`. For a minor, the guardian member grants it.
4. Revocation takes effect on the next request. Nothing is cached across requests.

## 6. Components

### 6.1 `MemberGate` (new, `backend/app/agentic.py` or `member_gate.py`)

```python
@dataclass
class Delegate:
    member_id: str
    status: Literal["self", "ok", "no_identity", "no_consent", "out_of_scope"]

class MemberGate:
    def __init__(self, ctx: FamilyContext, asker: str | None): ...
    def resolve(self, member_ids: list[str] | None, tool: str) -> list[Delegate]:
        """None → every member. Never raises for a refusal; refusals become statuses so the agent can explain them."""
```

It is built once in `ToolBox.__init__`, next to `visible_facts`.

### 6.2 `mf_family` tool

```json
{
  "name": "mf_family",
  "description": "Run one read-only MF Engine tool for several family members, each under their own login, in parallel. Members who have not linked MF Engine or not consented are returned as withheld; say so in the answer.",
  "input_schema": {
    "type": "object",
    "properties": {
      "tool": {"enum": ["getPortfolioSummary", "getLiquidityLadder", "getCapitalGainsSummary", "getHarvestingOpportunities",
                         "getSipBook", "getSipSchedule", "getPendingActions", "getPortfolioAllocationAnalysis"]},
      "args": {"type": "object"},
      "member_ids": {"type": "array", "items": {"type": "string"}, "description": "Omit for the whole family"}
    },
    "required": ["tool"]
  }
}
```

`FAMILY_SAFE` (the enum above) is a subset of `ALLOWLIST`. It excludes:
- **Identity and KYC tools**, for example `getInvestorProfile`.
- **Tools that already aggregate a family on the MF Engine side** (`getFamilyPortfolioSummary`, `getHouseholdTaxHarvestPlan`), to avoid double counting.

Result shape:

```json
{
  "tool": "getLiquidityLadder",
  "members": {
    "M001": {"status": "ok", "source_id": "MF:M001:getLiquidityLadder:familyos-3f2a…", "data": {...}},
    "M002": {"status": "no_consent"},
    "M003": {"status": "error", "code": "TIMEOUT"}
  },
  "withheld": ["M002"]
}
```

Execution details:
- `ThreadPoolExecutor(max_workers=4)`. Each worker calls `mf_engine.call(...)`, which already creates its own event loop through `mcp_source._run`.
- **Per-member timeout:** 60s for `HEAVY` tools, 30s otherwise. One member failing never fails the call.
- **Memo:** within one request, cache by `(member_id, tool, json(args))`. Nothing is cached across requests (see §5, rule 4).

### 6.3 Redaction

Applied to every other member's payload before it reaches the model:
- **Removed:** `pan`, `phone`, `email`, `address`, bank account numbers, and `access_token`-like fields.
- **Masked:** folio numbers, to the last 4 digits.

The asker's own payload is unchanged.

### 6.4 `family_liquidity_plan` (deterministic, `finance.py`)

This is the first combined tool, because it is the demo question.
1. `mf_family("getLiquidityLadder", {needed_by})` with **no target**, so each member's `redemption_order` is a full-exit list with `exit_load` and `estimated_tax` for each scheme. Each member's own ₹1.25L LTCG exemption is already applied by MF Engine.
2. Rank every member's tranches together by cost ratio `(exit_load + estimated_tax) / redeem_gross`, using `Decimal`. Take tranches until `target_amount` net is reached, with a partial last tranche.
3. Return a `finance.Calc` for each step. The `record_ids` are the `source_id`s, so the Calculations and Evidence tabs show which member's MF Engine call each rupee came from.

Output: the combined order (member, scheme, gross, load, tax, net), `target_met`, `shortfall`, and what each member contributes.

### 6.5 Provenance and tracing

- `box.live_calls` entries gain `member_id` and `source_id`. The `respond` step passes them through as `live_sources`.
- Trace line: `tool · mf_family[getLiquidityLadder] → Rajesh ✓ · Arjun ✓ · Priya 🔒 not shared`.
- **Contract C additions:** `withheld_live_members: [{member_id, reason}]` and `live_sources[].member_id`. These are additive and backward compatible.

### 6.6 Audit log

There's one line of append-only JSONL for each delegated call, stored in `backend/data/audit/F00x.live_mf.jsonl` (add `backend/data/audit/` to `.gitignore`):
`{ts, family_id, asker, on_behalf_of, tool, args_hash, request_id, status}`.
The `request_id` is the same `x-request-id` sent to MF Engine, so both sides can be matched up.

### 6.7 `verify` additions

| Check | Logic |
|---|---|
| Live data only from consenting members | Every `live_calls` entry with `member_id != asker` has gate status `ok` at call time. This is a structural guarantee, re-checked here. |
| Withheld members acknowledged | If `withheld` is non-empty and the answer talks about the whole family ("our family", "everyone", "household"), `missing_information` must name the withheld members. On failure, the existing repair loop asks Claude to add them. |
| Existing number grounding | Unchanged. Combined figures come from `family_liquidity_plan`'s Calcs. |

### 6.8 System prompt additions

> For questions about the whole family's mutual funds, use mf_family (or family_liquidity_plan for raising cash) instead
> of the mf_ tools, which cover only the asker. Members marked withheld have not shared live data: never estimate their
> holdings. Say they were excluded and use their FamilyOS records only if those are visible.

## 7. Failure modes

| Situation | Behaviour |
|---|---|
| No member except the asker has consented | `mf_family` returns only the asker. The answer says the others were not included. |
| A member's MF Engine token has expired | Status `error` with code `AUTH`. The answer says live data was unavailable for that member. A UI nudge asks them to re-link. |
| MF Engine is down | Every member returns `error`, and the agent falls back to FamilyOS records, as today. |
| Asker is the MF Engine family head and the question is family tax harvesting | Prefer `mf_getHouseholdTaxHarvestPlan` (one call, MF Engine-scoped). Use `mf_family` only for members outside the MF Engine family. |
| A member's MF data also exists as a Contract A record (for example, an imported CAS) | Prefer live data for that member and mark the static record as superseded for this answer (assumption listed), to avoid double counting. |

## 8. Testing

Unit tests (`backend/tests/test_familyos.py`, with `mf_engine.call` mocked):
- No consent: `mf_engine.call` is **never invoked** for that member, and the status is `no_consent`.
- A tool outside `FAMILY_SAFE` or outside the member's `scope` is refused.
- Redaction: PAN, phone and email are absent and folios are masked in another member's payload. The asker's payload is unchanged.
- `family_liquidity_plan`: cheapest-first across members, partial last tranche, and exact `Decimal` totals.
- `verify`: an answer that talks about the whole family without naming the withheld members triggers one repair.
- Revoking consent between two requests: the second request makes no call.

Golden end-to-end check (F001): Rajesh and Arjun consent, Priya doesn't. Ask "Can we raise ₹30 lakh by March for
the down payment?" Expect `mf_family` or `family_liquidity_plan` in the trace, Priya in `withheld_live_members`, and
every rupee traced to a `source_id`.

## 9. Rollout

| Phase | Scope |
|---|---|
| 1 | Consent field and endpoint, `MemberGate`, `mf_family` with redaction, audit log, verify consent check, Contract C fields |
| 2 | `family_liquidity_plan`, a Members tab toggle ("Share live mutual-fund data for family questions"), trace badges |
| 3 | Optional LLM member agents through LangGraph `Send`, for open-ended reviews of each member. They reuse `MemberGate` and get a per-member `ToolBox` whose asker is that member, so no new consent paths. |
| 4 | Compliance and suitability critic node after `verify` (separate design) |

### Phase 1 as built

| Piece | Where |
|---|---|
| `MemberGate`, `FAMILY_SAFE` with scopes, `redact`, `audit`, `fan_out` | `backend/app/member_gate.py` |
| `mf_family` tool, prompt rule, `consent_check` and `withheld_check` in `verify`, `withheld_live_members` and per-member `live_sources` in Contract C | `backend/app/agentic.py` |
| `x-request-id` passed in, so audit lines match MF Engine logs | `mf_engine.call(..., request_id=)` |
| Consent store | `ContextStore.set_live_consent` writes `F00x.access.json` and updates the live context in place |
| API | `GET /api/family/{id}/consent`, `POST /api/family/{id}/consent/{member_id}` with `{viewer_id, granted, scope}` |

Phase 1 differs from the design above in three ways:
- **When `mf_family` is offered:** only when MF Engine is configured and at least one member other than the asker
  has opted in. Otherwise the asker-only `mf_` tools are enough.
- **Combining members' figures:** this uses the existing `calculate` tool on returned numbers until
  `family_liquidity_plan` lands in phase 2.
- **A refused tool doesn't count as withheld:** a member's status for a refused tool (`out_of_scope`, `no_identity`) is
  reported in `withheld`, the same as `no_consent`.

**Caveat:** FamilyOS has no user authentication yet. `viewer_id` on the consent endpoint, like `member_id` on `/ask`,
is self-asserted, so the "only the member can grant" rule holds only once requests carry a verified identity. This
must be fixed before any non-demo use.

## 10. Open questions

1. **MF Engine policy on delegated reads.** The integration guide requires one identity per request, and this design
   respects that. But FamilyOS would use member B's stored login to answer member A's question. Confirm with the MF Engine
   team whether this needs a distinct header (for example, `x-on-behalf-of-asker`) or partner-level consent on their side.
2. **Token custody.** Logins for other members currently live in `backend/.secrets/mf_engine_identities.json`. Production
   needs encrypted storage, refresh handling, and revocation when consent is withdrawn.
3. **Minors and dependents.** Guardian consent, and whether a minor's holdings should count towards family goals by default.
4. **Should consent be per goal?** For example, "share for the house goal only" instead of a blanket scope.
