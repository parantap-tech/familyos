"""Seed a self-contained live-MF demo: a copy of the family data with consent states, plus an MF Engine identities
file that points FamilyOS members at personas in tools/mock_mf_engine_mcp.py.

Never touches the tracked files in backend/data: everything goes to backend/.demo/ (gitignored), and the app is
pointed there with FAMILYOS_DATA_DIR. Re-running starts the demo fresh.

    .venv/bin/python tools/seed_live_demo.py            # writes backend/.demo/, prints the story
    source .demo/demo.env                                # FAMILYOS_DATA_DIR, MF_ENGINE_IDENTITIES, MF_ENGINE_MCP_URL
    (or just run ./demo-live.sh from the repo root)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
SRC = BACKEND / "data"
ALL = ["portfolio", "tax", "liquidity", "sips"]

# member -> (mock identity or None, consent scope: None = not shared, list = shared with that scope)
CAST: dict[str, dict[str, tuple[str | None, list[str] | None, str]]] = {
    "F001": {  # the story: Arjun asks about the family's down payment
        "M001": ("demo-rajesh", ALL, "shares everything; MF Engine family head; ₹40k LTCG already booked this FY"),
        "M002": ("demo-priya", None, "linked but has not shared → withheld; her SIP mandate has expired"),
        "M003": ("demo-arjun", ALL, "the asker; midcap lot 35 days from LTCG; ELSS partly locked"),
    },
    "F002": {  # the full consent matrix
        "M203": ("demo-suresh", ALL, "the asker; MF Engine family head"),
        "M204": ("demo-meera", ALL, "shares everything; LTCG exemption already used up"),
        "M201": ("demo-venkatesh", ["liquidity"], "shares liquidity only → tax and portfolio tools refused (out_of_scope)"),
        "M202": ("demo-kamala", None, "linked, has not shared (no_consent)"),
        "M205": ("demo-expired", ALL, "shares, but the MF Engine login has expired → error, answer must fall back"),
        "M206": ("demo-empty", ALL, "minor; shares (guardian), empty portfolio"),
        "M207": (None, None, "never linked MF Engine and keeps finances private (no_identity)"),
    },
}

QUESTIONS = {
    "F001": [("M003", "Can our family raise ₹30 lakh from mutual funds by March 2027 for the house down payment, at the lowest cost?"),
             ("M003", "How much long-term capital gain can each of us book tax-free this year?"),
             ("M001", "Are any of our SIPs about to fail?")],
    "F002": [("M203", "What is the whole family's mutual-fund position and who has not shared?"),
             ("M203", "Can we raise ₹40 lakh from mutual funds by June 2027?")],
}


def seed(out: Path, mock_url: str) -> dict:
    if out.exists():
        shutil.rmtree(out)
    data = out / "data"
    data.mkdir(parents=True)
    for f in SRC.glob("*.json"):
        if f.name.startswith("P") or f.name.startswith("private") or ".ingested." in f.name or ".goals." in f.name:
            continue  # never copy real-person data or local state
        shutil.copy(f, data / f.name)

    identities = {"mcp_token": "local-demo", "members": {}}
    for fam, cast in CAST.items():
        path = data / f"{fam}.access.json"
        doc = json.loads(path.read_text())
        for mid, (ident, scope, _) in cast.items():
            if ident:
                identities["members"][f"{fam}:{mid}"] = {"access_token": ident}
            pol = doc["policies"].setdefault(mid, {"share_with_family": False, "private_record_ids": []})
            pol["share_live_mf_with_family"] = scope is not None
            if scope is not None:
                pol["live_mf_consent"] = {"granted_at": "2026-11-26", "granted_by": mid, "scope": scope}
        path.write_text(json.dumps(doc, indent=2))

    ids_path = out / "mf_engine_identities.json"
    ids_path.write_text(json.dumps(identities, indent=2))
    env = {"FAMILYOS_DATA_DIR": str(data), "MF_ENGINE_IDENTITIES": str(ids_path), "MF_ENGINE_MCP_URL": mock_url}
    (out / "demo.env").write_text("".join(f'export {k}="{v}"\n' for k, v in env.items()))
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=BACKEND / ".demo")
    ap.add_argument("--mock-url", default="http://127.0.0.1:9300/mcp")
    a = ap.parse_args()
    env = seed(a.out, a.mock_url)
    print(f"Seeded live-MF demo in {a.out}  (tracked backend/data untouched)\n")
    for fam, cast in CAST.items():
        print(f"{fam}:")
        for mid, (ident, scope, note) in cast.items():
            share = "not shared" if scope is None else ("shares " + ", ".join(scope))
            print(f"  {mid:5} {ident or '— not linked':16} {share:40} {note}")
    print("\nTry, viewing as the member shown:")
    for fam, qs in QUESTIONS.items():
        for mid, q in qs:
            print(f"  [{fam} as {mid}] {q}")
    print(f"\nEnvironment ({a.out / 'demo.env'}):")
    for k, v in env.items():
        print(f"  {k}={v}")


if __name__ == "__main__":
    main()
