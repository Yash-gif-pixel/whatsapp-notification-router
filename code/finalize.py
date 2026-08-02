"""Phase 6 - safety override, merge, and output.csv.

Combines the Phase 4 rule decisions with the Phase 5 LLM decisions, runs one
last deterministic safety pass over *all* of them, and writes
``dataset/output.csv``.

Run from the terminal::

    py code/finalize.py                 # write dataset/output.csv
    py code/finalize.py --dry-run       # report only, write nothing
    py code/finalize.py --out other.csv

Three things this module guarantees, none of which are assumed from earlier
phases:

1. Any message matching the credential-extraction pattern (rule 1a) or the
   router-injection pattern (rule 1d) ends as ``mute``/``scam``, whatever
   Phase 4 or Phase 5 said. Every actual change is reported, never silent.
2. The rule engine is the single source of truth for which phase owns a
   message. A message the rules resolve takes the rule decision even if a
   stale LLM answer for the same id is still sitting in the Phase 5 cache.
3. No message ends up with two decisions, and none ends up with none.

No routing logic lives here. The safety pass re-runs the existing rule
functions; it does not define new ones.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import Dataset, load_dataset  # noqa: E402
from llm_reasoner import DecisionStore  # noqa: E402
from media_normalizer import (  # noqa: E402
    NO_CREDENTIALS_HELP,
    build_normalized_contexts,
)
from rule_engine import (  # noqa: E402
    apply_rules_to_all,
    hard_safety_finding,
    image_mismatch_finding,
)

__all__ = ["build_final_decisions", "write_output", "OUTPUT_COLUMNS"]

OUTPUT_COLUMNS = [
    "message_id", "action", "message_type", "reason", "confidence",
    "evidence_message_ids",
]

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "dataset" / "output.csv"

#: Confidence to use when the safety pass forces a decision. Slightly above the
#: rule band's middle: a confirmed credential grab is not an ambiguous call.
OVERRIDE_CONFIDENCE = 0.93


def build_final_decisions(
    ds: Dataset | None = None,
    contexts: list[dict[str, Any]] | None = None,
    store: DecisionStore | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge, override, and return (rows in messages.csv order, report)."""
    ds = ds if ds is not None else load_dataset()
    contexts = contexts if contexts is not None else build_normalized_contexts(ds)
    store = store if store is not None else DecisionStore.load()

    rule_decisions, unresolved = apply_rules_to_all(contexts)
    by_ctx = {c["message_id"]: c for c in contexts}
    by_rule = {d["message_id"]: d for d in rule_decisions}
    unresolved_set = set(unresolved)

    report: dict[str, Any] = {
        "overrides": [],
        "stale_cache_ignored": [],
        "missing": [],
        "conflicts": [],
        "mismatch_not_muted": [],
        "mention_conflict_checks": [],
        "sources": {"rules": 0, "llm": 0},
    }

    # Any cached LLM answer for a message the rules now own is stale.
    for message_id in store.entries:
        if message_id not in unresolved_set and message_id in by_rule:
            report["stale_cache_ignored"].append(message_id)
    report["stale_cache_ignored"].sort()

    rows: list[dict[str, Any]] = []
    for message_id in ds.messages["message_id"]:
        ctx = by_ctx[message_id]

        # -- step 2: exactly one source of truth per message ---------------
        rule_decision = by_rule.get(message_id)
        llm_entry = store.get(message_id) if message_id in unresolved_set else None

        if rule_decision is not None:
            decision = {
                "action": rule_decision["action"],
                "message_type": rule_decision["message_type"],
                "reason": rule_decision["reason"],
                "confidence": rule_decision["confidence"],
                "evidence_message_ids": rule_decision["evidence_message_ids"],
            }
            source = f"rules:{rule_decision['rule']}"
            report["sources"]["rules"] += 1
        elif llm_entry and llm_entry.get("action"):
            decision = {
                "action": llm_entry["action"],
                "message_type": llm_entry["message_type"],
                "reason": llm_entry["reason"],
                "confidence": llm_entry["confidence"],
                "evidence_message_ids": llm_entry["evidence_message_ids"],
            }
            source = "llm"
            report["sources"]["llm"] += 1
        else:
            report["missing"].append(message_id)
            continue

        if rule_decision is not None and message_id in store.entries and (
                store.entries[message_id].get("action")):
            cached = store.entries[message_id]
            if (cached["action"], cached["message_type"]) != (
                    decision["action"], decision["message_type"]):
                report["conflicts"].append({
                    "message_id": message_id,
                    "rules": f"{decision['action']}/{decision['message_type']}",
                    "stale_llm": f"{cached['action']}/{cached['message_type']}",
                    "used": "rules",
                })

        # -- step 1: the final safety net ---------------------------------
        finding = hard_safety_finding(ctx)
        if finding is not None and not (
                decision["action"] == "mute" and decision["message_type"] == "scam"):
            report["overrides"].append({
                "message_id": message_id,
                "rule": finding["rule"],
                "was": f"{decision['action']}/{decision['message_type']}",
                "now": "mute/scam",
                "source": source,
                "signals": finding["signals"],
            })
            decision.update({
                "action": "mute",
                "message_type": "scam",
                "reason": finding["reason"],
                "confidence": OVERRIDE_CONFIDENCE,
            })
            source += "+override"

        # Rule 1b is reported, not enforced - an image/text mismatch is strong
        # evidence but the spec scopes the hard override to 1a and 1d.
        if (image_mismatch_finding(ctx) is not None
                and decision["message_type"] != "scam"):
            report["mismatch_not_muted"].append({
                "message_id": message_id,
                "decision": f"{decision['action']}/{decision['message_type']}",
            })

        # A mention must never have overridden a hard safety finding.
        if ctx.get("mention_check") and finding is not None:
            report["mention_conflict_checks"].append({
                "message_id": message_id,
                "final": f"{decision['action']}/{decision['message_type']}",
                "safety_rule": finding["rule"],
            })

        evidence = (decision["evidence_message_ids"] or "").strip() or "none"
        rows.append({
            "message_id": message_id,
            "action": decision["action"],
            "message_type": decision["message_type"],
            "reason": " ".join(str(decision["reason"]).split()),
            "confidence": round(float(decision["confidence"]), 2),
            "evidence_message_ids": evidence,
            "_source": source,
        })

    return rows, report


def write_output(rows: Sequence[dict[str, Any]], path: Path | str | None = None) -> Path:
    """Write output.csv with exactly the required columns, in order."""
    target = Path(path) if path else DEFAULT_OUTPUT
    frame = pd.DataFrame(rows)[OUTPUT_COLUMNS]
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, encoding="utf-8", lineterminator="\n")
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 6 finalization.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    print("=" * 78)
    print("PHASE 6 - SAFETY OVERRIDE, MERGE, OUTPUT")
    print("=" * 78)

    ds = load_dataset()
    rows, report = build_final_decisions(ds)

    print(f"  messages          : {len(ds.messages)}")
    print(f"  decisions produced: {len(rows)}")
    print(f"    from rules      : {report['sources']['rules']}")
    print(f"    from the LLM    : {report['sources']['llm']}")

    print()
    print("-- step 1: safety override pass (rules 1a / 1d over all 110)")
    if report["overrides"]:
        print(f"  {len(report['overrides'])} decision(s) CHANGED by the override:")
        for o in report["overrides"]:
            print(f"    {o['message_id']}: {o['was']} -> {o['now']} "
                  f"(rule {o['rule']}, was decided by {o['source']})")
            print(f"      signals: {'; '.join(o['signals'])}")
    else:
        print("  no decision changed - every 1a/1d message was already mute/scam")
    if report["mention_conflict_checks"]:
        print(f"  mentioned messages with a hard safety finding: "
              f"{len(report['mention_conflict_checks'])}")
        for c in report["mention_conflict_checks"]:
            print(f"    {c['message_id']}: final {c['final']} "
                  f"(safety rule {c['safety_rule']}) - safety held")
    else:
        print("  no mentioned message carries a 1a/1d finding, so nothing to arbitrate")
    if report["mismatch_not_muted"]:
        print(f"  [NOTE] {len(report['mismatch_not_muted'])} image-mismatch (1b) "
              f"message(s) not typed scam - reported, not overridden:")
        for m in report["mismatch_not_muted"]:
            print(f"    {m['message_id']}: {m['decision']}")

    print()
    print("-- step 2: stale cache resolution")
    if report["stale_cache_ignored"]:
        print(f"  {len(report['stale_cache_ignored'])} stale LLM cache entr(ies) "
              f"ignored in favour of the rule decision:")
        print(f"    {', '.join(report['stale_cache_ignored'])}")
    else:
        print("  no stale cache entries")
    if report["conflicts"]:
        print(f"  {len(report['conflicts'])} message(s) had a differing stale answer:")
        for c in report["conflicts"]:
            print(f"    {c['message_id']}: rules={c['rules']} "
                  f"stale_llm={c['stale_llm']} -> used {c['used']}")
    else:
        print("  no message had two conflicting decisions")
    if report["missing"]:
        print(f"  [WARN] {len(report['missing'])} message(s) have NO decision: "
              f"{', '.join(report['missing'])}")

    print()
    print("-- step 3: output")
    if report["missing"]:
        # Writing a short file would look like success and fail evaluation
        # later. Refuse, say why, and exit non-zero.
        print(f"  REFUSING TO WRITE - {len(report['missing'])} of {len(ds.messages)} "
              f"message(s) have no decision.")
        for message_id in report["missing"][:10]:
            print("      " + NO_CREDENTIALS_HELP.format(item=message_id))
            break
        print(f"      affected: {', '.join(report['missing'][:12])}"
              + (f" (+{len(report['missing']) - 12} more)"
                 if len(report["missing"]) > 12 else ""))
        print("      output.csv left unchanged.")
        return 1
    if args.dry_run:
        print("  --dry-run: nothing written")
        return 0
    target = write_output(rows, args.out)
    print(f"  wrote {len(rows)} rows to {target}")
    print(f"  columns: {', '.join(OUTPUT_COLUMNS)}")
    print()
    print(f"  generated at {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
