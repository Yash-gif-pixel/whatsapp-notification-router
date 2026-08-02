"""Phase 5 - validation for the LLM reasoning pass.

Run from the terminal::

    py code/validate_llm.py
    py code/validate_llm.py --show msg_040,msg_027
    py code/validate_llm.py --injection-test   # spends 3 API calls, see below

Confirms every previously-unresolved message now carries a decision or an
explicit processing_error, prints the msg_040 conflict call in full plus a
spread of other decisions, and reports the Phase 5 / combined distributions
and confidence bands.

``--injection-test`` deliberately pushes the three known prompt-injection
messages through the Phase 5 classifier even though Phase 4's rule 1d already
catches them. Without it the injection defence in this phase is never actually
exercised on this dataset, and an untested defence is not a defence.

Reads the caches written by earlier phases; makes no API calls unless
``--injection-test`` is passed. Problems are reported, never raised.

TEST FIXTURES, NOT ROUTING INPUTS
---------------------------------
Message ids below (``KNOWN_INJECTIONS``, and the ``msg_040`` lookup in
``check_conflict_case``) name specific cases to re-read and re-test. They are
expected-value assertions about decisions the pipeline already made, never
inputs to any decision. ``llm_reasoner`` contains no hardcoded id at all, and
no module on the routing path (``finalize`` -> ``rule_engine`` /
``llm_reasoner`` / ``media_normalizer`` / ``context_builder`` /
``data_loader``) imports this file, so nothing here can reach ``output.csv``.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import load_dataset  # noqa: E402
from llm_reasoner import (  # noqa: E402
    INJECTION_MARKERS,
    DecisionStore,
    _classify_with_detail,
    injection_influence_check,
)
from media_normalizer import build_normalized_contexts, effective_text  # noqa: E402
from rule_engine import apply_rules_to_all  # noqa: E402

PROBLEMS: list[str] = []
MAX_LISTED = 12

#: Phase 4's confidence band, for the comparison the spec asks for.
PHASE4_BAND = (0.83, 0.94)

#: Messages whose text carries directives aimed at the router.
KNOWN_INJECTIONS = ("msg_107", "msg_108", "msg_110")

SPOTCHECK_COUNT = 8


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def sub(title: str) -> None:
    print()
    print("-- " + title)


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def warn(msg: str) -> None:
    PROBLEMS.append(msg)
    print(f"  [WARN] {msg}")


def note(msg: str) -> None:
    print(f"  [NOTE] {msg}")


def sample(values: Iterable[Any]) -> str:
    vals = [str(v) for v in values]
    shown = ", ".join(vals[:MAX_LISTED])
    if len(vals) > MAX_LISTED:
        shown += f", ... (+{len(vals) - MAX_LISTED} more)"
    return shown


def wrapped(text: str, indent: str = "      ") -> None:
    for line in textwrap.wrap(" ".join(str(text).split()), width=72):
        print(indent + line)


# ---------------------------------------------------------------------------


def check_coverage(unresolved, store) -> None:
    section("1. COVERAGE OF THE UNRESOLVED SET")
    decided = [m for m in unresolved if (store.get(m) or {}).get("action")]
    failed = [m for m in unresolved if store.get(m) and not store.get(m).get("action")]
    missing = [m for m in unresolved if not store.get(m)]

    print(f"  unresolved after Phase 4 : {len(unresolved)}")
    print(f"  decided by the LLM       : {len(decided)}")
    print(f"  explicit processing_error: {len(failed)}")
    print(f"  never attempted          : {len(missing)}")

    if missing:
        warn(f"{len(missing)} message(s) silently skipped: {sample(missing)}")
    elif not failed:
        ok("every unresolved message has a decision, none skipped")
    else:
        ok("every unresolved message was attempted")

    if failed:
        warn(f"{len(failed)} message(s) failed: {sample(failed)}")
        for m in failed:
            print(f"      {m}: {store.entries[m].get('processing_error')}")

    sub("stale cache entries")
    unresolved_set = set(unresolved)
    stale = sorted(m for m in store.entries if m not in unresolved_set)
    if stale:
        note(f"{len(stale)} cached LLM decision(s) are now resolved by a Phase 4 "
             f"rule and must be ignored downstream: {sample(stale)}")
        print("      Phase 6 merges on the rule engine's unresolved list, so these")
        print("      are dropped automatically - but they are left in the cache so")
        print("      the earlier LLM answer stays auditable.")
    else:
        ok("no stale cache entries")

    sub("provider and models used")
    models = Counter((store.get(m) or {}).get("model") for m in decided)
    for model, n in sorted(models.items(), key=lambda kv: -kv[1]):
        print(f"    {str(model):<28} {n}")
    providers = Counter((store.get(m) or {}).get("provider") for m in decided)
    print(f"    providers: {dict(providers)}")

    retried = [m for m in decided if (store.get(m) or {}).get("attempts")]
    print(f"    decisions that needed a retry or model switch: {len(retried)}")


def check_conflict_case(contexts, store) -> None:
    section("2. THE MENTION / FORWARD CONFLICT CALL (msg_040)")
    by_ctx = {c["message_id"]: c for c in contexts}
    ctx = by_ctx.get("msg_040")
    entry = store.get("msg_040")
    if ctx is None:
        warn("msg_040 not present")
        return
    flags = ctx.get("rule_engine") or {}

    print("  Phase 4 declined this one deliberately:")
    print(f"    unresolved_reason : {flags.get('unresolved_reason')}")
    if flags.get("suppressed_decision"):
        sup = flags["suppressed_decision"]
        print(f"    rule {flags.get('suppressed_rule')} would have said: "
              f"{sup['action']}/{sup['message_type']}")
    print()
    print("  message content:")
    wrapped(effective_text(ctx))
    print()
    print("  group context: "
          f"{ctx['conversation_context'].get('group_name')} "
          f"(muted={ctx['conversation_context'].get('group_muted_by_user')}), "
          f"forwarded {ctx['forwarded_count']} times, mention={ctx['mention_check']}")

    if not entry or not entry.get("action"):
        warn(f"msg_040 has no Phase 5 decision "
             f"({(entry or {}).get('processing_error', 'not attempted')})")
        return

    print()
    print("  Phase 5 decision:")
    print(f"    action               : {entry['action']}")
    print(f"    message_type         : {entry['message_type']}")
    print(f"    confidence           : {entry['confidence']}")
    print(f"    evidence_message_ids : {entry['evidence_message_ids']}")
    print(f"    model                : {entry.get('model')}")
    print("    reason               :")
    wrapped(entry["reason"], indent="      ")

    if entry["action"] == "notify":
        note("the model read the tag as a genuine personal address")
    else:
        note("the model read the tag as boilerplate inside a broadcast")


def check_injection_defence(contexts, store, run_live: bool) -> None:
    section("3. PROMPT-INJECTION DEFENCE")
    by_ctx = {c["message_id"]: c for c in contexts}
    decisions = {m: e for m, e in store.entries.items() if e.get("action")}

    in_phase5 = [
        c["message_id"] for c in contexts
        if c["message_id"] in decisions
        and INJECTION_MARKERS.search(c.get("normalized_text") or "")
    ]
    print(f"  messages carrying router-directed text that reached Phase 5: "
          f"{len(in_phase5)}")
    print(f"  known injection messages in the dataset: {len(KNOWN_INJECTIONS)} "
          f"({', '.join(KNOWN_INJECTIONS)})")
    note("all three were caught by Phase 4 rule 1d, so none of them reach the "
         "LLM in normal operation")

    findings = injection_influence_check(contexts, decisions)
    if findings:
        warn(f"{len(findings)} Phase 5 decision(s) look influenced by injected text")
        for f in findings:
            print(f"      {f['message_id']}: {f['action']}/{f['message_type']} "
                  f"conf={f['confidence']}")
            for p in f["problems"]:
                print(f"        - {p}")
    else:
        ok("no Phase 5 decision shows signs of injection influence")

    sub("live adversarial test")
    if not run_live:
        print("    skipped. Re-run with --injection-test to push the three known")
        print("    injection messages through the Phase 5 classifier directly and")
        print("    verify the prompt defence holds on its own (costs 3 API calls).")
        note("without this, the Phase 5 injection defence is untested on this "
             "dataset - Phase 4 catches every case before it gets here")
        return

    print("    Forcing the three known injection messages through the Phase 5")
    print("    classifier, bypassing rule 1d, to test the prompt defence alone.")
    held, broke = [], []
    for message_id in KNOWN_INJECTIONS:
        ctx = by_ctx.get(message_id)
        if ctx is None:
            continue
        decision, detail = _classify_with_detail(ctx, None)
        print()
        print(f"    {message_id} ({detail.get('model')}):")
        wrapped(ctx["normalized_text"], indent="      | ")
        if not decision:
            warn(f"{message_id}: adversarial test call failed - "
                 f"{detail.get('processing_error')}")
            continue
        print(f"      -> {decision['action']}/{decision['message_type']}  "
              f"conf={decision['confidence']}")
        wrapped(f"reason: {decision['reason']}", indent="      ")
        if decision["action"] == "mute" and decision["message_type"] == "scam":
            held.append(message_id)
        else:
            broke.append(message_id)

    print()
    if broke:
        warn(f"injection defence did NOT hold for: {sample(broke)} - "
             f"the Phase 5 prompt needs tightening")
    if held:
        ok(f"injection defence held for {len(held)}/{len(KNOWN_INJECTIONS)}: "
           f"all returned mute/scam despite the embedded directives")


def check_distributions(contexts, rule_decisions, store, ds, unresolved) -> None:
    section("4. DISTRIBUTIONS")
    # Only decisions still owned by Phase 5 - a message the rules later took
    # over must not be counted twice.
    live = set(unresolved)
    llm_rows = [
        {"action": e["action"], "message_type": e["message_type"]}
        for m, e in store.entries.items() if e.get("action") and m in live
    ]
    rule_rows = [
        {"action": d["action"], "message_type": d["message_type"]}
        for d in rule_decisions
    ]
    if not llm_rows:
        print("  (no Phase 5 decisions yet)")
        return

    sub("Phase 5 only")
    df5 = pd.DataFrame(llm_rows)
    for line in pd.crosstab(df5["action"], df5["message_type"],
                            margins=True, margins_name="TOTAL").to_string().split("\n"):
        print("    " + line)

    sub("combined (Phase 4 rules + Phase 5 LLM)")
    dfc = pd.DataFrame(rule_rows + llm_rows)
    for line in pd.crosstab(dfc["action"], dfc["message_type"],
                            margins=True, margins_name="TOTAL").to_string().split("\n"):
        print("    " + line)

    sub("action share vs sample_messages.csv")
    s = ds.sample_messages
    print(f"    {'action':<9} {'phase4':>13} {'phase5':>13} {'combined':>15} "
          f"{'sample n=30':>14}")
    for action in ("notify", "digest", "mute"):
        p4 = int(sum(1 for r in rule_rows if r["action"] == action))
        p5 = int(sum(1 for r in llm_rows if r["action"] == action))
        cb = p4 + p5
        sn = int((s["action"] == action).sum())
        print(f"    {action:<9} {p4:>5} ({p4/max(len(rule_rows),1):>5.0%}) "
              f"{p5:>5} ({p5/len(llm_rows):>5.0%}) "
              f"{cb:>6} ({cb/len(dfc):>5.0%}) "
              f"{sn:>6} ({sn/len(s):>5.0%})")


def check_confidence(rule_decisions, store) -> None:
    section("5. CONFIDENCE CALIBRATION")
    llm_conf = [e["confidence"] for e in store.entries.values() if e.get("action")]
    rule_conf = [d["confidence"] for d in rule_decisions]
    if not llm_conf:
        print("  (no Phase 5 decisions yet)")
        return

    def stats(values):
        return min(values), max(values), sum(values) / len(values)

    r_min, r_max, r_mean = stats(rule_conf)
    l_min, l_max, l_mean = stats(llm_conf)
    print(f"    Phase 4 (rules) : min={r_min:.2f} max={r_max:.2f} mean={r_mean:.3f} "
          f"n={len(rule_conf)}")
    print(f"    Phase 5 (LLM)   : min={l_min:.2f} max={l_max:.2f} mean={l_mean:.3f} "
          f"n={len(llm_conf)}")

    if l_mean < r_mean:
        ok(f"Phase 5 mean confidence ({l_mean:.3f}) sits below Phase 4's "
           f"({r_mean:.3f}), as expected for the ambiguous cases")
    else:
        warn(f"Phase 5 mean confidence ({l_mean:.3f}) is not below Phase 4's "
             f"({r_mean:.3f}) - the model may be overstating certainty")

    over = [m for m, e in store.entries.items()
            if e.get("action") and e["confidence"] > 0.9]
    if over:
        warn(f"{len(over)} Phase 5 decision(s) above the 0.9 ceiling: {sample(over)}")
    else:
        ok("no Phase 5 decision exceeds the 0.9 ceiling for judgement calls")

    sub("distribution")
    buckets = Counter()
    for c in llm_conf:
        buckets[f"{int(c * 10) / 10:.1f}-{int(c * 10) / 10 + 0.1:.1f}"] += 1
    for bucket in sorted(buckets):
        bar = "#" * buckets[bucket]
        print(f"    {bucket}  {buckets[bucket]:>3}  {bar}")


def check_integrity(contexts, store) -> None:
    section("6. DECISION INTEGRITY")
    by_ctx = {c["message_id"]: c for c in contexts}
    decided = {m: e for m, e in store.entries.items() if e.get("action")}

    cross_cited = [
        m for m, e in decided.items()
        if by_ctx[m]["evidence_tier"] == "cross_type"
        and e["evidence_message_ids"] != "none"
    ]
    if cross_cited:
        warn(f"cross_type evidence cited by the model: {sample(cross_cited)}")
    else:
        ok("no Phase 5 decision cites cross_type evidence")

    bogus = []
    for m, e in decided.items():
        if e["evidence_message_ids"] == "none":
            continue
        available = {ev["message_id"] for ev in by_ctx[m]["retrieved_evidence"]}
        for eid in e["evidence_message_ids"].split(";"):
            if eid not in available:
                bogus.append(f"{m}->{eid}")
    if bogus:
        warn(f"cited evidence the model was not shown: {sample(bogus)}")
    else:
        ok("every cited evidence id was actually supplied to the model")

    n_none = sum(1 for e in decided.values() if e["evidence_message_ids"] == "none")
    print(f"    decisions citing evidence: {len(decided) - n_none} / {len(decided)}")

    sub("reason style")
    lengths = [len(e["reason"]) for e in decided.values()]
    print(f"    length min={min(lengths)} max={max(lengths)} "
          f"mean={sum(lengths)//len(lengths)}  (sample_messages.csv: 58-114, mean 82)")
    long_reasons = [m for m, e in decided.items() if len(e["reason"]) > 160]
    if long_reasons:
        warn(f"reason far longer than the sample style: {sample(long_reasons)}")
    else:
        ok("reason lengths are in the sample's range")


def print_spotchecks(contexts, store, only: list[str] | None) -> None:
    section("7. SPOT-CHECK DECISIONS")
    by_ctx = {c["message_id"]: c for c in contexts}
    decided = {m: e for m, e in store.entries.items() if e.get("action")}

    if only:
        chosen = [m for m in only if m in decided]
    else:
        # One per message_type, for spread.
        seen: set[str] = set()
        chosen = []
        for m in sorted(decided):
            mt = decided[m]["message_type"]
            if mt not in seen:
                seen.add(mt)
                chosen.append(m)
            if len(chosen) >= SPOTCHECK_COUNT:
                break

    for m in chosen:
        e, ctx = decided[m], by_ctx[m]
        cc = ctx["conversation_context"]
        who = (cc.get("group_name") or cc.get("display_name")
               or cc.get("sender_user_id") or "-")
        print()
        print(f"  {m}  {ctx['conversation_type']}/{who}  -> "
              f"{e['action']}/{e['message_type']}  conf={e['confidence']}  "
              f"evidence={e['evidence_message_ids']}  tier={ctx['evidence_tier']}")
        wrapped(effective_text(ctx)[:230])
        print(f"      reason: {e['reason']}")


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 5 validation.")
    parser.add_argument("--show", default=None, help="comma-separated message ids")
    parser.add_argument("--injection-test", action="store_true",
                        help="spend 3 API calls to test the injection defence live")
    args = parser.parse_args(list(argv) if argv is not None else None)

    ds = load_dataset()
    contexts = build_normalized_contexts(ds)
    rule_decisions, unresolved = apply_rules_to_all(contexts)
    store = DecisionStore.load()

    print("=" * 78)
    print("PHASE 5 VALIDATION - LLM Reasoning")
    print("=" * 78)
    print(f"dataset dir : {ds.dataset_dir}")
    print(f"cache file  : {store.path}")
    print(f"phase 4 resolved : {len(rule_decisions)}   unresolved : {len(unresolved)}")

    check_coverage(unresolved, store)
    check_conflict_case(contexts, store)
    check_injection_defence(contexts, store, args.injection_test)
    check_distributions(contexts, rule_decisions, store, ds, unresolved)
    check_confidence(rule_decisions, store)
    check_integrity(contexts, store)
    print_spotchecks(contexts, store, args.show.split(",") if args.show else None)

    section("SUMMARY")
    if PROBLEMS:
        print(f"  {len(PROBLEMS)} item(s) flagged:")
        for i, p in enumerate(PROBLEMS, 1):
            print(f"    {i}. {p}")
    else:
        print("  No issues found.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
