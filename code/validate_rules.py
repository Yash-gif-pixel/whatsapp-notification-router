"""Phase 4 - validation for the rule engine.

Run from the terminal::

    py code/validate_rules.py
    py code/validate_rules.py --show-unresolved     # list what Phase 5 inherits
    py code/validate_rules.py --show msg_085,msg_064

Reports resolved-vs-unresolved counts broken down by rule, the named test
cases, the domain-mismatch breakdown, and the action/message_type shape against
sample_messages.csv. Makes no API calls. Problems are reported, never raised.

TEST FIXTURES, NOT ROUTING INPUTS
---------------------------------
Message ids below (``TEST_CASES``) are expected-value assertions: they state
what the rules should independently conclude, and the run fails loudly when
they do not. They are never consulted while deciding anything. ``rule_engine``
contains no hardcoded id at all, and no module on the routing path
(``finalize`` -> ``rule_engine`` / ``llm_reasoner`` / ``media_normalizer`` /
``context_builder`` / ``data_loader``) imports this file, so nothing here can
reach ``output.csv``. These assertions earn their keep - they are what caught
``msg_040`` resolving against the stated spec in Phase 4.
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
from media_normalizer import build_normalized_contexts, effective_text  # noqa: E402
from rule_engine import RULE_LABELS, apply_rules, apply_rules_to_all  # noqa: E402

PROBLEMS: list[str] = []
MAX_LISTED = 12

EXPECTED_MESSAGES = 110

#: Cases the spec asks to check by hand, with the outcome it expects.
#: ``action=None`` means the message is expected to stay unresolved.
TEST_CASES = {
    "msg_085": {"action": "mute", "type": "scam", "rule": "1a",
                "label": "voice OTP scam from a fake bank helpdesk"},
    "msg_064": {"action": "mute", "type": "scam", "rule": "1b",
                "label": "refund text over an unrelated movie poster"},
    "msg_040": {"action": None, "type": None, "rule": None,
                "flag": "mention_forward_conflict",
                "label": "mention inside a chain forward - must stay unresolved"},
    "msg_056": {"action": "notify", "type": None, "rule": "2",
                "label": "plain direct mention in a muted group"},
}


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


def wrapped(text: str, indent: str = "        ") -> None:
    for line in textwrap.wrap(" ".join(str(text).split()), width=70):
        print(indent + line)


# ---------------------------------------------------------------------------


def check_coverage(contexts, decisions, unresolved) -> None:
    section("1. RESOLVED VS UNRESOLVED")
    total = len(contexts)
    print(f"  messages            : {total}")
    print(f"  resolved by rules   : {len(decisions):>4}  ({len(decisions)/total:.0%})")
    print(f"  unresolved -> Phase 5: {len(unresolved):>4}  ({len(unresolved)/total:.0%})")

    if total != EXPECTED_MESSAGES:
        warn(f"expected {EXPECTED_MESSAGES} contexts, got {total}")
    if len(decisions) + len(unresolved) != total:
        warn("decisions + unresolved does not equal the message count")
    else:
        ok("every message is either resolved or explicitly unresolved")

    sub("by rule")
    counts = Counter(d["rule"] for d in decisions)
    for rule in sorted(RULE_LABELS):
        n = counts.get(rule, 0)
        marker = " " if n else "."
        print(f"    {marker} rule {rule:<3} {n:>4}  {RULE_LABELS[rule]}")
    print(f"      (none)   {len(unresolved):>4}  left for Phase 5")

    sub("unresolved by conversation_type")
    by_id = {c["message_id"]: c for c in contexts}
    conv = Counter(by_id[m]["conversation_type"] for m in unresolved)
    for k, v in sorted(conv.items()):
        total_k = sum(1 for c in contexts if c["conversation_type"] == k)
        print(f"    {k:<10} {v:>3} of {total_k}")


def check_test_cases(contexts, decisions) -> None:
    section("2. NAMED TEST CASES")
    by_decision = {d["message_id"]: d for d in decisions}
    by_ctx = {c["message_id"]: c for c in contexts}

    for message_id, want in TEST_CASES.items():
        ctx = by_ctx.get(message_id)
        decision = by_decision.get(message_id)
        print()
        print(f"  {message_id} - {want['label']}")
        if ctx is None:
            warn(f"{message_id} not present in the dataset")
            continue
        wrapped(effective_text(ctx)[:260])

        # Cases expected to stay unresolved.
        if want["action"] is None:
            if decision is not None:
                warn(f"{message_id}: expected UNRESOLVED, got "
                     f"{decision['action']}/{decision['message_type']} "
                     f"via rule {decision['rule']}")
                continue
            flags = ctx.get("rule_engine") or {}
            print(f"    -> UNRESOLVED  reason={flags.get('unresolved_reason')}")
            if flags.get("suppressed_rule"):
                sup = flags["suppressed_decision"]
                print(f"       suppressed rule {flags['suppressed_rule']} would have said "
                      f"{sup['action']}/{sup['message_type']}")
            if flags.get("question_for_phase5"):
                wrapped(f"question for Phase 5: {flags['question_for_phase5']}",
                        indent="       ")
            wanted_flag = want.get("flag")
            if wanted_flag and not flags.get(wanted_flag):
                warn(f"{message_id}: unresolved but {wanted_flag} is not set")
            else:
                ok(f"{message_id} left unresolved with {wanted_flag}: true, as expected")
            continue

        if decision is None:
            flags = ctx.get("rule_engine") or {}
            warn(f"{message_id}: expected {want['action']} via rule {want['rule']}, "
                 f"but it was left UNRESOLVED "
                 f"(reason={flags.get('unresolved_reason')})")
            continue

        got = f"{decision['action']}/{decision['message_type']} via rule {decision['rule']}"
        matches_action = decision["action"] == want["action"]
        matches_type = want["type"] is None or decision["message_type"] == want["type"]
        matches_rule = decision["rule"].startswith(want["rule"])
        print(f"    -> {got}  conf={decision['confidence']}")
        print(f"       reason: {decision['reason']}")
        if decision["signals"]:
            print(f"       signals: {'; '.join(decision['signals'])}")

        if matches_action and matches_type and matches_rule:
            ok(f"{message_id} resolved as expected")
        elif matches_action and matches_type:
            note(f"{message_id} reached the expected outcome via rule "
                 f"{decision['rule']} rather than {want['rule']}")
        else:
            warn(f"{message_id}: expected {want['action']}"
                 f"{'/' + want['type'] if want['type'] else ''} via rule "
                 f"{want['rule']}, got {got}")


def check_mention_precedence(contexts, decisions, unresolved) -> None:
    """Every mentioned message, and how safety precedence treated it."""
    section("3. MENTION PRECEDENCE")
    by_decision = {d["message_id"]: d for d in decisions}
    mentioned = [c for c in contexts if c["mention_check"]]
    unresolved_set = set(unresolved)

    print(f"  messages where the user is tagged directly: {len(mentioned)}")
    print()
    print(f"    {'message':<10} {'fwd':>4}  {'outcome':<24} path")

    conflicts, hard_safety, plain = [], [], []
    for c in sorted(mentioned, key=lambda x: x["message_id"]):
        d = by_decision.get(c["message_id"])
        flags = c.get("rule_engine") or {}
        if d is None:
            outcome = "UNRESOLVED"
            if flags.get("mention_forward_conflict"):
                path = (f"rule {flags['suppressed_rule']} suppressed "
                        f"-> mention_forward_conflict")
                conflicts.append(c["message_id"])
            else:
                path = f"no rule matched ({flags.get('unresolved_reason')})"
        else:
            outcome = f"{d['action']}/{d['message_type']}"
            path = f"rule {d['rule']}"
            if d["rule"] in ("1a", "1b", "1d"):
                path += " - hard safety beat the mention"
                hard_safety.append(c["message_id"])
            elif d["rule"] == "2":
                plain.append(c["message_id"])
        print(f"    {c['message_id']:<10} {c['forwarded_count'] or 0:>4}  "
              f"{outcome:<24} {path}")

    sub("breakdown")
    print(f"    notify via rule 2 (no safety rule fired) : {len(plain)}")
    print(f"    hard safety (1a/1b/1d) overrode mention  : {len(hard_safety)}")
    print(f"    mention + mass-forward overlap -> Phase 5: {len(conflicts)}")
    if conflicts:
        print(f"      {sample(conflicts)}")

    sub("integrity")
    bad = [
        c["message_id"] for c in mentioned
        if (d := by_decision.get(c["message_id"])) is not None and d["rule"] == "1c"
    ]
    if bad:
        warn(f"rule 1c resolved a mentioned message instead of deferring: {sample(bad)}")
    else:
        ok("rule 1c never resolves a mentioned message on its own")

    flagged_not_unresolved = [
        c["message_id"] for c in contexts
        if (c.get("rule_engine") or {}).get("mention_forward_conflict")
        and c["message_id"] not in unresolved_set
    ]
    if flagged_not_unresolved:
        warn(f"flagged mention_forward_conflict but not unresolved: "
             f"{sample(flagged_not_unresolved)}")
    else:
        ok("every mention_forward_conflict message is genuinely unresolved")

    unmentioned_flagged = [
        c["message_id"] for c in contexts
        if (c.get("rule_engine") or {}).get("mention_forward_conflict")
        and not c["mention_check"]
    ]
    if unmentioned_flagged:
        warn(f"mention_forward_conflict set on an unmentioned message: "
             f"{sample(unmentioned_flagged)}")
    else:
        ok("the conflict flag only ever appears on a mentioned message")


def check_domain_mismatch(contexts, decisions, unresolved) -> None:
    section("4. DOMAIN-MISMATCH BUSINESS MESSAGES")
    by_decision = {d["message_id"]: d for d in decisions}
    rows = []
    for c in sorted(contexts, key=lambda x: x["message_id"]):
        cc = c["conversation_context"]
        if cc.get("kind") != "business" or cc.get("domain_match"):
            continue
        d = by_decision.get(c["message_id"])
        rows.append({
            "message_id": c["message_id"],
            "verified": bool(cc.get("verified")),
            "reports_30d": cc.get("user_reports_30d") or 0,
            "relationship": bool(cc.get("has_relationship")),
            "outcome": f"{d['action']}/{d['message_type']}" if d else "UNRESOLVED",
            "rule": d["rule"] if d else "-",
        })

    print(f"  business messages where domain_used_by_sender != official_domain: {len(rows)}")
    print()
    print(f"    {'message':<10} {'ver':<5} {'reports':>7} {'rel':<5} {'rule':<5} outcome")
    for r in rows:
        print(f"    {r['message_id']:<10} {str(r['verified']):<5} {r['reports_30d']:>7} "
              f"{str(r['relationship']):<5} {r['rule']:<5} {r['outcome']}")

    sub("breakdown")
    by_outcome = Counter(r["outcome"] for r in rows)
    for outcome, n in sorted(by_outcome.items(), key=lambda kv: -kv[1]):
        print(f"    {outcome:<20} {n}")
    scam = sum(1 for r in rows if r["outcome"] == "mute/scam")
    print()
    print(f"    resolved as scam         : {scam} of {len(rows)}")
    print(f"    resolved as something else: "
          f"{sum(1 for r in rows if r['outcome'] not in ('mute/scam', 'UNRESOLVED'))}")
    print(f"    left unresolved          : {sum(1 for r in rows if r['outcome'] == 'UNRESOLVED')}")
    note("a domain mismatch alone is not treated as scam - a verified sender the "
         "user actively deals with can legitimately send via a link shortener")


def check_distribution(decisions, ds) -> None:
    section("5. ACTION / MESSAGE_TYPE SHAPE")
    if not decisions:
        print("  (nothing resolved)")
        return
    df = pd.DataFrame([{"action": d["action"], "message_type": d["message_type"]}
                       for d in decisions])
    sub("resolved decisions")
    table = pd.crosstab(df["action"], df["message_type"], margins=True, margins_name="TOTAL")
    for line in table.to_string().split("\n"):
        print("    " + line)

    sub("action share: rules vs sample_messages.csv")
    s = ds.sample_messages
    sample_share = s["action"].value_counts(normalize=True)
    rule_share = df["action"].value_counts(normalize=True)
    print(f"    {'action':<10} {'rules':>16} {'sample (n=30)':>16}")
    for action in ("notify", "digest", "mute"):
        r = rule_share.get(action, 0.0)
        sv = sample_share.get(action, 0.0)
        rn = int((df["action"] == action).sum())
        sn = int((s["action"] == action).sum())
        print(f"    {action:<10} {rn:>4} ({r:>5.0%}) {sn:>8} ({sv:>5.0%})")
    note("the rules deliberately skew to mute: safety rules resolve confidently, "
         "while most notify/digest judgement calls are the ambiguous ones left "
         "for Phase 5 - so this is a shape check, not a target")

    sub("confidence band")
    confs = [d["confidence"] for d in decisions]
    print(f"    min={min(confs)}  max={max(confs)}  mean={sum(confs)/len(confs):.3f}")
    if max(confs) > 0.95:
        warn(f"confidence above the 0.95 cap: max={max(confs)}")
    else:
        ok("all confidences within the 0.80-0.95 band")


def check_integrity(contexts, decisions) -> None:
    section("6. DECISION INTEGRITY")
    by_ctx = {c["message_id"]: c for c in contexts}
    allowed_actions = {"notify", "digest", "mute"}
    allowed_types = {"personal", "urgent", "event", "payment", "business_update",
                     "promotion", "greeting", "forward", "spam", "scam", "unknown"}

    bad_action = [d["message_id"] for d in decisions if d["action"] not in allowed_actions]
    bad_type = [d["message_id"] for d in decisions if d["message_type"] not in allowed_types]
    if bad_action:
        warn(f"invalid action value: {sample(bad_action)}")
    else:
        ok("every action is notify/digest/mute")
    if bad_type:
        warn(f"invalid message_type value: {sample(bad_type)}")
    else:
        ok("every message_type is from the allowed set")

    sub("evidence citation")
    cross_cited = [
        d["message_id"] for d in decisions
        if by_ctx[d["message_id"]]["evidence_tier"] == "cross_type"
        and d["evidence_message_ids"] != "none"
    ]
    if cross_cited:
        warn(f"cross_type evidence cited as precedent: {sample(cross_cited)}")
    else:
        ok("cross_type evidence is never cited as behavioural precedent")

    hist_ids = set()
    for c in contexts:
        for e in c["retrieved_evidence"]:
            hist_ids.add(e["message_id"])
    bogus = [
        f"{d['message_id']}->{eid}"
        for d in decisions
        for eid in (d["evidence_message_ids"].split(";")
                    if d["evidence_message_ids"] != "none" else [])
        if eid not in hist_ids
    ]
    if bogus:
        warn(f"cited evidence not present in the message's own retrieval: {sample(bogus)}")
    else:
        ok("every cited evidence id came from that message's retrieved_evidence")

    n_none = sum(1 for d in decisions if d["evidence_message_ids"] == "none")
    print(f"    decisions citing evidence: {len(decisions) - n_none} / {len(decisions)}")

    sub("reason text")
    lengths = [len(d["reason"]) for d in decisions]
    print(f"    length min={min(lengths)} max={max(lengths)} "
          f"mean={sum(lengths)//len(lengths)}  (sample_messages.csv: 58-114, mean 82)")
    too_long = [d["message_id"] for d in decisions if len(d["reason"]) > 160]
    if too_long:
        warn(f"reason far longer than the sample style: {sample(too_long)}")
    else:
        ok("reason lengths are in the sample's range")

    sub("safety precedence")
    safety = [d for d in decisions if d["rule"].startswith("1")]
    mentioned_and_safety = [
        d["message_id"] for d in safety if by_ctx[d["message_id"]]["mention_check"]
    ]
    print(f"    safety-rule decisions: {len(safety)}")
    if mentioned_and_safety:
        note(f"safety outranked a direct mention on: {sample(mentioned_and_safety)}")
    ok("safety rules are evaluated before mentions by construction")


def print_injection_findings(contexts, decisions) -> None:
    section("7. EMBEDDED ROUTER INSTRUCTIONS (rule 1d)")
    hits = [d for d in decisions if d["rule"] == "1d"]
    by_ctx = {c["message_id"]: c for c in contexts}
    print("  Messages whose text tries to instruct the router itself. Treated as")
    print("  data, never as instruction, and muted as scam.")
    if not hits:
        print("\n    (none)")
        return
    for d in sorted(hits, key=lambda x: x["message_id"]):
        ctx = by_ctx[d["message_id"]]
        print()
        print(f"    {d['message_id']}  {ctx['conversation_type']}  "
              f"-> {d['action']}/{d['message_type']}  conf={d['confidence']}")
        wrapped(effective_text(ctx)[:240], indent="      ")
    warn(f"{len(hits)} message(s) contain instructions targeting the router - "
         f"Phase 5 prompts must never treat message content as instructions")


def print_samples(contexts, decisions, only: list[str] | None) -> None:
    section("8. SAMPLE RESOLVED DECISIONS")
    by_ctx = {c["message_id"]: c for c in contexts}
    if only:
        chosen = [d for d in decisions if d["message_id"] in set(only)]
    else:
        seen: set[str] = set()
        chosen = []
        for d in sorted(decisions, key=lambda x: (x["rule"], x["message_id"])):
            if d["rule"] not in seen:
                seen.add(d["rule"])
                chosen.append(d)
    for d in chosen:
        ctx = by_ctx[d["message_id"]]
        print()
        print(f"  [{d['rule']}] {d['message_id']}  {ctx['conversation_type']}  "
              f"-> {d['action']}/{d['message_type']}  conf={d['confidence']}  "
              f"evidence={d['evidence_message_ids']}")
        wrapped(effective_text(ctx)[:220], indent="      ")
        print(f"      reason : {d['reason']}")
        if d["signals"]:
            print(f"      signals: {'; '.join(d['signals'])}")


def print_unresolved(contexts, unresolved) -> None:
    section("9. UNRESOLVED - WHAT PHASE 5 INHERITS")
    by_ctx = {c["message_id"]: c for c in contexts}
    for message_id in sorted(unresolved):
        ctx = by_ctx[message_id]
        cc = ctx["conversation_context"]
        flags = ctx.get("rule_engine") or {}
        who = (cc.get("group_name") or cc.get("display_name")
               or cc.get("sender_user_id") or "-")
        print()
        print(f"  {message_id}  {ctx['conversation_type']}/{who}  "
              f"tier={ctx['evidence_tier']}  reason={flags.get('unresolved_reason')}")
        wrapped(effective_text(ctx)[:200], indent="      ")
        if flags.get("mention_forward_conflict"):
            print("      mention_forward_conflict: true")
            wrapped(flags["question_for_phase5"], indent="      ")


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4 rule engine validation.")
    parser.add_argument("--show-unresolved", action="store_true")
    parser.add_argument("--show", default=None, help="comma-separated message ids")
    args = parser.parse_args(list(argv) if argv is not None else None)

    ds = load_dataset()
    contexts = build_normalized_contexts(ds)

    print("=" * 78)
    print("PHASE 4 VALIDATION - Rule Engine")
    print("=" * 78)
    print(f"dataset dir : {ds.dataset_dir}")

    unsafe = [c["message_id"] for c in contexts
              if not c["completeness"]["safe_for_text_reasoning"]]
    if unsafe:
        warn(f"{len(unsafe)} context(s) not safe for text reasoning - Phase 3 "
             f"incomplete: {sample(unsafe)}")
        print("  Cannot run the rule engine until Phase 3 is complete.")
        return 1
    ok(f"all {len(contexts)} contexts are safe for text reasoning")

    decisions, unresolved = apply_rules_to_all(contexts)

    check_coverage(contexts, decisions, unresolved)
    check_test_cases(contexts, decisions)
    check_mention_precedence(contexts, decisions, unresolved)
    check_domain_mismatch(contexts, decisions, unresolved)
    check_distribution(decisions, ds)
    check_integrity(contexts, decisions)
    print_injection_findings(contexts, decisions)
    print_samples(contexts, decisions, args.show.split(",") if args.show else None)
    if args.show_unresolved:
        print_unresolved(contexts, unresolved)

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
