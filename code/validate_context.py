"""Phase 2 - validation for the assembled context objects.

Run from the terminal::

    py code/validate_context.py

Checks coverage, evidence-tier distribution, the transcription flag, and the
internal consistency of every context object, then prints three full context
objects (a Tier 1 case, a Tier 2 fallback case, and a needs_transcription
case) for manual verification.

Nothing here writes to disk. Problems are reported, never raised.

TEST FIXTURES, NOT ROUTING INPUTS
---------------------------------
Message ids below (``EXPECTED_CROSS_TYPE_IDS``) are expected-value assertions
about which messages fall through to the weakest evidence tier. They are never
inputs to any decision. ``context_builder`` contains no hardcoded id at all,
and no module on the routing path (``finalize`` -> ``rule_engine`` /
``llm_reasoner`` / ``media_normalizer`` / ``context_builder`` /
``data_loader``) imports this file.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from context_builder import (  # noqa: E402
    EVIDENCE_TIER_STRENGTH,
    EVIDENCE_TIERS,
    MAX_EVIDENCE_PER_TIER,
    build_all_contexts,
    to_jsonable,
)
from data_loader import load_dataset  # noqa: E402

PROBLEMS: list[str] = []
MAX_LISTED = 10

# Expectations stated in the Phase 2 spec, checked explicitly below.
EXPECTED_MESSAGES = 110
EXPECTED_NEEDS_TRANSCRIPTION = 8
EXPECTED_TIER_COUNTS = {"exact": 104, "fallback": 3, "cross_type": 3, "none": 0}
#: The three personal messages whose users have no personal history at all.
EXPECTED_CROSS_TYPE_IDS = {"msg_089", "msg_090", "msg_096"}


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


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def check_coverage(contexts: list[dict[str, Any]], ds) -> None:
    section("1. COVERAGE")
    n = len(contexts)
    print(f"  context objects built : {n}")
    print(f"  rows in messages.csv  : {len(ds.messages)}")

    if n != len(ds.messages):
        warn(f"built {n} contexts for {len(ds.messages)} messages")
    elif n != EXPECTED_MESSAGES:
        warn(f"expected {EXPECTED_MESSAGES} messages, dataset has {n}")
    else:
        ok(f"all {n} messages have a context object")

    ctx_ids = [c["message_id"] for c in contexts]
    msg_ids = list(ds.messages["message_id"])
    if set(ctx_ids) != set(msg_ids):
        missing = set(msg_ids) - set(ctx_ids)
        extra = set(ctx_ids) - set(msg_ids)
        if missing:
            warn(f"no context for {len(missing)} message(s): {sample(sorted(missing))}")
        if extra:
            warn(f"context for {len(extra)} unknown message(s): {sample(sorted(extra))}")
    else:
        ok("context message_ids match messages.csv exactly")

    dupes = [k for k, v in Counter(ctx_ids).items() if v > 1]
    if dupes:
        warn(f"duplicate context objects for: {sample(dupes)}")
    else:
        ok("no duplicate context objects")

    missing_profile = [c["message_id"] for c in contexts if c["user_profile"] is None]
    if missing_profile:
        warn(f"{len(missing_profile)} context(s) have no user_profile: {sample(missing_profile)}")
    else:
        ok("every context has a resolved user_profile")

    bad_conv = [
        c["message_id"]
        for c in contexts
        if c["conversation_context"].get("kind") != c["conversation_type"]
    ]
    if bad_conv:
        warn(f"conversation_context kind mismatch: {sample(bad_conv)}")
    else:
        ok("conversation_context kind matches conversation_type for every message")

    unresolved = [
        c["message_id"]
        for c in contexts
        if c["conversation_context"].get("group_found") is False
        or c["conversation_context"].get("business_found") is False
    ]
    if unresolved:
        warn(f"conversation_context could not resolve its entity: {sample(unresolved)}")
    else:
        ok("every group/business conversation_context resolved its entity")


def check_evidence_tiers(contexts: list[dict[str, Any]], ds) -> None:
    section("2. EVIDENCE TIERS")
    counts = Counter(c["evidence_tier"] for c in contexts)
    total = len(contexts)

    labels = {
        "exact": "TIER 1  (exact)     ",
        "fallback": "TIER 2  (fallback)  ",
        "cross_type": "TIER 2b (cross_type)",
        "none": "TIER 3  (none)      ",
    }
    for tier in EVIDENCE_TIERS:
        print(f"  {labels[tier]} : {counts.get(tier, 0):>4} / {total}   "
              f"(strength {EVIDENCE_TIER_STRENGTH[tier]})")

    sub("tier by conversation_type")
    df = pd.DataFrame(
        [{"conversation_type": c["conversation_type"], "tier": c["evidence_tier"]} for c in contexts]
    )
    table = pd.crosstab(df["conversation_type"], df["tier"], margins=True, margins_name="TOTAL")
    for line in table.to_string().split("\n"):
        print("    " + line)

    sub("spec expectation: exact=104, fallback=3, cross_type=3, none=0")
    for tier, expected in EXPECTED_TIER_COUNTS.items():
        got = counts.get(tier, 0)
        if got == expected:
            ok(f"{tier}: {got} as expected")
        else:
            ids = sample(c["message_id"] for c in contexts if c["evidence_tier"] == tier)
            warn(f"{tier}: expected {expected}, got {got} ({ids})")

    sub("spec expectation: no message is left without evidence")
    stranded = [c for c in contexts if c["evidence_tier"] == "none"]
    if not stranded:
        ok("every message resolved evidence at some tier")
    else:
        warn(
            f"{len(stranded)} message(s) still at TIER 3 (none): "
            f"{sample(c['message_id'] for c in stranded)}"
        )
        print()
        print(f"    {'message':<10} {'conv_type':<10} {'user':<7} {'exact':>6} {'same_ct':>8} "
              f"{'any':>5}  history mix")
        for c in stranded:
            d = c["evidence_diagnostics"]
            mix = ", ".join(f"{k}={v}" for k, v in sorted(d["history_conversation_type_mix"].items()))
            print(f"    {c['message_id']:<10} {c['conversation_type']:<10} {c['user_id']:<7} "
                  f"{d['n_exact_matches']:>6} {d['n_same_conversation_type']:>8} "
                  f"{d['n_any_history_for_user']:>5}  {mix}")

    sub("fallback cases (Tier 2 - same conversation_type)")
    fb = [c for c in contexts if c["evidence_tier"] == "fallback"]
    if not fb:
        print("    (none)")
    for c in fb:
        d = c["evidence_diagnostics"]
        print(f"    {c['message_id']}  {c['conversation_type']:<9} user={c['user_id']}  "
              f"same_ct_history={d['n_same_conversation_type']}  "
              f"kept={len(c['retrieved_evidence'])}")

    sub("cross-type cases (Tier 2b - other conversation_types)")
    xt = [c for c in contexts if c["evidence_tier"] == "cross_type"]
    if not xt:
        print("    (none)")
    for c in xt:
        d = c["evidence_diagnostics"]
        mix = ", ".join(f"{k}={v}" for k, v in sorted(d["history_conversation_type_mix"].items()))
        srcs = [e["source_conversation_type"] for e in c["retrieved_evidence"]]
        print(f"    {c['message_id']}  {c['conversation_type']:<9} user={c['user_id']}  "
              f"any_history={d['n_any_history_for_user']}  kept={len(c['retrieved_evidence'])}")
        print(f"      user history mix : {mix}")
        print(f"      evidence sourced from : {srcs}")

    sub("cross-type integrity")
    got_ids = {c["message_id"] for c in xt}
    if got_ids == EXPECTED_CROSS_TYPE_IDS:
        ok(f"cross_type set is exactly the expected {sorted(EXPECTED_CROSS_TYPE_IDS)}")
    else:
        warn(
            f"cross_type set is {sorted(got_ids)}, expected {sorted(EXPECTED_CROSS_TYPE_IDS)}"
        )

    unlabelled = [
        f"{c['message_id']}->{e['message_id']}"
        for c in contexts
        for e in c["retrieved_evidence"]
        if not e.get("source_conversation_type")
    ]
    if unlabelled:
        warn(f"evidence item(s) with no source_conversation_type: {sample(unlabelled)}")
    else:
        ok("every evidence item names the conversation_type it came from")

    mislabelled = [
        f"{c['message_id']}->{e['message_id']}"
        for c in contexts
        for e in c["retrieved_evidence"]
        if e["source_conversation_type"] != e["conversation_type"]
    ]
    if mislabelled:
        warn(f"source_conversation_type disagrees with the history row: {sample(mislabelled)}")
    else:
        ok("source_conversation_type matches the underlying message_history row")

    # cross_type only fires when no same-type history exists, so by
    # construction no cross_type item may share the message's own type.
    contaminated = [
        f"{c['message_id']}->{e['message_id']}({e['source_conversation_type']})"
        for c in xt
        for e in c["retrieved_evidence"]
        if e["source_conversation_type"] == c["conversation_type"]
    ]
    if contaminated:
        warn(
            "cross_type evidence drawn from the message's OWN conversation_type - "
            f"that should have been Tier 2: {sample(contaminated)}"
        )
    else:
        ok("no cross_type evidence comes from the message's own conversation_type")

    bad_strength = [
        c["message_id"]
        for c in contexts
        if c["evidence_strength"] != EVIDENCE_TIER_STRENGTH[c["evidence_tier"]]
        or any(
            e["evidence_strength"] != EVIDENCE_TIER_STRENGTH[e["evidence_tier"]]
            for e in c["retrieved_evidence"]
        )
    ]
    if bad_strength:
        warn(f"evidence_strength disagrees with evidence_tier: {sample(bad_strength)}")
    else:
        ok("evidence_strength agrees with evidence_tier everywhere "
           "(exact=3 > fallback=2 > cross_type=1 > none=0)")

    sub("evidence integrity")
    over_cap = [
        c["message_id"] for c in contexts if len(c["retrieved_evidence"]) > MAX_EVIDENCE_PER_TIER
    ]
    if over_cap:
        warn(f"more than {MAX_EVIDENCE_PER_TIER} evidence items on: {sample(over_cap)}")
    else:
        ok(f"no context exceeds the cap of {MAX_EVIDENCE_PER_TIER} evidence items")

    tier_mismatch = [
        c["message_id"]
        for c in contexts
        if any(e["evidence_tier"] != c["evidence_tier"] for e in c["retrieved_evidence"])
    ]
    if tier_mismatch:
        warn(f"per-item tier disagrees with context tier: {sample(tier_mismatch)}")
    else:
        ok("every evidence item carries the same tier label as its context")

    fabricated = [
        c["message_id"]
        for c in contexts
        if c["evidence_tier"] == "none" and c["retrieved_evidence"]
    ]
    if fabricated:
        warn(f"tier 'none' but evidence present (fabrication): {sample(fabricated)}")
    else:
        ok("tier 'none' always means an empty evidence list - nothing fabricated")

    empty_non_none = [
        c["message_id"]
        for c in contexts
        if c["evidence_tier"] != "none" and not c["retrieved_evidence"]
    ]
    if empty_non_none:
        warn(f"non-'none' tier but no evidence items: {sample(empty_non_none)}")
    else:
        ok("every non-'none' tier has at least one evidence item")

    hist_ids = set(ds.message_history["message_id"].dropna())
    bogus = [
        f"{c['message_id']}->{e['message_id']}"
        for c in contexts
        for e in c["retrieved_evidence"]
        if e["message_id"] not in hist_ids
    ]
    if bogus:
        warn(f"evidence ids not in message_history.csv: {sample(bogus)}")
    else:
        ok("every evidence message_id exists in message_history.csv")

    leaks = [
        f"{c['message_id']}->{e['message_id']}"
        for c in contexts
        for e in c["retrieved_evidence"]
        if e["created_at"] is not None
        and c["created_at"] is not None
        and e["created_at"] >= c["created_at"]
    ]
    if leaks:
        warn(f"evidence dated at/after its message (future leak): {sample(leaks)}")
    else:
        ok("all evidence is strictly older than its message - no future leak")

    unordered = []
    for c in contexts:
        dates = [e["created_at"] for e in c["retrieved_evidence"] if e["created_at"]]
        if dates != sorted(dates, reverse=True):
            unordered.append(c["message_id"])
    if unordered:
        warn(f"evidence not sorted newest-first: {sample(unordered)}")
    else:
        ok("evidence is sorted newest-first everywhere")

    no_reaction = [
        f"{c['message_id']}->{e['message_id']}"
        for c in contexts
        for e in c["retrieved_evidence"]
        if not e["reaction"]["recorded"]
    ]
    if no_reaction:
        warn(f"evidence with no joined reaction row: {sample(no_reaction)}")
    else:
        ok("every evidence item has a joined message_events reaction")

    sizes = [len(c["retrieved_evidence"]) for c in contexts]
    print(f"    evidence per message: min={min(sizes)} max={max(sizes)} "
          f"mean={sum(sizes) / len(sizes):.2f}  (cap {MAX_EVIDENCE_PER_TIER})")


def check_media_flags(contexts: list[dict[str, Any]], ds) -> None:
    section("3. MEDIA STATUS / TRANSCRIPTION FLAG")

    by_status = Counter(c["media_status"]["status"] for c in contexts)
    for status, n in sorted(by_status.items()):
        print(f"  {status:<26} : {n:>4}")

    needs_tr = [c for c in contexts if c["media_status"]["needs_transcription"]]
    needs_img = [c for c in contexts if c["media_status"]["needs_image_description"]]

    sub(f"spec expectation: exactly {EXPECTED_NEEDS_TRANSCRIPTION} needs_transcription")
    if len(needs_tr) == EXPECTED_NEEDS_TRANSCRIPTION:
        ok(f"{len(needs_tr)} message(s) flagged needs_transcription, as expected")
    else:
        warn(
            f"expected {EXPECTED_NEEDS_TRANSCRIPTION} needs_transcription, "
            f"got {len(needs_tr)}: {sample(c['message_id'] for c in needs_tr)}"
        )

    wrong_type = [c["message_id"] for c in needs_tr if c["media_status"]["media_type"] != "voice"]
    if wrong_type:
        warn(f"needs_transcription set on non-voice message(s): {sample(wrong_type)}")
    else:
        ok("every needs_transcription message has media_type == 'voice'")

    with_text = [c["message_id"] for c in needs_tr if c["completeness"]["text_usable"]]
    if with_text:
        warn(f"needs_transcription message(s) unexpectedly have text: {sample(with_text)}")
    else:
        ok("every needs_transcription message has empty message_text")

    all_voice = [c for c in contexts if c["media_status"]["media_type"] == "voice"]
    unflagged = [
        c["message_id"] for c in all_voice if not c["media_status"]["needs_transcription"]
    ]
    if unflagged:
        warn(f"voice message(s) not flagged needs_transcription: {sample(unflagged)}")
    else:
        ok(f"all {len(all_voice)} voice message(s) are flagged")

    sub("the text-reasoning guard")
    unsafe = [c for c in contexts if not c["completeness"]["safe_for_text_reasoning"]]
    print(f"    safe_for_text_reasoning == False on {len(unsafe)} message(s)")
    if {c["message_id"] for c in unsafe} == {c["message_id"] for c in needs_tr}:
        ok("the guard covers exactly the needs_transcription set - "
           "no empty text reaches a text rule")
    else:
        warn(
            "safe_for_text_reasoning does not line up with needs_transcription: "
            f"guard={sample(sorted(c['message_id'] for c in unsafe))}"
        )
    print(f"    message ids: {sample(sorted(c['message_id'] for c in unsafe))}")

    sub("media file resolution")
    unresolved = [
        c["message_id"] for c in contexts
        if c["media_status"]["media_resolved"] is False
    ]
    missing_file = [
        c["message_id"] for c in contexts
        if c["media_status"]["file_exists"] is False
    ]
    if unresolved:
        warn(f"media_id did not resolve for: {sample(unresolved)}")
    else:
        ok("every media_id resolved to a row in images.csv / voice_notes.csv")
    if missing_file:
        warn(f"resolved media file missing on disk for: {sample(missing_file)}")
    else:
        ok("every resolved media file exists on disk")

    print(f"    needs_image_description : {len(needs_img)} "
          f"(all retain caption text: "
          f"{all(c['completeness']['text_usable'] for c in needs_img)})")


def check_mentions_and_passthrough(contexts: list[dict[str, Any]], ds) -> None:
    section("4. MENTIONS, TIMING, PASSTHROUGH")

    mentioned = [c for c in contexts if c["mention_check"]]
    other_only = [c for c in contexts if c["mention_detail"]["mentions_other_user_only"]]
    print(f"  mention_check True (this user tagged) : {len(mentioned)}")
    print(f"  tagged someone else only             : {len(other_only)}")
    note("users.csv has no display-name column, so mentions can only be matched "
         "on @<user_id>")
    for c in mentioned:
        muted = c["conversation_context"].get("group_muted_by_user")
        print(f"    {c['message_id']}  {c['conversation_type']:<9} user={c['user_id']}  "
              f"tokens={c['mention_detail']['mention_tokens']}  "
              f"group_muted_by_user={muted}")

    sub("mention integrity")
    bad = [
        c["message_id"]
        for c in contexts
        if c["mention_check"]
        and str(c["user_id"]).lower() not in {t.lower() for t in c["mention_detail"]["mention_tokens"]}
    ]
    if bad:
        warn(f"mention_check True but user_id not among the tokens: {sample(bad)}")
    else:
        ok("mention_check True only when the receiving user_id is actually tagged")

    blank_searched = [
        c["message_id"] for c in contexts
        if c["mention_check"] and not c["mention_detail"]["text_searchable"]
    ]
    if blank_searched:
        warn(f"mention detected in blank text: {sample(blank_searched)}")
    else:
        ok("no mention claimed on an empty message_text")

    sub("forwarded_count passthrough")
    src = dict(zip(ds.messages["message_id"], ds.messages["forwarded_count"]))
    drift = [
        c["message_id"]
        for c in contexts
        if c["forwarded_count"] != (None if pd.isna(src[c["message_id"]]) else int(src[c["message_id"]]))
    ]
    if drift:
        warn(f"forwarded_count altered for: {sample(drift)}")
    else:
        ok("forwarded_count matches messages.csv exactly for all rows")
    nonzero = [c for c in contexts if (c["forwarded_count"] or 0) > 0]
    print(f"    forwarded_count > 0 : {len(nonzero)} message(s), "
          f"max={max((c['forwarded_count'] or 0) for c in contexts)}")

    sub("timing")
    in_dnd = [c["message_id"] for c in contexts if c["timing"]["in_dnd_window"]]
    print(f"    arriving inside the user's DND window : {len(in_dnd)}")
    if in_dnd:
        print(f"      {sample(in_dnd)}")
    unknown_dnd = [c["message_id"] for c in contexts if c["timing"]["in_dnd_window"] is None]
    if unknown_dnd:
        warn(f"DND window could not be evaluated for: {sample(unknown_dnd)}")
    else:
        ok("DND window evaluated for every message")

    sub("conversation_context spot checks")
    groups = [c for c in contexts if c["conversation_type"] == "group"]
    non_members = [c["message_id"] for c in groups if not c["conversation_context"]["is_member"]]
    if non_members:
        warn(f"group message where the user is not a listed member: {sample(non_members)}")
    else:
        ok(f"all {len(groups)} group recipients are listed members of their group")
    muted = [c["message_id"] for c in groups if c["conversation_context"]["group_muted_by_user"]]
    print(f"    group messages arriving in a group the user muted : {len(muted)}")

    biz = [c for c in contexts if c["conversation_type"] == "business"]
    no_rel = [c["message_id"] for c in biz if not c["conversation_context"]["has_relationship"]]
    mismatch = [c["message_id"] for c in biz if not c["conversation_context"]["domain_match"]]
    unverified = [c["message_id"] for c in biz if not c["conversation_context"]["verified"]]
    print(f"    business messages with no prior relationship : {len(no_rel)}")
    print(f"    business messages where sender domain != official domain : {len(mismatch)}")
    print(f"    business messages from an unverified account : {len(unverified)}")

    pers = [c for c in contexts if c["conversation_type"] == "personal"]
    no_shared = [c["message_id"] for c in pers if c["conversation_context"]["shared_group_count"] == 0]
    print(f"    personal messages with no shared group with the sender : {len(no_shared)}")
    if no_shared:
        print(f"      {sample(no_shared)}")


# ---------------------------------------------------------------------------
# examples
# ---------------------------------------------------------------------------


def print_context(ctx: dict[str, Any], caption: str) -> None:
    print()
    print("+" + "-" * 76 + "+")
    print(f"| {caption[:74].ljust(74)} |")
    print("+" + "-" * 76 + "+")
    print(json.dumps(to_jsonable(ctx), indent=2, ensure_ascii=False))


def pick_examples(contexts: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str]]:
    """One per tier plus one needs_transcription case. Deterministic."""
    picks: list[tuple[dict[str, Any], str]] = []
    by_id = {c["message_id"]: c for c in contexts}

    def first(pred) -> dict[str, Any] | None:
        for mid in sorted(by_id):
            if pred(by_id[mid]):
                return by_id[mid]
        return None

    tier1 = first(lambda c: c["evidence_tier"] == "exact" and len(c["retrieved_evidence"]) >= 2)
    if tier1 is None:
        tier1 = first(lambda c: c["evidence_tier"] == "exact")
    if tier1:
        picks.append((tier1, f"EXAMPLE A - TIER 1 (exact) evidence: {tier1['message_id']}"))

    tier2 = first(lambda c: c["evidence_tier"] == "fallback")
    if tier2:
        picks.append((tier2, f"EXAMPLE B - TIER 2 (fallback) evidence: {tier2['message_id']}"))

    tier2b = first(lambda c: c["evidence_tier"] == "cross_type")
    if tier2b:
        picks.append(
            (tier2b, f"EXAMPLE C - TIER 2b (cross_type) evidence: {tier2b['message_id']}")
        )

    stranded = first(lambda c: c["evidence_tier"] == "none")
    if stranded:
        picks.append((stranded, f"EXAMPLE D - TIER 3 (none): {stranded['message_id']}"))

    voice = first(lambda c: c["media_status"]["needs_transcription"])
    if voice:
        label = "EXAMPLE E" if stranded else "EXAMPLE D"
        picks.append((voice, f"{label} - needs_transcription: {voice['message_id']}"))

    return picks


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ds = load_dataset(argv[0] if argv else None)

    print("=" * 78)
    print("PHASE 2 VALIDATION - Context Assembly")
    print("=" * 78)
    print(f"dataset dir : {ds.dataset_dir}")

    contexts = build_all_contexts(ds)

    check_coverage(contexts, ds)
    check_evidence_tiers(contexts, ds)
    check_media_flags(contexts, ds)
    check_mentions_and_passthrough(contexts, ds)

    section("5. FULL EXAMPLE CONTEXT OBJECTS")
    for ctx, caption in pick_examples(contexts):
        print_context(ctx, caption)

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
