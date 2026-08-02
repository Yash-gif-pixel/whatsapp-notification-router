"""Phase 6 - final validation of dataset/output.csv.

Run from the terminal::

    py code/validate_output.py
    py code/validate_output.py --sample 10   # extra spot-check rows

Reads the written output.csv back off disk and checks it against
messages.csv, message_history.csv and the assembled contexts. Nothing is
decided or rewritten here - this pass only reports.

TEST FIXTURES, NOT ROUTING INPUTS
---------------------------------
Message ids below (``FINAL_CASES``) select rows to print for a last
read-through before submission. They are expected-value assertions about
decisions already written to ``output.csv``, never inputs to any decision. No
module on the routing path (``finalize`` -> ``rule_engine`` / ``llm_reasoner``
/ ``media_normalizer`` / ``context_builder`` / ``data_loader``) imports this
file. This module imports ``finalize`` in order to check its output, not the
other way round.
"""

from __future__ import annotations

import argparse
import random
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import load_dataset  # noqa: E402
from finalize import DEFAULT_OUTPUT, OUTPUT_COLUMNS, build_final_decisions  # noqa: E402
from llm_reasoner import ACTIONS, MESSAGE_TYPES, VAGUE_QUALIFIER  # noqa: E402
from media_normalizer import build_normalized_contexts, effective_text  # noqa: E402
from rule_engine import apply_rules_to_all  # noqa: E402

PROBLEMS: list[str] = []
MAX_LISTED = 12

EXPECTED_ROWS = 110
REASON_MIN, REASON_MAX = 40, 130
SPOTCHECK_SEED = 20260802

#: Final read-through set: the named test cases plus the rule 2b trio.
FINAL_CASES = [
    ("msg_085", "voice OTP scam from a fake bank helpdesk (rule 1a)"),
    ("msg_064", "refund text over an unrelated movie poster (rule 1b)"),
    ("msg_056", "plain direct mention in a muted group (rule 2)"),
    ("msg_040", "mention inside a chain forward - resolved by the LLM"),
    ("msg_005", "personal hand-off, marketplace group (rule 2b)"),
    ("msg_103", "personal hand-off, personal chat (rule 2b)"),
    ("msg_104", "personal hand-off, personal chat (rule 2b)"),
]


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


def check_schema(out: pd.DataFrame, ds) -> None:
    section("1. SCHEMA AND COMPLETENESS")

    if list(out.columns) == OUTPUT_COLUMNS:
        ok(f"columns exact and in order: {', '.join(OUTPUT_COLUMNS)}")
    else:
        warn(f"columns are {list(out.columns)}, expected {OUTPUT_COLUMNS}")

    print(f"  rows: {len(out)}")
    if len(out) != EXPECTED_ROWS:
        warn(f"expected {EXPECTED_ROWS} rows, got {len(out)}")
    else:
        ok(f"{EXPECTED_ROWS} rows present")

    expected_ids = list(ds.messages["message_id"])
    got_ids = list(out["message_id"])
    if got_ids == expected_ids:
        ok("message_id order matches messages.csv exactly")
    elif set(got_ids) == set(expected_ids):
        note("same ids as messages.csv but in a different order "
             "(the blank template preserves messages.csv order)")
    else:
        missing = set(expected_ids) - set(got_ids)
        extra = set(got_ids) - set(expected_ids)
        if missing:
            warn(f"{len(missing)} message_id(s) missing: {sample(sorted(missing))}")
        if extra:
            warn(f"{len(extra)} unknown message_id(s): {sample(sorted(extra))}")

    dupes = out["message_id"][out["message_id"].duplicated()].tolist()
    if dupes:
        warn(f"duplicate message_id(s): {sample(dupes)}")
    else:
        ok("no duplicate message_id")

    sub("column values")
    for col in OUTPUT_COLUMNS:
        blank = out[out[col].isna() | (out[col].astype(str).str.strip() == "")]
        if len(blank):
            warn(f"{col}: {len(blank)} blank value(s): "
                 f"{sample(blank['message_id'].tolist())}")
    if not PROBLEMS or all("blank" not in p for p in PROBLEMS):
        ok("no blank or missing values in any column")

    bad_action = out[~out["action"].isin(ACTIONS)]
    if len(bad_action):
        warn(f"invalid action: {sample(bad_action['message_id'].tolist())}")
    else:
        ok(f"every action is one of {'/'.join(ACTIONS)}")

    bad_type = out[~out["message_type"].isin(MESSAGE_TYPES)]
    if len(bad_type):
        warn(f"invalid message_type: {sample(bad_type['message_id'].tolist())}")
    else:
        ok(f"every message_type is one of the {len(MESSAGE_TYPES)} allowed values")

    conf = pd.to_numeric(out["confidence"], errors="coerce")
    if conf.isna().any():
        warn(f"non-numeric confidence: "
             f"{sample(out.loc[conf.isna(), 'message_id'].tolist())}")
    elif not ((conf >= 0) & (conf <= 1)).all():
        warn("confidence outside 0-1")
    else:
        ok("confidence is numeric and within 0-1 everywhere")

    empty_ev = out[out["evidence_message_ids"].astype(str).str.strip() == ""]
    if len(empty_ev):
        warn(f"empty evidence_message_ids (should be the literal 'none'): "
             f"{sample(empty_ev['message_id'].tolist())}")
    else:
        ok("evidence_message_ids is never empty - 'none' is used explicitly")


def check_evidence(out: pd.DataFrame, ds, contexts) -> None:
    section("2. EVIDENCE INTEGRITY")
    by_ctx = {c["message_id"]: c for c in contexts}
    history_ids = set(ds.message_history["message_id"].dropna())

    cited_rows = out[out["evidence_message_ids"] != "none"]
    print(f"  rows citing evidence: {len(cited_rows)} / {len(out)}")

    bogus, not_supplied = [], []
    for _, row in cited_rows.iterrows():
        ctx = by_ctx[row["message_id"]]
        supplied = {e["message_id"] for e in ctx["retrieved_evidence"]}
        for eid in str(row["evidence_message_ids"]).split(";"):
            if eid not in history_ids:
                bogus.append(f"{row['message_id']}->{eid}")
            elif eid not in supplied:
                not_supplied.append(f"{row['message_id']}->{eid}")

    if bogus:
        warn(f"cited ids absent from message_history.csv: {sample(bogus)}")
    else:
        ok("every cited evidence id exists in message_history.csv")
    if not_supplied:
        warn(f"cited ids not in that message's own retrieval: {sample(not_supplied)}")
    else:
        ok("every cited id came from that message's retrieved_evidence")

    sub("cross_type evidence")
    cross = [c["message_id"] for c in contexts if c["evidence_tier"] == "cross_type"]
    print(f"    messages whose retrieval is cross_type: {len(cross)} "
          f"({', '.join(cross)})")
    cited_cross = [
        m for m in cross
        if out.loc[out["message_id"] == m, "evidence_message_ids"].iloc[0] != "none"
    ]
    if cited_cross:
        warn(f"cross_type evidence cited in output: {sample(cited_cross)}")
    else:
        ok("no cross_type message cites evidence as behavioural justification")

    leaks = []
    for _, row in cited_rows.iterrows():
        ctx = by_ctx[row["message_id"]]
        by_id = {e["message_id"]: e for e in ctx["retrieved_evidence"]}
        for eid in str(row["evidence_message_ids"]).split(";"):
            item = by_id.get(eid)
            if item and item["created_at"] and ctx["created_at"]:
                if item["created_at"] >= ctx["created_at"]:
                    leaks.append(f"{row['message_id']}->{eid}")
    if leaks:
        warn(f"evidence dated at/after its message: {sample(leaks)}")
    else:
        ok("all cited evidence predates its message - no future leak")


def check_distribution(out: pd.DataFrame, ds, rows) -> None:
    section("3. FINAL DISTRIBUTION")
    sub("action x message_type, all 110")
    table = pd.crosstab(out["action"], out["message_type"],
                        margins=True, margins_name="TOTAL")
    for line in table.to_string().split("\n"):
        print("    " + line)

    sub("action share vs sample_messages.csv")
    s = ds.sample_messages
    print(f"    {'action':<9} {'output (n=110)':>16} {'sample (n=30)':>16}")
    for action in ACTIONS:
        n = int((out["action"] == action).sum())
        sn = int((s["action"] == action).sum())
        print(f"    {action:<9} {n:>6} ({n/len(out):>5.0%}) {sn:>8} ({sn/len(s):>5.0%})")

    sub("message_type share")
    counts = Counter(out["message_type"])
    sample_counts = Counter(s["message_type"])
    for mt in sorted(set(counts) | set(sample_counts)):
        print(f"    {mt:<17} output {counts.get(mt, 0):>3}   sample {sample_counts.get(mt, 0):>2}")

    sub("decision source")
    src = Counter(r["_source"].split(":")[0] for r in rows)
    detail = Counter(r["_source"] for r in rows)
    print(f"    {dict(src)}")
    for k, v in sorted(detail.items()):
        print(f"      {k:<16} {v}")

    sub("confidence")
    conf = pd.to_numeric(out["confidence"])
    print(f"    min={conf.min():.2f}  max={conf.max():.2f}  mean={conf.mean():.3f}")
    print(f"    sample_messages.csv: min={s['confidence'].min():.2f} "
          f"max={s['confidence'].max():.2f} mean={s['confidence'].mean():.3f}")
    if conf.max() > 0.95:
        warn(f"confidence above 0.95: max={conf.max()}")
    else:
        ok("no confidence exceeds 0.95")


def check_reasons(out: pd.DataFrame) -> None:
    section("4. REASON TEXT")
    lengths = out["reason"].str.len()
    print(f"    min={lengths.min()}  max={lengths.max()}  mean={int(lengths.mean())}")
    print(f"    sample_messages.csv: 58-114, mean 82")

    long_ones = out[lengths > REASON_MAX]
    short_ones = out[lengths < REASON_MIN]
    if len(long_ones):
        warn(f"{len(long_ones)} reason(s) over {REASON_MAX} chars: "
             f"{sample(long_ones['message_id'].tolist())}")
        for _, r in long_ones.iterrows():
            print(f"      {r['message_id']} ({len(r['reason'])}): {r['reason']}")
    else:
        ok(f"no reason exceeds {REASON_MAX} characters")
    if len(short_ones):
        warn(f"{len(short_ones)} reason(s) under {REASON_MIN} chars: "
             f"{sample(short_ones['message_id'].tolist())}")
        for _, r in short_ones.iterrows():
            print(f"      {r['message_id']} ({len(r['reason'])}): {r['reason']}")
    else:
        ok(f"no reason is shorter than {REASON_MIN} characters")

    vague = out[out["reason"].str.contains(VAGUE_QUALIFIER, regex=True)]
    if len(vague):
        note(f"{len(vague)} reason(s) contain a hedging phrase - review by eye:")
        for _, r in vague.iterrows():
            print(f"      {r['message_id']}: {r['reason']}")
    else:
        ok("no reason uses a vague qualifier without cited text")

    dupes = out["reason"].value_counts()
    repeated = dupes[dupes > 2]
    if len(repeated):
        note(f"{len(repeated)} reason string(s) reused more than twice "
             f"(expected - rule templates are shared):")
        for text, n in repeated.items():
            print(f"      {n}x  {text[:78]}")


def print_final_cases(out: pd.DataFrame, contexts, rows) -> None:
    section("5. FINAL FORM OF THE KNOWN TEST CASES")
    by_ctx = {c["message_id"]: c for c in contexts}
    by_row = {r["message_id"]: r for r in rows}
    for message_id, label in FINAL_CASES:
        row = out[out["message_id"] == message_id]
        if row.empty:
            warn(f"{message_id} not in output.csv")
            continue
        r = row.iloc[0]
        ctx = by_ctx[message_id]
        print()
        print(f"  {message_id} - {label}")
        print(f"    source   : {by_row[message_id]['_source']}")
        print(f"    context  : {ctx['conversation_type']}, tier={ctx['evidence_tier']}, "
              f"mention={ctx['mention_check']}, forwarded={ctx['forwarded_count']}")
        wrapped(effective_text(ctx)[:230], indent="      | ")
        print(f"    -> {r['action']}/{r['message_type']}  conf={r['confidence']}  "
              f"evidence={r['evidence_message_ids']}")
        print(f"    reason: {r['reason']}")


def print_spotchecks(out: pd.DataFrame, contexts, rows, n: int) -> None:
    section(f"6. SPOT-CHECK - {n} ROWS ACROSS TYPES AND ACTIONS")
    by_ctx = {c["message_id"]: c for c in contexts}
    by_row = {r["message_id"]: r for r in rows}
    named = {m for m, _ in FINAL_CASES}

    rng = random.Random(SPOTCHECK_SEED)
    pool = out[~out["message_id"].isin(named)]
    # One per (action, message_type) pair first, so the sample spans the space.
    picks: list[str] = []
    for _, group in pool.groupby(["action", "message_type"]):
        picks.append(rng.choice(group["message_id"].tolist()))
    rng.shuffle(picks)
    picks = picks[:n]
    if len(picks) < n:
        rest = [m for m in pool["message_id"] if m not in picks]
        picks += rng.sample(rest, min(n - len(picks), len(rest)))

    for message_id in picks:
        r = out[out["message_id"] == message_id].iloc[0]
        ctx = by_ctx[message_id]
        cc = ctx["conversation_context"]
        who = (cc.get("group_name") or cc.get("display_name")
               or cc.get("sender_user_id") or "-")
        print()
        print(f"  {message_id}  {ctx['conversation_type']}/{who}  "
              f"[{by_row[message_id]['_source']}]")
        wrapped(effective_text(ctx)[:200], indent="      | ")
        print(f"    -> {r['action']}/{r['message_type']}  conf={r['confidence']}  "
              f"evidence={r['evidence_message_ids']}")
        print(f"    reason: {r['reason']}")


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 6 output validation.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--sample", type=int, default=10)
    args = parser.parse_args(list(argv) if argv is not None else None)

    path = Path(args.out) if args.out else DEFAULT_OUTPUT
    ds = load_dataset()
    contexts = build_normalized_contexts(ds)
    apply_rules_to_all(contexts)
    rows, report = build_final_decisions(ds, contexts)

    print("=" * 78)
    print("PHASE 6 VALIDATION - FINAL OUTPUT")
    print("=" * 78)
    print(f"output file : {path}")
    if not path.is_file():
        warn(f"{path} does not exist - run py code/finalize.py first")
        return 1

    out = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    print(f"read back   : {len(out)} rows, {len(out.columns)} columns")

    check_schema(out, ds)
    check_evidence(out, ds, contexts)
    check_distribution(out, ds, rows)
    check_reasons(out)
    print_final_cases(out, contexts, rows)
    print_spotchecks(out, contexts, rows, args.sample)

    section("SUMMARY")
    if report["overrides"]:
        print(f"  safety override changed {len(report['overrides'])} decision(s)")
    if PROBLEMS:
        print(f"  {len(PROBLEMS)} item(s) flagged:")
        for i, p in enumerate(PROBLEMS, 1):
            print(f"    {i}. {p}")
    else:
        print("  No issues found. output.csv is ready.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
