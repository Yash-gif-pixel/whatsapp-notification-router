"""Phase 1 - validation / sanity checks for the loaded dataset.

Run from the terminal::

    py code/validate_data.py

Reports duplicates, broken references, media resolution, dataset counts, and
history coverage, then prints three fully resolved example messages so the
joins can be eyeballed before anything is built on top of them.

Nothing here writes to disk. Problems are reported, never raised.

TEST FIXTURES, NOT ROUTING INPUTS
---------------------------------
This file contains no hardcoded message, user, group or business id, and no
module on the routing path (``finalize`` -> ``rule_engine`` / ``llm_reasoner``
/ ``media_normalizer`` / ``context_builder`` / ``data_loader``) imports it.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import Dataset, load_dataset  # noqa: E402

PROBLEMS: list[str] = []
MAX_LISTED = 10


# ---------------------------------------------------------------------------
# printing helpers (ASCII only - Windows consoles are not always UTF-8)
# ---------------------------------------------------------------------------


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


def sample(values: Iterable[Any]) -> str:
    vals = [str(v) for v in values]
    shown = ", ".join(vals[:MAX_LISTED])
    if len(vals) > MAX_LISTED:
        shown += f", ... (+{len(vals) - MAX_LISTED} more)"
    return shown


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, time):
        return value.strftime("%H:%M")
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    return str(value)


def block(text: str, indent: str = "    ") -> None:
    for line in text.split("\n"):
        print(indent + line)


def kv(pairs: dict[str, Any], indent: str = "    ", skip_none: bool = False) -> None:
    if not pairs:
        print(indent + "(none)")
        return
    width = max(len(k) for k in pairs)
    for k, v in pairs.items():
        if skip_none and v is None:
            continue
        print(f"{indent}{k.ljust(width)} : {fmt(v)}")


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


_FILE_STEMS = {"output_template": "output"}


def check_row_counts(ds: Dataset) -> None:
    section("1. ROW COUNTS")
    print("  parsed rows vs raw file lines - a gap means quoted multi-line fields,")
    print("  which is expected and confirms the CSV parser handled them correctly.")
    print()
    counts = ds.row_counts()
    width = max(len(k) for k in counts)
    for name, n in counts.items():
        path = ds.dataset_dir / f"{_FILE_STEMS.get(name, name)}.csv"
        raw = _raw_line_count(path)
        note = f"  (raw file lines incl. header: {raw})" if raw is not None and raw != n + 1 else ""
        print(f"  {name.ljust(width)} : {n:>6} rows{note}")
    print()
    print(f"  media files on disk : images={_count_files(ds, 'images')} "
          f"audio={_count_files(ds, 'audio')}")


def _raw_line_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8", newline="") as fh:
        return sum(1 for _ in fh)


def _count_files(ds: Dataset, subdir: str) -> int:
    path = ds.dataset_dir / "media" / subdir
    return len([p for p in path.iterdir() if p.is_file()]) if path.is_dir() else 0


def check_uniqueness(ds: Dataset) -> None:
    section("2. PRIMARY KEY UNIQUENESS")
    single = [
        ("messages", ds.messages, "message_id"),
        ("message_history", ds.message_history, "message_id"),
        ("users", ds.users, "user_id"),
        ("groups", ds.groups, "group_id"),
        ("business_accounts", ds.business_accounts, "business_id"),
        ("images", ds.images, "image_id"),
        ("voice_notes", ds.voice_notes, "voice_note_id"),
        ("sample_messages", ds.sample_messages, "message_id"),
        ("output_template", ds.output_template, "message_id"),
    ]
    for name, df, col in single:
        dupes = df[col][df[col].duplicated(keep=False)].dropna().unique().tolist()
        if dupes:
            warn(f"{name}.{col} has {len(dupes)} duplicated value(s): {sample(dupes)}")
        else:
            ok(f"{name}.{col} unique ({len(df)} rows)")

    composite = [
        ("group_members", ds.group_members, ["group_id", "user_id"]),
        ("user_business_history", ds.user_business_history, ["user_id", "business_id"]),
        ("message_events", ds.message_events, ["user_id", "message_id"]),
        ("daily_notification_summary", ds.daily_notification_summary, ["user_id", "date"]),
    ]
    for name, df, cols in composite:
        dup_mask = df.duplicated(subset=cols, keep=False)
        if dup_mask.any():
            pairs = df.loc[dup_mask, cols].astype(str).agg("/".join, axis=1).unique().tolist()
            warn(f"{name} has {len(pairs)} duplicated key(s) on {'+'.join(cols)}: {sample(pairs)}")
        else:
            ok(f"{name} unique on {'+'.join(cols)} ({len(df)} rows)")


def _check_fk(
    label: str,
    values: pd.Series,
    known: set[str],
    *,
    required: bool = False,
) -> None:
    """Report ids in ``values`` that are absent from ``known``."""
    present = values.dropna()
    missing = sorted({v for v in present if v not in known})
    n_null = int(len(values) - len(present))
    detail = f"{len(present)} non-empty" + (f", {n_null} empty" if n_null else "")
    if missing:
        warn(f"{label}: {len(missing)} unknown id(s) [{detail}] -> {sample(missing)}")
    elif required and n_null:
        warn(f"{label}: {n_null} row(s) have an empty value but it is required")
    else:
        ok(f"{label}: all resolve ({detail})")


def check_referential_integrity(ds: Dataset) -> None:
    section("3. REFERENTIAL INTEGRITY")

    user_ids = set(ds.users["user_id"].dropna())
    group_ids = set(ds.groups["group_id"].dropna())
    business_ids = set(ds.business_accounts["business_id"].dropna())
    history_ids = set(ds.message_history["message_id"].dropna())
    media_ids = set(ds._media_by_id)

    sub("messages.csv -> lookup files")
    _check_fk("messages.user_id -> users", ds.messages["user_id"], user_ids, required=True)
    _check_fk("messages.group_id -> groups", ds.messages["group_id"], group_ids)
    _check_fk(
        "messages.business_id -> business_accounts", ds.messages["business_id"], business_ids
    )
    _check_fk("messages.sender_user_id -> users", ds.messages["sender_user_id"], user_ids)
    _check_fk("messages.media_id -> images+voice_notes", ds.messages["media_id"], media_ids)

    sub("supporting files -> lookup files")
    _check_fk("message_history.user_id -> users", ds.message_history["user_id"], user_ids)
    _check_fk("message_history.group_id -> groups", ds.message_history["group_id"], group_ids)
    _check_fk(
        "message_history.business_id -> business_accounts",
        ds.message_history["business_id"],
        business_ids,
    )
    _check_fk(
        "message_history.sender_user_id -> users", ds.message_history["sender_user_id"], user_ids
    )
    _check_fk("message_history.media_id -> images+voice_notes",
             ds.message_history["media_id"], media_ids)
    _check_fk("message_events.user_id -> users", ds.message_events["user_id"], user_ids)
    _check_fk(
        "message_events.message_id -> message_history",
        ds.message_events["message_id"],
        history_ids,
    )
    _check_fk("group_members.group_id -> groups", ds.group_members["group_id"], group_ids)
    _check_fk("group_members.user_id -> users", ds.group_members["user_id"], user_ids)
    _check_fk("user_business_history.user_id -> users", ds.user_business_history["user_id"],
             user_ids)
    _check_fk(
        "user_business_history.business_id -> business_accounts",
        ds.user_business_history["business_id"],
        business_ids,
    )
    _check_fk(
        "daily_notification_summary.user_id -> users",
        ds.daily_notification_summary["user_id"],
        user_ids,
    )

    sub("output.csv template vs messages.csv")
    msg_ids = set(ds.messages["message_id"].dropna())
    out_ids = set(ds.output_template["message_id"].dropna())
    if msg_ids == out_ids:
        ok(f"output.csv template covers exactly the {len(msg_ids)} message ids")
    else:
        if msg_ids - out_ids:
            warn(f"output.csv missing {len(msg_ids - out_ids)} id(s): {sample(sorted(msg_ids - out_ids))}")
        if out_ids - msg_ids:
            warn(f"output.csv has {len(out_ids - msg_ids)} extra id(s): {sample(sorted(out_ids - msg_ids))}")


def check_shape_consistency(ds: Dataset) -> None:
    """conversation_type should agree with which id column is populated."""
    section("4. CONVERSATION / MEDIA SHAPE CONSISTENCY")
    m = ds.messages

    expectations = {
        "group": ("group_id", ["business_id"]),
        "business": ("business_id", ["group_id"]),
        "personal": ("sender_user_id", ["group_id", "business_id"]),
    }
    for conv, (needed, forbidden) in expectations.items():
        rows = m[m["conversation_type"] == conv]
        if rows.empty:
            continue
        missing = rows[rows[needed].isna()]["message_id"].tolist()
        if missing:
            warn(f"{conv} messages missing {needed}: {sample(missing)}")
        else:
            ok(f"all {len(rows)} '{conv}' messages have {needed}")
        for col in forbidden:
            extra = rows[rows[col].notna()]["message_id"].tolist()
            if extra:
                warn(f"{conv} messages unexpectedly carry {col}: {sample(extra)}")

    unknown_conv = sorted(set(m["conversation_type"]) - set(expectations))
    if unknown_conv:
        warn(f"unexpected conversation_type value(s): {sample(unknown_conv)}")
    else:
        ok(f"conversation_type values limited to {sorted(expectations)}")

    sub("media")
    allowed_media = {"", "image", "voice"}
    bad_media = sorted(set(m["media_type"]) - allowed_media)
    if bad_media:
        warn(f"unexpected media_type value(s): {sample(bad_media)}")
    else:
        ok("media_type values limited to '', 'image', 'voice'")

    with_media_type = m[m["media_type"] != ""]
    no_id = with_media_type[with_media_type["media_id"].isna()]["message_id"].tolist()
    if no_id:
        warn(f"messages with media_type but no media_id: {sample(no_id)}")
    else:
        ok(f"all {len(with_media_type)} media messages carry a media_id")

    orphan_id = m[(m["media_type"] == "") & m["media_id"].notna()]["message_id"].tolist()
    if orphan_id:
        warn(f"messages with media_id but empty media_type: {sample(orphan_id)}")

    # Does the media_id resolve to a file that is actually on disk?
    unresolved, missing_file, kind_mismatch = [], [], []
    for _, row in with_media_type.iterrows():
        media = ds.get_media(row["media_id"])
        if media is None:
            unresolved.append(f"{row['message_id']}({row['media_id']})")
            continue
        if not media["file_exists"]:
            missing_file.append(f"{row['message_id']}->{media['file_path']}")
        if media["media_kind"] != row["media_type"]:
            kind_mismatch.append(
                f"{row['message_id']}: media_type={row['media_type']} but id is {media['media_kind']}"
            )
    if unresolved:
        warn(f"{len(unresolved)} media_id(s) not present in images.csv/voice_notes.csv: {sample(unresolved)}")
    else:
        ok("every media_id resolves to a row in images.csv / voice_notes.csv")
    if missing_file:
        warn(f"{len(missing_file)} media file(s) referenced but not on disk: {sample(missing_file)}")
    else:
        ok("every resolved media file exists under dataset/media/")
    if kind_mismatch:
        warn(f"media kind mismatch: {sample(kind_mismatch)}")

    sub("text")
    empty_text = m[(m["message_text"].str.strip() == "")]
    by_media = empty_text["media_type"].value_counts().to_dict()
    print(f"    messages with empty message_text: {len(empty_text)} (by media_type: {by_media})")
    text_no_media = empty_text[empty_text["media_type"] == ""]["message_id"].tolist()
    if text_no_media:
        warn(f"messages with neither text nor media: {sample(text_no_media)}")
    else:
        ok("every message has text and/or media")

    multiline = m[m["message_text"].str.contains("\n", regex=False)]
    ok(f"{len(multiline)} message(s) contain embedded newlines (multi-line CSV parsed correctly)")
    max_row = m.loc[m["message_text"].str.len().idxmax()]
    print(f"    longest message_text: {max_row['message_id']} "
          f"({len(max_row['message_text'])} chars, "
          f"{max_row['message_text'].count(chr(10)) + 1} lines)")


def check_breakdowns(ds: Dataset) -> None:
    section("5. BREAKDOWNS")
    m = ds.messages.copy()
    m["media_type"] = m["media_type"].replace({"": "(none)"})

    sub("messages.csv by conversation_type x media_type")
    table = pd.crosstab(m["conversation_type"], m["media_type"], margins=True, margins_name="TOTAL")
    block(table.to_string())

    sub("messages.csv timestamp range")
    print(f"    created_at: {fmt(m['created_at'].min())}  ->  {fmt(m['created_at'].max())}")
    print(f"    distinct receiving users : {m['user_id'].nunique()}")
    print(f"    distinct groups referenced   : {m['group_id'].nunique()}")
    print(f"    distinct businesses referenced: {m['business_id'].nunique()}")
    print(f"    forwarded_count > 0          : {int((m['forwarded_count'] > 0).sum())}")

    sub("message_history.csv by conversation_type x media_type")
    h = ds.message_history.copy()
    h["media_type"] = h["media_type"].replace({"": "(none)"})
    ht = pd.crosstab(h["conversation_type"], h["media_type"], margins=True, margins_name="TOTAL")
    block(ht.to_string())
    print(f"    created_at: {fmt(h['created_at'].min())}  ->  {fmt(h['created_at'].max())}")

    sub("message_events.csv reaction mix")
    ev = ds.message_events
    for col in [
        "message_opened",
        "message_replied",
        "notification_dismissed",
        "muted_after_message",
        "message_reported",
    ]:
        true_n = int(ev[col].fillna(False).sum())
        print(f"    {col.ljust(24)} true={true_n:>4}  ({true_n / len(ev):.0%} of {len(ev)})")
    covered = ds.message_history["message_id"].isin(set(ev["message_id"])).sum()
    print(f"    history rows with a recorded reaction: {covered} / {len(ds.message_history)}")


def check_history_coverage(ds: Dataset) -> None:
    section("6. HISTORY / EVIDENCE COVERAGE")
    print("  'exact-key history' = message_history rows for the same receiving user AND")
    print("  the same counterpart (sender_user_id / group_id / business_id), dated strictly")
    print("  before the incoming message. This is what retrieval will use in Phase 2.")
    print()

    rows = []
    for _, msg in ds.messages.iterrows():
        exact = ds.get_message_history_for_sender(
            msg["user_id"],
            sender_user_id=msg["sender_user_id"] if pd.notna(msg["sender_user_id"]) else None,
            group_id=msg["group_id"] if pd.notna(msg["group_id"]) else None,
            business_id=msg["business_id"] if pd.notna(msg["business_id"]) else None,
            before=msg["created_at"],
        )
        counterpart_only = ds.get_message_history_for_sender(
            msg["user_id"],
            group_id=msg["group_id"] if pd.notna(msg["group_id"]) else None,
            business_id=msg["business_id"] if pd.notna(msg["business_id"]) else None,
            before=msg["created_at"],
        )
        user_any = ds.get_message_history_for_sender(msg["user_id"], before=msg["created_at"])
        rows.append(
            {
                "message_id": msg["message_id"],
                "conversation_type": msg["conversation_type"],
                "n_exact": len(exact),
                "n_counterpart": len(counterpart_only),
                "n_user_any": len(user_any),
            }
        )
    cov = pd.DataFrame(rows)

    total = len(cov)
    zero_exact = cov[cov["n_exact"] == 0]
    print(f"  messages with SOME exact-key history : {total - len(zero_exact):>4} / {total}")
    print(f"  messages with ZERO exact-key history : {len(zero_exact):>4} / {total}")
    print()
    sub("zero exact-key history, by conversation_type")
    if zero_exact.empty:
        print("    (none)")
    else:
        for conv, n in zero_exact["conversation_type"].value_counts().items():
            in_conv = int((cov["conversation_type"] == conv).sum())
            print(f"    {conv.ljust(10)} {n:>3} of {in_conv}")

    sub("fallback breadth for the zero-exact-key messages")
    if zero_exact.empty:
        print("    (not needed)")
    else:
        rescued_counterpart = int((zero_exact["n_counterpart"] > 0).sum())
        rescued_user = int((zero_exact["n_user_any"] > 0).sum())
        print(f"    have history for the same group/business (ignoring sender): {rescued_counterpart}")
        print(f"    have any history at all for that user                     : {rescued_user}")
        stranded = zero_exact[zero_exact["n_user_any"] == 0]["message_id"].tolist()
        print(f"    no history whatsoever for that user                       : {len(stranded)}")
        if stranded:
            print(f"      -> {sample(stranded)}")

    sub("exact-key history size distribution")
    desc = cov["n_exact"].describe()
    print(f"    min={int(desc['min'])}  median={int(desc['50%'])}  "
          f"mean={desc['mean']:.1f}  max={int(desc['max'])}")


def check_media_inventory(ds: Dataset) -> None:
    section("7. MEDIA INVENTORY")
    for kind, df, id_col, subdir in (
        ("image", ds.images, "image_id", "images"),
        ("voice", ds.voice_notes, "voice_note_id", "audio"),
    ):
        listed = set(df[id_col].dropna())
        on_disk_dir = ds.dataset_dir / "media" / subdir
        on_disk = {p.stem for p in on_disk_dir.iterdir() if p.is_file()} if on_disk_dir.is_dir() else set()
        print(f"  {kind}: {len(listed)} listed in csv, {len(on_disk)} file(s) in media/{subdir}")
        missing = sorted(listed - on_disk)
        unlisted = sorted(on_disk - listed)
        if missing:
            warn(f"{kind} ids listed in csv but with no file on disk: {sample(missing)}")
        if unlisted:
            print(f"    files on disk not listed in the csv: {sample(unlisted)}")

        used_msgs = set(ds.messages.loc[ds.messages["media_type"] == kind, "media_id"].dropna())
        used_hist = set(
            ds.message_history.loc[ds.message_history["media_type"] == kind, "media_id"].dropna()
        )
        print(f"    referenced by messages.csv: {len(used_msgs)}, "
              f"by message_history.csv: {len(used_hist)}")
        unresolved = sorted((used_msgs | used_hist) - listed)
        if unresolved:
            warn(f"{kind} media_id(s) referenced but absent from the csv: {sample(unresolved)}")


# ---------------------------------------------------------------------------
# resolved examples
# ---------------------------------------------------------------------------


def print_resolved_message(ds: Dataset, message_id: str) -> None:
    resolved = ds.resolve_message(message_id, history_limit=5)
    msg = resolved["message"]

    print()
    print("+" + "-" * 76 + "+")
    print(f"| {('RESOLVED MESSAGE: ' + message_id).ljust(74)} |")
    print("+" + "-" * 76 + "+")

    sub("message (messages.csv)")
    kv(
        {
            "message_id": msg["message_id"],
            "user_id": msg["user_id"],
            "conversation_type": msg["conversation_type"],
            "group_id": msg["group_id"],
            "business_id": msg["business_id"],
            "sender_user_id": msg["sender_user_id"],
            "created_at": msg["created_at"],
            "media_type": msg["media_type"] or None,
            "media_id": msg["media_id"],
            "forwarded_count": msg["forwarded_count"],
        }
    )
    text = msg["message_text"]
    print("    message_text     :")
    if not text.strip():
        print("      (empty - media-only message)")
    else:
        shown = text if len(text) <= 700 else text[:700] + " ...[truncated]"
        for line in shown.split("\n"):
            print("      | " + line)

    sub("receiving user (get_user)")
    user = resolved["user"]
    if user is None:
        print("    (user not found)")
    else:
        kv(user)
        print(f"    in_dnd_window    : {fmt(resolved['in_dnd_window'])} (at message time)")

    if msg["conversation_type"] == "group":
        sub("group context (get_group_context)")
        gc = resolved["group_context"]
        if gc is None:
            print("    (group not found)")
        else:
            kv(gc)
        sub("sender profile (get_user on sender_user_id)")
        sender = resolved["sender"]
        if sender is None:
            print("    (sender not in users.csv)")
        else:
            kv(sender)
        if sender and msg["group_id"]:
            sender_member = ds.get_group_context(msg["group_id"], msg["sender_user_id"])
            print(f"    sender role in this group : "
                  f"{fmt(sender_member['member_role']) if sender_member else '-'}")

    elif msg["conversation_type"] == "business":
        sub("business context (get_business_context)")
        bc = resolved["business_context"]
        if bc is None:
            print("    (business not found)")
        else:
            kv(bc)

    else:
        sub("sender profile (get_user on sender_user_id)")
        sender = resolved["sender"]
        if sender is None:
            print("    (sender not in users.csv)")
        else:
            kv(sender)

    if msg["media_id"]:
        sub("media (get_media)")
        media = resolved["media"]
        if media is None:
            print(f"    media_id {msg['media_id']} does not resolve")
        else:
            kv({k: v for k, v in media.items() if k != "abs_path"})
            print(f"    abs_path         : {media['abs_path']}")

    sub("notification load (get_daily_notification_load, last 3 days on file)")
    load = ds.get_daily_notification_load(msg["user_id"]).tail(3)
    if load.empty:
        print("    (no rows)")
    else:
        for _, r in load.iterrows():
            print(f"    {fmt(r['date'])[:10]}  sent={r['notifications_sent']}  "
                  f"dismissed={r['notifications_dismissed']}")

    sub("history + reactions (get_message_history_for_sender, newest 5 before this message)")
    hist = resolved["history"]
    if hist.empty:
        print("    (no exact-key history for this user + counterpart)")
    else:
        for _, h in hist.iterrows():
            flags = []
            for col, label in (
                ("message_opened", "opened"),
                ("message_replied", "replied"),
                ("notification_dismissed", "dismissed"),
                ("muted_after_message", "muted"),
                ("message_reported", "reported"),
            ):
                if bool(h[col]) if pd.notna(h[col]) else False:
                    flags.append(label)
            rt = h["reaction_time_minutes"]
            if pd.notna(rt):
                flags.append(f"{int(rt)}min")
            snippet = " ".join(str(h["message_text"]).split())
            if len(snippet) > 96:
                snippet = snippet[:96] + "..."
            if not snippet:
                snippet = f"({h['media_type'] or 'no text'})"
            print(f"    {h['message_id']}  {fmt(h['created_at'])}  "
                  f"[{', '.join(flags) if flags else 'no reaction recorded'}]")
            print(f"      {snippet}")


def pick_examples(ds: Dataset) -> list[str]:
    """One message per conversation_type, deterministic (lowest message_id)."""
    picks: list[str] = []
    for conv in ["personal", "group", "business"]:
        rows = ds.messages[ds.messages["conversation_type"] == conv]
        if not rows.empty:
            picks.append(sorted(rows["message_id"].tolist())[0])
    if not picks:
        picks = sorted(ds.messages["message_id"].tolist())[:3]
    return picks[:3]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    dataset_dir = argv[0] if argv else None

    ds = load_dataset(dataset_dir)
    print("=" * 78)
    print("PHASE 1 VALIDATION - Message Notification Router")
    print("=" * 78)
    print(f"dataset dir : {ds.dataset_dir}")
    print(f"pandas      : {pd.__version__}")

    check_row_counts(ds)
    check_uniqueness(ds)
    check_referential_integrity(ds)
    check_shape_consistency(ds)
    check_breakdowns(ds)
    check_history_coverage(ds)
    check_media_inventory(ds)

    section("8. FULLY RESOLVED EXAMPLE MESSAGES")
    for message_id in pick_examples(ds):
        print_resolved_message(ds, message_id)

    section("SUMMARY")
    if PROBLEMS:
        print(f"  {len(PROBLEMS)} issue(s) flagged (none fatal - loading succeeded):")
        for i, p in enumerate(PROBLEMS, 1):
            print(f"    {i}. {p}")
    else:
        print("  No issues found.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
