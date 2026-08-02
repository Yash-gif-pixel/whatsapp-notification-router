"""Phase 1 - Data loading and lookup layer for the Message Notification Router.

This module does two things and nothing else:

1. Loads every CSV in ``dataset/`` into typed pandas DataFrames.
2. Exposes generic join/lookup functions keyed by id.

There is deliberately no routing logic, no scoring, no LLM calls and no media
decoding here. Those belong to later phases.

Typical use::

    from data_loader import load_dataset

    ds = load_dataset()
    user = ds.get_user("u_011")
    grp  = ds.get_group_context("group_002", "u_011")
    hist = ds.get_message_history_for_sender("u_011", group_id="group_002")

Every lookup is id-driven; there are no hardcoded ids anywhere in this file.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "Dataset",
    "load_dataset",
    "default_dataset_dir",
]


# ---------------------------------------------------------------------------
# Column type specs
# ---------------------------------------------------------------------------
# Anything not listed for a file is kept as a plain string column. Empty cells
# in id/datetime/int/bool columns become NA; empty cells in text columns stay
# as "" so that downstream `.str` operations never trip over NaN.

_MESSAGE_SPEC: dict[str, Sequence[str]] = {
    "id": ["message_id", "user_id", "group_id", "business_id", "sender_user_id", "media_id"],
    "datetime": ["created_at"],
    "int": ["forwarded_count"],
    "bool": [],
    "float": [],
}

_SPECS: dict[str, dict[str, Sequence[str]]] = {
    "messages": _MESSAGE_SPEC,
    "message_history": _MESSAGE_SPEC,
    "sample_messages": {
        **_MESSAGE_SPEC,
        "float": ["confidence"],
    },
    "users": {
        "id": ["user_id"],
        "datetime": [],
        "int": [
            "messages_opened_30d",
            "messages_replied_30d",
            "notifications_dismissed_30d",
            "messages_reported_30d",
        ],
        "bool": [],
        "float": [],
    },
    "groups": {
        "id": ["group_id"],
        "datetime": ["created_at"],
        "int": ["member_count", "admin_count", "messages_30d"],
        "bool": [],
        "float": [],
    },
    "group_members": {
        "id": ["group_id", "user_id"],
        "datetime": ["joined_at"],
        "int": [
            "messages_sent_30d",
            "messages_read_30d",
            "replies_sent_30d",
            "notifications_dismissed_30d",
        ],
        "bool": ["group_muted_by_user"],
        "float": [],
    },
    "business_accounts": {
        "id": ["business_id"],
        "datetime": [],
        "int": [
            "account_age_days",
            "messages_sent_30d",
            "user_reports_30d",
            "domain_used_by_sender_age_days",
        ],
        "bool": ["verified"],
        "float": [],
    },
    "user_business_history": {
        "id": ["user_id", "business_id"],
        "datetime": ["last_activity_at", "promotions_opted_out_at", "last_reply_at"],
        "int": [
            "activity_count_180d",
            "messages_opened_30d",
            "messages_dismissed_30d",
            "messages_replied_30d",
        ],
        "bool": ["allows_promotions"],
        "float": [],
    },
    "message_events": {
        "id": ["user_id", "message_id"],
        "datetime": [],
        "int": ["reaction_time_minutes"],
        "bool": [
            "message_opened",
            "message_replied",
            "notification_dismissed",
            "muted_after_message",
            "message_reported",
        ],
        "float": [],
    },
    "images": {"id": ["image_id"], "datetime": [], "int": [], "bool": [], "float": []},
    "voice_notes": {"id": ["voice_note_id"], "datetime": [], "int": [], "bool": [], "float": []},
    "daily_notification_summary": {
        "id": ["user_id"],
        "datetime": ["date"],
        "int": ["notifications_sent", "notifications_dismissed"],
        "bool": [],
        "float": [],
    },
    "output": {
        "id": ["message_id"],
        "datetime": [],
        "int": [],
        "bool": [],
        "float": ["confidence"],
    },
}

# Datetime formats seen in the dataset, tried in order.
_DATETIME_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d")


# ---------------------------------------------------------------------------
# Low-level parsing helpers
# ---------------------------------------------------------------------------


def default_dataset_dir() -> Path:
    """Resolve ``dataset/`` relative to this file, overridable by env var."""
    env = os.environ.get("ORCHESTRATE_DATASET_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parent.parent / "dataset").resolve()


def _parse_datetime(series: pd.Series) -> pd.Series:
    """Parse a string column to datetime64, trying each known format."""
    cleaned = series.astype("string").str.strip().replace({"": pd.NA})
    expected = int(cleaned.notna().sum())
    if expected == 0:
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for fmt in _DATETIME_FORMATS:
        parsed = pd.to_datetime(cleaned, format=fmt, errors="coerce")
        if int(parsed.notna().sum()) == expected:
            return parsed
    # Mixed formats within one column: fall back to per-value inference.
    return pd.to_datetime(cleaned, format="mixed", errors="coerce")


def _parse_int(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip().replace({"": pd.NA})
    return pd.to_numeric(cleaned, errors="coerce").astype("Int64")


def _parse_float(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip().replace({"": pd.NA})
    return pd.to_numeric(cleaned, errors="coerce").astype("Float64")


def _parse_bool(series: pd.Series) -> pd.Series:
    """0/1 (or true/false) flag columns -> nullable boolean."""
    cleaned = series.astype("string").str.strip().str.lower().replace({"": pd.NA})
    mapped = cleaned.map(
        {"1": True, "0": False, "true": True, "false": False, "yes": True, "no": False}
    )
    return mapped.astype("boolean")


def _parse_id(series: pd.Series) -> pd.Series:
    """Id columns: empty means 'not applicable', not an empty id."""
    return series.astype("string").str.strip().replace({"": pd.NA})


def _read_csv(path: Path, spec: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """Read one CSV with the shared conventions.

    Everything is read as text first (``keep_default_na=False``) so that free
    text such as ``message_text`` is never reinterpreted as NaN, and so that
    quoted multi-line fields survive verbatim. Typed columns are then coerced
    explicitly per the spec.
    """
    if not path.exists():
        raise FileNotFoundError(f"Expected dataset file not found: {path}")

    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_values=[],
        encoding="utf-8",
    )
    # Normalise header whitespace.
    df.columns = [str(c).strip() for c in df.columns]

    parsers = {
        "id": _parse_id,
        "datetime": _parse_datetime,
        "int": _parse_int,
        "float": _parse_float,
        "bool": _parse_bool,
    }
    for kind, parser in parsers.items():
        for col in spec.get(kind, ()):
            if col in df.columns:
                df[col] = parser(df[col])

    typed = {c for kind in parsers for c in spec.get(kind, ())}
    for col in df.columns:
        if col not in typed:
            df[col] = df[col].astype("string").fillna("")

    return df


def _parse_dnd_window(window: Any) -> tuple[time | None, time | None]:
    """``"22:00-07:00"`` -> ``(time(22, 0), time(7, 0))``."""
    if window is None or (isinstance(window, str) and not window.strip()):
        return None, None
    text = str(window).strip()
    if "-" not in text:
        return None, None
    start_txt, _, end_txt = text.partition("-")

    def _one(part: str) -> time | None:
        part = part.strip()
        for fmt in ("%H:%M", "%H:%M:%S"):
            try:
                return datetime.strptime(part, fmt).time()
            except ValueError:
                continue
        return None

    return _one(start_txt), _one(end_txt)


def _clean(value: Any) -> Any:
    """Convert pandas/numpy scalars into plain Python, NA -> None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NaT or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


def _row_to_dict(row: pd.Series, prefix: str = "") -> dict[str, Any]:
    return {f"{prefix}{k}": _clean(v) for k, v in row.items()}


def _first_row(df: pd.DataFrame) -> pd.Series | None:
    return None if df.empty else df.iloc[0]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    """Typed in-memory view of ``dataset/`` plus id-driven lookup functions."""

    dataset_dir: Path

    messages: pd.DataFrame
    users: pd.DataFrame
    groups: pd.DataFrame
    group_members: pd.DataFrame
    business_accounts: pd.DataFrame
    user_business_history: pd.DataFrame
    message_history: pd.DataFrame
    message_events: pd.DataFrame
    images: pd.DataFrame
    voice_notes: pd.DataFrame
    daily_notification_summary: pd.DataFrame
    sample_messages: pd.DataFrame
    output_template: pd.DataFrame

    # Lookup indexes, built in __post_init__.
    _users_by_id: dict[str, pd.Series] = field(default_factory=dict, repr=False)
    _groups_by_id: dict[str, pd.Series] = field(default_factory=dict, repr=False)
    _members_by_key: dict[tuple[str, str], pd.Series] = field(default_factory=dict, repr=False)
    _business_by_id: dict[str, pd.Series] = field(default_factory=dict, repr=False)
    _ubh_by_key: dict[tuple[str, str], pd.Series] = field(default_factory=dict, repr=False)
    _messages_by_id: dict[str, pd.Series] = field(default_factory=dict, repr=False)
    _history_by_id: dict[str, pd.Series] = field(default_factory=dict, repr=False)
    _events_by_key: dict[tuple[str, str], pd.Series] = field(default_factory=dict, repr=False)
    _media_by_id: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    # ---- construction -----------------------------------------------------

    def __post_init__(self) -> None:
        self._users_by_id = self._index_by(self.users, "user_id")
        self._groups_by_id = self._index_by(self.groups, "group_id")
        self._business_by_id = self._index_by(self.business_accounts, "business_id")
        self._messages_by_id = self._index_by(self.messages, "message_id")
        self._history_by_id = self._index_by(self.message_history, "message_id")
        self._members_by_key = self._index_by_pair(self.group_members, "group_id", "user_id")
        self._ubh_by_key = self._index_by_pair(self.user_business_history, "user_id", "business_id")
        self._events_by_key = self._index_by_pair(self.message_events, "user_id", "message_id")

        self._media_by_id = {}
        for kind, df, id_col in (
            ("image", self.images, "image_id"),
            ("voice", self.voice_notes, "voice_note_id"),
        ):
            for _, row in df.iterrows():
                media_id = _clean(row[id_col])
                if media_id is None:
                    continue
                rel = str(row["file_path"]).strip()
                abs_path = (self.dataset_dir / rel).resolve()
                self._media_by_id[media_id] = {
                    "media_id": media_id,
                    "media_kind": kind,
                    "file_path": rel,
                    "abs_path": abs_path,
                    "file_exists": abs_path.is_file(),
                }

    @staticmethod
    def _index_by(df: pd.DataFrame, key: str) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for _, row in df.iterrows():
            k = _clean(row[key])
            if k is not None and k not in out:  # first row wins; dupes reported by validator
                out[k] = row
        return out

    @staticmethod
    def _index_by_pair(df: pd.DataFrame, k1: str, k2: str) -> dict[tuple[str, str], pd.Series]:
        out: dict[tuple[str, str], pd.Series] = {}
        for _, row in df.iterrows():
            a, b = _clean(row[k1]), _clean(row[k2])
            if a is None or b is None:
                continue
            if (a, b) not in out:
                out[(a, b)] = row
        return out

    # ---- single-entity lookups -------------------------------------------

    def get_user(self, user_id: str | None) -> dict[str, Any] | None:
        """Return the ``users.csv`` row for ``user_id``, or None if unknown."""
        if user_id is None:
            return None
        row = self._users_by_id.get(str(user_id))
        if row is None:
            return None
        out = _row_to_dict(row)
        start, end = _parse_dnd_window(out.get("do_not_disturb_window"))
        out["dnd_start"] = start
        out["dnd_end"] = end
        return out

    def get_group(self, group_id: str | None) -> dict[str, Any] | None:
        if group_id is None:
            return None
        row = self._groups_by_id.get(str(group_id))
        return None if row is None else _row_to_dict(row)

    def get_business(self, business_id: str | None) -> dict[str, Any] | None:
        if business_id is None:
            return None
        row = self._business_by_id.get(str(business_id))
        return None if row is None else _row_to_dict(row)

    def get_message(self, message_id: str | None) -> dict[str, Any] | None:
        """Look up an incoming message, falling back to history."""
        if message_id is None:
            return None
        key = str(message_id)
        row = self._messages_by_id.get(key)
        if row is None:
            row = self._history_by_id.get(key)
        return None if row is None else _row_to_dict(row)

    def get_media(self, media_id: str | None) -> dict[str, Any] | None:
        """Resolve an image/voice id to its path. Does not read the file."""
        if media_id is None:
            return None
        return self._media_by_id.get(str(media_id))

    # ---- merged-context lookups ------------------------------------------

    def get_group_context(
        self, group_id: str | None, user_id: str | None
    ) -> dict[str, Any] | None:
        """Merge ``groups.csv`` with this user's ``group_members.csv`` row.

        Returns None when ``group_id`` is not applicable (personal/business
        messages) or unknown. Membership fields are prefixed ``member_`` and are
        None when the user is not a listed member of the group.
        """
        if group_id is None:
            return None
        group_row = self._groups_by_id.get(str(group_id))
        if group_row is None:
            return None

        ctx: dict[str, Any] = _row_to_dict(group_row)
        member_row = (
            self._members_by_key.get((str(group_id), str(user_id)))
            if user_id is not None
            else None
        )
        ctx["user_id"] = None if user_id is None else str(user_id)
        ctx["is_member"] = member_row is not None
        member_cols = [c for c in self.group_members.columns if c not in ("group_id", "user_id")]
        if member_row is not None:
            ctx.update({f"member_{c}": _clean(member_row[c]) for c in member_cols})
        else:
            ctx.update({f"member_{c}": None for c in member_cols})
        return ctx

    def get_business_context(
        self, user_id: str | None, business_id: str | None
    ) -> dict[str, Any] | None:
        """Merge ``business_accounts.csv`` with this user's relationship row.

        Relationship fields are prefixed ``rel_``; they are None when the user
        has no prior history with the business (a cold sender). ``domain_match``
        is a plain equality check on the two domain columns already in the file,
        surfaced here so callers do not re-derive it.
        """
        if business_id is None:
            return None
        biz_row = self._business_by_id.get(str(business_id))
        if biz_row is None:
            return None

        ctx: dict[str, Any] = _row_to_dict(biz_row)
        ctx["user_id"] = None if user_id is None else str(user_id)

        official = (ctx.get("official_domain") or "").strip().lower()
        used = (ctx.get("domain_used_by_sender") or "").strip().lower()
        ctx["domain_match"] = bool(official) and bool(used) and official == used

        rel_row = (
            self._ubh_by_key.get((str(user_id), str(business_id)))
            if user_id is not None
            else None
        )
        rel_cols = [
            c for c in self.user_business_history.columns if c not in ("user_id", "business_id")
        ]
        ctx["has_relationship"] = rel_row is not None
        if rel_row is not None:
            ctx.update({f"rel_{c}": _clean(rel_row[c]) for c in rel_cols})
        else:
            ctx.update({f"rel_{c}": None for c in rel_cols})
        return ctx

    # ---- history retrieval ------------------------------------------------

    def get_message_history_for_sender(
        self,
        user_id: str,
        sender_user_id: str | None = None,
        group_id: str | None = None,
        business_id: str | None = None,
        *,
        conversation_type: str | None = None,
        before: datetime | pd.Timestamp | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Historical messages received by ``user_id``, joined to their reactions.

        All supplied filters are ANDed. With no filters beyond ``user_id`` you
        get that user's whole history. Result columns are the
        ``message_history.csv`` columns plus the ``message_events.csv`` reaction
        columns (``message_opened``, ``message_replied``,
        ``reaction_time_minutes``, ``notification_dismissed``,
        ``muted_after_message``, ``message_reported``), which are NA when the
        user has no recorded reaction to that message.

        Sorted newest first. ``before`` restricts to messages strictly older than
        a timestamp, which is what you want when retrieving precedent for an
        incoming message without leaking the future.
        """
        hist = self.message_history
        mask = hist["user_id"] == str(user_id)
        if sender_user_id is not None:
            mask &= hist["sender_user_id"] == str(sender_user_id)
        if group_id is not None:
            mask &= hist["group_id"] == str(group_id)
        if business_id is not None:
            mask &= hist["business_id"] == str(business_id)
        if conversation_type is not None:
            mask &= hist["conversation_type"] == str(conversation_type)
        if before is not None:
            mask &= hist["created_at"] < pd.Timestamp(before)

        subset = hist[mask.fillna(False)]

        events = self.message_events
        joined = subset.merge(
            events[events["user_id"] == str(user_id)].drop(columns=["user_id"]),
            on="message_id",
            how="left",
            validate="many_to_one",
        )
        joined = joined.sort_values("created_at", ascending=False, kind="stable")
        joined = joined.reset_index(drop=True)
        if limit is not None:
            joined = joined.head(limit)
        return joined

    def get_message_events(self, user_id: str, message_id: str) -> dict[str, Any] | None:
        row = self._events_by_key.get((str(user_id), str(message_id)))
        return None if row is None else _row_to_dict(row)

    def get_daily_notification_load(
        self, user_id: str, on_date: datetime | pd.Timestamp | None = None
    ) -> pd.DataFrame:
        """Daily notification counts for a user, optionally for one date."""
        df = self.daily_notification_summary
        mask = df["user_id"] == str(user_id)
        if on_date is not None:
            day = pd.Timestamp(on_date).normalize()
            mask &= df["date"] == day
        return df[mask.fillna(False)].sort_values("date").reset_index(drop=True)

    # ---- convenience ------------------------------------------------------

    def is_in_dnd_window(self, user_id: str, when: datetime | pd.Timestamp) -> bool | None:
        """Is ``when`` inside the user's do-not-disturb window? None if unknown.

        Handles windows that wrap past midnight (e.g. 22:00-07:00).
        """
        user = self.get_user(user_id)
        if user is None:
            return None
        start, end = user.get("dnd_start"), user.get("dnd_end")
        if start is None or end is None:
            return None
        moment = pd.Timestamp(when).time()
        if start <= end:
            return start <= moment < end
        return moment >= start or moment < end

    def resolve_message(self, message_id: str, *, history_limit: int | None = 10) -> dict[str, Any]:
        """Bundle every lookup for one incoming message into a single dict.

        This is a thin assembly of the lookups above - no derived features, no
        ranking, no summarisation. Those are Phase 2's job.
        """
        message = self.get_message(message_id)
        if message is None:
            raise KeyError(f"Unknown message_id: {message_id}")

        user_id = message.get("user_id")
        created_at = message.get("created_at")

        history = self.get_message_history_for_sender(
            user_id,
            sender_user_id=message.get("sender_user_id"),
            group_id=message.get("group_id"),
            business_id=message.get("business_id"),
            before=created_at,
            limit=history_limit,
        )

        return {
            "message": message,
            "user": self.get_user(user_id),
            "group_context": self.get_group_context(message.get("group_id"), user_id),
            "business_context": self.get_business_context(user_id, message.get("business_id")),
            "sender": self.get_user(message.get("sender_user_id")),
            "media": self.get_media(message.get("media_id")),
            "in_dnd_window": (
                self.is_in_dnd_window(user_id, created_at) if created_at is not None else None
            ),
            "history": history,
        }

    # ---- introspection ----------------------------------------------------

    def table_names(self) -> list[str]:
        return [
            "messages",
            "users",
            "groups",
            "group_members",
            "business_accounts",
            "user_business_history",
            "message_history",
            "message_events",
            "images",
            "voice_notes",
            "daily_notification_summary",
            "sample_messages",
            "output_template",
        ]

    def row_counts(self) -> dict[str, int]:
        return {name: len(getattr(self, name)) for name in self.table_names()}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_dataset(dataset_dir: str | Path | None = None) -> Dataset:
    """Load every participant-facing CSV into a :class:`Dataset`."""
    base = Path(dataset_dir).expanduser().resolve() if dataset_dir else default_dataset_dir()
    if not base.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {base}")

    def read(stem: str) -> pd.DataFrame:
        return _read_csv(base / f"{stem}.csv", _SPECS[stem])

    return Dataset(
        dataset_dir=base,
        messages=read("messages"),
        users=read("users"),
        groups=read("groups"),
        group_members=read("group_members"),
        business_accounts=read("business_accounts"),
        user_business_history=read("user_business_history"),
        message_history=read("message_history"),
        message_events=read("message_events"),
        images=read("images"),
        voice_notes=read("voice_notes"),
        daily_notification_summary=read("daily_notification_summary"),
        sample_messages=read("sample_messages"),
        output_template=read("output"),
    )
