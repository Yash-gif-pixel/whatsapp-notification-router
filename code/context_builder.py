"""Phase 2 - Context assembly for the Message Notification Router.

Turns one row of ``messages.csv`` into a structured context object holding
every fact a later phase needs to route it. This module assembles facts only:
no routing decision, no rule engine, no LLM calls, no media decoding.

All file loading and joining is delegated to :mod:`data_loader` (Phase 1).

Typical use::

    from data_loader import load_dataset
    from context_builder import build_context, build_all_contexts

    ds = load_dataset()
    ctx = build_context("msg_001", ds)
    all_ctx = build_all_contexts(ds)          # one per row of messages.csv

Two invariants the later phases depend on:

* ``ctx["evidence_tier"]`` is one of ``exact`` / ``fallback`` / ``cross_type``
  / ``none`` and every item in ``ctx["retrieved_evidence"]`` carries the same
  tier label plus its own ``source_conversation_type``. Nothing is fabricated:
  tier ``none`` means an empty list. ``ctx["evidence_strength"]`` gives the
  same ordering as a number (3/2/1/0) for weighting.
* ``ctx["completeness"]["safe_for_text_reasoning"]`` is False whenever
  ``message_text`` is blank because the payload is unread media. Do not feed
  such a message to any text rule or prompt as if the empty string were
  evidence of "no urgency" or "no risk" - it is unknown, pending Phase 3.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Mapping, Sequence

import pandas as pd

from data_loader import Dataset, load_dataset

__all__ = [
    "build_context",
    "build_all_contexts",
    "to_jsonable",
    "MAX_EVIDENCE_PER_TIER",
    "EVIDENCE_TIERS",
    "EVIDENCE_TIER_STRENGTH",
]

#: Most historical messages kept per message, per the Phase 2 spec.
MAX_EVIDENCE_PER_TIER = 3

#: Tier labels, strongest first.
#:
#: ``exact``      a true precedent from this exact counterpart.
#: ``fallback``   same conversation_type for this user, any counterpart.
#: ``cross_type`` this user's general history from *other* conversation types.
#:                Weakest evidence available - it says nothing about how this
#:                user behaves in this channel, only how they behave in
#:                general. Only ever preferable to having nothing.
#: ``none``       no evidence exists. Always an empty list.
EVIDENCE_TIERS = ("exact", "fallback", "cross_type", "none")

#: Machine-readable strength ordering so a later phase can weight or discount
#: evidence without hardcoding the tier names. Higher is stronger.
EVIDENCE_TIER_STRENGTH = {"exact": 3, "fallback": 2, "cross_type": 1, "none": 0}

#: ``users.csv`` has no display-name column, so the only mention token the
#: dataset can express is the user id itself (e.g. ``@u_010``).
_MENTION_RE = re.compile(r"@\s*([A-Za-z0-9_]+)")

_REACTION_FLAGS = (
    "message_opened",
    "message_replied",
    "notification_dismissed",
    "muted_after_message",
    "message_reported",
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _opt(value: Any) -> Any:
    """Normalise a pandas scalar to a plain Python value or None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


def _flag(value: Any) -> bool | None:
    v = _opt(value)
    return None if v is None else bool(v)


def _int(value: Any) -> int | None:
    v = _opt(value)
    return None if v is None else int(v)


def _text(value: Any) -> str:
    v = _opt(value)
    return "" if v is None else str(v)


def _resolve_message_row(ds: Dataset, message: Any) -> dict[str, Any]:
    """Accept a message_id, a DataFrame row, or a plain dict."""
    if isinstance(message, str):
        row = ds.get_message(message)
        if row is None:
            raise KeyError(f"Unknown message_id: {message}")
        return row
    if isinstance(message, pd.Series):
        return {k: _opt(v) for k, v in message.items()}
    if isinstance(message, Mapping):
        return {k: _opt(v) for k, v in message.items()}
    raise TypeError(f"Cannot build context from {type(message).__name__}")


# ---------------------------------------------------------------------------
# section builders
# ---------------------------------------------------------------------------


def _build_user_profile(ds: Dataset, user_id: str | None) -> dict[str, Any] | None:
    """Spec field 1 - the five notification-behaviour columns from users.csv."""
    user = ds.get_user(user_id)
    if user is None:
        return None
    return {
        "user_id": user["user_id"],
        "do_not_disturb_window": user["do_not_disturb_window"],
        "dnd_start": user["dnd_start"],
        "dnd_end": user["dnd_end"],
        "messages_opened_30d": _int(user["messages_opened_30d"]),
        "messages_replied_30d": _int(user["messages_replied_30d"]),
        "notifications_dismissed_30d": _int(user["notifications_dismissed_30d"]),
        "messages_reported_30d": _int(user["messages_reported_30d"]),
    }


def _build_group_context(ds: Dataset, msg: Mapping[str, Any]) -> dict[str, Any]:
    """Spec field 2a - group metadata plus THIS user's standing in THIS group."""
    group_id = msg.get("group_id")
    user_id = msg.get("user_id")
    sender_id = msg.get("sender_user_id")

    gc = ds.get_group_context(group_id, user_id)
    if gc is None:
        return {"kind": "group", "group_id": group_id, "group_found": False}

    sender_gc = ds.get_group_context(group_id, sender_id) if sender_id else None
    sender_role = sender_gc.get("member_role") if sender_gc else None

    return {
        "kind": "group",
        "group_found": True,
        "group_id": gc["group_id"],
        "group_name": gc["group_name"],
        "group_type": gc["group_type"],
        "member_count": _int(gc["member_count"]),
        "admin_count": _int(gc["admin_count"]),
        "group_created_at": gc["created_at"],
        "group_messages_30d": _int(gc["messages_30d"]),
        # this user, in this group specifically
        "is_member": bool(gc["is_member"]),
        "user_role": gc["member_role"],
        "user_is_admin": gc["member_role"] == "admin",
        "group_muted_by_user": _flag(gc["member_group_muted_by_user"]),
        "user_joined_at": gc["member_joined_at"],
        "user_messages_sent_30d": _int(gc["member_messages_sent_30d"]),
        "user_messages_read_30d": _int(gc["member_messages_read_30d"]),
        "user_replies_sent_30d": _int(gc["member_replies_sent_30d"]),
        "user_notifications_dismissed_30d": _int(gc["member_notifications_dismissed_30d"]),
        # who sent it, and their standing in this group
        "sender_user_id": sender_id,
        "sender_is_member": sender_gc is not None and bool(sender_gc["is_member"]),
        "sender_role_in_group": sender_role,
        "sender_is_admin": sender_role == "admin",
    }


def _build_business_context(ds: Dataset, msg: Mapping[str, Any]) -> dict[str, Any]:
    """Spec field 2b - business identity plus THIS user's relationship to it."""
    business_id = msg.get("business_id")
    user_id = msg.get("user_id")

    bc = ds.get_business_context(user_id, business_id)
    if bc is None:
        return {"kind": "business", "business_id": business_id, "business_found": False}

    return {
        "kind": "business",
        "business_found": True,
        "business_id": bc["business_id"],
        "display_name": bc["display_name"],
        "brand_name": bc["brand_name"],
        "category": bc["category"],
        "verified": _flag(bc["verified"]),
        # identity signals
        "official_domain": bc["official_domain"] or None,
        "domain_used_by_sender": bc["domain_used_by_sender"] or None,
        "domain_match": bool(bc["domain_match"]),
        "domain_used_by_sender_age_days": _int(bc["domain_used_by_sender_age_days"]),
        "account_age_days": _int(bc["account_age_days"]),
        "messages_sent_30d": _int(bc["messages_sent_30d"]),
        "user_reports_30d": _int(bc["user_reports_30d"]),
        # this user's relationship with this business specifically
        "has_relationship": bool(bc["has_relationship"]),
        "why_user_knows_account": bc["rel_why_user_knows_account"],
        "last_activity_at": bc["rel_last_activity_at"],
        "allows_promotions": _flag(bc["rel_allows_promotions"]),
        "promotions_opted_out_at": bc["rel_promotions_opted_out_at"],
        "activity_count_180d": _int(bc["rel_activity_count_180d"]),
        "user_messages_opened_30d": _int(bc["rel_messages_opened_30d"]),
        "user_messages_dismissed_30d": _int(bc["rel_messages_dismissed_30d"]),
        "user_messages_replied_30d": _int(bc["rel_messages_replied_30d"]),
        "user_last_reply_at": bc["rel_last_reply_at"],
    }


def _build_personal_context(ds: Dataset, msg: Mapping[str, Any]) -> dict[str, Any]:
    """Spec field 2c - minimal, plus the optional shared-group check."""
    user_id = msg.get("user_id")
    sender_id = msg.get("sender_user_id")

    sender = ds.get_user(sender_id)
    ctx: dict[str, Any] = {
        "kind": "personal",
        "sender_user_id": sender_id,
        "sender_known_user": sender is not None,
    }

    gm = ds.group_members
    mine = set(gm.loc[gm["user_id"] == str(user_id), "group_id"].dropna())
    theirs = set(gm.loc[gm["user_id"] == str(sender_id), "group_id"].dropna()) if sender_id else set()
    shared = sorted(mine & theirs)

    ctx["shared_group_count"] = len(shared)
    ctx["shared_group_ids"] = shared
    ctx["shared_groups"] = [
        {
            "group_id": g,
            "group_name": (ds.get_group(g) or {}).get("group_name"),
            "group_type": (ds.get_group(g) or {}).get("group_type"),
        }
        for g in shared
    ]
    if sender is not None:
        ctx["sender_profile"] = {
            "messages_opened_30d": _int(sender["messages_opened_30d"]),
            "messages_replied_30d": _int(sender["messages_replied_30d"]),
            "notifications_dismissed_30d": _int(sender["notifications_dismissed_30d"]),
            "messages_reported_30d": _int(sender["messages_reported_30d"]),
        }
    return ctx


def _build_conversation_context(ds: Dataset, msg: Mapping[str, Any]) -> dict[str, Any]:
    conv = msg.get("conversation_type")
    if conv == "group":
        return _build_group_context(ds, msg)
    if conv == "business":
        return _build_business_context(ds, msg)
    if conv == "personal":
        return _build_personal_context(ds, msg)
    return {"kind": "unknown", "conversation_type": conv}


def _build_mention_check(msg: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Spec field 3 - is THIS user directly mentioned in the text?

    ``users.csv`` carries no display name, so the only mention token the
    dataset can express is ``@<user_id>``. Mentions of other users are recorded
    separately so a later phase can tell "someone was tagged, but not you"
    apart from "nobody was tagged".
    """
    user_id = str(msg.get("user_id") or "")
    text = _text(msg.get("message_text"))
    tokens = [t for t in _MENTION_RE.findall(text)]
    lowered = {t.lower() for t in tokens}

    mentions_user = user_id.lower() in lowered
    detail = {
        "mention_tokens": tokens,
        "mentions_this_user": mentions_user,
        "mentions_other_user_only": bool(tokens) and not mentions_user,
        "matched_on": "user_id" if mentions_user else None,
        # No name column exists in users.csv, so name-based matching is not
        # possible with this dataset. Recorded so it is not silently assumed.
        "name_matching_available": False,
        "text_searchable": bool(text.strip()),
    }
    return mentions_user, detail


def _reaction(row: Mapping[str, Any]) -> dict[str, Any]:
    recorded = any(_opt(row.get(f)) is not None for f in _REACTION_FLAGS)
    return {
        "recorded": recorded,
        **{f: _flag(row.get(f)) for f in _REACTION_FLAGS},
        "reaction_time_minutes": _int(row.get("reaction_time_minutes")),
    }


def _evidence_items(
    frame: pd.DataFrame, tier: str, match_reason: Any, limit: int
) -> list[dict[str, Any]]:
    """History rows (already newest-first) -> evidence dicts, capped at ``limit``.

    ``match_reason`` may be a plain string or a callable taking the row, which
    lets a tier explain itself per item (cross-type evidence names the
    conversation it actually came from).
    """
    items: list[dict[str, Any]] = []
    for _, row in frame.head(limit).iterrows():
        source_type = _opt(row["conversation_type"])
        items.append(
            {
                "message_id": _opt(row["message_id"]),
                "evidence_tier": tier,
                "evidence_strength": EVIDENCE_TIER_STRENGTH[tier],
                "match_reason": match_reason(row) if callable(match_reason) else match_reason,
                "created_at": _opt(row["created_at"]),
                "conversation_type": source_type,
                # Explicit duplicate of conversation_type: downstream reasoning
                # must be able to say which channel this precedent came from
                # without inferring it, especially for cross_type evidence.
                "source_conversation_type": source_type,
                "group_id": _opt(row["group_id"]),
                "business_id": _opt(row["business_id"]),
                "sender_user_id": _opt(row["sender_user_id"]),
                "message_text": _text(row["message_text"]),
                "media_type": _text(row["media_type"]) or None,
                "media_id": _opt(row["media_id"]),
                "forwarded_count": _int(row["forwarded_count"]),
                "reaction": _reaction(row),
            }
        )
    return items


def _build_evidence(
    ds: Dataset, msg: Mapping[str, Any]
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Spec field 4 - tiered retrieval, stopping at the first tier with hits.

    Tier 1  (``exact``)      same user AND same counterpart:
                             sender_user_id for personal, business_id for
                             business, group_id for group.
    Tier 2  (``fallback``)   same user AND same conversation_type, when tier 1
                             is empty. Weaker: this user's general behaviour in
                             that channel, not a precedent from this
                             counterpart.
    Tier 2b (``cross_type``) same user, ANY conversation_type, when tiers 1 and
                             2 are both empty. Weakest usable evidence: it
                             describes how this user behaves elsewhere, not in
                             this channel. Every item names its
                             ``source_conversation_type`` so a later phase can
                             attribute it honestly instead of implying it is a
                             same-channel precedent.
    Tier 3  (``none``)       nothing retrievable. Empty list, never fabricated.

    ``before`` keeps retrieval strictly in the past relative to the incoming
    message so no future information leaks into the evidence.
    """
    user_id = msg.get("user_id")
    conv = msg.get("conversation_type")
    created_at = msg.get("created_at")

    if conv == "group":
        exact_kwargs = {"group_id": msg.get("group_id")}
        exact_reason = f"same group_id ({msg.get('group_id')})"
    elif conv == "business":
        exact_kwargs = {"business_id": msg.get("business_id")}
        exact_reason = f"same business_id ({msg.get('business_id')})"
    else:
        exact_kwargs = {"sender_user_id": msg.get("sender_user_id")}
        exact_reason = f"same sender_user_id ({msg.get('sender_user_id')})"

    exact = ds.get_message_history_for_sender(user_id, before=created_at, **exact_kwargs)

    # Diagnostics: recorded regardless of which tier wins, so a "none" result
    # can be told apart from a retrieval bug without re-querying.
    same_type = ds.get_message_history_for_sender(
        user_id, conversation_type=conv, before=created_at
    )
    any_history = ds.get_message_history_for_sender(user_id, before=created_at)
    diagnostics = {
        "n_exact_matches": len(exact),
        "n_same_conversation_type": len(same_type),
        "n_any_history_for_user": len(any_history),
        "history_conversation_type_mix": (
            any_history["conversation_type"].value_counts().to_dict() if len(any_history) else {}
        ),
    }

    if len(exact):
        return "exact", _evidence_items(exact, "exact", exact_reason, MAX_EVIDENCE_PER_TIER), diagnostics

    if len(same_type):
        reason = f"no exact precedent; same conversation_type ({conv}) for this user"
        return "fallback", _evidence_items(same_type, "fallback", reason, MAX_EVIDENCE_PER_TIER), diagnostics

    if len(any_history):
        def reason(row: Mapping[str, Any]) -> str:
            src = _opt(row["conversation_type"])
            return (
                f"no exact precedent and no {conv} history for this user; "
                f"cross-type evidence from a {src} conversation - reflects this "
                f"user's general behaviour, not their behaviour in {conv} chats"
            )

        return (
            "cross_type",
            _evidence_items(any_history, "cross_type", reason, MAX_EVIDENCE_PER_TIER),
            diagnostics,
        )

    return "none", [], diagnostics


def _build_media_status(ds: Dataset, msg: Mapping[str, Any]) -> dict[str, Any]:
    """Spec field 5 - what still has to happen before the payload is readable."""
    media_type = _text(msg.get("media_type"))
    media_id = msg.get("media_id")
    has_text = bool(_text(msg.get("message_text")).strip())

    status: dict[str, Any] = {
        "media_type": media_type or None,
        "media_id": media_id,
        "needs_image_description": False,
        "needs_transcription": False,
        "file_path": None,
        "abs_path": None,
        "file_exists": None,
        "media_resolved": None,
    }

    if not media_type:
        status["status"] = "text"
        return status

    status["status"] = (
        "needs_image_description" if media_type == "image" else "needs_transcription"
    )
    status["needs_image_description"] = media_type == "image"
    status["needs_transcription"] = media_type == "voice"

    media = ds.get_media(media_id)
    status["media_resolved"] = media is not None
    if media is not None:
        status["file_path"] = media["file_path"]
        status["abs_path"] = str(media["abs_path"])
        status["file_exists"] = media["file_exists"]
        status["media_kind"] = media["media_kind"]
        status["media_kind_matches_media_type"] = media["media_kind"] == media_type

    # The hard guard the spec calls for: a voice note with no caption carries
    # no readable payload at all until Phase 3 transcribes it.
    status["has_caption_text"] = has_text
    status["payload_readable"] = has_text
    return status


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def build_context(message: Any, ds: Dataset | None = None) -> dict[str, Any]:
    """Assemble the full context object for one incoming message.

    ``message`` may be a ``message_id``, a row from ``ds.messages``, or a dict
    with the same columns. Works for any id - nothing is hardcoded.
    """
    ds = ds if ds is not None else load_dataset()
    msg = _resolve_message_row(ds, message)

    user_id = msg.get("user_id")
    created_at = msg.get("created_at")
    message_text = _text(msg.get("message_text"))
    text_usable = bool(message_text.strip())

    mention, mention_detail = _build_mention_check(msg)
    tier, evidence, evidence_diagnostics = _build_evidence(ds, msg)
    media_status = _build_media_status(ds, msg)

    pending_steps: list[str] = []
    if media_status["needs_transcription"]:
        pending_steps.append("transcription")
    if media_status["needs_image_description"]:
        pending_steps.append("image_description")

    return {
        # --- identity / passthrough -------------------------------------
        "message_id": msg.get("message_id"),
        "user_id": user_id,
        "conversation_type": msg.get("conversation_type"),
        "created_at": created_at,
        "message_text": message_text,
        "forwarded_count": _int(msg.get("forwarded_count")),  # spec field 6, as-is
        # --- spec field 1 -----------------------------------------------
        "user_profile": _build_user_profile(ds, user_id),
        # --- spec field 2 -----------------------------------------------
        "conversation_context": _build_conversation_context(ds, msg),
        # --- spec field 3 -----------------------------------------------
        "mention_check": mention,
        "mention_detail": mention_detail,
        # --- spec field 4 -----------------------------------------------
        "evidence_tier": tier,
        "evidence_strength": EVIDENCE_TIER_STRENGTH[tier],
        "retrieved_evidence": evidence,
        "evidence_diagnostics": evidence_diagnostics,
        # --- spec field 5 -----------------------------------------------
        "media_status": media_status,
        # Filled by Phase 3 (media_normalizer.apply_normalization). Declared
        # here so the context shape is stable whether or not Phase 3 has run.
        # Always separate from message_text: a later phase must be able to tell
        # original message content from text derived out of media.
        "normalized_text": None,
        "normalization": None,
        # --- assembly-time facts ----------------------------------------
        "timing": {
            "created_at": created_at,
            "hour": created_at.hour if isinstance(created_at, datetime) else None,
            "weekday": created_at.strftime("%a") if isinstance(created_at, datetime) else None,
            "in_dnd_window": (
                ds.is_in_dnd_window(user_id, created_at) if created_at is not None else None
            ),
        },
        "completeness": {
            "text_usable": text_usable,
            "pending_steps": pending_steps,
            # False => downstream must NOT read the empty message_text as
            # "no urgency signal" / "no risk signal". It is unknown.
            "safe_for_text_reasoning": text_usable,
            "context_complete": text_usable and not pending_steps,
        },
    }


def build_all_contexts(ds: Dataset | None = None) -> list[dict[str, Any]]:
    """One context object per row of ``messages.csv``, in file order."""
    ds = ds if ds is not None else load_dataset()
    return [build_context(row, ds) for _, row in ds.messages.iterrows()]


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------


def to_jsonable(value: Any) -> Any:
    """Deep-convert a context object into JSON-safe primitives."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="minutes")
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
