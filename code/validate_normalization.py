"""Phase 3 - validation for media normalization.

Run from the terminal::

    py code/validate_normalization.py

Confirms every image and voice note ended up with a non-null ``normalized_text``
(or an explicit ``processing_error``), prints all 8 transcripts in full and a
sample of image descriptions for spot-checking, and confirms that
``safe_for_text_reasoning`` is now True for all 110 messages.

Reads the cache written by ``media_normalizer.py``; makes no API calls of its
own. Problems are reported, never raised.

TEST FIXTURES, NOT ROUTING INPUTS
---------------------------------
This file contains no hardcoded message, user, group or business id, and no
module on the routing path (``finalize`` -> ``rule_engine`` / ``llm_reasoner``
/ ``media_normalizer`` / ``context_builder`` / ``data_loader``) imports it.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from context_builder import build_all_contexts  # noqa: E402
from data_loader import load_dataset  # noqa: E402
from media_normalizer import (  # noqa: E402
    NormalizationStore,
    apply_normalization,
    effective_text,
)

PROBLEMS: list[str] = []
MAX_LISTED = 10

EXPECTED_MESSAGES = 110
EXPECTED_IMAGE_MESSAGES = 15
EXPECTED_VOICE_MESSAGES = 8
EXPECTED_DISTINCT_IMAGES = 11
EXPECTED_DISTINCT_VOICE = 8
IMAGE_SAMPLE = 4


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


def wrapped(text: str, indent: str = "      ") -> None:
    for para in str(text).split("\n"):
        if not para.strip():
            print()
            continue
        for line in textwrap.wrap(para, width=72) or [""]:
            print(indent + line)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def check_coverage(contexts: list[dict[str, Any]], store: NormalizationStore) -> None:
    section("1. NORMALIZATION COVERAGE")

    images = [c for c in contexts if c["media_status"]["media_type"] == "image"]
    voices = [c for c in contexts if c["media_status"]["media_type"] == "voice"]
    distinct_img = {c["media_status"]["media_id"] for c in images}
    distinct_vn = {c["media_status"]["media_id"] for c in voices}

    print(f"  image messages : {len(images):>3}  ({len(distinct_img)} distinct files)")
    print(f"  voice messages : {len(voices):>3}  ({len(distinct_vn)} distinct files)")
    print(f"  API calls needed for full coverage: "
          f"{len(distinct_img) + len(distinct_vn)} "
          f"(not {len(images) + len(voices)} - media is shared across messages)")

    for label, got, expected in (
        ("image messages", len(images), EXPECTED_IMAGE_MESSAGES),
        ("voice messages", len(voices), EXPECTED_VOICE_MESSAGES),
        ("distinct images", len(distinct_img), EXPECTED_DISTINCT_IMAGES),
        ("distinct voice notes", len(distinct_vn), EXPECTED_DISTINCT_VOICE),
    ):
        if got == expected:
            ok(f"{label}: {got} as expected")
        else:
            warn(f"{label}: expected {expected}, got {got}")

    sub("per-message normalization result")
    for label, group in (("image", images), ("voice", voices)):
        done = [c for c in group if c.get("normalized_text")]
        errored = [c for c in group if c.get("normalization")
                   and c["normalization"]["status"] == "failed"]
        missing = [c for c in group if c.get("normalization") is None
                   or c["normalization"]["status"] == "not_attempted"]
        print(f"    {label}: {len(done)} normalized, {len(errored)} failed, "
              f"{len(missing)} not attempted (of {len(group)})")
        if missing:
            warn(f"{label} message(s) never attempted: "
                 f"{sample(c['message_id'] for c in missing)}")
        if errored:
            warn(f"{label} message(s) failed to normalize: "
                 f"{sample(c['message_id'] for c in errored)}")
        if group and not errored and not missing:
            ok(f"every {label} message has a non-null normalized_text")

    sub("failures in detail")
    failed_ids = store.failed_ids()
    if not failed_ids:
        print("    (none)")
    else:
        for media_id in failed_ids:
            entry = store.entries[media_id]
            print(f"    {media_id} ({entry.get('media_kind')}): "
                  f"{entry.get('processing_error')}")
        warn(f"{len(failed_ids)} media file(s) have an explicit processing_error")

    sub("nothing fabricated")
    # Every message carries normalized_text, but a message with no media must
    # carry its own text verbatim - anything else would be invented content.
    # (This check originally asserted that only media messages had the field,
    # which stopped being true once Phase 4 needed text for all 110.)
    invented = [
        c["message_id"] for c in contexts
        if not c["media_status"]["media_id"]
        and (c.get("normalized_text") or "") != (c["message_text"] or "")
    ]
    if invented:
        warn(f"non-media message whose normalized_text differs from its own "
             f"message_text: {sample(invented)}")
    else:
        ok("every non-media message carries its own text verbatim - "
           "nothing invented")

    wrong_source = [
        c["message_id"] for c in contexts
        if not c["media_status"]["media_id"]
        and (c.get("normalization") or {}).get("source") != "original_text"
    ]
    if wrong_source:
        warn(f"non-media message not labelled source=original_text: "
             f"{sample(wrong_source)}")
    else:
        ok("provenance is recorded on every message, media or not")

    both = [
        c["message_id"] for c in contexts
        if c.get("normalized_text") and (c.get("normalization") or {}).get("processing_error")
    ]
    if both:
        warn(f"both normalized_text and processing_error set: {sample(both)}")
    else:
        ok("normalized_text and processing_error are mutually exclusive")

    sub("original message_text preserved")
    ds_text = {c["message_id"]: c["message_text"] for c in contexts}
    voice_with_text = [c["message_id"] for c in voices if ds_text[c["message_id"]].strip()]
    if voice_with_text:
        warn(f"voice message_text was overwritten: {sample(voice_with_text)}")
    else:
        ok("message_text still empty for all voice messages - "
           "normalized_text is a separate field")


def check_guard(contexts: list[dict[str, Any]]) -> None:
    section("2. TEXT-REASONING GUARD")

    unsafe = [c for c in contexts if not c["completeness"]["safe_for_text_reasoning"]]
    pending = [c for c in contexts if c["completeness"]["pending_steps"]]

    print(f"  total messages                        : {len(contexts)}")
    print(f"  safe_for_text_reasoning == False      : {len(unsafe)}   (was 8 before Phase 3)")
    print(f"  still carrying a pending step         : {len(pending)}")

    if len(contexts) != EXPECTED_MESSAGES:
        warn(f"expected {EXPECTED_MESSAGES} contexts, got {len(contexts)}")

    if not unsafe:
        ok("safe_for_text_reasoning is True for every message")
    else:
        warn(f"{len(unsafe)} message(s) still unsafe for text reasoning: "
             f"{sample(c['message_id'] for c in unsafe)}")
        for c in unsafe:
            err = (c.get("normalization") or {}).get("processing_error") or "not attempted"
            print(f"      {c['message_id']}  {c['media_status']['media_type']}"
                  f"/{c['media_status']['media_id']}  -> {err}")

    if pending:
        by_step: dict[str, list[str]] = {}
        for c in pending:
            for step in c["completeness"]["pending_steps"]:
                by_step.setdefault(step, []).append(c["message_id"])
        for step, ids in sorted(by_step.items()):
            warn(f"{len(ids)} message(s) still pending '{step}': {sample(ids)}")
    else:
        ok("no message has a pending normalization step left")

    sub("guard consistency")
    mismatch = [
        c["message_id"] for c in contexts
        if c["completeness"]["safe_for_text_reasoning"]
        and not (c["message_text"].strip() or c.get("normalized_text"))
    ]
    if mismatch:
        warn(f"marked safe but has no usable text at all: {sample(mismatch)}")
    else:
        ok("every message marked safe actually has text to reason over")

    complete = [c for c in contexts if c["completeness"]["context_complete"]]
    print(f"    context_complete == True : {len(complete)} / {len(contexts)}")


def check_provenance(contexts: list[dict[str, Any]], store: NormalizationStore) -> None:
    section("3. PROVENANCE AND CONFIDENCE")

    normalized = [c for c in contexts if c.get("normalized_text")]
    by_provider: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for c in normalized:
        n = c["normalization"]
        by_provider[str(n.get("provider"))] = by_provider.get(str(n.get("provider")), 0) + 1
        by_source[str(n.get("source"))] = by_source.get(str(n.get("source")), 0) + 1

    print("  messages with normalized_text, by provider:")
    for k, v in sorted(by_provider.items()):
        print(f"    {k:<12} {v}")
    print("  by source:")
    for k, v in sorted(by_source.items()):
        print(f"    {k:<20} {v}")

    models = sorted({str(c["normalization"].get("model")) for c in normalized})
    print(f"  models used: {', '.join(models) if models else '(none)'}")

    sub("transcription confidence")
    voices = [c for c in contexts if c["media_status"]["media_type"] == "voice"]
    conf: dict[str, list[str]] = {}
    for c in voices:
        value = str((c.get("normalization") or {}).get("transcription_confidence"))
        conf.setdefault(value, []).append(c["message_id"])
    for value, ids in sorted(conf.items()):
        print(f"    {value:<8} {len(ids)}   {sample(ids)}")
    low = conf.get("low", [])
    if low:
        print(f"    -> {len(low)} transcript(s) flagged low confidence; treat their "
              f"content as uncertain in Phase 4")

    unflagged = [
        c["message_id"] for c in voices
        if c.get("normalized_text")
        and (c["normalization"].get("transcription_confidence") not in ("high", "low"))
    ]
    transcribed = [c for c in voices if c.get("normalized_text")]
    if unflagged:
        warn(f"transcript(s) with no confidence flag: {sample(unflagged)}")
    elif transcribed:
        ok(f"all {len(transcribed)} transcript(s) carry a high/low confidence flag")
    else:
        print("    (no transcripts yet - nothing to check)")

    sub("de-duplication")
    shared: dict[str, list[str]] = {}
    for c in contexts:
        media_id = c["media_status"].get("media_id")
        if media_id:
            shared.setdefault(media_id, []).append(c["message_id"])
    reused = {k: v for k, v in shared.items() if len(v) > 1}
    print(f"    media files referenced by more than one message: {len(reused)}")
    for media_id, ids in sorted(reused.items()):
        print(f"      {media_id} -> {sorted(ids)}")
    inconsistent = [
        media_id for media_id, ids in reused.items()
        if len({next((c["normalized_text"] for c in contexts if c["message_id"] == i), None)
                for i in ids}) > 1
    ]
    if inconsistent:
        warn(f"shared media produced different text per message: {sample(inconsistent)}")
    elif reused:
        ok("shared media resolves to identical text on every message that uses it")


# ---------------------------------------------------------------------------
# printed output for eyeballing
# ---------------------------------------------------------------------------


def print_transcripts(contexts: list[dict[str, Any]]) -> None:
    section("4. ALL 8 VOICE TRANSCRIPTS (full)")
    voices = sorted(
        (c for c in contexts if c["media_status"]["media_type"] == "voice"),
        key=lambda c: c["message_id"],
    )
    if not voices:
        print("  (none)")
        return
    for c in voices:
        n = c.get("normalization") or {}
        media = c["media_status"]
        conv = c["conversation_context"]
        who = (conv.get("group_name") if conv.get("kind") == "group"
               else conv.get("display_name") if conv.get("kind") == "business"
               else conv.get("sender_user_id"))
        print()
        print(f"  {c['message_id']}  [{media['media_id']}]  {c['conversation_type']} / {who}")
        print(f"    user={c['user_id']}  sent={c['created_at']:%Y-%m-%d %H:%M}  "
              f"confidence={n.get('transcription_confidence')}  language={n.get('language')}")
        if not c.get("normalized_text"):
            print(f"    NOT TRANSCRIBED - {n.get('processing_error') or 'not attempted'}")
            continue
        wrapped(c["normalized_text"])
        if n.get("notes"):
            print(f"      (notes: {n['notes']})")


def print_image_samples(contexts: list[dict[str, Any]], limit: int = IMAGE_SAMPLE) -> None:
    section(f"5. IMAGE DESCRIPTIONS (sample of {limit})")
    images = sorted(
        (c for c in contexts if c["media_status"]["media_type"] == "image"),
        key=lambda c: c["message_id"],
    )
    seen: set[str] = set()
    shown = 0
    for c in images:
        media_id = c["media_status"]["media_id"]
        if media_id in seen:
            continue
        seen.add(media_id)
        if shown >= limit:
            continue
        shown += 1
        print()
        print(f"  {c['message_id']}  [{media_id}]  file={c['media_status']['file_path']}")
        print("    caption (original message_text):")
        wrapped(" ".join(c["message_text"].split()), indent="      | ")
        print("    description (normalized_text):")
        if not c.get("normalized_text"):
            err = (c.get("normalization") or {}).get("processing_error") or "not attempted"
            print(f"      NOT DESCRIBED - {err}")
        else:
            wrapped(c["normalized_text"])
    print()
    print(f"  ({shown} of {len(seen)} distinct images shown; run with the cache to see all)")


def print_effective_text_example(contexts: list[dict[str, Any]]) -> None:
    section("6. EFFECTIVE TEXT (what Phase 4 will reason over)")
    picks = []
    for kind in ("voice", "image"):
        match = next(
            (c for c in sorted(contexts, key=lambda x: x["message_id"])
             if c["media_status"]["media_type"] == kind and c.get("normalized_text")),
            None,
        )
        if match:
            picks.append(match)
    if not picks:
        print("  (nothing normalized yet)")
        return
    for c in picks:
        print()
        print(f"  {c['message_id']} ({c['media_status']['media_type']}) - effective_text():")
        wrapped(effective_text(c))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ds = load_dataset(argv[0] if argv else None)
    store = NormalizationStore.load()

    print("=" * 78)
    print("PHASE 3 VALIDATION - Multimodal Normalization")
    print("=" * 78)
    print(f"dataset dir : {ds.dataset_dir}")
    print(f"cache file  : {store.path}")
    print(f"cache state : {len(store.entries)} entr(ies), "
          f"{len(store.ok_ids())} normalized, {len(store.failed_ids())} failed")
    if not store.entries:
        print()
        print("  NOTE: the cache is empty - media_normalizer.py has not produced any")
        print("  results yet. Every check below will report the un-normalized state.")

    contexts = apply_normalization(build_all_contexts(ds), store)

    check_coverage(contexts, store)
    check_guard(contexts)
    check_provenance(contexts, store)
    print_transcripts(contexts)
    print_image_samples(contexts)
    print_effective_text_example(contexts)

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
