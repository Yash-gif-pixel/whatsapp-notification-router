"""Phase 3 - Multimodal normalization for the Message Notification Router.

Turns the media flagged by Phase 2 into plain text:

* the 11 distinct images behind the 15 ``needs_image_description`` messages are
  described by a vision model;
* the 8 voice notes behind the 8 ``needs_transcription`` messages are
  transcribed by Gemini.

Audio is always Gemini because **the Claude API has no audio input** - its
multimodal content blocks are image, document/PDF and text. Images can go to
either provider: Claude is used when ``ANTHROPIC_API_KEY`` is set, and Gemini
otherwise, so a Gemini-only setup needs no Anthropic key at all. Override with
``ORCHESTRATE_VISION_PROVIDER=anthropic|gemini``.

Results are cached in ``code/artifacts/media_normalization.json`` keyed by
``media_id``, so re-runs cost nothing and the pipeline stays resumable. Media
is de-duplicated before calling: 3 images are shared across 2-3 messages each,
so 23 media messages need only 19 API calls.

Run from the terminal::

    py code/media_normalizer.py                 # normalize everything missing
    py code/media_normalizer.py --force         # ignore the cache and redo
    py code/media_normalizer.py --only img_002,vn_004
    py code/media_normalizer.py --dry-run       # show the plan, call nothing

Secrets come from the environment, or from a gitignored ``.env`` at the repo
root (paste keys there; a real environment variable always wins over it):

    GEMINI_API_KEY      required - transcription, and image description too
                        unless one of the two below is set (GOOGLE_API_KEY
                        is read as an alias)
    GEMINI_API_KEY_2    optional - a second Gemini key; when present, images
                        use it so the two modalities draw on separate quota
    ANTHROPIC_API_KEY   optional - only if you want Claude to do the images

Optional overrides: ``ORCHESTRATE_VISION_PROVIDER``,
``ORCHESTRATE_VISION_MODEL``, ``ORCHESTRATE_AUDIO_MODEL``.
No key is ever read from tracked files, printed, or written to the cache.

A file that fails is recorded as ``normalized_text: None`` plus a
``processing_error`` string; one bad file never aborts the run.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from context_builder import build_all_contexts  # noqa: E402
from data_loader import Dataset, load_dataset  # noqa: E402

__all__ = [
    "NormalizationStore",
    "normalize_media",
    "apply_normalization",
    "effective_text",
    "load_store",
    "load_env_file",
    "DEFAULT_VISION_MODEL",
    "DEFAULT_AUDIO_MODEL",
]

#: Gitignored file at the repo root where API keys are pasted.
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

#: Shown when work is needed but neither a cached result nor a key exists.
#: Shared with llm_reasoner so both phases say the same thing.
NO_CREDENTIALS_HELP = (
    "No API key found and no cached result for {item}.\n"
    "      Set ANTHROPIC_API_KEY or GEMINI_API_KEY to generate fresh results,\n"
    "      or ensure code/artifacts/ is present from the original submission.\n"
    "      See README for setup."
)

#: Claude, for images, when an Anthropic key is present. Vision only - the
#: Claude API accepts no audio input.
DEFAULT_VISION_MODEL = "claude-opus-5"

#: Gemini, for images, when there is no Anthropic key. Gemini is multimodal,
#: so a Gemini-only setup covers both modalities on its own.
#:
#: Pinned rather than the ``gemini-flash-latest`` alias so runs stay
#: reproducible. Verified against this dataset's media for both image and audio
#: input; ``gemini-2.5-flash`` still appears in models.list() but is closed to
#: new keys and 404s on generateContent.
DEFAULT_VISION_MODEL_GEMINI = "gemini-3.6-flash"

#: Gemini, for audio. Overridable because model availability varies per key.
DEFAULT_AUDIO_MODEL = "gemini-3.6-flash"

ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "media_normalization.json"

MAX_ATTEMPTS = 4
RETRY_BASE_SECONDS = 2.0
#: Ceiling on a single sleep, so a bad Retry-After can't stall the run.
MAX_RETRY_SLEEP_SECONDS = 90.0


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

IMAGE_PROMPT = """\
You are preparing a factual description of an image attached to a WhatsApp \
message, so that a downstream notification router can reason about it as text.

Write 2-4 sentences of plain text. Cover, in this order:

1. What type of content this is - choose the closest of: promotional poster, \
screenshot of a payment or QR code, official notice or document, casual photo, \
meme or chain forward, or other (say which).
2. Any text visible in the image - transcribe it, verbatim where short. Include \
amounts, dates, deadlines, phone numbers, URLs and account or UPI handles exactly \
as written.
3. Anything a spam or scam check would need: urgency or threat language, requests \
for payment or personal details, QR codes, shortened or lookalike links, and \
whether any brand logo shown matches the brand the text claims to be from.

Report only what is actually visible. Do not speculate about the sender's intent, \
do not guess at text you cannot read, and do not label the image as a scam or as \
safe - describe, and let the reader judge. If the image is unreadable, say so \
plainly. Reply with the description only, no preamble and no headings."""

VOICE_PROMPT = """\
Transcribe this voice note verbatim. It is a WhatsApp message and may be in \
English, Hindi, or a mix of both, and may include Indian names, places, amounts \
and dates.

Reply with a single JSON object and nothing else:

{
  "transcript": "<verbatim transcript; use the language actually spoken>",
  "language": "<e.g. english, hindi, hinglish>",
  "confidence": "<high or low>",
  "notes": "<brief note on audio quality or anything unclear, or empty string>"
}

Set "confidence" to "low" if the audio is noisy, clipped, heavily accented, \
partly inaudible, or if you had to guess at any meaningful words - a best-effort \
transcript flagged low is far better than a confident guess. Mark anything you \
could not make out as [inaudible] inside the transcript rather than inventing \
words."""


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------


@dataclass
class NormalizationStore:
    """Cache of media_id -> normalization result, persisted as JSON."""

    path: Path = ARTIFACT_PATH
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "NormalizationStore":
        p = Path(path) if path else ARTIFACT_PATH
        if p.is_file():
            with p.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            return cls(path=p, entries=raw.get("entries", {}))
        return cls(path=p, entries={})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "entries": self.entries,
        }
        with self.path.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.write("\n")

    def get(self, media_id: str | None) -> dict[str, Any] | None:
        return None if media_id is None else self.entries.get(str(media_id))

    def put(self, media_id: str, entry: dict[str, Any]) -> None:
        self.entries[str(media_id)] = entry

    # -- summaries ------------------------------------------------------

    def ok_ids(self) -> list[str]:
        return sorted(k for k, v in self.entries.items() if v.get("normalized_text"))

    def failed_ids(self) -> list[str]:
        return sorted(k for k, v in self.entries.items() if not v.get("normalized_text"))


def load_store(path: Path | str | None = None) -> NormalizationStore:
    return NormalizationStore.load(path)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def load_env_file(path: Path | str | None = None) -> list[str]:
    """Populate ``os.environ`` from a ``.env`` file. Returns the names loaded.

    An already-set environment variable always wins, so exporting a key in the
    shell overrides the file. Blank values are skipped, which means an untouched
    template does not shadow a real environment variable. Only names are ever
    returned or logged - never values.
    """
    if os.environ.get("ORCHESTRATE_SKIP_ENV_FILE"):
        # Lets a caller simulate a machine with no credentials at all, which is
        # how the cache-only reproducibility check is run. Without it, .env
        # would quietly put the keys back and the test would prove nothing.
        return []
    p = Path(path) if path else ENV_FILE
    loaded: list[str] = []
    if not p.is_file():
        return loaded
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or not value or os.environ.get(key):
            continue
        os.environ[key] = value
        loaded.append(key)
    return loaded


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_b64(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("ascii")


def _media_type_for(path: Path, default: str) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or default


def _error_entry(media_id: str, kind: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "media_id": media_id,
        "media_kind": kind,
        "normalized_text": None,
        "processing_error": reason,
        "created_at": _now(),
        **extra,
    }


def _is_transient(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = ("rate", "429", "overload", "timeout", "timed out", "503", "502", "500",
               "connection", "unavailable")
    return any(m in text for m in markers)


def _retry_after_seconds(exc: Exception) -> float | None:
    """Server-requested wait, if the error carries one.

    Gemini rate-limit errors say exactly how long to wait, e.g.
    ``'retryDelay': '26s'`` / ``Please retry in 26.9586s``. Honouring that beats
    guessing: plain exponential backoff caps out well below the free tier's
    per-minute window and burns the attempt budget for nothing.
    """
    text = str(exc)
    for pattern in (r"'retryDelay'\s*:\s*'(\d+(?:\.\d+)?)s'",
                    r"retry in (\d+(?:\.\d+)?)s"):
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return None


#: Public alias - later phases reuse this rather than re-parsing quota errors.
retry_after_seconds = _retry_after_seconds


def is_transient(exc: Exception) -> bool:
    """Public alias of the transient-failure test."""
    return _is_transient(exc)


def _retrying(fn, label: str):
    """Run ``fn`` with backoff on transient failures, honouring Retry-After."""
    last: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - recorded, never raised further
            last = exc
            if attempt == MAX_ATTEMPTS or not _is_transient(exc):
                break
            backoff = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
            asked = _retry_after_seconds(exc)
            delay = min(max(backoff, (asked or 0) + 1.0), MAX_RETRY_SLEEP_SECONDS)
            reason = "rate limited" if asked else "transient failure"
            print(f"      {reason} on {label} (attempt {attempt}/{MAX_ATTEMPTS}); "
                  f"retrying in {delay:.0f}s")
            time.sleep(delay)
    raise last  # type: ignore[misc]


# ---------------------------------------------------------------------------
# provider: Claude vision (images)
# ---------------------------------------------------------------------------


def _describe_image_anthropic(path: Path, model: str) -> dict[str, Any]:
    """Describe one image with Claude vision. Raises on unrecoverable failure."""
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    block = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": _media_type_for(path, "image/jpeg"),
            "data": _read_b64(path),
        },
    }
    request: dict[str, Any] = {
        "model": model,
        "max_tokens": 2000,
        "output_config": {"effort": "low"},
        "messages": [{"role": "user", "content": [block, {"type": "text", "text": IMAGE_PROMPT}]}],
    }

    def call_with_fallbacks():
        # Claude Opus 5 safety classifiers can decline; a server-side fallback
        # re-serves the request instead of losing it. Some of these images are
        # scam posters, so this is a realistic path, not a formality.
        return client.beta.messages.create(
            betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request
        )

    try:
        response = _retrying(call_with_fallbacks, path.name)
    except Exception as exc:  # noqa: BLE001
        # If the fallback beta is not enabled on this key, retry plainly rather
        # than failing the file over an optional feature.
        if "fallback" not in str(exc).lower() and "beta" not in str(exc).lower():
            raise
        print(f"      server-side fallbacks unavailable ({exc}); retrying without")
        response = _retrying(lambda: client.messages.create(**request), path.name)

    if response.stop_reason == "refusal":
        category = getattr(getattr(response, "stop_details", None), "category", None)
        raise RuntimeError(f"model declined to describe this image (category={category})")

    text = "\n".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        raise RuntimeError(f"empty description returned (stop_reason={response.stop_reason})")

    return {
        "normalized_text": text,
        "provider": "anthropic",
        "model": getattr(response, "model", model),
        "stop_reason": response.stop_reason,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }


# ---------------------------------------------------------------------------
# provider: Gemini (audio)
# ---------------------------------------------------------------------------


def _gemini_key(purpose: str = "audio") -> str | None:
    """The Gemini key to use for a modality.

    A second key is optional. When one is present it is used for image work, so
    the two modalities draw on separate quota instead of racing for the same
    rate limit.
    """
    primary = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    secondary = os.environ.get("GEMINI_API_KEY_2") or os.environ.get("GEMINI_IMAGE_API_KEY")
    return (secondary or primary) if purpose == "image" else primary


def _vision_provider() -> str:
    """Which provider describes images: ``anthropic`` or ``gemini``.

    Gemini handles both vision and audio, so a Gemini-only setup needs no
    Anthropic key at all. Claude is used for images only when an Anthropic key
    is actually present, or when explicitly demanded.
    """
    override = (os.environ.get("ORCHESTRATE_VISION_PROVIDER") or "").strip().lower()
    if override in ("anthropic", "gemini"):
        return override
    return "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "gemini"


def _describe_image_gemini(path: Path, model: str) -> dict[str, Any]:
    """Describe one image with Gemini vision. Raises on unrecoverable failure."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_gemini_key("image"))
    part = types.Part.from_bytes(
        data=path.read_bytes(), mime_type=_media_type_for(path, "image/jpeg")
    )

    response = _retrying(
        lambda: client.models.generate_content(model=model, contents=[part, IMAGE_PROMPT]),
        path.name,
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("empty description returned")

    return {
        "normalized_text": text,
        "provider": "gemini",
        "model": model,
    }


def _describe_image(path: Path, model: str, provider: str) -> dict[str, Any]:
    """Dispatch image description to the configured provider."""
    if provider == "anthropic":
        return _describe_image_anthropic(path, model)
    return _describe_image_gemini(path, model)


def _transcribe_audio(path: Path, model: str) -> dict[str, Any]:
    """Transcribe one voice note with Gemini. Raises on unrecoverable failure."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_gemini_key())
    part = types.Part.from_bytes(
        data=path.read_bytes(), mime_type=_media_type_for(path, "audio/mpeg")
    )

    def call():
        return client.models.generate_content(
            model=model,
            contents=[part, VOICE_PROMPT],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )

    response = _retrying(call, path.name)
    raw = (response.text or "").strip()
    if not raw:
        raise RuntimeError("empty transcription response")

    transcript, language, confidence, notes = raw, None, "low", ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Near-miss JSON is common (an unquoted key is enough to break the
        # parse). Recover the transcript rather than handing the whole blob
        # downstream as if it were speech - a later phase would otherwise
        # reason over JSON punctuation. Confidence stays low: the response did
        # not come back in the contracted shape.
        field = re.search(r'"transcript"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
        if field:
            transcript = json.loads(f'"{field.group(1)}"')
            notes = ("model returned malformed JSON; transcript field recovered, "
                     "confidence forced low")
            lang = re.search(r'"language"\s*:\s*"([^"]*)"', raw)
            language = lang.group(1) if lang else None
        else:
            notes = "response was not valid JSON; using raw text and flagging low confidence"
    else:
        if isinstance(parsed, dict):
            transcript = str(parsed.get("transcript", "")).strip()
            language = parsed.get("language") or None
            raw_conf = str(parsed.get("confidence", "")).strip().lower()
            confidence = raw_conf if raw_conf in ("high", "low") else "low"
            notes = str(parsed.get("notes") or "")

    if not transcript:
        raise RuntimeError("transcription returned an empty transcript")

    return {
        "normalized_text": transcript,
        "provider": "gemini",
        "model": model,
        "transcription_confidence": confidence,
        "language": language,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def _pending_media(ds: Dataset) -> list[dict[str, Any]]:
    """Distinct media referenced by messages.csv that needs normalizing.

    De-duplicated by media_id: three images are shared across several messages,
    so 23 media messages resolve to 19 distinct files.
    """
    rows: dict[str, dict[str, Any]] = {}
    for _, msg in ds.messages.iterrows():
        kind = str(msg["media_type"] or "")
        if kind not in ("image", "voice"):
            continue
        media_id = msg["media_id"]
        if media_id is None or media_id != media_id:  # NaN guard
            continue
        media_id = str(media_id)
        entry = rows.setdefault(
            media_id,
            {"media_id": media_id, "media_kind": kind, "message_ids": [], "media": ds.get_media(media_id)},
        )
        entry["message_ids"].append(str(msg["message_id"]))
    for entry in rows.values():
        entry["message_ids"].sort()
    return [rows[k] for k in sorted(rows)]


def normalize_media(
    ds: Dataset | None = None,
    *,
    store: NormalizationStore | None = None,
    only: Sequence[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
    vision_model: str | None = None,
    audio_model: str | None = None,
) -> NormalizationStore:
    """Describe/transcribe every pending media file, caching as it goes."""
    ds = ds if ds is not None else load_dataset()
    store = store if store is not None else NormalizationStore.load()

    from_env_file = load_env_file()
    if os.environ.get("ORCHESTRATE_SKIP_ENV_FILE"):
        print("  .env loading disabled (ORCHESTRATE_SKIP_ENV_FILE)")
    elif from_env_file:
        print(f"  loaded from .env : {', '.join(from_env_file)}")
    elif ENV_FILE.is_file():
        print(f"  .env present at {ENV_FILE} but no keys filled in")

    provider = _vision_provider()
    default_vision = (
        DEFAULT_VISION_MODEL if provider == "anthropic" else DEFAULT_VISION_MODEL_GEMINI
    )
    vision_model = vision_model or os.environ.get("ORCHESTRATE_VISION_MODEL") or default_vision
    audio_model = audio_model or os.environ.get("ORCHESTRATE_AUDIO_MODEL") or DEFAULT_AUDIO_MODEL

    pending = _pending_media(ds)
    if only:
        wanted = {s.strip() for s in only if s.strip()}
        pending = [p for p in pending if p["media_id"] in wanted]

    todo = [p for p in pending if force or not (store.get(p["media_id"]) or {}).get("normalized_text")]
    n_images = sum(1 for p in todo if p["media_kind"] == "image")
    n_voice = sum(1 for p in todo if p["media_kind"] == "voice")

    print(f"  distinct media referenced : {len(pending)}")
    print(f"  already normalized (cache): {len(pending) - len(todo)}")
    print(f"  to process now            : {len(todo)}  ({n_images} image, {n_voice} voice)")
    print(f"  images -> {provider:<9} {vision_model}")
    print(f"  audio  -> gemini    {audio_model}")
    if _gemini_key("image") and _gemini_key("image") != _gemini_key("audio"):
        print("  (second Gemini key detected: images and audio use separate quota)")

    if dry_run:
        print("\n  --dry-run: no API calls made. Planned:")
        for item in todo:
            print(f"    {item['media_id']:<8} {item['media_kind']:<6} "
                  f"-> {', '.join(item['message_ids'])}")
        return store

    if not todo:
        print("\n  Nothing to do.")
        return store

    have_vision_key = bool(
        os.environ.get("ANTHROPIC_API_KEY") if provider == "anthropic" else _gemini_key("image")
    )
    have_audio_key = bool(_gemini_key("audio"))
    missing_vision = "ANTHROPIC_API_KEY" if provider == "anthropic" else "GEMINI_API_KEY"
    if n_images and not have_vision_key:
        print(f"\n  [WARN] {missing_vision} is not set - image description will be "
              "recorded as failed.")
    if n_voice and not have_audio_key:
        print("\n  [WARN] GEMINI_API_KEY / GOOGLE_API_KEY is not set - transcription "
              "will be recorded as failed.")
    print()

    for i, item in enumerate(todo, 1):
        media_id = item["media_id"]
        kind = item["media_kind"]
        media = item["media"]
        label = f"[{i}/{len(todo)}] {media_id} ({kind})"

        if media is None:
            store.put(media_id, _error_entry(media_id, kind, "media_id not found in images.csv/voice_notes.csv"))
            print(f"  {label}: FAILED - unresolved media_id")
            continue

        path = Path(media["abs_path"])
        if not path.is_file():
            store.put(media_id, _error_entry(media_id, kind, f"file not found on disk: {media['file_path']}"))
            print(f"  {label}: FAILED - file missing")
            continue

        # No key and nothing cached for this file: say what to do and move on,
        # rather than raising or writing a bogus result.
        if (kind == "image" and not have_vision_key) or (
                kind == "voice" and not have_audio_key):
            print(f"  {label}: SKIPPED")
            print("      " + NO_CREDENTIALS_HELP.format(item=media_id))
            store.put(media_id, _error_entry(
                media_id, kind, "no API key set and no cached result"))
            continue

        print(f"  {label}: {media['file_path']}")
        try:
            if kind == "image":
                result = _describe_image(path, vision_model, provider)
            else:
                result = _transcribe_audio(path, audio_model)
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the run
            store.put(media_id, _error_entry(media_id, kind, f"{type(exc).__name__}: {exc}"))
            print(f"      FAILED - {type(exc).__name__}: {exc}")
        else:
            entry = {
                "media_id": media_id,
                "media_kind": kind,
                "file_path": media["file_path"],
                "message_ids": item["message_ids"],
                "processing_error": None,
                "created_at": _now(),
                **result,
            }
            store.put(media_id, entry)
            preview = " ".join(entry["normalized_text"].split())[:100]
            conf = entry.get("transcription_confidence")
            print(f"      ok{f' (confidence={conf})' if conf else ''}: {preview}...")
        store.save()  # checkpoint after every file so a crash loses at most one

    store.save()
    return store


# ---------------------------------------------------------------------------
# applying results to Phase 2 context objects
# ---------------------------------------------------------------------------


def apply_normalization(
    contexts: list[dict[str, Any]], store: NormalizationStore
) -> list[dict[str, Any]]:
    """Fold normalization results into Phase 2 contexts, in place.

    Sets ``normalized_text`` (and its provenance), clears the matching entry
    from ``completeness.pending_steps``, and flips ``text_usable`` /
    ``safe_for_text_reasoning`` to True once a transcript exists.
    ``message_text`` is never touched - it stays empty for voice notes so a
    later phase can always tell original text from derived text.
    """
    for ctx in contexts:
        media = ctx["media_status"]
        media_id = media.get("media_id")
        if not media_id:
            # No media: the original text is already the text to reason over.
            # Populating normalized_text here means every message carries one,
            # and `normalization.source` still says where it came from.
            ctx["normalized_text"] = ctx["message_text"] or None
            ctx["normalization"] = {
                "status": "ok" if ctx["message_text"].strip() else "empty",
                "media_id": None,
                "media_kind": None,
                "source": "original_text",
                "provider": None,
                "model": None,
                "transcription_confidence": None,
                "language": None,
                "notes": None,
                "processing_error": None,
                "created_at": None,
            }
            continue

        entry = store.get(media_id)
        if entry is None:
            ctx["normalized_text"] = None
            ctx["normalization"] = {"status": "not_attempted", "media_id": media_id}
            continue

        text = entry.get("normalized_text")
        ctx["normalized_text"] = text
        ctx["normalization"] = {
            "status": "ok" if text else "failed",
            "media_id": media_id,
            "media_kind": entry.get("media_kind"),
            "source": "image_description" if entry.get("media_kind") == "image" else "transcription",
            "provider": entry.get("provider"),
            "model": entry.get("model"),
            "transcription_confidence": entry.get("transcription_confidence"),
            "language": entry.get("language"),
            "notes": entry.get("notes"),
            "processing_error": entry.get("processing_error"),
            "created_at": entry.get("created_at"),
        }
        ctx["processing_error"] = entry.get("processing_error")

        media["normalized"] = bool(text)
        media["processing_error"] = entry.get("processing_error")

        completeness = ctx["completeness"]
        if text:
            step = "image_description" if entry.get("media_kind") == "image" else "transcription"
            completeness["pending_steps"] = [
                s for s in completeness["pending_steps"] if s != step
            ]
            completeness["text_usable"] = True
            completeness["safe_for_text_reasoning"] = True
            completeness["context_complete"] = not completeness["pending_steps"]
            media["payload_readable"] = True
    return contexts


def effective_text(ctx: dict[str, Any]) -> str:
    """The text a later phase should reason over: original, then derived.

    For an image, caption and description are both real signal - a scam often
    lives in the mismatch between them - so both are returned, labelled. For a
    voice note only the transcript exists. For a plain text message the two
    fields are the same string, so it is returned once.
    """
    original = (ctx.get("message_text") or "").strip()
    derived = (ctx.get("normalized_text") or "").strip()
    source = (ctx.get("normalization") or {}).get("source")

    if not derived or derived == original or source == "original_text":
        return original or derived

    parts = [original] if original else []
    parts.append(f"[{source or 'media'}] {derived}")
    return "\n\n".join(parts)


def build_normalized_contexts(
    ds: Dataset | None = None, store: NormalizationStore | None = None
) -> list[dict[str, Any]]:
    """Phase 2 contexts for all 110 messages with Phase 3 results folded in."""
    ds = ds if ds is not None else load_dataset()
    store = store if store is not None else NormalizationStore.load()
    return apply_normalization(build_all_contexts(ds), store)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 3 media normalization.")
    parser.add_argument("--force", action="store_true", help="ignore cached results")
    parser.add_argument("--only", default=None, help="comma-separated media ids")
    parser.add_argument("--dry-run", action="store_true", help="show the plan, call nothing")
    parser.add_argument("--dataset-dir", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    print("=" * 78)
    print("PHASE 3 - MEDIA NORMALIZATION")
    print("=" * 78)

    ds = load_dataset(args.dataset_dir)
    store = normalize_media(
        ds,
        only=args.only.split(",") if args.only else None,
        force=args.force,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        print()
        print(f"  cache written to : {store.path}")
        print(f"  normalized       : {len(store.ok_ids())}")
        failed = store.failed_ids()
        print(f"  failed           : {len(failed)}")
        if failed:
            for media_id in failed:
                print(f"    {media_id}: {store.entries[media_id].get('processing_error')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
