"""Message Notification Router - full pipeline in one command.

    py code/main.py

Runs all six phases in order and writes ``dataset/output.csv``, then validates
it. This module sequences the pipeline and nothing else: every step calls into
the phase module that owns it, so there is no logic here to drift out of sync
with the phase scripts.

    [1/7] data_loader          load and check the dataset
    [2/7] context_builder      assemble one context per message
    [3/7] media_normalizer     image descriptions + voice transcripts
    [4/7] rule_engine          deterministic routing
    [5/7] llm_reasoner         LLM for what the rules declined
    [6/7] finalize             safety override, merge, write output.csv
    [7/7] validate_output      final checks

Steps 3 and 5 are cache-first: with ``code/artifacts/*.json`` present the whole
run makes **zero API calls** and needs no API key. Delete those files, or pass
``--force-regenerate``, to rebuild them from the APIs instead - that does need
a key (see README).

Exit codes: 0 success, 1 a step failed, 2 the dataset is missing or unreadable.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

TOTAL_STEPS = 7
_START = time.monotonic()


def step(n: int, title: str) -> None:
    print(f"\n[{n}/{TOTAL_STEPS}] {title}")
    print("-" * 70)


def detail(text: str) -> None:
    print(f"      {text}")


def fail(message: str, code: int = 1) -> int:
    print()
    print("=" * 70)
    print("PIPELINE FAILED")
    print("=" * 70)
    for line in str(message).split("\n"):
        print(f"  {line}")
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the full Message Notification Router pipeline.")
    parser.add_argument("--dataset-dir", default=None,
                        help="override the dataset directory")
    parser.add_argument("--out", default=None,
                        help="write output.csv somewhere other than dataset/")
    parser.add_argument("--force-regenerate", action="store_true",
                        help="ignore caches and re-call the APIs (needs a key)")
    parser.add_argument("--quiet-validation", action="store_true",
                        help="print only the validation verdict, not the report")
    args = parser.parse_args(list(argv) if argv is not None else None)

    print("=" * 70)
    print("MESSAGE NOTIFICATION ROUTER")
    print("=" * 70)

    # Imports are inside main so a missing dependency reports cleanly rather
    # than exploding at module load.
    try:
        from context_builder import build_all_contexts
        from data_loader import load_dataset
        from finalize import build_final_decisions, write_output
        from llm_reasoner import DecisionStore, classify_all_unresolved
        from media_normalizer import (
            NormalizationStore,
            apply_normalization,
            normalize_media,
        )
        from rule_engine import apply_rules_to_all
        import validate_output as vo
    except ImportError as exc:
        return fail(
            f"A required package is missing: {exc}\n"
            f"Install dependencies with:\n"
            f"    pip install -r requirements.txt",
            code=1,
        )

    # ---- 1. dataset --------------------------------------------------
    step(1, "Loading dataset")
    try:
        ds = load_dataset(args.dataset_dir)
    except FileNotFoundError as exc:
        return fail(
            f"{exc}\n"
            f"The dataset/ directory must sit next to code/ and contain the\n"
            f"13 participant CSVs plus media/. Pass --dataset-dir to point\n"
            f"somewhere else.",
            code=2,
        )
    except Exception as exc:  # noqa: BLE001 - a clean message beats a traceback
        return fail(f"Could not read the dataset: {type(exc).__name__}: {exc}", code=2)

    counts = ds.row_counts()
    detail(f"dataset: {ds.dataset_dir}")
    detail(f"messages to route: {counts['messages']}")
    detail(f"history rows: {counts['message_history']}, "
           f"events: {counts['message_events']}")
    if counts["messages"] == 0:
        return fail("messages.csv is empty - nothing to route.", code=2)

    # ---- 2. context --------------------------------------------------
    step(2, "Building context objects")
    contexts = build_all_contexts(ds)
    detail(f"built {len(contexts)} context objects")

    # ---- 3. media ----------------------------------------------------
    step(3, "Normalizing media (cache-first)")
    media_store = NormalizationStore.load()
    cached_media = len(media_store.ok_ids())
    detail(f"cached media results: {cached_media}")
    media_store = normalize_media(ds, store=media_store, force=args.force_regenerate)
    contexts = apply_normalization(contexts, media_store)

    unreadable = [c["message_id"] for c in contexts
                  if not c["completeness"]["safe_for_text_reasoning"]]
    if unreadable:
        return fail(
            f"{len(unreadable)} message(s) still have no usable text: "
            f"{', '.join(unreadable[:8])}\n"
            f"Their media could not be described or transcribed. Restore\n"
            f"code/artifacts/media_normalization.json from the submission, or\n"
            f"set an API key to regenerate. See README.",
            code=1,
        )
    detail(f"all {len(contexts)} messages have usable text")

    # ---- 4. rules ----------------------------------------------------
    step(4, "Applying the rule engine")
    rule_decisions, unresolved = apply_rules_to_all(contexts)
    detail(f"resolved by rules: {len(rule_decisions)}")
    detail(f"left for the LLM : {len(unresolved)}")

    # ---- 5. llm ------------------------------------------------------
    step(5, "LLM reasoning for unresolved messages (cache-first)")
    decision_store = DecisionStore.load()
    classify_all_unresolved(
        contexts, unresolved, store=decision_store, force=args.force_regenerate)

    still_missing = [m for m in unresolved
                     if not (decision_store.get(m) or {}).get("action")]
    if still_missing:
        return fail(
            f"{len(still_missing)} message(s) have no decision: "
            f"{', '.join(still_missing[:8])}\n"
            f"Restore code/artifacts/llm_decisions.json from the submission,\n"
            f"or set an API key to regenerate. See README.",
            code=1,
        )

    # ---- 6. finalize -------------------------------------------------
    step(6, "Safety override, merge, and write output.csv")
    rows, report = build_final_decisions(ds, contexts, decision_store)
    if report["missing"]:
        return fail(
            f"{len(report['missing'])} message(s) have no final decision - "
            f"refusing to write a short output.csv.",
            code=1,
        )
    if report["overrides"]:
        detail(f"safety override changed {len(report['overrides'])} decision(s):")
        for o in report["overrides"]:
            detail(f"  {o['message_id']}: {o['was']} -> {o['now']} (rule {o['rule']})")
    else:
        detail("safety override: no change (all 1a/1d messages already mute/scam)")
    if report["stale_cache_ignored"]:
        detail(f"stale cache entries superseded by rules: "
               f"{len(report['stale_cache_ignored'])}")

    target = write_output(rows, args.out)
    detail(f"wrote {len(rows)} rows -> {target}")

    # ---- 7. validation -----------------------------------------------
    step(7, "Validating output.csv")
    buffer = io.StringIO()
    argv_for_validator = ["--out", str(target)] if args.out else []
    with redirect_stdout(buffer):
        vo.PROBLEMS.clear()
        code = vo.main(argv_for_validator)
    problems = list(vo.PROBLEMS)

    if not args.quiet_validation:
        print(buffer.getvalue().rstrip())
        print()
    if code != 0:
        return fail("validation could not complete - see the report above.", code=1)
    if problems:
        print(f"      validation flagged {len(problems)} item(s):")
        for i, p in enumerate(problems, 1):
            print(f"        {i}. {p}")
    else:
        detail("validation passed with no issues")

    # ---- summary -----------------------------------------------------
    from collections import Counter
    sources = Counter(r["_source"].split(":")[0] for r in rows)
    actions = Counter(r["action"] for r in rows)
    elapsed = time.monotonic() - _START

    print()
    print("=" * 70)
    print("PIPELINE COMPLETE" if not problems else "PIPELINE COMPLETE (with flags)")
    print("=" * 70)
    print(f"  output      : {target}")
    print(f"  rows        : {len(rows)}  "
          f"({sources.get('rules', 0)} via rules, {sources.get('llm', 0)} via LLM)")
    print(f"  actions     : " + ", ".join(
        f"{a}={actions.get(a, 0)}" for a in ("notify", "digest", "mute")))
    print(f"  validation  : {'PASS' if not problems else f'PASS with {len(problems)} flag(s)'}")
    print(f"  elapsed     : {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
