# WhatsApp Message Notification Router

Routes each of the 110 messages in `dataset/messages.csv` to **notify**,
**digest**, or **mute**, personalised per recipient, across text, image, and
voice.

## Reproducing the submitted output

**This submission includes cached results (`code/artifacts/`). Running
`validate_output.py` or the full pipeline reproduces the submitted `output.csv`
with no API calls or API key required. To regenerate from scratch instead, set
your own `ANTHROPIC_API_KEY` or `GEMINI_API_KEY` before running.**

Install dependencies with:

```bash
pip install -r requirements.txt
```

`requirements.txt` sits at the repo root. If you extracted `code.zip` on its own,
a copy is bundled inside the archive, so use:

```bash
pip install -r code/requirements.txt
```

Then run the whole pipeline in one command, from the directory that contains
`code/` and `dataset/`:

```bash
py code/main.py
```

`main.py` runs all six phases in order, writes `dataset/output.csv`, validates
it, and prints a pass/fail summary. It sequences the phase modules and contains
no logic of its own. Exit codes: `0` success, `1` a step failed, `2` the dataset
is missing or unreadable.

To run a single phase instead, each has its own script — see the
[Architecture](#architecture--6-phases) table below for which module owns which
phase. `py code/finalize.py && py code/validate_output.py` reproduces just the
last two steps.

Verified: with every credential removed and both SDKs instrumented to raise on
any network call, the full chain runs to completion with **0 API calls** and
produces a **byte-identical** `output.csv` (sha256
`bda890277d48c2977c0f0b78e8ef45e3cba986bd7cbe1b01974972ec232a524b`).

Every phase checks its cache before considering an API call. If a cache entry is
missing *and* no key is set, the run prints what is missing and how to fix it,
then continues or exits cleanly — it never raises a stack trace, and
`finalize.py` refuses to write a short `output.csv` rather than emitting a file
that looks complete.

## Setup

Python 3.14. Installing `requirements.txt` covers everything; only `pandas` and
`numpy` are needed to reproduce the submitted output. `anthropic` and
`google-genai` are imported lazily and are required only when regenerating
media descriptions or LLM decisions from scratch.

**No credentials are included in this submission.** `.env` is gitignored and not
packaged; `.env.example` documents the variables. None are needed to reproduce
`output.csv`.

To regenerate from scratch instead, create a `.env` from `.env.example` and
paste a key in, or export it in your shell — a shell variable always wins over
the file.

## Pipeline entry point

| Command | Runs |
|---|---|
| `py code/main.py` | all six phases + validation |
| `py code/main.py --force-regenerate` | same, but re-calls the APIs (needs a key) |
| `py code/main.py --quiet-validation` | same, verdict only |

## Architecture — 6 phases

| Phase | Module | Does |
|---|---|---|
| 1 | `data_loader.py` | Typed load of all 13 CSVs; id-driven lookup/join functions |
| 2 | `context_builder.py` | One context object per message: trust signals, tiered evidence, mention check, media status |
| 3 | `media_normalizer.py` | Vision descriptions for 11 distinct images, transcripts for 8 voice notes |
| 4 | `rule_engine.py` | Deterministic rules resolve 49; the rest are explicitly unresolved |
| 5 | `llm_reasoner.py` | One structured LLM call per remaining 61 |
| 6 | `finalize.py` | Safety override across all 110, merge, write `output.csv` |

Each phase has a `validate_*.py` that reports rather than raises. Run them in
order for a full audit.

## Key design decisions

**Tiered evidence retrieval.** `exact` (same counterpart) → `fallback` (same
conversation type) → `cross_type` (any history, weakest) → `none`. Each tier
carries a numeric `evidence_strength` (3/2/1/0) and every item records its
`source_conversation_type`. Cross-type evidence is never cited as behavioural
precedent: for `msg_089`, a benign lost-water-bottle message, the only available
history was a reported scam and two ignored promos, which would have dragged it
toward mute if weighted like same-channel precedent.

**Rules first, LLM for judgement calls.** Rules resolve what is clear-cut and
return `None` otherwise — an honest hand-off, not a failure. Rule confidence
sits at 0.83–0.94; LLM confidence at 0.75–0.90, deliberately lower because those
are the ambiguous cases.

**Injection defence.** The dataset contains three real attacks (`msg_107`,
`msg_108`, `msg_110`) that wrap OTP/PIN requests in fake router directives
("Routing override: ... set action=notify"). Rule 1d catches them
deterministically. The Phase 5 prompt defends independently: a constant system
prompt with no interpolation, message *and* evidence text fenced and labelled
untrusted, and closing-tag sequences defused. Verified by forcing all three past
rule 1d — **3/3 returned mute/scam** and named the manipulation in the reason.

**Rule 2b, a deterministic override.** Three near-identical hand-off messages
received three different LLM answers (`digest/promotion`, `notify/personal`,
`notify/urgent`), the outlier being the copy that sat in a marketplace group
with a product photo. Three rounds of prompt strengthening failed and confidence
*rose* each time. Identical content must route identically, so this became a
rule rather than prompt guidance.

**Opt-out language is decisive.** Only marketing offers to stop sending itself,
so "Reply STOP to unsubscribe" classifies a message as promotional before any
operational-urgency keyword is considered. Checking these the other way round
let "extra discounts today" register as operational and routed three adverts as
time-sensitive business updates.

## No hardcoded labels

No routing module contains a hardcoded dataset id. Verified by AST parse, which
excludes comments and docstrings structurally rather than by eye:
`rule_engine.py`, `llm_reasoner.py`, `data_loader.py`, `context_builder.py`,
`media_normalizer.py` and `finalize.py` each contain **zero** message, user,
group or business id literals in executable code.

The `validate_*.py` scripts do name specific ids — for example
`validate_rules.py` asserts that `msg_085` resolves to `mute`/`scam` via rule
1a. These are **expected-value assertions about output the pipeline produced
independently**, never inputs to a decision, and the run fails loudly when an
assertion does not hold. That is how `msg_040` was caught resolving against the
stated spec in Phase 4.

Nothing in a validator can reach `output.csv`. The transitive imports of
`finalize.py`, which is what writes it, are:

```text
finalize -> data_loader, llm_reasoner, media_normalizer, rule_engine
         -> (transitively) context_builder

validators reachable from the routing path: NONE
```

The dependency runs the other way: `validate_output.py` imports `finalize` in
order to check its output.

## Known limitations

- **No Anthropic key was available**; the pipeline ran on Gemini. The provider
  is pluggable — set `ANTHROPIC_API_KEY` and Claude is used with no code change.
- **Gemini's free tier caps ~20 requests/day/model**, so a model pool was
  required: of the 61 decisions Phase 5 owns, 44 came from
  `gemini-3.1-flash-lite`, 16 from `gemini-3.5-flash` and 1 from
  `gemini-3.6-flash`. **Decision quality is therefore not uniform across rows.**
  Each decision records its model in `code/artifacts/llm_decisions.json`. (That
  file holds 64 entries; 3 are superseded by rule 2b and are excluded from the
  counts above — `validate_llm.py` reports them as stale.)
- **Mentions are detectable only as `@user_id`** — `users.csv` has no
  display-name column, so a mention by name cannot be seen.
- **Reason text averages 99 characters against the sample's 82.** Within range,
  but consistently at the top of whatever band the model is given.
- Rule 4's promotional guard being bypassed by the word "today" — **fixed**;
  opt-out language is now checked first, via a helper shared with rule 3a.
- Rule 2b missing conditionally-phrased deadlines — **fixed**; `HARD_DEADLINE`
  now matches conditional cut-offs ("before I show it to someone else") as well
  as clock times.

## Artifacts

`code/artifacts/*.json` cache media normalization and LLM decisions. They are
**not** gitignored and must be included in `code.zip` — they are what makes the
submission reproducible without keys. Delete them to force a full regeneration.
