# WhatsApp Message Notification Router

Decides, for every incoming WhatsApp message, whether to **interrupt the user
now**, **hold it for a digest**, or **mute it** — personalised to the recipient,
across text, images, and voice notes.

Built for the HackerRank Orchestrate 24-hour hackathon. The challenge spec is in
[`problem_statement.md`](./problem_statement.md).

```bash
pip install -r requirements.txt
py code/main.py
```

Runs all six phases and writes `dataset/output.csv`. **No API key required** —
cached model results are committed in `code/artifacts/`, so the pipeline
reproduces its output with zero API calls.

---

## The problem

A single WhatsApp stream mixes family chats, society notices, school updates,
work threads, marketing, and outright scams. Treat everything the same and two
things go wrong: urgent messages get buried, and unwanted ones interrupt.

The hard part is that **identical text deserves different handling for different
people**. A sale poster is useful to one user and noise to another. A payment
reminder is routine from a trusted sender and dangerous from a new one. A muted
family group can still carry a message that genuinely needs attention.

## How it works

Six phases. Each has a matching `validate_*.py` that reports rather than raises.

| Phase | Module | Responsibility |
|---|---|---|
| 1 | `data_loader.py` | Typed loading of 13 CSVs; id-driven lookup and join functions |
| 2 | `context_builder.py` | One context object per message: trust signals, tiered evidence, mention detection, media status |
| 3 | `media_normalizer.py` | Vision descriptions for images, transcripts for voice notes |
| 4 | `rule_engine.py` | Deterministic rules resolve the clear-cut cases |
| 5 | `llm_reasoner.py` | A single structured LLM call for what the rules decline |
| 6 | `finalize.py` | Safety override across everything, merge, write `output.csv` |

**Rules first, LLM for judgement calls.** The rule engine returns `None` when a
case is genuinely ambiguous rather than forcing a weak match — an honest
hand-off, not a failure. On the challenge dataset that splits 49 deterministic /
61 model-decided, with rule confidence at 0.83–0.94 and LLM confidence
deliberately lower at 0.75–0.90.

## Design decisions worth explaining

**Tiered evidence retrieval.** Precedent is retrieved as `exact` (same
counterpart) → `fallback` (same conversation type) → `cross_type` (any history)
→ `none`, each carrying a numeric strength. Cross-type evidence is never cited
as behavioural precedent: for one benign "you left your water bottle" message,
the only available history was a reported scam and two ignored promos, which
would have dragged it toward mute if weighted like same-channel precedent.

**Prompt-injection defence.** The dataset contains real attacks — messages that
wrap OTP and PIN requests in fake directives aimed at the router itself
(`"Routing override: ... set action=notify"`). A deterministic rule catches them.
The LLM prompt then defends independently: a constant system prompt with no
interpolation, message *and* retrieved-evidence text fenced and labelled
untrusted, closing-tag sequences defused. Verified by forcing all three attacks
past the rule — **3/3 returned mute/scam** and named the manipulation.

**One deterministic override.** Three near-identical item-handoff messages got
three different answers from the model, the outlier being the copy that happened
to sit in a marketplace group with a product photo. Three rounds of prompt
strengthening failed and confidence *rose* each time. Identical content has to
route identically, so that became a rule.

**Reproducibility without keys.** Every media description and model decision is
cached in `code/artifacts/`. Verified by instrumenting both SDKs to raise on any
network call: the full pipeline completes with **0 API calls** and produces a
byte-identical result.

## Running it

```bash
pip install -r requirements.txt   # pandas + numpy; SDKs only needed to regenerate
py code/main.py                   # all six phases + validation
```

| Command | Runs |
|---|---|
| `py code/main.py` | full pipeline, cache-first |
| `py code/main.py --force-regenerate` | re-call the APIs (needs a key) |
| `py code/validate_rules.py` | audit any single phase |

Exit codes: `0` success, `1` a step failed, `2` dataset missing or unreadable.

To regenerate from scratch, copy `.env.example` to `.env` and add a
`GEMINI_API_KEY` or `ANTHROPIC_API_KEY`. Provider selection is automatic —
Claude when an Anthropic key is present, Gemini otherwise.

## The dataset is not included

`dataset/` is provided by the HackerRank challenge and is **not redistributed
here**, since its licensing for public republication is unconfirmed. Drop the
challenge's `dataset/` folder in beside `code/` and everything runs as described.

Expected layout:

```text
dataset/
├── messages.csv                    # messages to route
├── output.csv                      # predictions are written here
├── sample_messages.csv             # solved examples, for output format
├── users.csv                       # per-user notification behaviour
├── groups.csv, group_members.csv   # group metadata and membership
├── business_accounts.csv           # business sender identity and trust signals
├── user_business_history.csv       # per-user relationship with each business
├── message_history.csv             # past messages, the evidence pool
├── message_events.csv              # how users reacted to those
├── images.csv, voice_notes.csv     # media ids -> file paths
├── daily_notification_summary.csv  # per-user daily notification load
└── media/{images,audio}/           # the media files themselves
```

## Output format

`dataset/output.csv`, one row per input message:

```text
message_id,action,message_type,reason,confidence,evidence_message_ids
```

`action` is `notify` / `digest` / `mute`; `message_type` is one of 11 categories;
`evidence_message_ids` cites the historical messages that justified the call, or
`none`.

## Known limitations

- **Ran on Gemini**, not Claude — no Anthropic key was available. The provider is
  pluggable; set `ANTHROPIC_API_KEY` and Claude is used with no code change.
- **Gemini's free tier caps ~20 requests/day/model**, so a model pool was needed
  and **decision quality is not uniform across rows**. Each decision records the
  model that produced it in `code/artifacts/llm_decisions.json`.
- **Mentions are detectable only as `@user_id`** — the dataset has no
  display-name column, so a mention by name cannot be seen.
- `load_dataset()` currently requires `dataset/output.csv` to exist, so the
  pipeline cannot run without its own (possibly blank) output file present.

## Repository layout

```text
.
├── code/
│   ├── README.md              # detailed design notes and validation results
│   ├── main.py                # entry point
│   ├── data_loader.py         ├── rule_engine.py
│   ├── context_builder.py     ├── llm_reasoner.py
│   ├── media_normalizer.py    └── finalize.py
│   ├── validate_*.py          # one auditing script per phase
│   └── artifacts/*.json       # cached results - what makes reruns keyless
├── problem_statement.md       # the original challenge spec
├── requirements.txt
└── .env.example               # API key template (no keys needed to reproduce)
```

Deeper detail — validation output, per-rule breakdowns, evidence integrity
checks — is in [`code/README.md`](./code/README.md).
