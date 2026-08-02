# HackerRank Orchestrate — Message Notification Router

> ## Submission
>
> **Solution documentation: [`code/README.md`](./code/README.md)** — architecture,
> design decisions, and known limitations.
>
> Run everything with:
>
> ```bash
> pip install -r requirements.txt
> py code/main.py
> ```
>
> Predictions are written to `dataset/output.csv` (110 rows). Cached results are
> bundled in `code/artifacts/`, so the run reproduces the submitted output with
> **no API calls and no API key**.
>
> ### Note on `dataset/`
>
> **The dataset is not included in this repository.** It is provided by the
> HackerRank Orchestrate challenge and is not redistributed here, since its
> licensing for public republication has not been confirmed. The layout below
> documents exactly what the pipeline expects; drop the challenge's `dataset/`
> folder in beside `code/` and everything runs as described.

---

Challenge background follows, from the starter repository. The repository layout
section has been updated to reflect the submitted structure; the rest is
unchanged.

## About the challenge

Starter repository for the **HackerRank Orchestrate** 24-hour hackathon.

## Message Notification Router

Build an AI-powered system for WhatsApp that decides which messages deserve immediate attention, which should wait, and which should be muted.

The system must reason over multimodal messages, including text messages, image posters/screenshots, and voice notes.

WhatsApp is noisy. A user can receive family chats, society notices, school updates, co-worker messages, business account promotions, image posters, voice notes, and scams in the same message stream. Treating every message the same creates two bad outcomes: important messages get missed, and unwanted or risky messages interrupt the user.

Read [`problem_statement.md`](./problem_statement.md) for the full task spec, input/output schema, allowed values, and submission format.

---

## Repository Layout

```text
.
├── AGENTS.md                         # Rules for AI coding tools + transcript logging
├── problem_statement.md              # Full challenge statement
├── README.md                         # You are here
├── requirements.txt                  # Python dependencies
├── .env.example                      # Template for API keys (none needed to reproduce)
├── code/
│   ├── README.md                     # SOLUTION DOCUMENTATION - start here
│   ├── main.py                       # Entry point: runs all 6 phases end to end
│   ├── data_loader.py                # Phase 1: typed loading + lookup functions
│   ├── context_builder.py            # Phase 2: per-message context assembly
│   ├── media_normalizer.py           # Phase 3: image descriptions + transcripts
│   ├── rule_engine.py                # Phase 4: deterministic routing rules
│   ├── llm_reasoner.py               # Phase 5: LLM for unresolved messages
│   ├── finalize.py                   # Phase 6: safety override, merge, write output
│   ├── validate_*.py                 # One validation script per phase
│   └── artifacts/                    # Cached media + LLM results (enables no-key reruns)
└── dataset/
    ├── messages.csv                  # Messages to route
    ├── output.csv                    # SUBMISSION: 110 predictions
    ├── sample_messages.csv           # Solved examples
    ├── users.csv                     # User notification behavior
    ├── groups.csv                    # Group metadata
    ├── group_members.csv             # User-group relationships
    ├── business_accounts.csv         # Business sender metadata
    ├── user_business_history.csv     # User-business history
    ├── message_history.csv           # Historical messages
    ├── message_events.csv            # User reactions to historical messages
    ├── images.csv                    # Image IDs and media file paths
    ├── voice_notes.csv               # Voice note IDs and media file paths
    ├── daily_notification_summary.csv
    └── media/
        ├── images/
        └── audio/
```

---

## What You Need to Build

For every row in `dataset/messages.csv`, produce one row in `output.csv` with:

| Column | Meaning |
|---|---|
| `message_id` | Incoming message ID |
| `action` | One of `notify`, `digest`, or `mute` |
| `message_type` | Best-fit message category |
| `reason` | Short human-readable explanation |
| `confidence` | Number from `0` to `1` |
| `evidence_message_ids` | Historical message IDs used as evidence; write `none` if there is no useful evidence |

Your system should make personalized decisions using the provided message, user, group, business, media, and historical interaction data.
For image and voice-note messages, `images.csv` and `voice_notes.csv` only provide file paths; your system should inspect the media files themselves.

---

## Suggested Workflow

1. Inspect `dataset/sample_messages.csv` to understand the expected output format.
2. Load `dataset/messages.csv` and all relevant context files.
3. Build your routing system using any approach: LLMs, retrieval, rules, classifiers, agents, or hybrids.
4. Write predictions to `output.csv`.
5. Evaluate your approach on the solved sample rows before submitting.

You may use any language or runtime. Python, JavaScript, and TypeScript are all reasonable choices.

---

## Requirements

Your solution must:

- be runnable from the terminal
- read the provided files from `dataset/`
- produce a valid `output.csv`
- include one prediction for every `message_id` in `dataset/messages.csv`
- not use organizer-only files or hardcoded labels

If you use API keys or secrets, read them from environment variables. Never hardcode secrets in the repo.

---

## Evaluation

Your `output.csv` will be compared against hidden ground-truth labels.

The scoring will consider:

- correctness of `action`
- correctness of `message_type`
- usefulness and consistency of `reason`
- whether `evidence_message_ids` point to relevant historical messages
- reasonable confidence calibration

Strong systems will combine retrieval, structured metadata, behavioral history, safety checks, OCR/ASR handling, and contextual reasoning.

---

## Chat Transcript Logging

This repo includes an [`AGENTS.md`](./AGENTS.md) file for AI coding tools. It asks compatible tools to append conversation summaries to:

| Platform | Path |
|---|---|
| macOS / Linux | `$HOME/hackerrank_orchestrate_august26/log.txt` |
| Windows | `%USERPROFILE%\hackerrank_orchestrate_august26\log.txt` |

Upload this log as your chat transcript at submission time. Do not paste secrets into the chat.

---

## Submission

Submit the following files as instructed by HackerRank:

1. **Code zip**: full runnable solution, prompts/configs, README, and any evaluation files.
2. **Predictions CSV**: final `output.csv` for all rows in `dataset/messages.csv`.
3. **Chat transcript**: the `log.txt` described above.

Before submitting, confirm:

- `output.csv` has one row per row in `dataset/messages.csv`.
- `output.csv` has the exact required columns in the exact required order.
- Your runnable code and setup instructions are included in `code.zip`.
