# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

WealthDesk is an AI banking assistant (Bharat National Bank) built session by session across a 16-session course on Agentic AI Engineering. Each session adds a capability. Only released sessions are in the repo (currently through S1).

## Setup (run once)

```bash
pip install -r requirements.txt
# Copy .env.example to .env and fill in GROQ_API_KEY
python data/seed.py       # Creates data/bnb_data.db
python data/ingest.py     # Populates data/vectorstore/ (ChromaDB)
```

## Common Commands

```bash
# Run the terminal chatbot for the current session
cd s01/starter
python -m wealthdesk.agent

# Run tests for a single session (always run one session at a time)
pytest s01/tests/ -v

# Tests do not need a real API key — the LLM is mocked in conftest.py
```

**Important:** Never run `pytest s01/ s02/` together. Each session defines a `wealthdesk` package in its own directory; running multiple sessions in one pytest invocation causes the wrong module to be patched.

## Architecture

Each session folder (`s01/`, `s02/`, …) contains:
- `starter/` — code students fill in (TODO comments)
- `solution/` — reference implementation
- `tests/` — unit tests that import from `solution/`, not `starter/`

The `wealthdesk` package inside each session has the same five-file layout:

| File | Responsibility |
|---|---|
| `__init__.py` | Calls `load_dotenv()` before any other module loads |
| `config.py` | `SYSTEM_PROMPT`, `MODEL_NAME`, `TEMPERATURE`, `MAX_TOKENS` — no API calls |
| `state.py` | `WealthDeskState` TypedDict (only shape, no logic) |
| `tools.py` | `llm` instance (`ChatGroq`) and `@tool` functions (added in later sessions) |
| `nodes.py` | LangGraph node functions — each takes `WealthDeskState`, returns a partial dict |
| `agent.py` | `build_graph()` + `graph` module-level instance + `run()` terminal loop |

**LangGraph pattern:** Nodes return only the keys they changed; LangGraph merges the partial dict back into the full state. `graph` at module level in `agent.py` is required by `langgraph.json` for LangGraph Studio.

## Data Layer (added progressively)

- **SQLite** (`data/bnb_data.db`) — structured data: loan rates, FD rates, branches, rate history. Source of truth for all numbers. Created by `data/seed.py`.
- **ChromaDB** (`data/vectorstore/`) — unstructured policy documents from `data/documents/*.md`. Created by `data/ingest.py`. Rule: numbers go in SQLite, policy text goes in ChromaDB.

## LLM

- Provider: Groq (`langchain-groq`)
- Model: `meta-llama/llama-4-scout-17b-16e-instruct`
- `GROQ_API_KEY` must be in `.env`. `tools.py` raises `ValueError` at import time if it is missing.
- Tests set a dummy key via `conftest.py` and patch `_nodes.llm` with `unittest.mock` so no real API call is made.

## Key Rules

- Always run scripts from `wealthdesk/` (repo root), not from inside a session subfolder.
- The database is the source of truth for rates — not the system prompt, not the documents.
- `.env`, `data/bnb_data.db`, and `data/vectorstore/` are gitignored and must not be committed.
