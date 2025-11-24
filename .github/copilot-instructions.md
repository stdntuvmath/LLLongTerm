## Repo overview

This repository contains a very small, single-purpose Python script set under `LongTerm/` that is intended to fetch historical daily data (note in-source comment: "get historical day data from yahoo finance"). Key files:

- `LongTerm/LootLoader.py` — top-level runner that imports `LLLib_LongTerm` and contains the intent to fetch Yahoo Finance historical data.
- `LongTerm/LLLib_LongTerm.py` — currently empty (placeholder). This is where reusable functions and helpers should live.

The project has no build system, tests, or dependency manifest in the repo. The environment is expected to be a normal Python interpreter (Windows PowerShell is the developer shell). Keep changes minimal and runnable via `python`.

## What the agent should know and do

- Preserve the thin-script layout: keep a small entrypoint in `LongTerm/LootLoader.py` and put reusable logic in `LongTerm/LLLib_LongTerm.py`.
- `LLLib_LongTerm.py` is currently a placeholder — add functions there (for example `def get_historical(ticker, start, end):`) and call them from `LootLoader.py` rather than putting large blocks of code in the runner.
- Avoid introducing large frameworks or changing repository layout. Adding a small `requirements.txt` is acceptable if new dependencies are needed (e.g., `yfinance`).

## Coding patterns and examples (use these in edits)

- Example: add a helper in `LongTerm/LLLib_LongTerm.py`:

- Example usage in `LongTerm/LootLoader.py` should remain a simple call:
  - from `LLLib_LongTerm` import the helper, call it, and print or return results. Keep CLI parsing minimal (argparse only if required).

## Project-specific conventions

- Single directory (`LongTerm/`) holds the domain code. Keep tests or additional scripts in the same folder if they’re tightly coupled to the logic.
- File-level simplicity: prefer small functions with clear names rather than large nested code blocks in `LootLoader.py`.

## Integration points and dependencies

- The intended external integration is with Yahoo Finance (a library such as `yfinance` or direct web calls). No dependency file is present — if the agent adds third-party packages, also add a `requirements.txt` with pins.

## Developer workflows (what an agent should run)

- Run the script locally (PowerShell):

  python LongTerm\LootLoader.py

- If new dependencies are added, create `requirements.txt` and instruct the user to run (PowerShell):

  python -m pip install -r requirements.txt

## Safety and merge guidance

- If a `.github/copilot-instructions.md` already exists, merge by preserving repository-specific notes above and avoid overwriting any human-written guidance.
- Keep commits small and descriptive: "Add get_historical helper and basic runner wiring".

## What not to guess

- Do not invent missing project configuration (CI, tests) — only add them if requested. Only change the repo layout where absolutely necessary.

## Quick checklist for patches

1. Add helpers to `LongTerm/LLLib_LongTerm.py`.
2. Wire `LongTerm/LootLoader.py` to call the helpers.
3. Add `requirements.txt` if external libs are introduced.
4. Run `python LongTerm\LootLoader.py` to sanity-check.

---
If any section is unclear or you want different level of detail (examples, tests, or CI), tell me which parts to expand and I'll update the file.
