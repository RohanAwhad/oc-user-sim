## 2026-03-13

### Dataset extraction bootstrap
- Added `generate_dataset.py` to load Opencode `session` and `message` rows from the local SQLite DB.
- Added visible-text extraction from `part` rows and canonical `{role, content}` turn conversion.
- Added assistant non-text filtering, adjacent same-role turn merging, and `more than one user message` filtering.
- Added per-session conversation builders, reversed-role training dataset rendering, and a Sonnet quality-gate prompt using Anthropic Vertex env vars.
- Updated the Sonnet quality gate to use Vertex-safe extended thinking plus JSON text output parsed locally, since Vertex rejected `output_config` structured output.
- Added a repair pass for malformed or empty Sonnet JSON so the dataset pipeline does not crash on occasional non-JSON Vertex responses.
- Added a JSONL export path that writes reversed-role `{"messages": ...}` rows for `training_hub`-style LoRA SFT inputs to `exports/reversed_messages.jsonl`.
- Replaced the ad hoc `__main__` inspection loop with an end-to-end export flow plus progress and summary stats.
- Added outermost async export orchestration with a semaphore-limited max concurrency of `8` so session filtering and quality-gating can run in parallel.
- Tightened the quality-gate fallback so sessions with malformed empty repair output are rejected instead of crashing the full export.
- Added `train_lora.py`, a minimal `training_hub.lora_sft(...)` launcher pointed at `exports/reversed_messages.jsonl` with `chat_template` + `messages` defaults for the exported reversed dataset.
