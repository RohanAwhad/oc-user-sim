from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from anthropic import AnthropicVertex

VERTEX_PROJECT_ID = os.environ["ANTHROPIC_VERTEX_PROJECT_ID"]
VERTEX_REGION = os.environ.get(
    "VERTEX_LOCATION", os.environ.get("CLOUD_ML_REGION", "global")
)
LLM_CLIENT = AnthropicVertex(project_id=VERTEX_PROJECT_ID, region=VERTEX_REGION)
SONNET_MODEL = "claude-sonnet-4-5@20250929"
QUALITY_GATE_MAX_TURNS = 6
QUALITY_GATE_THINKING_BUDGET = 1024
QUALITY_GATE_REPAIR_MAX_TOKENS = 256
QUALITY_GATE_SYSTEM_PROMPT = """You filter Opencode conversation sessions for a user-simulator training dataset.

Think carefully about whether the transcript is useful training data for modeling a real user's next message.

Approve only when all of these are true:
- the visible turns form a coherent real human<->assistant conversation
- the user is making real requests, clarifications, or follow-ups
- the assistant replies are visible natural-language responses, not tool noise
- the first few turns provide a useful interaction pattern for training

Reject when any of these are true:
- the transcript is malformed, truncated, duplicated, or mostly boilerplate
- the visible turns are dominated by pasted logs, giant tables, or raw artifacts with little conversational value
- the conversation is too fragmentary or contextless to judge well
- the session is mostly workflow noise, synthetic control text, or otherwise poor training data

Return exactly one minified JSON object and nothing else.
Schema:
{"approval":"APPROVE|REJECT","reason":"string"}

The `approval` field must be either `APPROVE` or `REJECT`.
The `reason` field must be a short plain-English explanation."""
QUALITY_GATE_REPAIR_SYSTEM_PROMPT = """You repair malformed quality-gate outputs.

Return exactly one minified JSON object and nothing else.
Schema:
{"approval":"APPROVE|REJECT","reason":"string"}
"""


@dataclass
class State:
    opencode_db_path: Path = (
        Path.home() / ".local" / "share" / "opencode" / "opencode.db"
    )
    sessions: list[dict[str, Any]] = field(default_factory=list)
    messages_by_session_id: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict
    )


STATE = State()


def load_sessions() -> list[dict[str, Any]]:
    with sqlite3.connect(STATE.opencode_db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT *
            FROM session
            ORDER BY time_created
            """
        ).fetchall()

    sessions = [dict(row) for row in rows]
    return sessions


def load_messages_for_session(session: dict[str, Any]) -> list[dict[str, Any]]:
    session_id = session["id"]

    with sqlite3.connect(STATE.opencode_db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT *
            FROM message
            WHERE session_id = ?
            ORDER BY time_created, id
            """,
            (session_id,),
        ).fetchall()

    messages = [dict(row) for row in rows]
    return messages


def drop_non_text_assistant_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    assistant_message_ids = []

    for message in messages:
        message_data = json.loads(message["data"])
        if message_data["role"] == "assistant":
            assistant_message_ids.append(message["id"])

    if not assistant_message_ids:
        return list(messages)

    placeholders = ", ".join("?" for _ in assistant_message_ids)

    with sqlite3.connect(STATE.opencode_db_path) as connection:
        rows = connection.execute(
            f"""
            SELECT DISTINCT message_id
            FROM part
            WHERE message_id IN ({placeholders})
              AND json_extract(data, '$.type') = 'text'
            """,
            assistant_message_ids,
        ).fetchall()

    assistant_text_message_ids = {row[0] for row in rows}
    filtered_messages = []

    for message in messages:
        message_data = json.loads(message["data"])
        if message_data["role"] != "assistant":
            filtered_messages.append(message)
            continue

        if message["id"] in assistant_text_message_ids:
            filtered_messages.append(message)

    return filtered_messages


def get_message_content(message: dict[str, Any]) -> str:
    with sqlite3.connect(STATE.opencode_db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT data
            FROM part
            WHERE message_id = ?
            ORDER BY time_created, id
            """,
            (message["id"],),
        ).fetchall()

    text_parts = []
    for row in rows:
        part_data = json.loads(row["data"])
        if part_data["type"] == "text":
            text_parts.append(part_data["text"])

    return "".join(text_parts)


def merge_adjacent_same_role_turns(
    turns: list[dict[str, str]],
) -> list[dict[str, str]]:
    merged_turns = []

    for turn in turns:
        if not merged_turns:
            merged_turns.append(dict(turn))
            continue

        previous_turn = merged_turns[-1]
        if previous_turn["role"] != turn["role"]:
            merged_turns.append(dict(turn))
            continue

        previous_turn["content"] = f"{previous_turn['content']}\n\n{turn['content']}"

    return merged_turns


def do_have_more_than_one_user_message(turns: list[dict[str, str]]) -> bool:
    user_message_count = 0

    for turn in turns:
        if turn["role"] == "user":
            user_message_count += 1
            if user_message_count > 1:
                return True
    return False


def convert_messages_to_turns(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    filtered_messages = drop_non_text_assistant_messages(messages)
    turns = []

    for message in filtered_messages:
        message_data = json.loads(message["data"])
        content = get_message_content(message)
        if not content:
            continue

        turns.append(
            {
                "role": message_data["role"],
                "content": content,
            }
        )

    return turns


def get_anthropic_text_response(response: Any) -> str:
    text_blocks = []

    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            text_blocks.append(text)

    return "".join(text_blocks)


def extract_json_object(response_text: str) -> str:
    start_index = response_text.find("{")
    end_index = response_text.rfind("}")

    if start_index == -1 or end_index == -1 or end_index < start_index:
        raise ValueError("No JSON object found in quality gate response")

    return response_text[start_index : end_index + 1]


def parse_quality_gate_response(response_text: str) -> dict[str, str]:
    payload = json.loads(extract_json_object(response_text))
    approval = payload.get("approval")
    reason = payload.get("reason")

    if approval not in {"APPROVE", "REJECT"}:
        raise ValueError(f"Invalid approval value: {approval!r}")

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Quality gate response must include a non-empty reason")

    return {
        "approval": approval,
        "reason": reason,
    }


def repair_quality_gate_response(
    turns: list[dict[str, str]], response_text: str
) -> dict[str, str]:
    selected_turns = turns[:QUALITY_GATE_MAX_TURNS]
    prompt = "\n\n---\n\n".join(
        [f"{turn['role'].upper()}:\n{turn['content']}" for turn in selected_turns]
    )
    repair_input = (
        "Conversation:\n"
        f"{prompt}\n\n"
        "Malformed model output:\n"
        f"{response_text or '<empty>'}"
    )

    repair_response = LLM_CLIENT.messages.create(
        model=SONNET_MODEL,
        max_tokens=QUALITY_GATE_REPAIR_MAX_TOKENS,
        temperature=0,
        system=QUALITY_GATE_REPAIR_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": repair_input},
        ],
    )
    repair_text = get_anthropic_text_response(repair_response).strip()
    return parse_quality_gate_response(repair_text)


def does_sonnet_approve(turns: list[dict[str, str]]) -> bool:
    selected_turns = turns[:QUALITY_GATE_MAX_TURNS]
    prompt = "\n\n---\n\n".join(
        [f"{turn['role'].upper()}:\n{turn['content']}" for turn in selected_turns]
    )

    response = LLM_CLIENT.messages.create(
        model=SONNET_MODEL,
        max_tokens=4096,
        temperature=1,
        system=QUALITY_GATE_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": prompt},
        ],
        thinking={
            "type": "enabled",
            "budget_tokens": QUALITY_GATE_THINKING_BUDGET,
        },
    )
    response_text = get_anthropic_text_response(response).strip()
    try:
        quality_gate_response = parse_quality_gate_response(response_text)
    except (json.JSONDecodeError, ValueError):
        quality_gate_response = repair_quality_gate_response(turns, response_text)
    return quality_gate_response["approval"] == "APPROVE"


def reverse_roles_in_turns(turns: list[dict[str, str]]) -> list[dict[str, str]]:
    reversed_turns = []

    for turn in turns:
        reversed_role = turn["role"]
        if turn["role"] == "user":
            reversed_role = "assistant"
        elif turn["role"] == "assistant":
            reversed_role = "user"

        reversed_turns.append(
            {
                "role": reversed_role,
                "content": turn["content"],
            }
        )

    return reversed_turns


if __name__ == "__main__":
    STATE.sessions = load_sessions()
    print("Number of sessions:", len(STATE.sessions))

    for session in STATE.sessions:
        messages = load_messages_for_session(session)
        print("Number of messages:", len(messages))
        turns = convert_messages_to_turns(messages)
        print("Number of turns:", len(turns))
        turns = merge_adjacent_same_role_turns(turns)
        print("Number of turns post merge:", len(turns))
        for turn in turns:
            print(f"{turn['role']}: {turn['content']}")
            print()

        if not do_have_more_than_one_user_message(turns):
            print("Do not have more than one user message")
            continue

        if not does_sonnet_approve(turns):
            print("Sonnet did not approve")
            continue

        reversed_turns = reverse_roles_in_turns(turns)
        print("Number of reversed turns:", len(reversed_turns))
        for turn in reversed_turns:
            print(f"{turn['role']}: {turn['content']}")
            print()

        print("--------------------------------")
        exit()
