from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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


if __name__ == "__main__":
    STATE.sessions = load_sessions()
    print('Number of sessions:', len(STATE.sessions))
    messages = load_messages_for_session(STATE.sessions[4])
    print('Number of messages:', len(messages))
    turns = convert_messages_to_turns(messages)
    print('Number of turns:', len(turns))
    turns = merge_adjacent_same_role_turns(turns)
    print('Number of turns post merge:', len(turns))
    for turn in turns:
        print(f"{turn['role']}: {turn['content']}")
        print()
