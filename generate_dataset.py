from __future__ import annotations

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


if __name__ == "__main__":
    STATE.sessions = load_sessions()
    print(len(STATE.sessions))
    messages = load_messages_for_session(STATE.sessions[0])
    print(len(messages))
