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
    print(len(sessions))
    return sessions


if __name__ == "__main__":
    STATE.sessions = load_sessions()
    sess = STATE.sessions[0]

