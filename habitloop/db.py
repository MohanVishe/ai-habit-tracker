"""SQLite storage.

One file, no server, no ORM. A habit tracker is a few thousand rows at most —
anything heavier is infrastructure for its own sake.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

DEFAULT_DB = Path("data/habits.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS habits (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    category    TEXT    NOT NULL DEFAULT 'general',
    target_per_week INTEGER NOT NULL DEFAULT 7,
    created_on  TEXT    NOT NULL,
    archived    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS entries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    habit_id    INTEGER NOT NULL REFERENCES habits(id) ON DELETE CASCADE,
    on_date     TEXT    NOT NULL,
    done        INTEGER NOT NULL DEFAULT 1,
    note        TEXT,
    UNIQUE(habit_id, on_date)
);

CREATE INDEX IF NOT EXISTS idx_entries_date ON entries(on_date);
"""


@contextmanager
def connect(db_path: Path | str = DEFAULT_DB):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init(db_path: Path | str = DEFAULT_DB) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


# --- habits ---------------------------------------------------------------


def add_habit(name: str, category: str = "general", target_per_week: int = 7,
              db_path: Path | str = DEFAULT_DB) -> int:
    with connect(db_path) as conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO habits (name, category, target_per_week, created_on) "
            "VALUES (?, ?, ?, ?)",
            (name.strip(), category, target_per_week, date.today().isoformat()),
        )
        if cursor.lastrowid:
            return cursor.lastrowid
        row = conn.execute("SELECT id FROM habits WHERE name = ?", (name.strip(),)).fetchone()
        return row["id"]


def list_habits(include_archived: bool = False, db_path: Path | str = DEFAULT_DB) -> list[dict]:
    query = "SELECT * FROM habits"
    if not include_archived:
        query += " WHERE archived = 0"
    query += " ORDER BY name"
    with connect(db_path) as conn:
        return [dict(r) for r in conn.execute(query).fetchall()]


def archive_habit(habit_id: int, db_path: Path | str = DEFAULT_DB) -> None:
    with connect(db_path) as conn:
        conn.execute("UPDATE habits SET archived = 1 WHERE id = ?", (habit_id,))


# --- entries --------------------------------------------------------------


def log(habit_id: int, on_date: date | str, done: bool = True, note: str | None = None,
        db_path: Path | str = DEFAULT_DB) -> None:
    """Record a day. Re-logging the same day overwrites rather than duplicates."""
    iso = on_date.isoformat() if isinstance(on_date, date) else on_date
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO entries (habit_id, on_date, done, note) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(habit_id, on_date) DO UPDATE SET done = excluded.done, note = excluded.note",
            (habit_id, iso, int(done), note),
        )


def entries_between(start: date, end: date, db_path: Path | str = DEFAULT_DB) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT e.on_date, e.done, e.note, h.id AS habit_id, h.name, h.category, "
            "       h.target_per_week "
            "FROM entries e JOIN habits h ON h.id = e.habit_id "
            "WHERE e.on_date BETWEEN ? AND ? "
            "ORDER BY e.on_date",
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    return [dict(r) for r in rows]


def recent(days: int = 30, db_path: Path | str = DEFAULT_DB) -> list[dict]:
    today = date.today()
    return entries_between(today - timedelta(days=days - 1), today, db_path)
