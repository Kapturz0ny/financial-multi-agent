"""Per-user daily quota tracking against external LLM providers, backed by SQLite."""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Iterable, Optional

from dotenv import load_dotenv

load_dotenv()

DAILY_QUERY_LIMIT = int(os.getenv("DAILY_QUERY_LIMIT", "10"))
USAGE_DB_PATH = os.getenv("USAGE_DB_PATH", "./data/usage.db")

# Providers that consume the paid budget. Local model is intentionally excluded.
METERED_PROVIDERS: tuple[str, ...] = ("openai", "gemini")

_db_lock = Lock()


def _connect() -> sqlite3.Connection:
    Path(USAGE_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(USAGE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS usage_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            run_date TEXT NOT NULL,
            provider TEXT NOT NULL,
            ticker TEXT,
            mode TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_usage_user_date ON usage_log(username, run_date);
        """
    )
    conn.commit()


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def count_today(
    username: str,
    providers: Iterable[str] = METERED_PROVIDERS,
) -> int:
    """Number of metered analyses run by `username` today (UTC)."""
    providers = tuple(providers)
    placeholders = ",".join("?" * len(providers))
    sql = (
        f"SELECT COUNT(*) AS cnt FROM usage_log "
        f"WHERE username = ? AND run_date = ? AND provider IN ({placeholders})"
    )
    with _db_lock, _connect() as conn:
        _ensure_schema(conn)
        row = conn.execute(sql, (username, _today(), *providers)).fetchone()
        return int(row["cnt"]) if row else 0


def remaining(username: str) -> int:
    """How many more metered analyses the user can run today. Floored at 0."""
    used = count_today(username)
    return max(0, DAILY_QUERY_LIMIT - used)


def record_usage(
    username: str,
    provider: str,
    ticker: Optional[str] = None,
    mode: Optional[str] = None,
) -> None:
    """Append a usage row. Always records, regardless of provider — so we can audit local-model use too."""
    with _db_lock, _connect() as conn:
        _ensure_schema(conn)
        conn.execute(
            "INSERT INTO usage_log (username, run_date, provider, ticker, mode) VALUES (?, ?, ?, ?, ?)",
            (username, _today(), provider.lower(), ticker, mode),
        )
        conn.commit()


def reset_today_usage(
    username: str,
    providers: Iterable[str] = METERED_PROVIDERS,
) -> int:
    """Delete today's metered usage rows for a user and return how many rows were removed."""
    providers = tuple(providers)
    if not providers:
        return 0

    placeholders = ",".join("?" * len(providers))
    sql = (
        f"DELETE FROM usage_log "
        f"WHERE username = ? AND run_date = ? AND provider IN ({placeholders})"
    )
    with _db_lock, _connect() as conn:
        _ensure_schema(conn)
        cursor = conn.execute(sql, (username, _today(), *providers))
        conn.commit()
        return int(cursor.rowcount or 0)
