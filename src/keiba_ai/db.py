"""SQLite 接続と共通ユーティリティ。"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from importlib import resources
from pathlib import Path

from . import config

JST = timezone(timedelta(hours=9))


def now_jst() -> str:
    return datetime.now(JST).isoformat(timespec="seconds")


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    path = path or config.DB_PATH
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=60)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")
    con.executescript(resources.files("keiba_ai").joinpath("schema.sql").read_text(encoding="utf-8"))
    _migrate(con)
    return con


def _migrate(con: sqlite3.Connection) -> None:
    have = {r[1] for r in con.execute("PRAGMA table_info(model_version)")}
    for col, decl in (("status", "TEXT NOT NULL DEFAULT 'active'"), ("adopted_at", "TEXT"), ("comparison", "TEXT")):
        if col not in have:
            con.execute(f"ALTER TABLE model_version ADD COLUMN {col} {decl}")
    con.commit()


def upsert(con: sqlite3.Connection, table: str, row: dict, keys: list[str]) -> None:
    """キー一致時は NULL でない値だけ更新する (後から判明した情報で埋める)。"""
    cols = list(row)
    updates = [c for c in cols if c not in keys]
    sql = f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
    if updates:
        sets = ",".join(f"{c}=COALESCE(excluded.{c},{c})" for c in updates)
        sql += f" ON CONFLICT ({','.join(keys)}) DO UPDATE SET {sets}"
    else:
        sql += f" ON CONFLICT ({','.join(keys)}) DO NOTHING"
    con.execute(sql, [row[c] for c in cols])
