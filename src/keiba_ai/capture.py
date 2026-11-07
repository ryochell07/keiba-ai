"""発走前オッズを、どの時点で保存するかの規則。提供元への要求と保存量を抑えるためここで決める。"""
from __future__ import annotations

import sqlite3
from datetime import datetime

CLOSE_MIN = 45         # これより発走に近ければ毎回保存する
FAR_GAP_MIN = 55       # 発走まで遠いレースは約1時間おき
NEAR_GAP_MIN = 4
EXOTIC_GAP_MIN = 30


def _age_min(con: sqlite3.Connection, race_id: str, now: datetime, bets: str) -> float | None:
    row = con.execute(f"SELECT MAX(captured_at) FROM odds_snapshot WHERE race_id=? AND is_final=0 AND bet_type {bets}", (race_id,)).fetchone()
    return (now - datetime.fromisoformat(row[0])).total_seconds() / 60 if row and row[0] else None


def win_place_due(con: sqlite3.Connection, race_id: str, mins_to_post: float, now: datetime) -> bool:
    if mins_to_post <= 0:
        return False
    age = _age_min(con, race_id, now, "IN ('win','place')")
    return age is None or age >= (NEAR_GAP_MIN if mins_to_post <= CLOSE_MIN else FAR_GAP_MIN)


def exotics_due(con: sqlite3.Connection, race_id: str, mins_to_post: float, now: datetime, within_min: int = 15) -> bool:
    """組み合わせ券種は容量が大きいので、発走が近い時に1回だけ保存する。"""
    if not 0 < mins_to_post <= within_min:
        return False
    age = _age_min(con, race_id, now, "NOT IN ('win','place')")
    return age is None or age >= EXOTIC_GAP_MIN
