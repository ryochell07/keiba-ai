"""期待値計算と注意フラグ。オッズがない組番は期待値を計算せず「オッズ未取得」と明示する。"""
from __future__ import annotations

import sqlite3
from datetime import datetime

from .db import JST
from .providers.base import BET_LABEL

STATUS_NO_ODDS = "オッズ未取得"
STALE_MIN = 60
MIN_VERIFIED_BETS = 500
# JRA の控除率。損益分岐の目安として表示する。
TAKEOUT = {"win": 0.20, "place": 0.20, "bracket_quinella": 0.225, "quinella": 0.225, "wide": 0.225,
           "exacta": 0.25, "trio": 0.25, "trifecta": 0.275, "bracket_exacta": 0.25}


def expected(prob: float, odds: float | None, odds_max: float | None = None) -> tuple[float | None, float | None]:
    """期待値比率 = 確率 × オッズ − 1。複勝・ワイドは下限と上限の 2 値を返す。"""
    if odds is None:
        return None, None
    low = prob * odds - 1
    return low, (prob * odds_max - 1 if odds_max is not None else low)


def latest_odds(con: sqlite3.Connection, race_id: str, at: str | None = None) -> dict[tuple[str, str], dict]:
    """at 以前で最新のオッズ。発走前の表示に確定オッズは混ざらないが、確定後は 'final' を返す。"""
    at = at or datetime.now(JST).isoformat(timespec="seconds")
    rows = con.execute(
        """SELECT bet_type, combo, odds, odds_max, captured_at FROM odds_snapshot
           WHERE race_id=? AND (captured_at='final' OR captured_at<=?)""", (race_id, at)).fetchall()
    out: dict[tuple[str, str], dict] = {}
    for r in sorted(rows, key=lambda x: (x["captured_at"] == "final", x["captured_at"])):
        out[(r["bet_type"], r["combo"])] = dict(r)
    return out


def verified_bet_types(con: sqlite3.Connection, org: str) -> dict[str, dict]:
    """発走前オッズで判定した期待値戦略の、券種別の最新実績。"""
    row = con.execute("SELECT backtest_id FROM backtest_result WHERE org=? AND odds_basis='cutoff' "
                      "ORDER BY created_at DESC LIMIT 1", (org,)).fetchone()
    if not row:
        return {}
    out: dict[str, dict] = {}
    for r in con.execute("SELECT * FROM backtest_result WHERE backtest_id=? AND org=? AND odds_basis='cutoff' AND strategy LIKE 'ev_%'",
                         (row[0], org)):
        cur = out.get(r["bet_type"])
        if cur is None or r["n_bets"] > cur["n_bets"]:
            out[r["bet_type"]] = dict(r)
    return out


def make_flags(prob: float, prob_std: float | None, avg_runs: float | None, odds_row: dict | None, ev: float | None,
               verified: dict | None, now: datetime | None = None, past_roi: float | None = None) -> tuple[list[str], str]:
    """注意表示と信頼度 (A/B/C)。"""
    now = now or datetime.now(JST)
    flags: list[str] = []
    score = 2
    if prob_std is not None and prob > 0 and prob_std / prob > 0.25:
        flags.append("確率の不確実性大")
        score -= 1
    if avg_runs is not None and avg_runs < 3:
        flags.append("サンプル少")
        score -= 1
    if prob < 0.01:
        flags.append("低確率")
        score -= 1
    if odds_row and odds_row.get("captured_at") not in (None, "final"):
        age = (now - datetime.fromisoformat(odds_row["captured_at"])).total_seconds() / 60
        if age > STALE_MIN:
            flags.append(f"オッズが{int(age)}分前")
    if ev is not None and ev > 0:
        ok = (verified and verified["n_bets"] >= MIN_VERIFIED_BETS and verified["odds_basis"] == "cutoff"
              and (verified["roi_ci_low"] or 0) > 1.0)
        if not ok:
            flags.append("長期回収率は未検証")
        if past_roi is not None:
            flags.append(f"同条件の過去回収率 {past_roi:.0%}")
    return flags, ("A" if score >= 2 else "B" if score == 1 else "C")


def label(bet_type: str) -> str:
    return BET_LABEL.get(bet_type, bet_type)
