"""過去バックテストの発走10分前オッズは、購入可能な時刻と鮮度だけを採る。"""
import sqlite3

import pytest

from keiba_ai.backtest import _cutoff_odds


def _db():
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript("""
        CREATE TABLE race(race_id TEXT, org TEXT, race_date TEXT, post_time TEXT);
        CREATE TABLE odds_snapshot(
            race_id TEXT, bet_type TEXT, combo TEXT, odds REAL, odds_max REAL,
            captured_at TEXT, is_final INTEGER);
        INSERT INTO race VALUES ('fresh','JRA','2030-03-02','10:30');
        INSERT INTO race VALUES ('stale','JRA','2030-03-02','10:30');
        INSERT INTO race VALUES ('zero','JRA','2030-03-02','10:30');
        INSERT INTO race VALUES ('other','NAR','2030-03-02','10:30');
    """)
    return con


def test_cutoff_odds_excludes_future_old_naive_final_and_zero_prices():
    con = _db()
    con.executemany("INSERT INTO odds_snapshot VALUES (?,?,?,?,?,?,?)", [
        ("fresh", "win", "1", 4.0, None, "2030-03-02T10:10:00+09:00", 0),
        ("fresh", "win", "1", 3.0, None, "2030-03-02T01:18:00+00:00", 0),
        ("fresh", "win", "1", 2.0, None, "2030-03-02T01:21:00+00:00", 0),
        ("fresh", "win", "1", 1.0, None, "2030-03-02T01:19:00+00:00", 1),
        ("fresh", "win", "1", 1.5, None, "2030-03-02T10:19:00", 0),
        ("stale", "win", "1", 8.0, None, "2030-03-02T09:19:00+09:00", 0),
        ("zero", "win", "1", 0.0, None, "2030-03-02T10:19:00+09:00", 0),
        ("other", "win", "1", 10.0, None, "2030-03-02T10:15:00+09:00", 0),
    ])
    odds = _cutoff_odds(con, "JRA")
    assert odds == {("fresh", "win", "1"): (3.0, None)}
    assert _cutoff_odds(con, "NAR") == {("other", "win", "1"): (10.0, None)}


def test_cutoff_odds_accepts_exact_age_boundary_and_only_pre_cutoff_time():
    con = _db()
    con.executemany("INSERT INTO odds_snapshot VALUES (?,?,?,?,?,?,?)", [
        ("fresh", "win", "1", 9.0, None, "2030-03-02T09:20:00+09:00", 0),
        ("fresh", "win", "2", 4.0, None, "2030-03-02T10:20:00+09:00", 0),
        ("fresh", "place", "1", 2.2, 3.3, "2030-03-02T10:20:01+09:00", 0),
    ])
    odds = _cutoff_odds(con, "JRA")
    assert odds[("fresh", "win", "1")][0] == pytest.approx(9)
    assert odds[("fresh", "win", "2")][0] == pytest.approx(4)
    assert ("fresh", "place", "1") not in odds
