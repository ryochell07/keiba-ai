"""オッズ取得品質を「件数」ではなく同一時刻の出走馬番で判定する。架空データのみ。"""
import sqlite3
from datetime import datetime

import pytest

from keiba_ai.monitor import odds_quality


def _db():
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript("""
        CREATE TABLE race(
            race_id TEXT, org TEXT, race_date TEXT, post_time TEXT, course_code TEXT);
        CREATE TABLE entry(race_id TEXT, umaban INTEGER);
        CREATE TABLE result(race_id TEXT, umaban INTEGER, status TEXT);
        CREATE TABLE odds_snapshot(
            race_id TEXT, bet_type TEXT, combo TEXT, captured_at TEXT,
            odds REAL, is_final INTEGER);
        CREATE TABLE fetch_log(source TEXT, fetched_at TEXT, ok INTEGER);
        CREATE TABLE odds_capture_daily(
            race_date TEXT, org TEXT, n_races INTEGER, any_rate REAL,
            t30_rate REAL, t10_rate REAL, last_rate REAL,
            median_delay_t10_min REAL, p90_delay_t10_min REAL,
            missing_rate REAL, n_errors INTEGER, n_snapshots INTEGER,
            updated_at TEXT, PRIMARY KEY(race_date, org));
    """)
    return con


def _record_odds(con, rid, timestamp, numbers, *, bet="win", final=0):
    con.executemany(
        "INSERT INTO odds_snapshot VALUES (?,?,?,?,10.0,?)",
        [(rid, bet, str(u), timestamp, final) for u in numbers],
    )


def test_capture_quality_requires_every_active_horse_same_timestamp():
    con = _db()
    for rid in ("complete", "missing"):
        con.execute("INSERT INTO race VALUES (?,'JRA','2030-03-02','10:30','05')", (rid,))
        con.executemany("INSERT INTO entry VALUES (?,?)", [(rid, u) for u in range(1, 9)])
    # 完全取得のレース。古い部分取得も存在するが、後続の全頭同時刻を採用。
    _record_odds(con, "complete", "2030-03-02T10:12:00+09:00", range(1, 8))
    _record_odds(con, "complete", "2030-03-02T10:16:00+09:00", range(1, 9))
    # 1～8番のうち8番が未取得、9番は取消馬。件数8でも「全頭取得」ではない。
    con.execute("INSERT INTO entry VALUES ('missing',9)")
    con.execute("INSERT INTO result VALUES ('missing',9,'scratched')")
    _record_odds(con, "missing", "2030-03-02T10:16:00+09:00", [1,2,3,4,5,6,7,9])
    # 確定オッズがあっても発売中の完全取得として数えない。
    _record_odds(con, "missing", "final", range(1, 9), final=1)
    now = datetime.fromisoformat("2030-03-02T11:00:00+09:00")
    out = odds_quality(con, "2030-03-02", "JRA", now=now)
    assert out["n_races"] == 2
    assert out["any_rate"] == pytest.approx(0.5)
    assert out["t10_rate"] == pytest.approx(0.5)
    assert out["t30_rate"] == 0
    assert out["n_snapshots"] == 1
    assert out["missing_rate"] == pytest.approx(1 / 16)


def test_capture_quality_excludes_nonpositive_and_poststart_values():
    con = _db()
    con.execute("INSERT INTO race VALUES ('r','NAR','2030-03-02','10:30','N大井')")
    con.executemany("INSERT INTO entry VALUES ('r',?)", [(u,) for u in (1,2)])
    _record_odds(con, "r", "2030-03-02T10:18:00+09:00", [1])
    con.execute("INSERT INTO odds_snapshot VALUES ('r','win','2','2030-03-02T10:18:00+09:00',0,0)")
    _record_odds(con, "r", "2030-03-02T10:31:00+09:00", [1,2])
    out = odds_quality(con, "2030-03-02", "NAR",
                       now=datetime.fromisoformat("2030-03-02T11:00:00+09:00"))
    assert out["any_rate"] == 0
    assert out["t10_rate"] == 0
    assert out["n_snapshots"] == 0
    assert out["missing_rate"] == pytest.approx(0.5)
