"""予測台帳のタイムゾーン混在時の重複判定。データはすべて架空。"""
import sqlite3
from datetime import datetime

from keiba_ai.ledger import should_record, time_point


def _db():
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE prediction_run(
        race_id TEXT, time_point TEXT, model_version_id TEXT, input_hash TEXT,
        feature_hash TEXT, predicted_at TEXT)""")
    return con


def test_should_record_chooses_latest_actual_instant_not_latest_iso_string():
    con = _db()
    con.executemany("INSERT INTO prediction_run VALUES (?,?,?,?,?,?)", [
        ("race", "t10", "model", "old-hash", "features", "2030-03-02T10:00:00+09:00"),
        ("race", "t10", "model", "new-hash", "features", "2030-03-02T01:05:00+00:00"),
    ])
    now = datetime.fromisoformat("2030-03-02T10:10:00+09:00")
    # 10:05 JSTの記録が最新なので、同じハッシュなら重複記録しない。
    assert not should_record(con, "race", "t10", "model", "new-hash", "features",
                             now.isoformat(), now)


def test_should_record_ignores_future_predictions_with_different_offsets():
    con = _db()
    con.executemany("INSERT INTO prediction_run VALUES (?,?,?,?,?,?)", [
        ("race", "t10", "model", "correct-hash", "features", "2030-03-02T09:55:00+09:00"),
        ("race", "t10", "model", "future-hash", "features", "2030-03-02T02:00:00+00:00"),
    ])
    now = datetime.fromisoformat("2030-03-02T10:00:00+09:00")
    assert not should_record(con, "race", "t10", "model", "correct-hash", "features",
                             now.isoformat(), now)


def test_race_day_bucket_uses_jst_even_with_utc_prediction_timestamp():
    post = datetime.fromisoformat("2030-03-02T11:00:00+09:00")
    utc_morning = datetime.fromisoformat("2030-03-01T23:30:00+00:00")
    label, minutes = time_point(utc_morning, post, "2030-03-02")
    assert label == "morning"
    assert minutes == 150.0
    previous_day = datetime.fromisoformat("2030-03-01T13:00:00+00:00")
    label, _ = time_point(previous_day, post, "2030-03-02")
    assert label == "day_before"
