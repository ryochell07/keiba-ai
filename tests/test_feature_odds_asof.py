"""特徴量に予測時点より未来の市場オッズが入らないことを検証（架空データのみ）。"""
import sqlite3

import pandas as pd
import pytest

from keiba_ai.features import add_market


def _data():
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript("""
        CREATE TABLE race(race_id TEXT, org TEXT);
        CREATE TABLE odds_snapshot (
            race_id TEXT, bet_type TEXT, combo TEXT, captured_at TEXT,
            odds REAL, popularity INTEGER, is_final INTEGER);
        INSERT INTO race VALUES ('r','JRA');
    """)
    rows = [
        ("r", "win", "1", "2030-03-02T09:55:00+09:00", 3.0, 4, 0),
        ("r", "win", "1", "2030-03-02T00:58:00+00:00", 2.7, 3, 0),
        ("r", "win", "1", "2030-03-02T01:01:00+00:00", 2.3, 2, 0),
        ("r", "win", "1", "2030-03-02T10:18:00+09:00", 1.5, 1, 0),
        ("r", "win", "1", "2030-03-02T10:28:00+09:00", 0.9, 1, 0),
        ("r", "win", "1", "2030-03-02T10:15:00", 1.0, 1, 0),
        ("r", "win", "1", "final", 1.1, 1, 1),
        ("r", "win", "1", "2030-03-02T09:59:00+09:00", 1.0, 1, 1),
        ("r", "win", "2", "2030-03-02T10:19:00+09:00", 5.0, 6, 0),
    ]
    con.executemany("INSERT INTO odds_snapshot VALUES (?,?,?,?,?,?,?)", rows)
    frame = pd.DataFrame([
        {"race_id": "r", "org": "JRA", "race_date": "2030-03-02",
         "post_time": "10:30", "umaban": 1},
        {"race_id": "r", "org": "JRA", "race_date": "2030-03-02",
         "post_time": "10:30", "umaban": 2},
    ])
    return con, frame


def test_market_features_do_not_use_future_or_final_quotes():
    con, frame = _data()
    at_1000 = add_market(frame, con, as_of="2030-03-02T10:00:00+09:00")
    assert at_1000.loc[0, "market_win_odds"] == pytest.approx(2.7)
    assert pd.isna(at_1000.loc[1, "market_win_odds"])
    assert at_1000.loc[0, "market_drift"] == pytest.approx(2.7 / 3.0)

    at_1005 = add_market(frame, con, as_of="2030-03-02T10:05:00+09:00")
    assert at_1005.loc[0, "market_win_odds"] == pytest.approx(2.3)
    assert pd.isna(at_1005.loc[1, "market_win_odds"])
    # 学習時は従来と同様に発走10分前までの履歴から特徴量を作る
    historical = add_market(frame, con)
    assert historical.loc[0, "market_win_odds"] == pytest.approx(1.5)
    assert historical.loc[1, "market_win_odds"] == pytest.approx(5.0)


def test_market_features_obey_cutoff_and_reject_naive_asof():
    con, frame = _data()
    at_1029 = add_market(frame, con, as_of="2030-03-02T10:29:00+09:00")
    assert at_1029.loc[0, "market_win_odds"] == pytest.approx(1.5)
    with pytest.raises(ValueError, match="timezone"):
        add_market(frame, con, as_of="2030-03-02T10:00:00")
