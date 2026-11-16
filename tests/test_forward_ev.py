"""発走前評価で欠損オッズを購入可能な期待値と誤認しないための回帰テスト。"""
from keiba_ai.forward import _score


def test_ai_all_races_does_not_evaluate_ev_without_complete_odds():
    horses = [
        {"umaban": 1, "p_win_ai": 0.7, "win_odds": 2.0},
        {"umaban": 2, "p_win_ai": 0.3, "win_odds": None},
    ]
    result = _score(horses, "p_win_ai", {1: 1, 2: 2}, {"1": 200}, evaluate_ev=False)
    assert result["hit"] == 1
    assert result["ev_n"] == 0
    assert result["ev_ret"] == 0


def test_ev_still_available_for_eligible_same_time_odds():
    horses = [
        {"umaban": 1, "p_win_ai": 0.7, "win_odds": 2.0},
        {"umaban": 2, "p_win_ai": 0.3, "win_odds": 4.0},
    ]
    result = _score(horses, "p_win_ai", {1: 1, 2: 2}, {"1": 200})
    assert result["ev_n"] == 2
    assert result["ev_ret"] == 200


def test_place_ev_requires_matching_snapshot_for_every_horse():
    import sqlite3
    from keiba_ai.forward import _place_odds_same_time

    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE odds_snapshot (
        race_id TEXT, bet_type TEXT, combo TEXT, captured_at TEXT,
        odds REAL, is_final INTEGER)""")
    horses = [{"umaban": 1, "place_odds": 1.5}, {"umaban": 2, "place_odds": 2.0}]
    stamp = "2030-03-02T10:00:00+09:00"
    con.executemany("INSERT INTO odds_snapshot VALUES ('r','place',?,?,?,0)",
                    [("1", stamp, 1.5), ("2", stamp, 2.0)])
    assert _place_odds_same_time(con, "r", horses, stamp)
    assert not _place_odds_same_time(con, "r", horses, "2030-03-02T10:01:00+09:00")
    con.execute("UPDATE odds_snapshot SET odds=1.6 WHERE combo='1'")
    assert not _place_odds_same_time(con, "r", horses, stamp)
    con.execute("UPDATE odds_snapshot SET odds=1.5 WHERE combo='1'")
    con.execute("DELETE FROM odds_snapshot WHERE combo='2'")
    assert not _place_odds_same_time(con, "r", horses, stamp)
