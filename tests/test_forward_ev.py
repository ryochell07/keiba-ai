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


def test_fresh_odds_rejects_stale_future_and_poststart_quotes():
    from keiba_ai.forward import _fresh_prestart_odds

    base = {
        "odds_status": "complete",
        "odds_same_time": 1,
        "predicted_at": "2030-03-02T10:00:00+09:00",
        "scheduled_post_at": "2030-03-02T10:30:00+09:00",
        "odds_captured_at": "2030-03-02T09:59:00+09:00",
    }
    assert _fresh_prestart_odds(base)
    assert not _fresh_prestart_odds({**base, "odds_captured_at": "2030-03-02T08:59:00+09:00"})
    assert not _fresh_prestart_odds({**base, "odds_captured_at": "2030-03-02T10:01:00+09:00"})
    assert not _fresh_prestart_odds({**base, "odds_captured_at": "2030-03-02T10:31:00+09:00"})
    assert not _fresh_prestart_odds({**base, "odds_status": "partial"})
    assert not _fresh_prestart_odds({**base, "odds_same_time": 0})
    assert not _fresh_prestart_odds({**base, "odds_captured_at": None})
    assert not _fresh_prestart_odds({**base, "predicted_at": "2030-03-02T10:30:00+09:00"})


def test_paired_model_comparison_reports_same_race_deltas():
    import pytest
    from keiba_ai.forward import _paired_comparison

    rows = {
        "ai": [{"ll": 0.6, "brier": 0.3, "hit": 1}, {"ll": 1.4, "brier": 0.8, "hit": 0}],
        "market": [{"ll": 0.8, "brier": 0.4, "hit": 0}, {"ll": 1.2, "brier": 0.6, "hit": 1}],
        "blend": [{"ll": 0.7, "brier": 0.3, "hit": 1}, {"ll": 1.1, "brier": 0.5, "hit": 1}],
    }
    result = _paired_comparison(rows)
    assert result["n_races"] == 2
    assert result["sufficient_sample_for_screening"] is False
    assert result["minimum_races_for_screening"] == 100
    assert result["blend_minus_market"]["log_loss"] == pytest.approx(-0.1)
    assert result["blend_minus_market"]["top1_hit_rate"] == pytest.approx(0.5)
    assert result["ai_minus_market"]["log_loss"] == pytest.approx(0)
    assert _paired_comparison({}) == {"n_races": 0}
    with pytest.raises(ValueError):
        _paired_comparison({**rows, "blend": rows["blend"][:1]})


def test_win_ev_requires_nonfinal_snapshot_matching_all_horses():
    import sqlite3
    from keiba_ai.forward import _win_odds_same_time

    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE odds_snapshot (
        race_id TEXT, bet_type TEXT, combo TEXT, captured_at TEXT,
        odds REAL, is_final INTEGER)""")
    stamp = "2030-03-02T10:00:00+09:00"
    horses = [
        {"umaban": 1, "win_odds": 2.0, "odds_captured_at": stamp},
        {"umaban": 2, "win_odds": 4.0, "odds_captured_at": stamp},
    ]
    con.executemany("INSERT INTO odds_snapshot VALUES ('r','win',?,?,?,0)",
                    [("1", stamp, 2.0), ("2", stamp, 4.0)])
    assert _win_odds_same_time(con, "r", horses, stamp)
    assert not _win_odds_same_time(con, "r", horses, None)
    assert not _win_odds_same_time(con, "r", [], stamp)
    assert not _win_odds_same_time(con, "r", [{**horses[0], "odds_captured_at": "other"}, horses[1]], stamp)
    assert not _win_odds_same_time(con, "r", [{**horses[0], "win_odds": None}, horses[1]], stamp)
    assert not _win_odds_same_time(con, "r", [{**horses[0], "win_odds": 0.0}, horses[1]], stamp)
    con.execute("UPDATE odds_snapshot SET odds=2.1 WHERE combo='1'")
    assert not _win_odds_same_time(con, "r", horses, stamp)
    con.execute("UPDATE odds_snapshot SET odds=2.0 WHERE combo='1'")
    con.execute("UPDATE odds_snapshot SET is_final=1 WHERE combo='2'")
    assert not _win_odds_same_time(con, "r", horses, stamp)
    con.execute("UPDATE odds_snapshot SET is_final=0 WHERE combo='2'")
    con.execute("INSERT INTO odds_snapshot VALUES ('r','win','3',?,10.0,0)", (stamp,))
    assert not _win_odds_same_time(con, "r", horses, stamp)  # 取消馬の余分な正オッズ


def test_readonly_reconcile_does_not_touch_database_or_report(tmp_path, monkeypatch):
    import sqlite3
    from keiba_ai import config
    from keiba_ai.forward import reconcile

    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript("""
        CREATE TABLE prediction_run (
            race_id TEXT, org TEXT, predicted_at TEXT, scheduled_post_at TEXT);
        CREATE TABLE result (race_id TEXT, umaban INTEGER, finish_pos INTEGER);
        CREATE TABLE race (race_id TEXT, org TEXT, race_date TEXT, post_time TEXT);
        CREATE TABLE odds_snapshot (
            race_id TEXT, combo TEXT, odds REAL, is_final INTEGER, bet_type TEXT);
        CREATE TABLE backtest_result (backtest_id TEXT);
    """)
    con.execute("PRAGMA query_only = ON")
    monkeypatch.setattr(config, "REPORTS", tmp_path / "untouched")
    result = reconcile(con, "NAR", persist=False, write_report=False)
    assert result["n_races"] == 0
    assert result["evaluation_coverage"]["by_time_point"]["t10"]["selected_runs"] == 0
    assert not (tmp_path / "untouched").exists()


def test_readonly_cli_opens_db_without_schema_migration(tmp_path, monkeypatch, capsys):
    import sqlite3
    from keiba_ai import config
    from keiba_ai.cli import main

    db = tmp_path / "existing.sqlite"
    with sqlite3.connect(db) as con:
        con.executescript("""
            CREATE TABLE prediction_run (
                race_id TEXT, org TEXT, predicted_at TEXT, scheduled_post_at TEXT);
            CREATE TABLE result (race_id TEXT, umaban INTEGER, finish_pos INTEGER);
            CREATE TABLE race (race_id TEXT, org TEXT, race_date TEXT, post_time TEXT);
            CREATE TABLE odds_snapshot (
                race_id TEXT, combo TEXT, odds REAL, is_final INTEGER, bet_type TEXT);
        """)
    monkeypatch.setattr(config, "DB_PATH", db)
    monkeypatch.setattr(config, "REPORTS", tmp_path / "uncreated")
    assert main(["reconcile", "--org", "NAR", "--read-only"]) == 0
    assert '"n_races": 0' in capsys.readouterr().out
    assert not (tmp_path / "uncreated").exists()
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT name FROM sqlite_master WHERE name='backtest_result'").fetchone() is None


def test_latest_odds_excludes_final_flagged_timestamp_from_prestart_quotes():
    """'final' 文字列でなく実時刻を持つ確定オッズも予測に混ぜない。"""
    import sqlite3
    from keiba_ai.ev import latest_odds

    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE odds_snapshot(
        race_id TEXT, bet_type TEXT, combo TEXT, odds REAL, odds_max REAL,
        captured_at TEXT, is_final INTEGER)""")
    rows = [
        ("r", "win", "1", 3.0, None, "2030-03-02T10:00:00+09:00", 0),
        ("r", "win", "1", 2.0, None, "2030-03-02T10:05:00+09:00", 1),
        ("r", "win", "2", 5.0, None, "final", 1),
        ("r", "win", "3", 7.0, None, "2030-03-02T10:03:00+09:00", 0),
    ]
    con.executemany("INSERT INTO odds_snapshot VALUES (?,?,?,?,?,?,?)", rows)
    at = "2030-03-02T10:10:00+09:00"
    prestart = latest_odds(con, "r", at, include_final=False)
    assert prestart[("win", "1")]["odds"] == 3.0
    assert ("win", "2") not in prestart
    assert prestart[("win", "3")]["odds"] == 7.0
    with_final = latest_odds(con, "r", at, include_final=True)
    assert with_final[("win", "1")]["odds"] == 2.0
    assert with_final[("win", "2")]["odds"] == 5.0


def test_forward_rejects_invalid_probability_vectors():
    import math
    from keiba_ai.forward import _score

    base = [
        {"umaban": 1, "p_win_ai": 0.6, "win_odds": 2.0},
        {"umaban": 2, "p_win_ai": 0.4, "win_odds": 3.0},
    ]
    assert _score(base, "p_win_ai", {1: 1, 2: 2}, {}) is not None
    for invalid in (float("nan"), float("inf"), -0.1, 1.1, None):
        bad = [{**base[0], "p_win_ai": invalid}, base[1]]
        assert _score(bad, "p_win_ai", {1: 1, 2: 2}, {}) is None
    assert _score([{**base[0], "p_win_ai": 0.5}, base[1]],
                  "p_win_ai", {1: 1, 2: 2}, {}) is None
    assert _score([], "p_win_ai", {1: 1}, {}) is None


def test_paired_intervals_resample_race_dates_and_are_deterministic():
    """同日複数レースを独立標本として過大評価しない。"""
    import pytest
    from keiba_ai.forward import _paired_comparison

    rows = {"ai": [], "market": [], "blend": []}
    for day in range(1, 13):
        for race_no in range(1, 4):
            metadata = {"race_date": f"2030-05-{day:02d}",
                        "race_id": f"2030-05-{day:02d}-R{race_no}"}
            rows["market"].append({**metadata, "ll": 0.8, "brier": 0.5, "hit": 0})
            rows["ai"].append({**metadata, "ll": 0.7, "brier": 0.4, "hit": 1})
            rows["blend"].append({**metadata, "ll": 0.6, "brier": 0.3, "hit": 1})
    result = _paired_comparison(rows)
    assert result["n_races"] == 36
    assert result["n_race_dates"] == 12
    ci = result["blend_minus_market"]["ci95"]
    assert ci is not None
    assert ci["log_loss"] == pytest.approx([-0.2, -0.2])
    assert ci["brier"] == pytest.approx([-0.2, -0.2])
    assert _paired_comparison(rows)["blend_minus_market"]["ci95"] == ci
    # 100レース未満なら区間が存在しても採用判断しない
    assert result["sufficient_sample_for_screening"] is False


def test_paired_comparison_rejects_different_race_ids_and_too_few_dates():
    import pytest
    from keiba_ai.forward import _paired_comparison

    rows = {
        key: [{"ll": 0.8, "brier": 0.5, "hit": 1,
               "race_id": "r1", "race_date": "2030-05-01"}]
        for key in ("ai", "market", "blend")
    }
    assert _paired_comparison(rows)["ai_minus_market"]["ci95"] is None
    assert _paired_comparison(rows)["n_race_dates"] == 1
    rows["blend"][0]["race_id"] = "r2"
    with pytest.raises(ValueError, match="matching race IDs"):
        _paired_comparison(rows)


def test_verified_prestart_requires_known_time_and_timezone_aware_ordering():
    from keiba_ai.forward import _valid_prestart_run

    assert _valid_prestart_run({
        "predicted_at": "2030-03-02T10:00:00+09:00",
        "scheduled_post_at": "2030-03-02T01:30:00+00:00",
    })  # 同じ瞬間の比較。文字列としては逆転する
    assert not _valid_prestart_run({
        "predicted_at": "2030-03-02T10:30:00+09:00",
        "scheduled_post_at": "2030-03-02T01:15:00+00:00",
    })
    assert not _valid_prestart_run({
        "predicted_at": "2030-03-02T10:00:00+09:00",
        "scheduled_post_at": None,
    })
    assert not _valid_prestart_run({
        "predicted_at": "2030-03-02T10:00:00",
        "scheduled_post_at": "2030-03-02T10:30:00",
    })
    assert not _valid_prestart_run({
        "predicted_at": "invalid",
        "scheduled_post_at": "2030-03-02T10:30:00+09:00",
    })


def test_readonly_report_counts_predictions_missing_prestart_evidence(tmp_path, monkeypatch):
    import sqlite3
    from keiba_ai import config
    from keiba_ai.forward import reconcile

    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript("""
        CREATE TABLE race (race_id TEXT, org TEXT, race_date TEXT, post_time TEXT);
        CREATE TABLE result (race_id TEXT, finish_pos INTEGER);
        CREATE TABLE prediction_run (
            race_id TEXT, org TEXT, predicted_at TEXT, scheduled_post_at TEXT);
        CREATE TABLE odds_snapshot (
            race_id TEXT, combo TEXT, odds REAL, is_final INTEGER, bet_type TEXT);
        INSERT INTO race VALUES ('r','NAR','2030-03-02','10:30');
        INSERT INTO result VALUES ('r',1);
        INSERT INTO prediction_run VALUES ('r','NAR','2030-03-02T10:00:00+09:00',NULL);
        INSERT INTO prediction_run VALUES (
            'r','NAR','2030-03-02T10:31:00+09:00','2030-03-02T10:30:00+09:00');
    """)
    con.execute("PRAGMA query_only = ON")
    monkeypatch.setattr(config, "REPORTS", tmp_path / "untouched")
    out = reconcile(con, "NAR", persist=False, write_report=False)
    assert out["excluded_runs_without_verified_prestart_time"] == 2
    assert out["n_races"] == 0
    assert not (tmp_path / "untouched").exists()


def test_chronological_prestart_selection_uses_actual_utc_instant():
    from keiba_ai.forward import _chronological_prestart_runs, _select

    # 01:10 UTC は日本時間10:10、10:00 JST より後。
    # 辞書順（文字列順）では逆転し得るが、発走前の最後は後者を選ぶべき。
    candidates = [
        {"race_id": "r", "prediction_id": "late", "time_point": "last",
         "predicted_at": "2030-03-02T01:10:00+00:00",
         "scheduled_post_at": "2030-03-02T10:30:00+09:00"},
        {"race_id": "r", "prediction_id": "early", "time_point": "last",
         "predicted_at": "2030-03-02T10:00:00+09:00",
         "scheduled_post_at": "2030-03-02T10:30:00+09:00"},
        {"race_id": "r", "prediction_id": "after", "time_point": "last",
         "predicted_at": "2030-03-02T01:45:00+00:00",
         "scheduled_post_at": "2030-03-02T10:30:00+09:00"},
    ]
    ordered = _chronological_prestart_runs(candidates)
    assert [r["prediction_id"] for r in ordered] == ["early", "late"]
    assert _select(ordered)[("r", "last")]["prediction_id"] == "late"


def test_quote_is_fresh_rejects_stale_future_missing_and_poststart_odds():
    from datetime import datetime
    from keiba_ai.ev import quote_is_fresh

    post = datetime.fromisoformat("2030-03-02T10:30:00+09:00")
    now = datetime.fromisoformat("2030-03-02T10:00:00+09:00")
    base = {"captured_at": "2030-03-02T09:59:00+09:00", "odds": 3.0}
    assert quote_is_fresh(base, now, post)
    assert quote_is_fresh({**base, "captured_at": "2030-03-02T00:59:00+00:00"}, now, post)
    assert not quote_is_fresh({**base, "captured_at": "2030-03-02T08:59:00+09:00"}, now, post)
    assert not quote_is_fresh({**base, "captured_at": "2030-03-02T10:01:00+09:00"}, now, post)
    assert not quote_is_fresh({**base, "captured_at": "2030-03-02T10:30:00+09:00"}, now, post)
    assert not quote_is_fresh({**base, "odds": 0.0}, now, post)
    assert not quote_is_fresh({**base, "odds": float("nan")}, now, post)
    assert not quote_is_fresh({**base, "captured_at": "invalid"}, now, post)
    assert not quote_is_fresh({**base, "captured_at": "2030-03-02T09:59:00"}, now, post)
    assert not quote_is_fresh(base, now, None)


def test_latest_odds_compares_actual_utc_instants_not_timestamp_strings():
    import sqlite3
    from keiba_ai.ev import latest_odds

    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE odds_snapshot(
        race_id TEXT, bet_type TEXT, combo TEXT, odds REAL, odds_max REAL,
        captured_at TEXT, is_final INTEGER)""")
    con.executemany("INSERT INTO odds_snapshot VALUES ('r','win','1',?,NULL,?,0)", [
        (2.0, "2030-03-02T09:00:00+09:00"),
        (3.0, "2030-03-02T01:05:00+00:00"),  # 日本時間10:05、10:00の未来
        (4.0, "2030-03-02T00:55:00+00:00"),  # 日本時間09:55
    ])
    before = latest_odds(con, "r", "2030-03-02T10:00:00+09:00", include_final=False)
    assert before[("win", "1")]["odds"] == 4.0
    after = latest_odds(con, "r", "2030-03-02T10:06:00+09:00", include_final=False)
    assert after[("win", "1")]["odds"] == 3.0


def test_latest_complete_win_odds_recovers_consistent_snapshot_only():
    import sqlite3
    from datetime import datetime
    from keiba_ai.ev import latest_complete_win_odds

    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE odds_snapshot (
        race_id TEXT, bet_type TEXT, combo TEXT, captured_at TEXT,
        odds REAL, odds_max REAL, is_final INTEGER)""")
    records = [
        ("r", "win", "1", "2030-03-02T10:00:00+09:00", 3.0, None, 0),
        ("r", "win", "2", "2030-03-02T10:00:00+09:00", 6.0, None, 0),
        # 新しい部分取得は同時刻の完全記録より優先しない
        ("r", "win", "1", "2030-03-02T10:06:00+09:00", 4.0, None, 0),
        # 確定オッズや未来の値は対象外
        ("r", "win", "2", "2030-03-02T10:07:00+09:00", 8.0, None, 1),
        ("r", "win", "1", "2030-03-02T10:15:00+09:00", 5.0, None, 0),
        ("r", "win", "2", "2030-03-02T10:15:00+09:00", 9.0, None, 0),
    ]
    con.executemany("INSERT INTO odds_snapshot VALUES (?,?,?,?,?,?,?)", records)
    now = datetime.fromisoformat("2030-03-02T10:10:00+09:00")
    post = datetime.fromisoformat("2030-03-02T10:30:00+09:00")
    found = latest_complete_win_odds(con, "r", [1, 2], now, post)
    assert found is not None
    assert found["1"]["odds"] == 3.0
    assert found["2"]["odds"] == 6.0
    assert len({q["captured_at"] for q in found.values()}) == 1
    assert latest_complete_win_odds(con, "r", [1, 2, 3], now, post) is None
    assert latest_complete_win_odds(con, "r", [1, 2], now, None) is None
    assert latest_complete_win_odds(con, "r", [1, 1], now, post) is None
    stale = datetime.fromisoformat("2030-03-02T11:30:01+09:00")
    later_post = datetime.fromisoformat("2030-03-02T12:00:00+09:00")
    assert latest_complete_win_odds(con, "r", [1, 2], stale, later_post) is None
    # 当該レースに余分な正オッズの馬番がある場合、
    # 全頭の確率を残存馬だけで再正規化してしまうことを防ぐ。
    con.execute(
        "INSERT INTO odds_snapshot VALUES ('r','win','3','2030-03-02T10:00:00+09:00',12.0,NULL,0)"
    )
    assert latest_complete_win_odds(con, "r", [1, 2], now, post) is None
