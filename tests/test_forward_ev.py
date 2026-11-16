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
    con.execute("INSERT INTO odds_snapshot VALUES ('r','place','2',?,2.0,0)", (stamp,))
    con.execute("INSERT INTO odds_snapshot VALUES ('r','place','3',?,5.0,0)", (stamp,))
    assert not _place_odds_same_time(con, "r", horses, stamp)  # 余分な取消馬の価格
    con.execute("DELETE FROM odds_snapshot WHERE combo='3'")
    con.execute("UPDATE odds_snapshot SET odds=? WHERE combo='2'", (float("inf"),))
    assert not _place_odds_same_time(con, "r", horses, stamp)
    con.execute("UPDATE odds_snapshot SET odds=2.0 WHERE combo='2'")
    assert not _place_odds_same_time(con, "r", [{**horses[0], "place_odds": float("nan")}, horses[1]], stamp)
    assert _place_odds_same_time(con, "r", horses, stamp)


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
        "ai": [{"race_id": "r1", "ll": 0.6, "brier": 0.3, "hit": 1},
               {"race_id": "r2", "ll": 1.4, "brier": 0.8, "hit": 0}],
        "market": [{"race_id": "r1", "ll": 0.8, "brier": 0.4, "hit": 0},
                   {"race_id": "r2", "ll": 1.2, "brier": 0.6, "hit": 1}],
        "blend": [{"race_id": "r1", "ll": 0.7, "brier": 0.3, "hit": 1},
                  {"race_id": "r2", "ll": 1.1, "brier": 0.5, "hit": 1}],
    }
    result = _paired_comparison(rows)
    assert result["n_races"] == 2
    assert result["sufficient_sample_for_screening"] is False
    assert result["minimum_races_for_screening"] == 100
    assert result["blend_minus_market"]["log_loss"] == pytest.approx(-0.1)
    assert result["blend_minus_market"]["top1_hit_rate"] == pytest.approx(0.5)
    assert result["ai_minus_market"]["log_loss"] == pytest.approx(0)
    assert _paired_comparison({}) == {"n_races": 0}
    # 市場だけ逆順になっても、race_idをキーに比較して精度は同一。
    reversed_market = {**rows, "market": list(reversed(rows["market"]))}
    assert _paired_comparison(reversed_market) == result
    with pytest.raises(ValueError, match="matching race IDs"):
        _paired_comparison({**rows, "blend": rows["blend"][:1]})
    # 件数が同じでも対象レースが異なれば評価を拒否。
    with pytest.raises(ValueError, match="matching race IDs"):
        _paired_comparison({**rows, "market": [{**rows["market"][0], "race_id": "other"}, rows["market"][1]]})
    with pytest.raises(ValueError, match="unique non-empty race IDs"):
        _paired_comparison({**rows, "blend": [rows["blend"][0], rows["blend"][0]]})
    with pytest.raises(ValueError, match="all model predictions"):
        _paired_comparison({**rows, "market": []})


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


def test_forward_roi_excludes_missing_payouts_without_losing_probability_scores():
    from keiba_ai.forward import _stats

    horses = [
        {"umaban": 1, "p_win_ai": 0.7, "win_odds": 2.5},
        {"umaban": 2, "p_win_ai": 0.3, "win_odds": 5.0},
    ]
    fin = {1: 1, 2: 2}
    # 着順だけ判明、払戻が未取得。的中率・LogLossは有効でもROIは不明。
    missing = _score(horses, "p_win_ai", fin, {}, evaluate_ev=True)
    assert missing is not None and missing["hit"] == 1
    assert missing["ret"] is None
    assert missing["ev_n"] == 0
    no_payout_stats = _stats([missing])
    assert no_payout_stats["n_races"] == 1
    assert no_payout_stats["top1_hit"] == 1
    assert no_payout_stats["top1_roi"] is None
    assert no_payout_stats["top1_roi_races"] == 0
    assert no_payout_stats["payout_coverage_rate"] == 0
    assert no_payout_stats["ev100_roi"] is None

    # 勝ち馬の払戻があるなら、外れた馬の払戻0円は真正な不的中。
    verified = _score(horses, "p_win_ai", fin, {"1": 400}, evaluate_ev=True)
    assert verified["ret"] == 400
    assert verified["ev_n"] == 2
    mixed = _stats([missing, verified])
    assert mixed["n_races"] == 2  # スコアは両レースを評価
    assert mixed["top1_roi_races"] == 1
    assert mixed["payout_coverage_rate"] == 0.5
    assert mixed["top1_roi"] == 4.0
    assert mixed["ev100_roi"] == 2.0


def test_forward_payout_verification_handles_losing_top_and_multiple_winners():
    from keiba_ai.forward import _stats

    horses = [
        {"umaban": 1, "p_win_ai": 0.6, "win_odds": 2.0},
        {"umaban": 2, "p_win_ai": 0.3, "win_odds": 4.0},
        {"umaban": 3, "p_win_ai": 0.1, "win_odds": 10.0},
    ]
    loser_top = _score(horses, "p_win_ai", {1: 2, 2: 1, 3: 3}, {"2": 350})
    assert loser_top["ret"] == 0
    assert _stats([loser_top])["top1_roi"] == 0
    # 同着1着の片方の払戻が欠ける場合、回収額は不明。
    partial_dead_heat = _score(horses, "p_win_ai", {1: 1, 2: 1, 3: 3}, {"1": 220})
    assert partial_dead_heat is None
    complete_dead_heat = _score(horses, "p_win_ai", {1: 1, 2: 1, 3: 3}, {"1": 220, "2": 430})
    assert complete_dead_heat is None  # 単一正解のLogLossには使わない


def test_forward_probability_score_rejects_missing_duplicate_or_unmatched_horses():
    horses = [
        {"umaban": 1, "p_win_ai": 0.7, "win_odds": 2.0},
        {"umaban": 2, "p_win_ai": 0.3, "win_odds": 3.0},
    ]
    fin = {1: 1, 2: 2}
    assert _score(horses, "p_win_ai", fin, {}) is not None
    # 結果未取得馬を、暗黙に不的中として扱わない
    assert _score(horses, "p_win_ai", {1: 1}, {}) is None
    # 予測から欠けた馬を、正規化された2頭分の確率だけで採点しない
    assert _score(horses, "p_win_ai", {1: 1, 2: 2, 3: 3}, {}) is None
    duplicate = [horses[0], {**horses[0], "p_win_ai": 0.3}]
    assert _score(duplicate, "p_win_ai", fin, {}) is None
