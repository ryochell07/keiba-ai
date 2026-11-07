"""予測台帳のテスト。発走前の予測が消えず、書き換わらず、結果でも上書きされないことを確かめる。データはすべて架空。"""
import argparse
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from keiba_ai import config, ledger
from keiba_ai.bundles import active_bundle, guard_decision
from keiba_ai.capture import exotics_due, win_place_due
from keiba_ai.commands import cmd_daily
from keiba_ai.db import connect
from keiba_ai.features import build_features, load_frame
from keiba_ai.forward import reconcile
from keiba_ai.ingest import observed_as_of, save_odds, save_race
from keiba_ai.ledger import should_record
from keiba_ai.monitor import odds_quality
from keiba_ai.predict import predict_races
from keiba_ai.providers.base import EntryRec, OddsRec, PayoutRec, RaceRec
from keiba_ai.registry import retrain_with_gate
from datetime import datetime

DAY = "2030-03-02"
COURSES = ("01", "02", "03", "04", "05", "06")


def _future(rng, course, no, finished=False):
    n = 8
    horses = rng.choice(200, n, replace=False)
    order = rng.permutation(n) + 1
    post = f"{9 + no:02d}:30"  # 10:30〜21:30
    r = RaceRec(org="JRA", course_code=course, course_name=course, race_date=DAY, race_no=no, source="t", post_time=post, surface="芝", distance=1600, direction="左", class_name="オープン")
    for u, h in enumerate(horses, 1):
        r.entries.append(EntryRec(umaban=u, horse_id=f"H{h}", horse_name=f"架空{h}", waku=u, sex="牡", age=4, carried_weight=56.0, jockey_id=f"J{h % 9}", jockey_name="騎手", trainer_id=f"T{h % 7}", trainer_name="調教師"))
    return r, order


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ledger")
    old = config.MODELS, config.REPORTS, config.ROOT
    config.MODELS, config.REPORTS, config.ROOT = tmp / "models", tmp / "reports", tmp
    (tmp / "reports").mkdir()
    con = connect(tmp / "k.db")
    rng = np.random.default_rng(3)
    ability = rng.normal(0, 1.0, 200)
    start = np.datetime64("2029-01-06")
    for i in range(500):
        n = 8
        hs = rng.choice(200, n, replace=False)
        order = np.argsort(-(ability[hs] + rng.gumbel(0, 1.0, n)))
        rank = np.empty(n, int)
        rank[order] = np.arange(1, n + 1)
        r = RaceRec(org="JRA", course_code="05", course_name="05", race_date=str(start + np.timedelta64(i // 6, "D")), race_no=i % 6 + 1, source="t", post_time="15:00", surface="芝", distance=1600, direction="左", weather="晴", going="良", class_name="オープン")
        for u, h in enumerate(hs, 1):
            r.entries.append(EntryRec(umaban=u, horse_id=f"H{h}", horse_name=f"架空{h}", waku=u, sex="牡", age=4, carried_weight=56.0, jockey_id=f"J{h % 9}", jockey_name="騎手", trainer_id=f"T{h % 7}", trainer_name="調教師", body_weight=470.0,
                                      finish_pos=int(rank[u - 1]), status="finished", time_sec=96 + rank[u - 1] * .2, last3f=34.0, corners=f"{u},{u}", popularity_final=u))
        save_race(con, r)
    assert retrain_with_gate(con, "JRA", n_seeds=1)["decision"] == "採用"
    races = []
    for c in COURSES:
        for no in range(1, 13):
            races.append(_future(rng, c, no))
            save_race(con, races[-1][0], observed_at=f"{DAY}T08:00:00+09:00")
            save_odds(con, races[-1][0].race_id, [OddsRec("win", str(u), 2.0 + u) for u in range(1, 9)] + [OddsRec("place", str(u), 1.1 + u / 10, 1.5 + u / 10) for u in range(1, 9)],
                      f"{DAY}T08:50:00+09:00", False, "t")
    con.commit()
    yield con, races
    con.close()
    config.MODELS, config.REPORTS, config.ROOT = old


def _dump(con):
    runs = [tuple(r) for r in con.execute("SELECT * FROM prediction_run ORDER BY prediction_id")]
    horses = [tuple(r) for r in con.execute("SELECT * FROM prediction_horse ORDER BY prediction_id, umaban")]
    return runs, horses


def test_朝9時の予測は23時の再実行でも結果の取り込みでも変わらない(world):
    con, races = world
    a = predict_races(con, "JRA", DAY, at=f"{DAY}T09:00:00+09:00")
    assert a["races"] == 72 and a["recorded"] == 72
    ids = {r[0] for r in con.execute("SELECT DISTINCT race_id FROM prediction_run WHERE predicted_at=?", (f"{DAY}T09:00:00+09:00",))}
    assert len(ids) == 72
    before = _dump(con)
    # 23時の再実行: 全レース発走済みなので何も追加しない
    b = predict_races(con, "JRA", DAY, at=f"{DAY}T23:00:00+09:00")
    assert b["skipped_started"] == 72 and b["recorded"] == 0 and b["races"] == 0
    assert _dump(con) == before
    assert con.execute("SELECT COUNT(*) FROM prediction_run WHERE predicted_at>=scheduled_post_at").fetchone()[0] == 0
    # 結果を取り込む
    for race, order in races:
        for e, pos in zip(race.entries, order):
            e.finish_pos, e.status, e.time_sec, e.last3f, e.corners = int(pos), "finished", 96 + pos * .2, 34.0, "1,1"
            e.popularity_final = e.umaban
        w = int(np.argmin(order)) + 1
        race.payouts = [PayoutRec("win", str(w), 300, 1)]
        save_race(con, race, observed_at=f"{DAY}T23:30:00+09:00")
    con.commit()
    assert _dump(con) == before
    c = predict_races(con, "JRA", DAY, at=f"{DAY}T23:40:00+09:00")
    assert c["recorded"] == 0 and _dump(con) == before
    rep = reconcile(con, "JRA")
    assert rep["n_races"] == 72
    assert _dump(con) == before  # 照合でも変わらない
    bt = rep["by_time_point"]["morning"]
    assert bt["ai"]["n_races"] == bt["market"]["n_races"] == bt["blend"]["n_races"] == 72
    assert rep["final_odds_reference"]["purchasable"] is False


def test_台帳は削除も更新もできない(world):
    con, _ = world
    for sql in ("DELETE FROM prediction_run", "UPDATE prediction_run SET used_model='x'", "DELETE FROM prediction_horse", "UPDATE prediction_horse SET p_win_ai=0",
                "DELETE FROM prediction_input", "DELETE FROM field_observation", "DELETE FROM model_bundle", "UPDATE model_bundle SET note='x'",
                "INSERT INTO prediction(model_version_id,race_id,bet_type,combo,predicted_at,prob) VALUES ('a','b','c','d','e',0)"):
        with pytest.raises(sqlite3.IntegrityError):
            con.execute(sql)
    names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
    assert {"prediction_legacy_no_update", "prediction_legacy_no_delete", "prediction_legacy_no_insert"} <= names  # 旧テーブルの凍結


def test_保存した入力から同じ確率を再現できる(world):
    con, _ = world
    for r in con.execute("SELECT prediction_id FROM prediction_run LIMIT 5").fetchall():
        out = ledger.replay(con, r[0])
        assert out["reproduced"], out


def test_予測には必須項目がそろう(world):
    con, _ = world
    r = con.execute("SELECT * FROM prediction_run LIMIT 1").fetchone()
    for k in ("race_id", "prediction_id", "predicted_at", "scheduled_post_at", "time_point", "model_version_id", "bundle_id", "input_hash", "feature_hash", "odds_captured_at"):
        assert r[k], k
    h = con.execute("SELECT * FROM prediction_horse WHERE prediction_id=?", (r["prediction_id"],)).fetchall()
    assert len(h) == 8 and abs(sum(x["p_win_ai"] for x in h) - 1) < 1e-9 and abs(sum(x["p_top3_ai"] for x in h) - 3) < 1e-9
    assert abs(sum(x["p_win_market"] for x in h) - 1) < 1e-9 and all(x["win_odds"] for x in h)


def test_同じ入力は重ねて記録しないがオッズが変われば別の記録になる(tmp_path, world):
    con, races = world
    # 別の日の新しいレース
    rng = np.random.default_rng(5)
    r, _ = _future(rng, "07", 12)
    r.race_date = "2030-03-03"
    r.post_time = "16:00"
    save_race(con, r, observed_at="2030-03-03T08:00:00+09:00")
    rid = r.race_id
    n = lambda: con.execute("SELECT COUNT(*) FROM prediction_run WHERE race_id=?", (rid,)).fetchone()[0]
    predict_races(con, "JRA", "2030-03-03", at="2030-03-03T09:00:00+09:00")
    assert n() == 1
    predict_races(con, "JRA", "2030-03-03", at="2030-03-03T09:05:00+09:00")
    assert n() == 1  # 入力が同じ
    save_odds(con, rid, [OddsRec("win", str(u), 3.0 + u) for u in range(1, 9)], "2030-03-03T15:28:00+09:00", False, "t")
    out = predict_races(con, "JRA", "2030-03-03", at="2030-03-03T15:30:00+09:00")
    assert out["recorded"] == 1 and n() == 2
    row = con.execute("SELECT * FROM prediction_run WHERE race_id=? ORDER BY predicted_at DESC", (rid,)).fetchone()
    assert row["time_point"] == "t30" and row["odds_status"] == "complete" and row["odds_same_time"] == 1
    save_odds(con, rid, [OddsRec("win", str(u), 3.5 + u) for u in range(1, 9)], "2030-03-03T15:49:00+09:00", False, "t")
    predict_races(con, "JRA", "2030-03-03", at="2030-03-03T15:50:00+09:00")
    assert con.execute("SELECT time_point FROM prediction_run WHERE race_id=? ORDER BY predicted_at DESC", (rid,)).fetchone()[0] == "t10"
    assert n() == 3
    # 市場の確率はその時刻のオッズから作る
    h = con.execute("SELECT h.win_odds,h.odds_captured_at FROM prediction_horse h JOIN prediction_run r USING(prediction_id) WHERE r.race_id=? AND r.time_point='t10' AND h.umaban=1", (rid,)).fetchone()
    assert h["win_odds"] == 4.5 and h["odds_captured_at"].endswith("15:49:00+09:00")


def test_確定オッズは予測に使わない(world):
    con, _ = world
    rng = np.random.default_rng(6)
    r, _ = _future(rng, "08", 12)
    r.race_date = "2030-03-04"
    save_race(con, r, observed_at="2030-03-04T08:00:00+09:00")
    save_odds(con, r.race_id, [OddsRec("win", str(u), 9.0) for u in range(1, 9)], "final", True, "t")
    predict_races(con, "JRA", "2030-03-04", at="2030-03-04T09:00:00+09:00")
    row = con.execute("SELECT odds_status FROM prediction_run WHERE race_id=?", (r.race_id,)).fetchone()
    assert row["odds_status"] == "none"
    assert con.execute("SELECT COUNT(*) FROM prediction_horse h JOIN prediction_run p USING(prediction_id) WHERE p.race_id=? AND h.win_odds IS NOT NULL", (r.race_id,)).fetchone()[0] == 0


def test_取得時刻と値の変化を記録し過去を埋めない(con_empty=None):
    con = connect(":memory:")
    r, _ = _future(np.random.default_rng(1), "05", 1)
    save_race(con, r, observed_at=f"{DAY}T08:00:00+09:00")
    r.entries[0].body_weight = 480.0
    r.going = "稍重"
    save_race(con, r, observed_at=f"{DAY}T09:30:00+09:00")
    save_race(con, r, observed_at=f"{DAY}T09:35:00+09:00")  # 同じ値は増やさない
    assert con.execute("SELECT COUNT(*) FROM field_observation WHERE field='body_weight'").fetchone()[0] == 1
    early = observed_as_of(con, r.race_id, f"{DAY}T09:00:00+09:00")
    later = observed_as_of(con, r.race_id, f"{DAY}T10:00:00+09:00")
    assert (1, "body_weight") not in early and later[(1, "body_weight")] == "480.0"
    assert (0, "going") not in early and later[(0, "going")] == "稍重"
    assert con.execute("SELECT COUNT(*) FROM field_observation WHERE published_at IS NOT NULL").fetchone()[0] == 0
    # 過去レースの取り込みでは記録しない
    p, _ = _future(np.random.default_rng(2), "05", 2)
    p.race_date = "2029-01-01"
    save_race(con, p, observed_at=f"{DAY}T10:00:00+09:00")
    assert con.execute("SELECT COUNT(*) FROM field_observation WHERE race_id=?", (p.race_id,)).fetchone()[0] == 0


def test_オッズの状態と発走予定時刻を保存し取得履歴を上書きしない():
    con = connect(":memory:")
    r, _ = _future(np.random.default_rng(1), "05", 1)
    save_race(con, r, observed_at=f"{DAY}T08:00:00+09:00")
    save_odds(con, r.race_id, [OddsRec("win", "1", 3.0)], f"{DAY}T09:00:00+09:00", False, "t")
    save_odds(con, r.race_id, [OddsRec("win", "1", 3.5)], f"{DAY}T09:30:00+09:00", False, "t")
    save_odds(con, r.race_id, [OddsRec("win", "1", 3.7)], "final", True, "t")
    rows = con.execute("SELECT odds,sale_state,scheduled_post_at FROM odds_snapshot ORDER BY captured_at").fetchall()
    assert sorted((x[0], x[1]) for x in rows) == [(3.0, "open"), (3.5, "open"), (3.7, "final")]
    assert all(x[2] == f"{DAY}T10:30:00+09:00" for x in rows)


def test_取得間隔の規則():
    con = connect(":memory:")
    r, _ = _future(np.random.default_rng(1), "05", 1)
    save_race(con, r, observed_at=f"{DAY}T08:00:00+09:00")
    rid = r.race_id
    t = lambda s: datetime.fromisoformat(f"{DAY}T{s}:00+09:00")
    assert win_place_due(con, rid, 90, t("09:00"))
    save_odds(con, rid, [OddsRec("win", "1", 3.0)], f"{DAY}T09:00:00+09:00", False, "t")
    assert not win_place_due(con, rid, 80, t("09:10"))
    assert win_place_due(con, rid, 30, t("10:00"))
    assert not win_place_due(con, rid, 0, t("10:30"))
    assert not exotics_due(con, rid, 40, t("09:50"))
    assert exotics_due(con, rid, 12, t("10:18"))
    save_odds(con, rid, [OddsRec("trifecta", "1-2-3", 30.0)], f"{DAY}T10:18:00+09:00", False, "t")
    assert not exotics_due(con, rid, 7, t("10:23"))


def test_取得状況の集計():
    con = connect(":memory:")
    a, _ = _future(np.random.default_rng(1), "05", 1)   # 10:30 発走
    b, _ = _future(np.random.default_rng(2), "05", 2)   # 11:30 発走、オッズなし
    for r in (a, b):
        save_race(con, r, observed_at=f"{DAY}T08:00:00+09:00")
    for at in ("09:55", "10:15", "10:27"):
        save_odds(con, a.race_id, [OddsRec("win", str(u), 3.0) for u in range(1, 9)], f"{DAY}T{at}:00+09:00", False, "t")
    con.execute("INSERT INTO fetch_log(source,url,ref,fetched_at,http_status,bytes,attempts,ok,error) VALUES ('jra','x',NULL,?,500,0,1,0,'e')", (f"{DAY}T10:00:00+09:00",))
    q = odds_quality(con, DAY, "JRA", now=datetime.fromisoformat(f"{DAY}T23:00:00+09:00"))
    assert q["n_races"] == 2 and q["any_rate"] == 0.5 and q["t30_rate"] == 0.5 and q["t10_rate"] == 0.5 and q["last_rate"] == 0.5
    assert q["n_errors"] == 1 and q["missing_rate"] == 0.5 and q["median_delay_t10_min"] == pytest.approx(5.0)


def test_発走前のレースとオッズ0は取得状況に数えない():
    con = connect(":memory:")
    a, _ = _future(np.random.default_rng(1), "05", 1)   # 10:30 発走
    b, _ = _future(np.random.default_rng(2), "05", 2)   # 11:30 発走
    for r in (a, b):
        save_race(con, r, observed_at=f"{DAY}T08:00:00+09:00")
    win = lambda v: [OddsRec("win", str(u), v) for u in range(1, 9)]
    save_odds(con, a.race_id, win(0.0), f"{DAY}T09:55:00+09:00", False, "t")     # T-35 は発売前
    save_odds(con, a.race_id, win(3.0), f"{DAY}T10:15:00+09:00", False, "t")     # T-15
    save_odds(con, b.race_id, win(3.0), f"{DAY}T10:15:00+09:00", False, "t")
    state = {x[0]: x[1] for x in con.execute("SELECT captured_at,sale_state FROM odds_snapshot WHERE race_id=?", (a.race_id,))}
    assert state[f"{DAY}T09:55:00+09:00"] == "no_odds" and state[f"{DAY}T10:15:00+09:00"] == "open"
    q = odds_quality(con, DAY, "JRA", now=datetime.fromisoformat(f"{DAY}T10:40:00+09:00"))
    assert q["n_races"] == 1 and q["n_snapshots"] == 1 and q["any_rate"] == 1.0
    assert q["t30_rate"] == 0.0 and q["t10_rate"] == 1.0 and q["last_rate"] == 0.0 and q["missing_rate"] == 0.0


def test_オッズ0だけのレースは市場を使わずオッズなしとして記録する(world):
    con, _ = world
    r, _ = _future(np.random.default_rng(6), "09", 12)
    r.race_date, r.post_time = "2030-03-19", "16:00"
    save_race(con, r, observed_at="2030-03-19T08:00:00+09:00")
    save_odds(con, r.race_id, [OddsRec("win", str(u), 0.0) for u in range(1, 9)] + [OddsRec("place", str(u), 0.0, 0.0) for u in range(1, 9)], "2030-03-19T08:30:00+09:00", False, "t")
    predict_races(con, "JRA", "2030-03-19", at="2030-03-19T09:00:00+09:00")
    row = con.execute("SELECT * FROM prediction_run WHERE race_id=?", (r.race_id,)).fetchone()
    assert (row["odds_status"], row["used_model"], row["odds_captured_at"]) == ("none", "ai", None)
    h = con.execute("SELECT win_odds,place_odds,odds_captured_at,p_win_market FROM prediction_horse WHERE prediction_id=? AND umaban=1", (row["prediction_id"],)).fetchone()
    assert tuple(h) == (None, None, None, None)


def test_前日は特徴量が変わっても間隔を空けて記録する(world):
    con, _ = world
    r, _ = _future(np.random.default_rng(7), "09", 11)
    r.race_date, r.post_time = "2030-03-21", "16:00"
    save_race(con, r, observed_at="2030-03-20T08:00:00+09:00")
    predict_races(con, "JRA", "2030-03-21", at="2030-03-20T09:00:00+09:00")
    last = con.execute("SELECT * FROM prediction_run WHERE race_id=?", (r.race_id,)).fetchone()
    assert last["time_point"] == "day_before"
    check = lambda tp, at, fh: should_record(con, r.race_id, tp, last["model_version_id"], "別の入力", fh, f"2030-03-{at}+09:00", datetime.fromisoformat(f"2030-03-{at}+09:00"))
    assert not check("day_before", "20T10:00:00", "別の特徴量")
    assert check("day_before", "20T21:00:00", "別の特徴量")
    assert check("morning", "21T09:00:00", last["feature_hash"])


def test_合成のガード():
    assert guard_decision(-0.1, [-0.2, 0.15])[0] == "market_only"
    assert guard_decision(0.1, [-0.2, 0.15])[0] == "market_only"
    assert guard_decision(0.2, None)[0] == "market_only"
    assert guard_decision(0.2, [0.12, 0.24])[0] == "blend"


def test_モデルの版に再現情報がある(world):
    con, _ = world
    mid = con.execute("SELECT model_version_id FROM model_version WHERE status='active'").fetchone()[0]
    b = active_bundle(con, "JRA", mid)
    for k in ("model_hash", "train_start", "train_end", "tree_train_end", "valid_start", "valid_end", "features", "hyperparams", "n_seeds", "n_trees", "taus", "tau_period", "adoption_reason"):
        assert b[k] not in (None, ""), k
    assert json.loads(b["hyperparams"])["num_leaves"] == 58


def test_ガード有効時は市場単独になる(world, monkeypatch):
    con, _ = world
    monkeypatch.setattr(config, "guards", lambda: {"blend_enforce": True})
    rng = np.random.default_rng(8)
    r, _ = _future(rng, "09", 12)
    r.race_date = "2030-03-05"
    save_race(con, r, observed_at="2030-03-05T08:00:00+09:00")
    save_odds(con, r.race_id, [OddsRec("win", str(u), 2.0 + u) for u in range(1, 9)], "2030-03-05T08:50:00+09:00", False, "t")
    predict_races(con, "JRA", "2030-03-05", at="2030-03-05T09:00:00+09:00")
    run = con.execute("SELECT * FROM prediction_run WHERE race_id=?", (r.race_id,)).fetchone()
    # テストの世界では合成係数がないので guard は市場単独を勧める
    assert run["used_model"] == "market"
    h = con.execute("SELECT p_win_used,p_win_market FROM prediction_horse WHERE prediction_id=?", (run["prediction_id"],)).fetchall()
    assert all(abs(x[0] - x[1]) < 1e-9 for x in h)


def test_凍結中は自動で再学習しない(world, monkeypatch):
    con, _ = world
    monkeypatch.setattr(config, "guards", lambda: {"freeze_models": True})
    import keiba_ai.registry as reg
    monkeypatch.setattr(reg, "retrain_with_gate", lambda *a, **k: pytest.fail("再学習してはいけない"))
    n = con.execute("SELECT COUNT(*) FROM model_version").fetchone()[0]
    args = argparse.Namespace(no_fetch=True, retrain=True, org="JRA", days=1, odds=False)
    assert cmd_daily(args, con, {"sources": {"nar_csv": {}}}) == 0
    assert con.execute("SELECT COUNT(*) FROM model_version").fetchone()[0] == n
