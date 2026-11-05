"""モデルの比較・採用・ロールバックと、データ品質のテスト。データはすべて架空。"""
import numpy as np
import pandas as pd
import pytest

from conftest import make_race
from keiba_ai import config
from keiba_ai.features import FeatureSpec, build_features, load_frame
from keiba_ai.forward import reconcile
from keiba_ai.ingest import IntegrityError, save_odds, save_race, validate_race
from keiba_ai.providers.base import OddsRec
from keiba_ai.registry import _adopt, decide, list_models, rollback


def _insert(con, mid, status, adopted, path):
    con.execute("INSERT INTO model_version(model_version_id,org,algo,created_at,path,status,adopted_at) VALUES (?,?,?,?,?,?,?)",
                (mid, "JRA", "x", adopted or "2026-01-01", path, status, adopted))


def test_新モデルが悪ければ採用しない():
    old = {"win_logloss_model": 2.00, "brier": 0.85}
    assert decide({"win_logloss_model": 2.01, "brier": 0.85}, old)[0] is False
    assert decide({"win_logloss_model": 2.00, "brier": 0.86}, old)[0] is False
    assert decide({"win_logloss_model": 1.99, "brier": 0.84}, old)[0] is True
    assert decide({"win_logloss_model": 2.0005, "brier": 0.85}, old)[0] is True  # 誤差は同等


def test_ロールバック(con, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    for n in ("a", "b"):
        (tmp_path / f"{n}.pkl").write_bytes(b"x")
    _insert(con, "A", "retired", "2026-01-01T00:00:00", "a.pkl")
    _insert(con, "B", "active", "2026-02-01T00:00:00", "b.pkl")
    _insert(con, "R", "rejected", None, "a.pkl")
    assert "A" in rollback(con, "JRA")
    st = {m["model_version_id"]: m["status"] for m in list_models(con, "JRA")}
    assert st == {"A": "active", "B": "retired", "R": "rejected"}
    assert "B" in rollback(con, "JRA", "B")
    with pytest.raises(SystemExit):
        rollback(con, "JRA", "MISSING")


def test_重複登録しない(con):
    r = make_race("2024-01-01")
    save_race(con, r)
    save_race(con, r)
    assert con.execute("SELECT COUNT(*) FROM race").fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM entry").fetchone()[0] == 8
    assert con.execute("SELECT COUNT(*) FROM result").fetchone()[0] == 8
    assert con.execute("SELECT COUNT(*) FROM payout").fetchone()[0] == 1


def test_オッズは上書きせず履歴に残す(con):
    r = make_race("2024-01-01")
    save_race(con, r)
    save_odds(con, r.race_id, [OddsRec("win", "1", 3.0)], "2024-01-01T10:00:00+09:00", False, "t")
    save_odds(con, r.race_id, [OddsRec("win", "1", 2.5)], "2024-01-01T11:00:00+09:00", False, "t")
    save_odds(con, r.race_id, [OddsRec("win", "1", 2.5)], "2024-01-01T11:00:00+09:00", False, "t")  # 再送
    v = [x[0] for x in con.execute("SELECT odds FROM odds_snapshot ORDER BY captured_at")]
    assert v == [3.0, 2.5]


def test_後から判明した馬体重で埋まるがNULLで消えない(con):
    r = make_race("2024-01-01", finished=False)
    for e in r.entries:
        e.body_weight = None
    save_race(con, r)
    assert con.execute("SELECT body_weight FROM entry").fetchone()[0] is None
    for e in r.entries:
        e.body_weight = 480.0
    save_race(con, r)
    for e in r.entries:
        e.body_weight = None
    save_race(con, r)
    assert con.execute("SELECT body_weight FROM entry").fetchone()[0] == 480.0


def test_整合性チェック():
    r = make_race("2024-01-01")
    r.entries[1].umaban = r.entries[0].umaban
    with pytest.raises(IntegrityError):
        validate_race(r)
    r = make_race("2024-01-01")
    r.distance = 99999
    with pytest.raises(IntegrityError):
        validate_race(r)
    r = make_race("2024-01-01")
    for e in r.entries:
        e.finish_pos = (e.finish_pos or 0) + 1
    assert validate_race(r)  # 1着なしは警告


def test_欠損値でも特徴量を作れて未知カテゴリを受け入れる(filled, con):
    con.execute("UPDATE entry SET body_weight=NULL, jockey_id=NULL WHERE umaban=1")
    con.execute("UPDATE race SET going=NULL, class_name=NULL WHERE race_date='2024-03-15'")
    df = load_frame(con, "JRA")
    f = build_features(df)
    assert len(f) == len(df) and f["body_weight"].isna().sum() > 0
    spec = FeatureSpec.fit(f)
    f2 = f.copy()
    f2["course_code"] = "未知"
    X = spec.transform(f2)
    assert X["course_code"].isna().all()
    assert list(X.columns) == spec.columns


def test_取消馬は頭数と対象から除外(con):
    r = make_race("2024-01-01")
    r.entries[0].status, r.entries[0].finish_pos = "scratched", None
    save_race(con, r)
    df = load_frame(con, "JRA")
    assert len(df) == 7 and (df["n_runners"] == 7).all()


def test_予測と結果の照合(con, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORTS", tmp_path)
    r = make_race("2024-01-01", seed=1)
    save_race(con, r)
    con.execute("INSERT INTO model_version(model_version_id,org,algo,created_at) VALUES ('M','JRA','x','2024')")
    winner = next(p.combo for p in r.payouts)
    ins = "INSERT INTO prediction(model_version_id,race_id,bet_type,combo,predicted_at,prob,odds_captured_at,ev) VALUES ('M',?,?,?,?,?,?,?)"
    con.execute(ins, (r.race_id, "win", winner, "2024-01-01T14:00:00+09:00", .5, "2024-01-01T13:50:00+09:00", .2))
    other = "1" if winner != "1" else "2"
    con.execute(ins, (r.race_id, "win", other, "2024-01-01T14:00:00+09:00", .2, "final", 9.0))    # 確定オッズ基準は除外
    out = reconcile(con, "JRA")
    by = {(x["strategy"], x["bet_type"]): x for x in out["rows"]}
    assert by[("top1", "win")]["roi"] == 5.0
    assert by[("ev_1.0", "win")]["n_bets"] == 1
    # 発走後に作られた予測は集計しない
    con.execute("UPDATE prediction SET predicted_at='2024-01-01T16:00:00+09:00'")
    assert reconcile(con, "JRA")["rows"] == []
