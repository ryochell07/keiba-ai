"""未来情報の混入を検出するテスト。"""
import numpy as np
import pandas as pd
import pytest

from keiba_ai.backtest import assert_time_split
from keiba_ai.features import FORBIDDEN, add_market, assert_no_forbidden, build_features, load_frame
from keiba_ai.ingest import save_odds
from keiba_ai.providers.base import OddsRec

TARGET = "2024-06-15"


def _f(con):
    return load_frame(con, "JRA")


def test_当該レースの結果を変えても特徴量は変わらない(filled):
    df = _f(filled)
    base = build_features(df)
    mutated = df.copy()
    target = mutated["race_date"] == TARGET
    numeric = ["finish_pos", "time_sec", "last3f", "popularity_final"]
    for c in numeric:
        mutated.loc[target, c] = mutated.loc[target, c].to_numpy()[::-1]
    mutated.loc[target, "corners"] = "1,1"
    after = build_features(mutated)
    pd.testing.assert_frame_equal(base.loc[target], after.loc[target], check_exact=False, check_like=True)


def test_未来のレースを消しても過去の特徴量は変わらない(filled):
    df = _f(filled)
    full = build_features(df)
    cut = df[df["race_date"] <= TARGET].reset_index(drop=True)
    part = build_features(cut)
    keep = (df["race_date"] <= TARGET).to_numpy()
    a = full[keep].reset_index(drop=True)
    pd.testing.assert_frame_equal(a[part.columns], part, check_exact=False)


def test_同日の別レースの結果は使わない(filled, con):
    from conftest import make_race
    from keiba_ai.ingest import save_race
    save_race(con, make_race(TARGET, no=9, seed=500, n=6))
    con.commit()
    df = _f(con)
    r1 = df["race_id"].str.endswith("0501") & (df["race_date"] == TARGET)
    base = build_features(df)
    mutated = df.copy()
    later = df["race_id"].str.endswith("0509")
    mutated.loc[later, "finish_pos"] = mutated.loc[later, "finish_pos"].to_numpy()[::-1]
    after = build_features(mutated)
    cols = [c for c in base.columns if c.startswith(("jk", "tr", "hs_"))]
    pd.testing.assert_frame_equal(base.loc[r1, cols], after.loc[r1, cols], check_exact=False)


def test_禁止列の検出():
    assert_no_forbidden(["jk_h_win_rate", "fin_ratio_m3", "market_win_odds"])
    for bad in ["finish_pos", "popularity_final", "time_sec", "payout_amount", "last3f", "final_odds"]:
        assert FORBIDDEN.search(bad), bad
    with pytest.raises(ValueError):
        assert_no_forbidden(["finish_pos"])


def test_確定オッズと発走後オッズは市場特徴量に入らない(filled):
    rid = "J202406150501"
    save_odds(filled, rid, [OddsRec("win", "1", 2.0)], "2024-06-15T14:30:00+09:00", False, "t")  # 30分前
    save_odds(filled, rid, [OddsRec("win", "1", 3.0)], "2024-06-15T14:55:00+09:00", False, "t")  # 5分前(締切後扱い)
    save_odds(filled, rid, [OddsRec("win", "1", 9.9)], "final", True, "t")
    df = _f(filled)
    m = add_market(df, filled, cutoff_min=10)
    v = m.loc[(m.race_id == rid) & (m.umaban == 1), "market_win_odds"].iloc[0]
    assert v == 2.0
    other = m.loc[(m.race_id == rid) & (m.umaban == 2), "market_win_odds"].iloc[0]
    assert np.isnan(other)


def test_時系列分割の検査():
    with pytest.raises(AssertionError):
        assert_time_split(pd.Series(["2024-02-01"]), pd.Series(["2024-02-01"]))
    assert_time_split(pd.Series(["2024-01-31"]), pd.Series(["2024-02-01"]))


def test_追加特徴も当日の後のレースの結果を見ない(filled, con):
    from conftest import make_race
    from keiba_ai.ingest import save_race
    day = "2024-06-20"
    for no, seed in ((1, 301), (5, 302), (9, 303)):
        r = make_race(day, no=no, seed=seed, n=6)
        for e in r.entries:
            e.horse_id = f"D{no}{e.umaban}"
        save_race(con, r)
    con.commit()
    df = _f(con)
    base = build_features(df)
    later = df["race_id"].str.endswith("0509")
    mutated = df.copy()
    mutated.loc[later, "corners"] = "1,1"
    mutated.loc[later, "finish_pos"] = mutated.loc[later, "finish_pos"].to_numpy()[::-1]
    after = build_features(mutated)
    earlier = df["race_id"].str.endswith(("0501", "0505")) & (df["race_date"] == day)
    cols = [c for c in base.columns if c.startswith("x_day")]
    pd.testing.assert_frame_equal(base.loc[earlier, cols], after.loc[earlier, cols], check_exact=False)
    # 前のレースの結果は後のレースに反映される
    first = df["race_id"].str.endswith("0501") & (df["race_date"] == day)
    mutated2 = df.copy()
    mutated2.loc[first, "corners"] = "9,9"
    after2 = build_features(mutated2)
    last = later & (df["race_date"] == day)
    assert not base.loc[last, "x_day_win_corner"].equals(after2.loc[last, "x_day_win_corner"])


def test_封鎖期間は開発データに入らない():
    from keiba_ai.lab import LockedTestError, assert_dev_only, run_stage
    assert_dev_only(pd.DataFrame({"race_date": ["2026-02-28"]}))
    with pytest.raises(LockedTestError):
        assert_dev_only(pd.DataFrame({"race_date": ["2026-03-01"]}))
    with pytest.raises(LockedTestError):
        run_stage(None, "NAR", "final")


def test_当該レースの人気を変えても過去人気の特徴量は変わらない(filled):
    df = _f(filled)
    base = build_features(df)
    mutated = df.copy()
    target = mutated["race_date"] == TARGET
    mutated.loc[target, "popularity_final"] = 1
    after = build_features(mutated)
    cols = [c for c in base.columns if c.startswith("x_pastmkt")]
    pd.testing.assert_frame_equal(base.loc[target, cols], after.loc[target, cols], check_exact=False)


def test_最終評価は1回しか実行できない(con):
    from keiba_ai.lab_final import stage_final
    con.execute("INSERT INTO experiment(experiment_id,created_at,stage,name,org) VALUES ('x','2026','final','n','NAR')")
    with pytest.raises(SystemExit, match="実行済み"):
        stage_final(con, "NAR")


def test_採用グループの列だけがモデルに入る(filled):
    from keiba_ai.features import FeatureSpec
    f = build_features(_f(filled))
    spec = FeatureSpec.fit(f, groups=("speed",))
    x_cols = [c for c in spec.columns if c.startswith("x_")]
    assert x_cols and all(c.startswith("x_speed_") for c in x_cols)
    assert not [c for c in FeatureSpec.fit(f, groups=()).columns if c.startswith("x_")]
