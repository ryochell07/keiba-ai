"""過去の確定オッズ特徴 (x_pmo_) が当該レースのオッズを使わないことを確かめる。"""
import numpy as np
import pandas as pd

from keiba_ai.features import add_market, build_features, load_frame

TARGET = "2024-06-15"


def _with_odds(con):
    df = load_frame(con, "JRA")
    rng = np.random.default_rng(0)
    df = add_market(df, con)
    df["h_pmo_odds"] = rng.uniform(1.5, 80.0, len(df))
    return df


def test_当該レースのオッズを変えても過去オッズ特徴は変わらない(filled):
    df = _with_odds(filled)
    base = build_features(df)
    cols = [c for c in base.columns if c.startswith("x_pmo_")]
    assert cols
    mutated = df.copy()
    target = mutated["race_date"] == TARGET
    mutated.loc[target, "h_pmo_odds"] = mutated.loc[target, "h_pmo_odds"].to_numpy()[::-1] * 3
    mutated.loc[target, "finish_pos"] = mutated.loc[target, "finish_pos"].to_numpy()[::-1]
    after = build_features(mutated)
    pd.testing.assert_frame_equal(base.loc[target, cols], after.loc[target, cols], check_exact=False)
    # 後のレースには反映される
    later = df["race_date"] > TARGET
    assert not base.loc[later, cols].equals(after.loc[later, cols])


def test_未来のレースを消しても過去オッズ特徴は変わらない(filled):
    df = _with_odds(filled)
    full = build_features(df)
    cut = df[df["race_date"] <= TARGET].reset_index(drop=True)
    part = build_features(cut)
    keep = (df["race_date"] <= TARGET).to_numpy()
    cols = [c for c in part.columns if c.startswith("x_pmo_")]
    pd.testing.assert_frame_equal(full[keep].reset_index(drop=True)[cols], part[cols], check_exact=False)


def test_オッズがなくても動く(filled):
    f = build_features(load_frame(filled, "JRA"))
    assert f["x_pmo_m3"].isna().all()
