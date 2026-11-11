"""学習手順のテスト: 位置別の温度の推定と、検証分も含めた学習し直し。データはすべて架空。"""
import numpy as np
import pandas as pd
import pytest

from keiba_ai.db import connect
from keiba_ai.features import build_features, load_frame
from keiba_ai.gbm import REFIT_SCALE, ROUNDS, fit_taus, order_matrix, pl_nll, train_race_model
from keiba_ai.ingest import save_race
from keiba_ai.providers.base import EntryRec, RaceRec

# 乱数と並列の順序で結果が変わらないよう、木の作り方を固定する。小さなデータでも分割できる設定にする
FIXED = dict(min_data_in_leaf=20, learning_rate=0.1, num_threads=1, force_col_wise=True, deterministic=True)


def _sample(taus, n_races=4000, n=10, seed=0):
    """点数を位置別の温度で割った Plackett-Luce から 1〜3 着を作る。"""
    rng = np.random.default_rng(seed)
    scores = rng.normal(0, 1.0, n_races * n)
    pos = np.full(n_races * n, np.nan)
    for r in range(n_races):
        s = scores[r * n:(r + 1) * n]
        alive = np.ones(n, bool)
        for k, t in enumerate(taus):
            p = np.where(alive, np.exp(s / t), 0.0)
            pick = rng.choice(n, p=p / p.sum())
            pos[r * n + pick] = k + 1
            alive[pick] = False
    return scores, order_matrix(pd.DataFrame({"race_id": np.repeat(np.arange(n_races), n), "finish_pos": pos}))


@pytest.mark.parametrize("true,n_races,tol", [((0.8, 1.0, 1.3), 4000, 0.1), ((0.9, 1.0, 1.1), 500, 0.2)])
def test_温度は初期値のままにならず最適な値になる(true, n_races, tol):
    scores, (idx, mask, orders) = _sample(true, n_races=n_races)
    taus = fit_taus(scores, idx, mask, orders)
    assert taus == pytest.approx(true, abs=tol)
    # どの温度を少し動かしても良くならない。1 との差が小さいデータでは、以前は初期値の 1 で止まっていた
    best = pl_nll(scores, idx, mask, orders, taus)
    assert best < pl_nll(scores, idx, mask, orders)
    for k in range(3):
        for step in (0.99, 1.01):
            moved = tuple(t * step if i == k else t for i, t in enumerate(taus))
            assert best <= pl_nll(scores, idx, mask, orders, moved)


def test_点数に差がなければ温度は1のまま():
    scores, (idx, mask, orders) = _sample((1.0, 1.0, 1.0), n_races=200)
    assert fit_taus(np.zeros_like(scores), idx, mask, orders) == (1.0, 1.0, 1.0)


def test_3着まで決まらないレースがあっても温度を出せる():
    scores, (idx, mask, orders) = _sample((0.8, 1.0, 1.3), n_races=1500)
    orders[::2, 2] = -1   # 半分のレースは 2 着まで
    orders[::5, 1:] = -1  # 一部は 1 着だけ
    taus = fit_taus(scores, idx, mask, orders)
    assert taus == pytest.approx((0.8, 1.0, 1.3), abs=0.2)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """馬ごとの潜在能力で着順が決まる小さな世界。"""
    con = connect(tmp_path_factory.mktemp("training") / "k.db")
    rng = np.random.default_rng(11)
    ability = rng.normal(0, 1.0, 150)
    start = np.datetime64("2028-01-08")
    for i in range(400):
        hs = rng.choice(150, 8, replace=False)
        order = np.argsort(-(ability[hs] + rng.gumbel(0, 1.0, 8)))
        rank = np.empty(8, int)
        rank[order] = np.arange(1, 9)
        r = RaceRec(org="JRA", course_code="05", course_name="05", race_date=str(start + np.timedelta64(i // 6, "D")), race_no=i % 6 + 1, source="t", post_time="15:00",
                    surface="芝", distance=1600, direction="左", weather="晴", going="良", class_name="オープン")
        for u, h in enumerate(hs, 1):
            r.entries.append(EntryRec(umaban=u, horse_id=f"H{h}", horse_name=f"架空{h}", waku=u, sex="牡", age=4, carried_weight=56.0, jockey_id=f"J{h % 9}", jockey_name="騎手",
                                      trainer_id=f"T{h % 7}", trainer_name="調教師", body_weight=470.0, finish_pos=int(rank[u - 1]), status="finished",
                                      time_sec=96 + rank[u - 1] * .2, last3f=34.0, corners=f"{u},{u}", popularity_final=u))
        save_race(con, r)
    con.commit()
    frame = load_frame(con, "JRA")
    yield frame, build_features(frame)
    con.close()


def test_検証分も含めて学習し直したモデルを使う(data):
    frame, feats = data
    m = train_race_model(frame, feats, "JRA", n_seeds=2, params=FIXED)
    info = m.metrics
    assert info["refit"] is True and info["rounds_cap"] == ROUNDS
    assert info["refit_rounds"] == [max(1, round(b * REFIT_SCALE)) for b in info["best_rounds"]]
    assert [b.num_trees() for b in m.boosters] == info["refit_rounds"]
    assert max(info["best_rounds"]) > 5    # 早期終了が効く前に止まっていない
    # 木は検証期間の最後まで学習している。検証期間そのものは記録に残る
    assert info["tree_train_end"] == m.train_end == info["valid_end"] == frame["race_date"].max()
    assert info["valid_start"] <= info["valid_end"]
    assert info["n_train_races"] == frame["race_id"].nunique() == 400 and info["n_valid_races"] == 60


def test_学習し直さない指定では従来どおり検証分を学習しない(data):
    frame, feats = data
    plain = train_race_model(frame, feats, "JRA", n_seeds=2, params=FIXED, refit=False)
    info = plain.metrics
    assert info["refit"] is False and info["refit_rounds"] == []
    assert [b.num_trees() for b in plain.boosters] == info["best_rounds"]
    assert info["tree_train_end"] <= info["valid_start"]
    assert info["n_train_races"] == 400 - 60
    # 温度と検証の指標は学習し直す前のモデルで決めるので、学習し直しても変わらない
    again = train_race_model(frame, feats, "JRA", n_seeds=2, params=FIXED)
    assert again.metrics["best_rounds"] == info["best_rounds"]
    assert again.taus == pytest.approx(plain.taus, rel=1e-9)
    assert again.metrics["valid_pl_nll"] == pytest.approx(info["valid_pl_nll"], rel=1e-9)
    # 学習し直したモデルは検証分を学習済みなので、検証分への当てはまりは必ず良くなる
    valid = frame["race_id"].isin(frame.drop_duplicates("race_id")["race_id"].tolist()[-60:]).to_numpy()
    idx, mask, orders = order_matrix(frame[valid].reset_index(drop=True).assign(finish_pos=lambda d: pd.to_numeric(d["finish_pos"])))
    vx = feats[valid].reset_index(drop=True)
    assert pl_nll(again.scores(vx), idx, mask, orders) < pl_nll(plain.scores(vx), idx, mask, orders)
