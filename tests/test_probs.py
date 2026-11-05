import numpy as np
import pytest

from keiba_ai.ev import STATUS_NO_ODDS, expected
from keiba_ai.probs import all_bet_probs, joint_top3, place_slots, prune_combos

WK = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]


@pytest.fixture
def probs():
    s = np.random.default_rng(1).normal(size=12)
    return all_bet_probs(list(range(1, 13)), WK, s, (1.0, 1.2, 1.4)), s


def test_合計が理論値になる(probs):
    p, _ = probs
    for b, want in dict(win=1, place=3, quinella=1, exacta=1, wide=3, trio=1, trifecta=1, bracket_quinella=1).items():
        assert sum(p[b].values()) == pytest.approx(want), b


def test_組み合わせ数(probs):
    p, _ = probs
    assert (len(p["quinella"]), len(p["exacta"]), len(p["trio"]), len(p["trifecta"])) == (66, 132, 220, 1320)


def test_券種間の整合(probs):
    p, _ = probs
    assert p["quinella"]["1-2"] == pytest.approx(p["exacta"]["1-2"] + p["exacta"]["2-1"])
    trio = sum(v for k, v in p["trifecta"].items() if set(k.split("-")) == {"1", "2", "3"})
    assert p["trio"]["1-2-3"] == pytest.approx(trio)
    wide = sum(v for k, v in p["trio"].items() if {"1", "2"} <= set(k.split("-")))
    assert p["wide"]["1-2"] == pytest.approx(wide)
    place1 = sum(v for k, v in p["trio"].items() if "1" in k.split("-"))
    assert p["place"]["1"] == pytest.approx(place1)


def test_単純な掛け算ではない(probs):
    p, _ = probs
    naive = p["win"]["1"] * p["win"]["2"]
    assert p["exacta"]["1-2"] != pytest.approx(naive)


def test_シミュレーションと一致():
    s = np.array([1.2, .4, 0., -.3, -.8, .9, .1, -1.])
    taus = (1., 1.3, 1.6)
    p1, p2, p3 = joint_top3(s, taus)
    rng = np.random.default_rng(0)
    N, c = 200000, 0
    e1, e2, e3 = [np.exp(s / t) for t in taus]
    for _ in range(N):
        a = rng.choice(8, p=e1 / e1.sum())
        q = e2.copy(); q[a] = 0
        b = rng.choice(8, p=q / q.sum())
        q = e3.copy(); q[a] = q[b] = 0
        d = rng.choice(8, p=q / q.sum())
        c += (a, b, d) == (0, 5, 1)
    assert c / N == pytest.approx(p3[0, 5, 1], abs=0.002)


def test_複勝の払戻着数():
    assert [place_slots(n) for n in (4, 5, 7, 8, 18)] == [0, 2, 2, 3, 3]


def test_枠連は8頭以下で発売なし():
    r = all_bet_probs(list(range(1, 9)), [1, 2, 3, 4, 5, 6, 7, 8], np.zeros(8))
    assert "bracket_quinella" not in r and "bracket_exacta" not in r
    assert sum(r["place"].values()) == pytest.approx(3)


def test_地方は枠単あり():
    r = all_bet_probs(list(range(1, 10)), [1, 2, 3, 4, 5, 6, 7, 8, 8], np.zeros(9), org="NAR")
    assert sum(r["bracket_exacta"].values()) == pytest.approx(1)


def test_期待値():
    assert expected(0.25, 5.0) == (0.25, 0.25)
    lo, hi = expected(0.5, 1.5, 2.5)
    assert lo == pytest.approx(-0.25) and hi == pytest.approx(0.25)
    assert expected(0.5, None) == (None, None)
    assert STATUS_NO_ODDS == "オッズ未取得"


def test_候補絞り込み():
    combos = {"1-2": .5, "1-3": .3, "2-9": .2}
    assert prune_combos(combos, {1, 2, 3}, 1) == {"1-2": .5}


def test_合成はオッズの順位と金額を反映する():
    from keiba_ai.probs import blend_strength
    s = np.array([0.0, 0.0, 0.0, 0.0])
    odds = [2.0, 4.0, 8.0, 16.0]
    p = all_bet_probs([1, 2, 3, 4], [1, 2, 3, 4], blend_strength(s, odds, 0.1, 1.0, 1.0))["win"]
    assert p["1"] == pytest.approx(0.5 / 0.9375)
    assert p["1"] > p["2"] > p["3"] > p["4"]


def test_合成係数の推定は情報のある方に重みを置く():
    import pandas as pd
    from keiba_ai.probs import fit_blend
    rng = np.random.default_rng(0)
    rows = []
    for r in range(400):
        strength = rng.normal(0, 1, 8)
        p = np.exp(strength) / np.exp(strength).sum()
        w = rng.choice(8, p=p)
        odds = 0.8 / p
        noise = rng.normal(0, 1, 8)  # AI は完全なノイズ
        for i in range(8):
            rows.append(dict(race_id=r, score=noise[i], tau1=1.0, odds=odds[i], win=int(i == w)))
    a, b = fit_blend(pd.DataFrame(rows))
    assert abs(a) < 0.15 and 0.8 < b < 1.2
