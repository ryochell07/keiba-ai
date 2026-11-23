import numpy as np
import pandas as pd

from keiba_ai.probs import win_probs_from_history


def _races(n_races, rng, with_result=True):
    rows = []
    for r in range(n_races):
        n = int(rng.integers(6, 13))
        skill = rng.normal(size=n)
        pm = np.exp(skill + rng.normal(scale=0.5, size=n))
        pm /= pm.sum()
        win = rng.choice(n, p=np.exp(skill) / np.exp(skill).sum())
        for i in range(n):
            row = {"race_id": f"R{r}", "umaban": i + 1, "score": skill[i] + rng.normal(scale=0.5), "tau1": 1.0,
                   "p_mkt": pm[i], "odds": 1 / pm[i], "mkt_ok": True}
            if with_result:
                row["won"] = int(i == win)
            rows.append(row)
    return pd.DataFrame(rows)


def test_probs_normalized_and_uses_only_history():
    rng = np.random.default_rng(0)
    hist = _races(600, rng)
    cur = _races(50, rng, with_result=False)
    p = win_probs_from_history(hist, cur)
    assert p.shape == (len(cur),)
    assert np.all(p > 0)
    sums = pd.Series(p).groupby(cur["race_id"]).sum()
    assert np.allclose(sums, 1.0)


def test_falls_back_without_history_and_handles_missing_market():
    rng = np.random.default_rng(1)
    cur = _races(5, rng, with_result=False)
    cur.loc[cur.index[:3], "mkt_ok"] = False
    p = win_probs_from_history(cur.iloc[:0].assign(won=[]), cur)
    assert np.isfinite(p).all()
    assert np.allclose(pd.Series(p).groupby(cur["race_id"]).sum(), 1.0)
