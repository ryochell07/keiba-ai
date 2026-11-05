"""全券種の確率計算。一般化 Plackett-Luce で 1→2→3 着の同時確率を厳密に列挙する。

単勝確率の単純な掛け算ではなく、上位馬が抜けた後の条件付き確率を位置別の温度で計算するので、
着順の相関 (強い馬が 1 着なら他馬の 2 着確率が上がる等) が反映される。18 頭でも 4896 通りなので列挙できる。
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-300


def joint_top3(strength: np.ndarray, taus=(1.0, 1.0, 1.0)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P(1着=i), P(1着=i,2着=j), P(1着=i,2着=j,3着=k) を返す。"""
    n = len(strength)
    s = strength - strength.max()
    e = [np.exp(s / t) for t in taus]
    p1 = e[0] / e[0].sum()
    p2 = np.zeros((n, n))
    p3 = np.zeros((n, n, n)) if n >= 3 else np.zeros((n, n, max(n, 1)))
    for i in range(n):
        d2 = e[1].sum() - e[1][i]
        if d2 <= 0:
            continue
        cond2 = e[1] / d2
        cond2[i] = 0.0
        p2[i] = p1[i] * cond2
        for j in range(n):
            if j == i or n < 3:
                continue
            d3 = e[2].sum() - e[2][i] - e[2][j]
            if d3 <= 0:
                continue
            c3 = e[2] / d3
            c3[i] = c3[j] = 0.0
            p3[i, j] = p2[i, j] * c3
    return p1, p2, p3


def place_slots(n: int) -> int:
    """複勝の払戻対象着数。8 頭以上 3 着まで、5〜7 頭 2 着まで、4 頭以下は発売なし。"""
    return 3 if n >= 8 else (2 if n >= 5 else 0)


def all_bet_probs(umaban: list[int], waku: list[int | None], strength: np.ndarray, taus=(1.0, 1.0, 1.0),
                  org: str = "JRA") -> dict[str, dict[str, float]]:
    """券種 → 組番 → 確率。組番は馬番 (枠連は枠番) で、順序なし券種は昇順。"""
    n = len(umaban)
    u = np.asarray(umaban)
    p1, p2, p3 = joint_top3(np.asarray(strength, dtype=float), taus)
    out: dict[str, dict[str, float]] = {b: {} for b in
                                        ("win", "place", "quinella", "exacta", "wide", "trio", "trifecta",
                                         "bracket_quinella", "bracket_exacta")}
    for i in range(n):
        out["win"][str(u[i])] = float(p1[i])
    slots = place_slots(n)
    if slots == 3:
        m = p3.sum(axis=(1, 2)) + p3.sum(axis=(0, 2)) + p3.sum(axis=(0, 1))
    elif slots == 2:
        m = p2.sum(axis=1) + p2.sum(axis=0)
    else:
        m = None
    if m is not None:
        for i in range(n):
            out["place"][str(u[i])] = float(m[i])
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            out["exacta"][f"{u[i]}-{u[j]}"] = float(p2[i, j])
            if i < j:
                a, b = sorted((u[i], u[j]))
                out["quinella"][f"{a}-{b}"] = float(p2[i, j] + p2[j, i])
                # ワイド: 2 頭とも 3 着以内 = 3 頭目が誰であれ 6 通りの並びの合計
                w = (p3[i, j, :].sum() + p3[j, i, :].sum() + p3[i, :, j].sum() + p3[j, :, i].sum()
                     + p3[:, i, j].sum() + p3[:, j, i].sum())
                out["wide"][f"{a}-{b}"] = float(w)
    if n >= 3:
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if len({i, j, k}) == 3:
                        out["trifecta"][f"{u[i]}-{u[j]}-{u[k]}"] = float(p3[i, j, k])
                        if i < j < k:
                            a, b, c = sorted((u[i], u[j], u[k]))
                            out["trio"][f"{a}-{b}-{c}"] = float(
                                p3[i, j, k] + p3[i, k, j] + p3[j, i, k] + p3[j, k, i] + p3[k, i, j] + p3[k, j, i])
    if all(w is not None for w in waku):
        wk = [int(w) for w in waku]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                key = "-".join(str(x) for x in sorted((wk[i], wk[j])))
                out["bracket_quinella"][key] = out["bracket_quinella"].get(key, 0.0) + float(p2[i, j])
                ek = f"{wk[i]}-{wk[j]}"
                out["bracket_exacta"][ek] = out["bracket_exacta"].get(ek, 0.0) + float(p2[i, j])
    if org == "JRA":
        out.pop("bracket_exacta")          # JRA に枠単なし
        if n < 9:
            out["bracket_quinella"] = {}   # JRA は 8 頭以下で枠連発売なし
    return {k: v for k, v in out.items() if v}


def blend_strength(score: np.ndarray, win_odds: np.ndarray, alpha: float, beta: float, tau1: float) -> np.ndarray:
    """AI のスコアと単勝オッズを合成した強さ。オッズは市場が評価した確率の対数として足す。"""
    return alpha * np.asarray(score) / tau1 + beta * np.log(1.0 / np.asarray(win_odds))


def fit_blend(frame) -> tuple[float, float]:
    """race_id / score / tau1 / odds / win 列を持つ過去データで合成係数を最尤推定する。"""
    from scipy.optimize import minimize

    ms = (frame["score"] / frame["tau1"]).to_numpy()
    lm = np.log(1 / frame["odds"].to_numpy())
    group = frame["race_id"].to_numpy()
    _, inverse = np.unique(group, return_inverse=True)
    n = inverse.max() + 1
    won = frame["win"].to_numpy() == 1

    def nll(t):
        s = t[0] * ms + t[1] * lm
        top = np.full(n, -np.inf)
        np.maximum.at(top, inverse, s)
        e = np.exp(s - top[inverse])
        total = np.zeros(n)
        np.add.at(total, inverse, e)
        return float(-np.log(np.maximum(e[won] / total[inverse[won]], 1e-12)).mean())

    r = minimize(nll, [0.5, 0.5], method="Nelder-Mead", options=dict(maxiter=400, xatol=1e-3, fatol=1e-6))
    return float(r.x[0]), float(r.x[1])


def candidate_horses(umaban: list[int], strength: np.ndarray, k: int) -> set[int]:
    order = np.argsort(-np.asarray(strength))
    return {int(umaban[i]) for i in order[:k]}


def prune_combos(combos: dict[str, float], candidates: set[int], top_n: int) -> dict[str, float]:
    """候補馬だけで構成される組番のうち、確率上位 top_n 件を残す。"""
    kept = [(c, p) for c, p in combos.items() if all(int(x) in candidates for x in c.split("-"))]
    kept.sort(key=lambda x: -x[1])
    return dict(kept[:top_n])
