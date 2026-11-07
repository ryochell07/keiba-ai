"""Plackett-Luce 目的関数の LightGBM ランキングモデル。"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from .features import FeatureSpec, mask_late_info

# 調整期間 (2025-03〜08) の Optuna で選択。封鎖テストは使っていない
PARAMS = dict(learning_rate=0.0312, num_leaves=58, min_data_in_leaf=364, feature_fraction=0.592, bagging_fraction=0.856,
              bagging_freq=1, lambda_l2=2.68, verbosity=-1, num_threads=0)
TOP_K_WEIGHTS = (1.0, 0.7, 0.5)


def order_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """race_id ごとに 行番号パディング行列・マスク・着順順の位置 を作る。frame の race_id は連続していること。"""
    sizes = frame.groupby("race_id", sort=False).size().to_numpy()
    max_n = sizes.max()
    idx = np.full((len(sizes), max_n), -1, dtype=np.int64)
    orders = np.full((len(sizes), 3), -1, dtype=np.int64)
    pos = frame["finish_pos"].to_numpy(dtype=float)
    start = 0
    for r, n in enumerate(sizes):
        idx[r, :n] = np.arange(start, start + n)
        p = pos[start:start + n]
        valid = np.flatnonzero(~np.isnan(p))
        ranked = valid[np.argsort(p[valid], kind="stable")][:3]
        orders[r, :len(ranked)] = ranked
        start += n
    return idx, idx >= 0, orders


def pl_grad_hess(scores: np.ndarray, idx: np.ndarray, mask: np.ndarray, orders: np.ndarray):
    grad, hess = np.zeros_like(scores), np.zeros_like(scores)
    s = np.where(mask, scores[np.clip(idx, 0, None)], -np.inf)
    alive = mask.copy()
    rows = np.arange(len(idx))
    for k, w in enumerate(TOP_K_WEIGHTS):
        pick = orders[:, k]
        has = pick >= 0
        if not has.any():
            break
        lg = np.where(alive, s, -np.inf)
        top = lg.max(axis=1, keepdims=True)
        p = np.exp(lg - np.where(np.isfinite(top), top, 0.0))
        p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-300)
        p = np.where(has[:, None], p, 0.0)
        np.add.at(grad, idx[mask], (w * p)[mask])
        np.add.at(hess, idx[mask], (w * p * (1 - p))[mask])
        np.add.at(grad, idx[rows[has], pick[has]], -w)
        alive[rows[has], pick[has]] = False
    return grad, np.maximum(hess, 1e-6)


def pl_nll(scores: np.ndarray, idx, mask, orders, taus=(1.0, 1.0, 1.0), top_k: int = 3) -> float:
    s = np.where(mask, scores[np.clip(idx, 0, None)], -np.inf)
    alive = mask.copy()
    total, cnt = 0.0, 0
    rows = np.arange(len(idx))
    for k in range(top_k):
        pick = orders[:, k]
        has = pick >= 0
        if not has.any():
            break
        lg = np.where(alive, s / taus[k], -np.inf)
        total += float((logsumexp(lg, axis=1)[has] - lg[rows[has], pick[has]]).sum())
        cnt += int(has.sum())
        alive[rows[has], pick[has]] = False
    return total / max(cnt, 1)


def fit_taus(scores, idx, mask, orders) -> tuple[float, float, float]:
    """着順が下がるほどランダム性が増す効果を位置別の温度として推定する (Henery/Stern 型の補正)。"""
    def f(lt):
        return pl_nll(scores, idx, mask, orders, tuple(np.exp(lt)))
    r = minimize(f, np.zeros(3), method="Nelder-Mead", options=dict(maxiter=300, xatol=1e-3, fatol=1e-5))
    return tuple(float(x) for x in np.exp(r.x))


def softmax_by_race(scores: np.ndarray, race_ids: pd.Series, tau: float = 1.0) -> np.ndarray:
    s = pd.Series(scores / tau, index=race_ids.index)
    m = s.groupby(race_ids, sort=False).transform("max")
    e = np.exp(s - m)
    return (e / e.groupby(race_ids, sort=False).transform("sum")).to_numpy()


@dataclass
class RaceModel:
    org: str
    spec: FeatureSpec
    boosters: list = field(default_factory=list)
    taus: tuple = (1.0, 1.0, 1.0)
    metrics: dict = field(default_factory=dict)
    train_start: str = ""
    train_end: str = ""

    def member_scores(self, feats: pd.DataFrame) -> np.ndarray:
        return self.member_scores_x(self.spec.transform(feats))

    def member_scores_x(self, X: pd.DataFrame) -> np.ndarray:
        """変換済みの入力から採点する。記録した入力の再現に使う。"""
        return np.column_stack([b.predict(X) for b in self.boosters])

    def scores_from(self, X: pd.DataFrame) -> np.ndarray:
        return self.member_scores_x(X).mean(axis=1)

    def scores(self, feats: pd.DataFrame) -> np.ndarray:
        return self.member_scores(feats).mean(axis=1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**self.__dict__, "boosters": [b.model_to_string() for b in self.boosters]}
        path.write_bytes(pickle.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> "RaceModel":
        d = pickle.loads(path.read_bytes())
        d["boosters"] = [lgb.Booster(model_str=s) for s in d["boosters"]]
        return cls(**d)


def trainable(frame: pd.DataFrame, feats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1着馬が確定したレースだけを学習対象にする。"""
    done = frame.groupby("race_id")["finish_pos"].transform(lambda s: (pd.to_numeric(s) == 1).any())
    keep = done.to_numpy()
    return frame[keep].reset_index(drop=True), feats[keep].reset_index(drop=True)


def train_race_model(frame: pd.DataFrame, feats: pd.DataFrame, org: str, n_seeds: int = 3, rounds: int = 800,
                     valid_share: float = 0.15, groups=None, params: dict | None = None, offset: np.ndarray | None = None) -> RaceModel:
    """時系列の末尾を検証に使い、早期終了と温度を決める。ランダム分割はしない。"""
    frame, feats = trainable(frame, feats)
    frame = frame.copy()
    frame["finish_pos"] = pd.to_numeric(frame["finish_pos"], errors="coerce")
    races = frame.drop_duplicates("race_id")["race_id"].tolist()
    if len(races) < 100:
        raise ValueError(f"{org}: 学習レース数が不足しています ({len(races)})")
    n_valid = max(30, int(len(races) * valid_share))
    valid_ids = set(races[-n_valid:])
    is_valid = frame["race_id"].isin(valid_ids).to_numpy()
    spec = FeatureSpec.fit(feats[~is_valid], groups=groups)
    model = RaceModel(org=org, spec=spec, train_start=frame["race_date"].min(), train_end=frame["race_date"].max())
    tr_f, va_f = frame[~is_valid].reset_index(drop=True), frame[is_valid].reset_index(drop=True)
    Xv = spec.transform(feats[is_valid].reset_index(drop=True))
    tr_idx, tr_mask, tr_ord = order_matrix(tr_f)
    va_idx, va_mask, va_ord = order_matrix(va_f)
    best_rounds = []
    for seed in range(n_seeds):
        Xt = spec.transform(mask_late_info(feats[~is_valid].reset_index(drop=True), tr_f["race_id"], seed=seed))
        dtrain = lgb.Dataset(Xt, free_raw_data=False, categorical_feature=spec.categorical)
        dvalid = lgb.Dataset(Xv, reference=dtrain, free_raw_data=False, categorical_feature=spec.categorical)

        def obj(preds, _ds):
            return pl_grad_hess(preds, tr_idx, tr_mask, tr_ord)

        def metric(preds, _ds):
            return "pl_nll", pl_nll(preds, va_idx, va_mask, va_ord), False

        booster = lgb.train({**PARAMS, **(params or {}), "objective": obj, "seed": seed + 1}, dtrain, rounds, valid_sets=[dvalid],
                            feval=metric, callbacks=[lgb.early_stopping(60, verbose=False)])
        best_rounds.append(booster.best_iteration)
        booster = _trim(booster)
        model.boosters.append(booster)
    va_scores = model.scores(feats[is_valid].reset_index(drop=True))
    model.taus = fit_taus(va_scores, va_idx, va_mask, va_ord)
    model.metrics = {"best_rounds": best_rounds, "n_train_races": len(races) - n_valid, "n_valid_races": n_valid,
                     "valid_pl_nll": pl_nll(va_scores, va_idx, va_mask, va_ord, model.taus),
                     "tree_train_end": tr_f["race_date"].max(), "valid_start": va_f["race_date"].min(), "valid_end": va_f["race_date"].max(),
                     "hyperparams": {k: v for k, v in {**PARAMS, **(params or {})}.items() if k not in ("verbosity", "num_threads")}}
    return model


def _trim(booster: lgb.Booster) -> lgb.Booster:
    return lgb.Booster(model_str=booster.model_to_string(num_iteration=max(booster.best_iteration, 1)))
