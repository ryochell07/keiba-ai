"""精度改善の実験環境。

封鎖ルール: 2026-03-01 以降は最終評価専用。final 以外のステージはこの期間を読み込まない。
開発期間の区分:
  学習   ... 2025-02-28 まで
  調整用 ... 2025-03-01〜2025-08-31 (特徴量の採否、パラメータ)
  選択用 ... 2025-09-01〜2026-02-28 (調整で使っていない期間で確認)
"""
from __future__ import annotations

import json
import logging
import pickle
import shutil
import sqlite3
import time
from datetime import datetime

import numpy as np
import pandas as pd

from . import config
from .db import JST, now_jst
from .features import ADOPTED_GROUPS, FeatureSpec, GROUPS, add_market, build_features, load_frame
from .gbm import softmax_by_race, train_race_model, trainable

log = logging.getLogger(__name__)
LOCKED_FROM = "2026-03-01"
TUNE = ("2025-03-01", "2025-09-01")
SELECT = ("2025-09-01", LOCKED_FROM)
MAX_OPTUNA_TRIALS = 15
MIN_SAMPLE = 200


class LockedTestError(RuntimeError):
    pass


def lab_dir():
    d = config.DATA / "lab"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_dev(con, org: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """封鎖期間を除いた開発データ。特徴量は重いのでキャッシュする。"""
    cache = lab_dir() / f"dev_{org}.pkl"
    n = con.execute("SELECT COUNT(*) FROM race WHERE org=? AND race_date<?", (org, LOCKED_FROM)).fetchone()[0]
    if cache.exists():
        saved = pickle.loads(cache.read_bytes())
        if saved["n"] == n and saved["v"] == 2:
            return saved["frame"], saved["feats"]
    frame = load_frame(con, org, until="2026-02-28")
    frame = frame[frame["status"].notna()].reset_index(drop=True)
    assert_dev_only(frame)
    feats = build_features(add_market(frame, con))
    frame, feats = trainable(frame, feats)
    cache.write_bytes(pickle.dumps({"n": n, "v": 2, "frame": frame, "feats": feats}, protocol=5))
    return frame, feats


def assert_dev_only(frame: pd.DataFrame) -> None:
    if len(frame) and frame["race_date"].max() >= LOCKED_FROM:
        raise LockedTestError(f"封鎖期間のデータが開発に混入しています: {frame['race_date'].max()}")


def win_payouts(con, org: str) -> dict[tuple[str, int], int]:
    return {(r[0], int(r[1])): r[2] for r in con.execute(
        "SELECT p.race_id,p.combo,p.amount FROM payout p JOIN race r USING(race_id) WHERE r.org=? AND p.bet_type='win'", (org,))}


def race_metrics(ev: pd.DataFrame, p: np.ndarray, payouts: dict) -> pd.DataFrame:
    """レース単位の指標。ev は評価対象の行 (race_id, umaban, finish_pos)。"""
    d = ev[["race_id", "umaban", "finish_pos", "race_date"]].copy()
    d["p"] = p
    d["y"] = (pd.to_numeric(d["finish_pos"], errors="coerce") == 1).astype(float)
    d["se"] = (d["p"] - d["y"]) ** 2
    d["pay"] = [payouts.get((r, int(u)), 0) for r, u in zip(d["race_id"], d["umaban"])]
    g = d.groupby("race_id", sort=False)
    ll = -np.log(np.maximum(d.loc[d["y"] == 1].groupby("race_id", sort=False)["p"].first(), 1e-9))
    top = d.loc[g["p"].idxmax()].set_index("race_id")
    return pd.DataFrame({"ll": ll, "brier": g["se"].sum(), "hit": top["y"], "ret": top["pay"], "date": top["race_date"]}).dropna()


def summarize(m: pd.DataFrame, base: pd.DataFrame | None = None, n_boot: int = 1000) -> dict:
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(m), size=(n_boot, len(m)))
    roi = m["ret"].to_numpy()[idx].sum(axis=1) / (100 * len(m))
    cum = (m["ret"] - 100).cumsum()
    out = dict(n_races=len(m), log_loss=float(m["ll"].mean()), brier=float(m["brier"].mean()), top1_hit=float(m["hit"].mean()),
               roi=float(m["ret"].sum() / (100 * len(m))), roi_ci_low=float(np.percentile(roi, 2.5)), roi_ci_high=float(np.percentile(roi, 97.5)),
               max_drawdown=float((cum.cummax().clip(lower=0) - cum).max()))
    if base is not None:
        common = m.index.intersection(base.index)
        diff = (m.loc[common, "ll"] - base.loc[common, "ll"]).to_numpy()
        i2 = rng.integers(0, len(diff), size=(n_boot, len(diff)))
        boot = diff[i2].mean(axis=1)
        out.update(delta_log_loss=float(diff.mean()), delta_ci_low=float(np.percentile(boot, 2.5)), delta_ci_high=float(np.percentile(boot, 97.5)))
    return out


def record(con, stage: str, name: str, org: str, train: str, ev: str, res: dict, algo="lightgbm-pl", groups=(), params=None, calibration=None,
           decision=None, locked=False, seconds=None) -> None:
    eid = f"{stage}-{name}-{org}-{datetime.now(JST).strftime('%Y%m%d%H%M%S%f')}"
    con.execute("""INSERT INTO experiment(experiment_id,created_at,stage,name,org,train_period,eval_period,features,algo,params,calibration,n_races,log_loss,brier,top1_hit,roi,
                   n_bets,max_drawdown,roi_ci_low,roi_ci_high,delta_log_loss,delta_ci_low,delta_ci_high,decision,used_locked_test,seconds)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (eid, now_jst(), stage, name, org, train, ev, json.dumps(list(groups)), algo, json.dumps(params or {}), calibration, res.get("n_races"), res.get("log_loss"), res.get("brier"),
                 res.get("top1_hit"), res.get("roi"), res.get("n_races"), res.get("max_drawdown"), res.get("roi_ci_low"), res.get("roi_ci_high"), res.get("delta_log_loss"),
                 res.get("delta_ci_low"), res.get("delta_ci_high"), decision, int(locked), seconds))
    con.commit()


def fit_predict(frame, feats, since: str, train_end: str, ev_mask, groups=(), params=None, n_seeds=1) -> tuple[np.ndarray, object]:
    tr = ((frame["race_date"] >= since) & (frame["race_date"] < train_end)).to_numpy()
    model = train_race_model(frame[tr].reset_index(drop=True), feats[tr].reset_index(drop=True), "lab", n_seeds=n_seeds, groups=groups, params=params)
    ev_f = frame[ev_mask].reset_index(drop=True)
    s = model.scores(feats[ev_mask].reset_index(drop=True))
    return softmax_by_race(s, ev_f["race_id"], model.taus[0]), model


def period_mask(frame, period) -> np.ndarray:
    return ((frame["race_date"] >= period[0]) & (frame["race_date"] < period[1])).to_numpy()


def pop_market(frame, train_mask, ev_mask) -> np.ndarray:
    """確定人気から作る市場確率の代用。オッズがない期間の比較専用 (発走後の情報なので特徴量ではなく基準線)。"""
    tr = frame[train_mask].copy()
    tr["pop"] = pd.to_numeric(tr["popularity_final"], errors="coerce")
    tr["y"] = (pd.to_numeric(tr["finish_pos"], errors="coerce") == 1).astype(float)
    tr["nb"] = pd.cut(tr["n_runners"], [0, 8, 10, 12, 99], labels=False)
    table = tr.groupby(["nb", "pop"])["y"].agg(["sum", "count"])
    rate = ((table["sum"] + 0.5) / (table["count"] + 5)).to_dict()
    ev = frame[ev_mask].reset_index(drop=True)
    pop = pd.to_numeric(ev["popularity_final"], errors="coerce").fillna(99)
    nb = pd.cut(ev["n_runners"], [0, 8, 10, 12, 99], labels=False)
    p = pd.Series([rate.get((b, q), 0.005) for b, q in zip(nb, pop)], index=ev.index)
    return (p / p.groupby(ev["race_id"]).transform("sum")).to_numpy()


# ---- 弱点分析 ------------------------------------------------------------------------------
def _bucket_table(d: pd.DataFrame, key) -> list[dict]:
    rows = []
    for k, g in d.groupby(key, dropna=True, observed=True):
        y, p = g["y"].to_numpy(), np.clip(g["p"].to_numpy(), 1e-6, 1 - 1e-6)
        rows.append({"key": str(k), "n_races": int(g["race_id"].nunique()), "n_horses": len(g), "hit": float(y.mean()), "pred": float(p.mean()),
                     "gap": float(p.mean() - y.mean()), "roi": float(g["pay"].sum() / (100 * len(g))),
                     "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), "brier": float(((p - y) ** 2).mean()),
                     "market_pred": float(g["mp"].mean()), "low_sample": len(g) < MIN_SAMPLE})
    return rows


def stage_analyze(con, org):
    frame, feats = load_dev(con, org)
    pay = win_payouts(con, org)
    period = (TUNE[0], LOCKED_FROM)
    ev = period_mask(frame, period)
    t0 = time.time()
    p, model = fit_predict(frame, feats, "2024-01-01", TUNE[0], ev, groups=())
    tr = ((frame["race_date"] >= "2024-01-01") & (frame["race_date"] < TUNE[0])).to_numpy()
    f = frame[ev].reset_index(drop=True)
    x = feats[ev].reset_index(drop=True)
    mp = pop_market(frame, tr, ev)
    m = race_metrics(f, p, pay)
    mm = race_metrics(f, mp, pay)
    res = summarize(m, mm)
    record(con, "analyze", "baseline_v1_dev", org, f"2024-01-01..{TUNE[0]}", f"{period}", res, seconds=time.time() - t0)
    d = pd.DataFrame({"race_id": f["race_id"], "p": p, "mp": mp, "y": (pd.to_numeric(f["finish_pos"]) == 1).astype(float),
                      "pay": [pay.get((r, int(u)), 0) for r, u in zip(f["race_id"], f["umaban"])]})
    pop = pd.to_numeric(f["popularity_final"], errors="coerce")
    dims = {
        "競馬場": f["course_code"].str.replace("N", "", n=1), "距離": pd.cut(f["distance"], [0, 1000, 1300, 1500, 1700, 2100, 9999]).astype(str), "芝ダート": f["surface"], "馬場": f["going"], "天候": f["weather"],
        "頭数": pd.cut(f["n_runners"], [0, 7, 9, 11, 13, 99]).astype(str), "クラス": x["class_ord"].round(), "年齢条件": x["x_cond_age"],
        "確定人気": pd.cut(pop, [0, 1, 2, 3, 6, 9, 99], labels=["1番人気", "2番人気", "3番人気", "4-6番人気", "7-9番人気", "10番人気以下"]),
        "馬齢": f["age"].clip(upper=9), "性別": f["sex"], "前走着順": pd.cut(x["last_finish"], [0, 1, 3, 5, 9, 99]).astype(str),
        "走数": pd.cut(x["runs_count"], [-1, 0, 2, 5, 10, 30, 999]).astype(str), "脚質": pd.cut(x["corner_first_m5"], [0, .2, .45, .75, 1.01], labels=["逃げ", "先行", "差し", "追込"]),
        "騎手変更": x["jockey_changed"], "休養": pd.cut(x["rest_days"], [0, 14, 30, 60, 120, 9999]).astype(str),
        "距離適性(同距離帯の複勝率)": pd.cut(x["hs_dist_h_top3_rate"], [-.01, 0, .2, .4, .6, 1.01]).astype(str), "馬場適性(同馬場の複勝率)": pd.cut(x["hs_going_h_top3_rate"], [-.01, 0, .2, .4, .6, 1.01]).astype(str),
        "AI勝率": pd.cut(d["p"], [0, .02, .05, .1, .15, .2, .3, .5, 1.01]).astype(str),
        "AIと人気代用の差": pd.cut(d["p"] - d["mp"], [-1, -.1, -.05, -.02, .02, .05, .1, 1]).astype(str), "AIの評価": np.where(d["p"] > d["mp"], "市場より高い", "市場より低い"),
    }
    report = {"period": period, "overall": res, "market_proxy": summarize(mm), "taus": model.taus, "dims": {}, "note": "市場は確定人気の代用で、評価専用。オッズ帯別はオッズのない期間なので対象外"}
    for name, key in dims.items():
        report["dims"][name] = _bucket_table(d.assign(_k=np.asarray(key)), "_k")
    for name, col in (("騎手", "jockey_id"), ("調教師", "trainer_id")):  # 上位の件数だけ
        counts = f[col].value_counts()
        big = counts[counts >= 500].index
        t = _bucket_table(d[f[col].isin(big).to_numpy()].assign(_k=f.loc[f[col].isin(big), col].to_numpy()), "_k")
        report["dims"][name] = sorted(t, key=lambda r: r["gap"])[:8] + sorted(t, key=lambda r: r["gap"])[-8:]
    (config.REPORTS / f"lab_analyze_{org}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return {"overall": res, "market_proxy": report["market_proxy"]}


def stage_baseline(con, org):
    row = con.execute("SELECT * FROM model_version WHERE org=? AND status='active'", (org,)).fetchone()
    dest = config.DATA / "baseline" / org
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(config.ROOT / row["path"], dest / "model.pkl")
    report = config.REPORTS / f"backtest_{org}.json"
    if report.exists():
        shutil.copy(report, dest / "backtest.json")
    (dest / "meta.json").write_text(json.dumps({k: row[k] for k in row.keys() if k != "comparison"}, ensure_ascii=False, indent=1), encoding="utf-8")
    b = json.loads(report.read_text(encoding="utf-8"))["summary"]
    res = dict(n_races=b["n_races"], log_loss=b["win_logloss_model"], brier=b["brier"], top1_hit=b["top1_hit"], roi=b["top1_win_roi"])
    record(con, "baseline", "baseline_registered", org, row["train_start"], "2026-03-01..2026-11-04", res, decision="baseline", locked=True)
    return {"saved": str(dest.relative_to(config.ROOT)), "metrics": res, "note": "既存結果の登録のみ。封鎖期間の再評価はしていない"}


def stage_history(con, org):
    frame, feats = load_dev(con, org)
    pay = win_payouts(con, org)
    ev = period_mask(frame, TUNE)
    f = frame[ev].reset_index(drop=True)
    base, out = None, []
    for since in ("2024-01-01", "2022-01-01", "2019-01-01"):
        t0 = time.time()
        p, _ = fit_predict(frame, feats, since, TUNE[0], ev, groups=())
        m = race_metrics(f, p, pay)
        base = m if base is None else base
        r = summarize(m, base)
        record(con, "history", f"since_{since[:4]}", org, f"{since}..{TUNE[0]}", str(TUNE), r, seconds=time.time() - t0)
        out.append({"since": since, **r, "seconds": round(time.time() - t0)})
    return out


def best_since(con, org) -> str:
    row = con.execute("SELECT train_period FROM experiment WHERE stage='history' AND org=? ORDER BY log_loss LIMIT 1", (org,)).fetchone()
    return row[0].split("..")[0] if row else "2024-01-01"


def stage_groups(con, org):
    """各グループを1つずつ足して調整期間で評価し、改善が確かなものだけ採用する。最後に選択期間で確認する。"""
    frame, feats = load_dev(con, org)
    pay = win_payouts(con, org)
    since = best_since(con, org)
    tune = period_mask(frame, TUNE)
    tf = frame[tune].reset_index(drop=True)
    p0, _ = fit_predict(frame, feats, since, TUNE[0], tune, groups=())
    base = race_metrics(tf, p0, pay)
    results, adopted = [], []
    base_res = summarize(base)
    for g in GROUPS:
        t0 = time.time()
        p, _ = fit_predict(frame, feats, since, TUNE[0], tune, groups=(g,))
        r = summarize(race_metrics(tf, p, pay), base)
        keep = r["delta_ci_high"] < 0
        adopted += [g] if keep else []
        record(con, "groups", g, org, f"{since}..{TUNE[0]}", str(TUNE), r, groups=(g,), decision="採用候補" if keep else "不採用", seconds=time.time() - t0)
        results.append({"group": g, "adopt": keep, **{k: r[k] for k in ("log_loss", "delta_log_loss", "delta_ci_low", "delta_ci_high", "top1_hit", "roi")}})
        log.info("%s", results[-1])
    sel = period_mask(frame, SELECT)
    sf = frame[sel].reset_index(drop=True)
    pa, _ = fit_predict(frame, feats, since, SELECT[0], sel, groups=())
    a = race_metrics(sf, pa, pay)
    pb, _ = fit_predict(frame, feats, since, SELECT[0], sel, groups=tuple(adopted))
    conf = summarize(race_metrics(sf, pb, pay), a)
    record(con, "groups", "adopted_set_confirm", org, f"{since}..{SELECT[0]}", str(SELECT), conf, groups=adopted, decision="確認")
    out = {"base_tune": base_res, "groups": results, "adopted": adopted, "confirm_on_select": conf, "base_select": summarize(a)}
    (config.REPORTS / f"lab_groups_{org}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def adopted_groups(org) -> tuple[str, ...]:
    f = config.REPORTS / f"lab_groups_{org}.json"
    return tuple(json.loads(f.read_text(encoding="utf-8"))["adopted"]) if f.exists() else ADOPTED_GROUPS


def _binary_models(frame, feats, tr, ev, groups):
    """二値分類の比較手法。レース内で正規化して確率にする。"""
    spec = FeatureSpec.fit(feats[tr], groups=groups)
    Xt, Xv = spec.transform(feats[tr]), spec.transform(feats[ev])
    y = (pd.to_numeric(frame.loc[tr, "finish_pos"]) == 1).astype(int).to_numpy()
    logit = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
    out = {}
    import lightgbm as lgb
    b = lgb.train({"objective": "binary", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 60, "feature_fraction": 0.7, "verbosity": -1},
                  lgb.Dataset(Xt, y, categorical_feature=spec.categorical), 400)
    out["LightGBM 二値"] = logit(b.predict(Xv))
    import xgboost as xgb
    m = xgb.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.7, tree_method="hist", n_jobs=8)
    m.fit(Xt.to_numpy(), y)
    out["XGBoost 二値"] = logit(m.predict_proba(Xv.to_numpy())[:, 1])
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=400, learning_rate=0.05, depth=6, verbose=False, allow_writing_files=False, thread_count=8)
    m.fit(Xt.to_numpy(), y)
    out["CatBoost 二値"] = logit(m.predict_proba(Xv.to_numpy())[:, 1])
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    numeric = [c for c in spec.columns if c not in spec.vocab]
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=300, C=0.1))
    m.fit(Xt[numeric].to_numpy(), y)
    out["ロジスティック回帰"] = logit(m.predict_proba(Xv[numeric].to_numpy())[:, 1])
    return out


def _lambdarank(frame, feats, tr, ev, groups):
    import lightgbm as lgb
    spec = FeatureSpec.fit(feats[tr], groups=groups)
    Xt, Xv = spec.transform(feats[tr]), spec.transform(feats[ev])
    pos = pd.to_numeric(frame.loc[tr, "finish_pos"], errors="coerce").fillna(99)
    label = np.select([pos == 1, pos == 2, pos == 3], [3, 2, 1], 0)
    sizes = frame[tr].groupby("race_id", sort=False).size().to_numpy()
    b = lgb.train({"objective": "lambdarank", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 60, "feature_fraction": 0.7, "verbosity": -1, "lambdarank_truncation_level": 3},
                  lgb.Dataset(Xt, label, group=sizes, categorical_feature=spec.categorical), 300)
    return b.predict(Xv)


def _fit_temp(score, race_ids, y):
    """スコアから確率への温度だけを1つ推定する。"""
    from scipy.optimize import minimize_scalar
    def f(lt):
        p = softmax_by_race(score, race_ids, np.exp(lt))
        return -np.log(np.maximum(p[y == 1], 1e-9)).mean()
    return float(np.exp(minimize_scalar(f, bounds=(-3, 3), method="bounded").x))


def stage_models(con, org):
    """調整期間でパラメータを探し、選択期間 (未使用) で全手法を比較する。"""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    frame, feats = load_dev(con, org)
    pay = win_payouts(con, org)
    since, groups = best_since(con, org), adopted_groups(org)
    tune, sel = period_mask(frame, TUNE), period_mask(frame, SELECT)
    tf, sf = frame[tune].reset_index(drop=True), frame[sel].reset_index(drop=True)
    y_tune = (pd.to_numeric(tf["finish_pos"]) == 1).to_numpy()
    y_sel = (pd.to_numeric(sf["finish_pos"]) == 1).to_numpy()

    def objective(trial):
        params = dict(learning_rate=trial.suggest_float("learning_rate", 0.015, 0.08, log=True), num_leaves=trial.suggest_int("num_leaves", 15, 63),
                      min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 30, 400, log=True), feature_fraction=trial.suggest_float("feature_fraction", 0.4, 0.9),
                      bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 0.95), lambda_l2=trial.suggest_float("lambda_l2", 0.1, 50, log=True))
        p, _ = fit_predict(frame, feats, max(since, "2022-01-01"), TUNE[0], tune, groups=groups, params=params)
        return float(-np.log(np.maximum(p[y_tune], 1e-9)).mean())

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=MAX_OPTUNA_TRIALS)
    best = study.best_params
    log.info("optuna best %s %s", study.best_value, best)
    all_tr = ((frame["race_date"] >= since) & (frame["race_date"] < SELECT[0])).to_numpy()
    preds: dict[str, np.ndarray] = {}
    timer = {}
    def run(name, fn):
        t0 = time.time()
        preds[name] = fn()
        timer[name] = time.time() - t0
        log.info("%s %.0fs", name, timer[name])
    run("現行 (追加特徴なし)", lambda: fit_predict(frame, feats, since, SELECT[0], sel, groups=())[0])
    run("追加特徴 LightGBM PL", lambda: fit_predict(frame, feats, since, SELECT[0], sel, groups=groups)[0])
    run("最適化 LightGBM PL", lambda: fit_predict(frame, feats, since, SELECT[0], sel, groups=groups, params=best)[0])
    bins = _binary_models(frame, feats, all_tr, sel, groups)
    # 二値モデルの温度は調整期間で決めると学習をやり直す必要があるので、温度1で比較する
    for name, s in bins.items():
        preds[name] = softmax_by_race(s, sf["race_id"])
    rank = _lambdarank(frame, feats, all_tr, sel, groups)
    # ランキング学習のスコアは確率ではないので、選択期間の前半で温度を決め後半で評価する形にせず、的中率だけ意味を持つ
    half = sf["race_date"] < "2025-12-01"
    t = _fit_temp(rank[half.to_numpy()], sf.loc[half, "race_id"], y_sel[half.to_numpy()])
    preds["LambdaRank (温度は前半で推定)"] = softmax_by_race(rank, sf["race_id"], t)
    log_ens = np.mean([np.log(np.maximum(preds[k], 1e-9)) for k in ("最適化 LightGBM PL", "CatBoost 二値", "XGBoost 二値")], axis=0)
    preds["アンサンブル (PL+Cat+XGB)"] = softmax_by_race(log_ens, sf["race_id"])
    tr_mask = ((frame["race_date"] >= "2024-01-01") & (frame["race_date"] < SELECT[0])).to_numpy()
    preds["基準: 確定人気の代用"] = pop_market(frame, tr_mask, sel)
    base = race_metrics(sf, preds["現行 (追加特徴なし)"], pay)
    table = []
    for name, p in preds.items():
        r = summarize(race_metrics(sf, p, pay), base)
        table.append({"model": name, **{k: r[k] for k in ("n_races", "log_loss", "delta_log_loss", "delta_ci_low", "delta_ci_high", "brier", "top1_hit", "roi", "roi_ci_low", "roi_ci_high", "max_drawdown")}})
        record(con, "models", name, org, f"{since}..{SELECT[0]}", str(SELECT), r, algo=name, groups=groups, params=best if "最適化" in name else None, seconds=timer.get(name))
    out = {"since": since, "groups": groups, "optuna": {"best": best, "best_tune_log_loss": study.best_value, "trials": MAX_OPTUNA_TRIALS}, "table": table}
    (config.REPORTS / f"lab_models_{org}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    pickle.dump({"preds": preds, "race": sf[["race_id", "umaban", "finish_pos", "race_date", "popularity_final"]]}, open(lab_dir() / f"sel_preds_{org}.pkl", "wb"))
    return out


def best_params(org) -> dict | None:
    f = config.REPORTS / f"lab_models_{org}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    t = {r["model"]: r for r in d["table"]}
    if t["最適化 LightGBM PL"]["log_loss"] < t["追加特徴 LightGBM PL"]["log_loss"]:
        return d["optuna"]["best"]
    return None


def stage_calibrate(con, org):
    """選択期間の前半で校正を学習し、後半で評価する。"""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    frame, feats = load_dev(con, org)
    pay = win_payouts(con, org)
    since, groups = best_since(con, org), adopted_groups(org)
    sel = period_mask(frame, SELECT)
    sf = frame[sel].reset_index(drop=True)
    p, _ = fit_predict(frame, feats, since, SELECT[0], sel, groups=groups, params=best_params(org))
    y = (pd.to_numeric(sf["finish_pos"]) == 1).astype(int).to_numpy()
    fit = (sf["race_date"] < "2025-12-01").to_numpy()
    ev = ~fit
    ids = sf.loc[ev, "race_id"]
    evf = sf[ev].reset_index(drop=True)
    norm = lambda q: (pd.Series(q, index=ids.index) / pd.Series(q, index=ids.index).groupby(ids).transform("sum")).to_numpy()
    lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    methods = {"校正なし": p[ev]}
    platt = LogisticRegression(C=1e6).fit(lg[fit, None], y[fit])
    methods["Platt"] = norm(platt.predict_proba(lg[ev, None])[:, 1])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-5, y_max=1 - 1e-5).fit(p[fit], y[fit])
    methods["Isotonic"] = norm(iso.predict(p[ev]))
    methods["温度スケーリング"] = softmax_by_race(np.log(np.maximum(p, 1e-9)), sf["race_id"], _fit_temp(np.log(np.maximum(p[fit], 1e-9)), sf.loc[fit, "race_id"], y[fit]))[ev]
    base = race_metrics(evf, methods["校正なし"], pay)
    out = {}
    edges = np.array([0, .02, .05, .1, .15, .2, .3, .5, 1.01])
    for name, q in methods.items():
        r = summarize(race_metrics(evf, q, pay), base)
        record(con, "calibrate", name, org, f"{since}..{SELECT[0]}", "2025-12-01..2026-02-28", r, groups=groups, calibration=name)
        rel = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (q >= a) & (q < b)
            if m.sum():
                rel.append({"bin": f"{a:.0%}-{min(b, 1):.0%}", "n": int(m.sum()), "pred": float(q[m].mean()), "actual": float(y[ev][m].mean())})
        out[name] = {**r, "reliability": rel}
    (config.REPORTS / f"lab_calibrate_{org}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return {k: {x: v[x] for x in ("log_loss", "brier", "delta_log_loss", "delta_ci_low", "delta_ci_high")} for k, v in out.items()}


def stage_split(con, org):
    """共通モデルと、データが十分な条件の別モデルを同じレースで比較する。"""
    frame, feats = load_dev(con, org)
    pay = win_payouts(con, org)
    since, groups, params = best_since(con, org), adopted_groups(org), best_params(org)
    sel = period_mask(frame, SELECT)
    sf = frame[sel].reset_index(drop=True)
    p_global, _ = fit_predict(frame, feats, since, SELECT[0], sel, groups=groups, params=params)
    tr_all = ((frame["race_date"] >= since) & (frame["race_date"] < SELECT[0]))
    out = []
    splits = {"競馬場": frame["course_code"], "距離帯": feats["dist_bucket"].astype(str)}
    for dim, col in splits.items():
        counts = frame.loc[tr_all].drop_duplicates("race_id").groupby(col[tr_all]).size()
        for key, n in counts.items():
            if n < 5000:
                out.append({"dim": dim, "key": str(key).replace("N", ""), "n_train_races": int(n), "skipped": "データ不足 (5000レース未満)"})
                continue
            mask = (col == key).to_numpy()
            ev = sel & mask
            if frame.loc[ev, "race_id"].nunique() < 200:
                continue
            f2, x2 = frame[mask].reset_index(drop=True), feats[mask].reset_index(drop=True)
            ev2 = period_mask(f2, SELECT)
            p, _ = fit_predict(f2, x2, since, SELECT[0], ev2, groups=groups, params=params)
            evf = f2[ev2].reset_index(drop=True)
            lookup = pd.Series(p_global, index=pd.MultiIndex.from_frame(sf[["race_id", "umaban"]]))
            pg = lookup.reindex(pd.MultiIndex.from_frame(evf[["race_id", "umaban"]])).to_numpy()
            base = race_metrics(evf, pg, pay)
            r = summarize(race_metrics(evf, p, pay), base)
            record(con, "split", f"{dim}_{key}", org, f"{since}..{SELECT[0]}", str(SELECT), r, groups=groups)
            out.append({"dim": dim, "key": str(key).replace("N", ""), "n_train_races": int(n), "n_races": r["n_races"], "global_log_loss": float(base["ll"].mean()), "split_log_loss": r["log_loss"],
                        "delta": r["delta_log_loss"], "ci": [r["delta_ci_low"], r["delta_ci_high"]], "better": r["delta_ci_high"] < 0})
            log.info("%s", out[-1])
    (config.REPORTS / f"lab_split_{org}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def stage_market(con, org):
    """市場を基準にした補正を、開発期間では確定人気の代用で検証する。"""
    from .probs import fit_blend
    frame, feats = load_dev(con, org)
    pay = win_payouts(con, org)
    since, groups, params = best_since(con, org), adopted_groups(org), best_params(org)
    sel = period_mask(frame, SELECT)
    sf = frame[sel].reset_index(drop=True)
    tr = ((frame["race_date"] >= since) & (frame["race_date"] < SELECT[0])).to_numpy()
    y = (pd.to_numeric(sf["finish_pos"]) == 1).astype(int).to_numpy()
    p_ai, model = fit_predict(frame, feats, since, SELECT[0], sel, groups=groups, params=params)
    mp = pop_market(frame, tr, sel)
    fit = (sf["race_date"] < "2025-12-01").to_numpy()
    d = pd.DataFrame({"race_id": sf["race_id"], "score": np.log(np.maximum(p_ai, 1e-9)), "tau1": 1.0, "odds": 1 / np.maximum(mp, 1e-9), "win": y})
    a, b = fit_blend(d[fit])
    blend = softmax_by_race(a * d["score"].to_numpy() + b * np.log(np.maximum(mp, 1e-9)), sf["race_id"])
    # D: 市場を起点にして AI の特徴量で残差を学習する
    tr_idx = np.flatnonzero(tr)
    mid = frame["race_date"].iloc[tr_idx].quantile if False else None
    t_mask = ((frame["race_date"] >= "2022-01-01") & (frame["race_date"] < "2024-07-01")).to_numpy()
    mp_tr = pop_market(frame, t_mask, tr)   # 同じ行で推定しないよう、前の期間で作った代用確率を使う
    f_tr = frame[tr].reset_index(drop=True)
    x_tr = feats[tr].reset_index(drop=True)
    later = (f_tr["race_date"] >= "2024-07-01").to_numpy()
    res_model = _train_residual(f_tr[later].reset_index(drop=True), x_tr[later].reset_index(drop=True), np.log(np.maximum(mp_tr[later], 1e-9)), groups)
    x_ev = feats[sel].reset_index(drop=True)
    resid = softmax_by_race(res_model(x_ev) + np.log(np.maximum(mp, 1e-9)), sf["race_id"])
    ev = ~fit
    evf = sf[ev].reset_index(drop=True)
    base = race_metrics(evf, mp[ev], pay)
    table = {}
    for name, p in {"A AI独自": p_ai, "B 確定人気の代用": mp, f"C 単純合成 (AI {a:.2f} / 市場 {b:.2f})": blend, "D 市場起点でAIが補正": resid}.items():
        r = summarize(race_metrics(evf, p[ev], pay), base)
        record(con, "market", name, org, f"{since}..{SELECT[0]}", "2025-12-01..2026-02-28", r, groups=groups, algo=name)
        table[name] = {k: r[k] for k in ("n_races", "log_loss", "delta_log_loss", "delta_ci_low", "delta_ci_high", "brier", "top1_hit", "roi")}
    out = {"table": table, "note": "基準は確定人気の順位だけから作った代用確率。実オッズより情報が少なく、改善幅は過大に見えやすい"}
    (config.REPORTS / f"lab_market_{org}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def _train_residual(frame, feats, offset, groups, rounds=300):
    """offset (市場の対数確率) からの差だけを LightGBM に学習させる。"""
    import lightgbm as lgb
    from .gbm import PARAMS, order_matrix, pl_grad_hess
    frame = frame.copy()
    frame["finish_pos"] = pd.to_numeric(frame["finish_pos"], errors="coerce")
    spec = FeatureSpec.fit(feats, groups=groups)
    X = spec.transform(feats)
    idx, mask, orders = order_matrix(frame)
    ds = lgb.Dataset(X, free_raw_data=False, categorical_feature=spec.categorical, init_score=offset)

    def obj(preds, _d):
        return pl_grad_hess(preds, idx, mask, orders)

    b = lgb.train({**PARAMS, "objective": obj, "seed": 1, "learning_rate": 0.02, "num_leaves": 15}, ds, rounds)
    return lambda x: b.predict(spec.transform(x))


def stage_list(con, org):
    rows = con.execute("SELECT stage,name,eval_period,n_races,round(log_loss,4) ll,round(delta_log_loss,4) d,round(top1_hit,3) hit,round(roi,3) roi,decision,used_locked_test FROM experiment WHERE org=? ORDER BY created_at", (org,)).fetchall()
    return [dict(r) for r in rows]


def stage_summary(con, org):
    """公開しても問題ない集計だけを docs/lab_summary.json にまとめる (騎手・調教師・馬の個別情報は含めない)。"""
    load = lambda n: json.loads((config.REPORTS / f"lab_{n}_{org}.json").read_text(encoding="utf-8"))
    analyze = load("analyze")
    analyze["dims"] = {k: v for k, v in analyze["dims"].items() if k not in ("騎手", "調教師")}
    final = load("final")
    final = {k: final[k] for k in ("note", "groups", "params", "since", "baseline", "new", "market_reference_final_odds", "adopted", "market_compare", "blend",
                                   "calibration", "monthly", "by_course", "by_going", "ev_bands", "rows", "n_tests", "passed_candidates")}
    out = {"org": org, "updated_at": now_jst(), "final": final, "models": load("models"), "groups": load("groups"), "calibrate": load("calibrate"), "split": load("split"),
           "market_dev": load("market"), "analyze": analyze,
           "experiments": [dict(r) for r in con.execute("SELECT stage,name,eval_period,n_races,log_loss,delta_log_loss,delta_ci_low,delta_ci_high,brier,top1_hit,roi,decision,used_locked_test FROM experiment WHERE org=? ORDER BY created_at", (org,))]}
    path = config.ROOT / "docs" / "lab_summary.json"
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":"), default=str), encoding="utf-8")
    return {"written": str(path.relative_to(config.ROOT)), "bytes": path.stat().st_size}


def run_stage(con: sqlite3.Connection, org: str, stage: str, confirm: bool = False):
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    fn = {"baseline": stage_baseline, "analyze": stage_analyze, "history": stage_history, "groups": stage_groups, "models": stage_models,
          "calibrate": stage_calibrate, "split": stage_split, "market": stage_market, "list": stage_list, "summary": stage_summary}
    if stage == "final":
        if not confirm:
            raise LockedTestError("final は封鎖テストを使います。--confirm-locked-test を指定してください")
        from .lab_final import stage_final
        return stage_final(con, org)
    return fn[stage](con, org)
