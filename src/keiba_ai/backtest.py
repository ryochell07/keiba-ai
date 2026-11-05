"""時系列ウォークフォワードのバックテストとモデル比較。常に「過去で学習→未来で評価」のみ。"""
from __future__ import annotations

import json
import logging
import sqlite3
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import config
from .db import now_jst
from .features import FeatureSpec, add_market, build_features, load_frame, mask_late_info
from .gbm import softmax_by_race, train_race_model, trainable
from .probs import all_bet_probs, blend_strength, candidate_horses, fit_blend

log = logging.getLogger(__name__)
EV_THRESHOLDS = (1.0, 1.2, 1.5)


def month_starts(first: str, last: str, step: int) -> list[str]:
    y, m = int(first[:4]), int(first[5:7])
    out = []
    while f"{y}-{m:02d}-01" <= last:
        out.append(f"{y}-{m:02d}-01")
        m += step
        y, m = (y + (m - 1) // 12, (m - 1) % 12 + 1)
    return out


def assert_time_split(train_dates: pd.Series, test_dates: pd.Series) -> None:
    if train_dates.max() >= test_dates.min():
        raise AssertionError(f"時系列分割違反: 学習末 {train_dates.max()} >= 評価頭 {test_dates.min()}")


def walk_forward_scores(frame: pd.DataFrame, feats: pd.DataFrame, org: str, test_start: str, step_months: int,
                        n_seeds: int = 1, min_train_races: int = 300) -> tuple[pd.DataFrame, dict]:
    """fold ごとに再学習し、評価期間のスコアを返す。"""
    results = []
    folds = []
    last = frame["race_date"].max()
    starts = month_starts(test_start, last, step_months)
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else "9999-12-31"
        tr = frame["race_date"] < start
        te = (frame["race_date"] >= start) & (frame["race_date"] < end)
        if te.sum() == 0 or frame[tr]["race_id"].nunique() < min_train_races:
            continue
        assert_time_split(frame.loc[tr, "race_date"], frame.loc[te, "race_date"])
        model = train_race_model(frame[tr].reset_index(drop=True), feats[tr].reset_index(drop=True), org, n_seeds=n_seeds)
        test_frame = frame[te].reset_index(drop=True)
        s = model.scores(feats[te].reset_index(drop=True))
        results.append(test_frame.assign(score=s, tau1=model.taus[0], tau2=model.taus[1], tau3=model.taus[2]))
        folds.append({"start": start, "end": end, "n_train_races": model.metrics["n_train_races"],
                      "taus": model.taus, "n_test_races": test_frame["race_id"].nunique()})
        log.info("fold %s: train=%s test=%s taus=%s", start, model.metrics["n_train_races"],
                 test_frame["race_id"].nunique(), [round(t, 3) for t in model.taus])
    if not results:
        raise ValueError("評価できる fold がありません。データ期間を増やしてください")
    return pd.concat(results, ignore_index=True), {"folds": folds}


def _final_odds(con, org: str) -> dict[tuple[str, str, str], tuple[float, float | None]]:
    out = {}
    for r in con.execute("SELECT o.race_id,o.bet_type,o.combo,o.odds,o.odds_max FROM odds_snapshot o "
                         "JOIN race r USING(race_id) WHERE o.is_final=1 AND r.org=?", (org,)):
        out[(r[0], r[1], r[2])] = (r[3], r[4])
    return out


def _cutoff_odds(con, org: str, cutoff_min: int = 10) -> dict[tuple[str, str, str], tuple[float, float | None]]:
    """発走 cutoff 分前までの最後のスナップショットのみ。"""
    rows = con.execute(
        """SELECT o.race_id,o.bet_type,o.combo,o.odds,o.odds_max,o.captured_at,r.race_date,r.post_time
           FROM odds_snapshot o JOIN race r USING(race_id)
           WHERE o.is_final=0 AND o.captured_at<>'final' AND r.org=? AND r.post_time IS NOT NULL
           ORDER BY o.captured_at""", (org,)).fetchall()
    out = {}
    for r in rows:
        limit = pd.Timestamp(f"{r['race_date']} {r['post_time']}") - pd.Timedelta(minutes=cutoff_min)
        if pd.Timestamp(r["captured_at"]).tz_localize(None) <= limit:
            out[(r[0], r[1], r[2])] = (r[3], r[4])
    return out


def _bootstrap_roi(per_race: dict[str, list[float]], stakes: dict[str, float], n: int = 1000, seed: int = 0):
    keys = list(per_race)
    if not keys:
        return None, None
    ret = np.array([sum(per_race[k]) for k in keys])
    st = np.array([stakes[k] for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n, len(keys)))
    rois = ret[idx].sum(axis=1) / np.maximum(st[idx].sum(axis=1), 1e-9)
    return float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5))


class Ledger:
    """戦略 × 券種ごとの収支を集計する。レースは時系列順に渡すこと。"""

    def __init__(self):
        self.ret = defaultdict(lambda: defaultdict(list))
        self.stake = defaultdict(lambda: defaultdict(float))
        self.hits = defaultdict(int)
        self.bets = defaultdict(int)

    def bet(self, key, race_id: str, payout: float) -> None:
        self.ret[key][race_id].append(payout)
        self.stake[key][race_id] += 100.0
        self.bets[key] += 1
        self.hits[key] += int(payout > 0)

    def curve(self, key, max_points: int = 120) -> list[dict]:
        """累積回収率の推移。"""
        cum_ret, cum_st, out = 0.0, 0.0, []
        for i, (rid, rets) in enumerate(self.ret[key].items()):
            cum_ret += sum(rets)
            cum_st += self.stake[key][rid]
            if i >= 50 or len(self.ret[key]) <= 100:  # 序盤は1レースで大きく振れるので描かない
                out.append({"race": rid, "date": f"{rid[1:5]}-{rid[5:7]}-{rid[7:9]}", "roi": cum_ret / cum_st})
        step = max(1, len(out) // max_points)
        return out[::step] + (out[-1:] if out and (len(out) - 1) % step else [])

    def max_drawdown(self, key) -> float:
        """累積損益の最大下落幅 (円、1点100円)。"""
        cum, peak, dd = 0.0, 0.0, 0.0
        for rid, rets in self.ret[key].items():
            cum += sum(rets) - self.stake[key][rid]
            peak = max(peak, cum)
            dd = max(dd, peak - cum)
        return dd

    def rows(self):
        for key in self.ret:
            total_ret = sum(sum(v) for v in self.ret[key].values())
            total_stake = sum(self.stake[key].values())
            lo, hi = _bootstrap_roi(self.ret[key], self.stake[key])
            yield key, dict(n_races=len(self.ret[key]), n_bets=self.bets[key], n_hits=self.hits[key],
                            stake=total_stake, payout=total_ret, roi=total_ret / total_stake if total_stake else None,
                            roi_ci_low=lo, roi_ci_high=hi, max_drawdown=self.max_drawdown(key))


EV_BANDS = [(0, .8), (.8, 1.0), (1.0, 1.2), (1.2, 1.5), (1.5, 2.0), (2.0, 1e9)]


def _group_table(df: pd.DataFrame, col: str) -> list[dict]:
    out = []
    for key, g in df.groupby(col, dropna=True, sort=True):
        out.append({"key": str(key), "n_races": len(g), "roi": float(g["top1_ret"].sum() / (100 * len(g))),
                    "top1_hit": float(g["top1_hit"].mean()), "logloss": float(g["ll"].mean())})
    return out


def win_odds_frame(con, scored: pd.DataFrame, org: str) -> pd.DataFrame:
    """全頭に確定単勝オッズがあるレースだけを、合成係数の推定用に取り出す。"""
    od = pd.read_sql_query("SELECT o.race_id, CAST(o.combo AS INTEGER) AS umaban, o.odds FROM odds_snapshot o JOIN race r USING(race_id) "
                           "WHERE r.org=? AND o.bet_type='win' AND o.is_final=1", con, params=[org])
    d = scored.merge(od, on=["race_id", "umaban"], how="left")
    d["win"] = (pd.to_numeric(d["finish_pos"], errors="coerce") == 1).astype(int)
    ok = d.groupby("race_id")["odds"].transform(lambda x: x.notna().all() and (x > 0).all())
    # 取消馬のオッズが残るレースは市場確率が歪むので除く (人気馬の取消で見かけの期待値が 1 を超える)
    counts = od.groupby("race_id")["umaban"].count()
    same = d["race_id"].map(counts) == d.groupby("race_id")["umaban"].transform("count")
    d = d[ok & same]
    return d[d.groupby("race_id")["win"].transform("sum") >= 1]


def evaluate(con: sqlite3.Connection, scored: pd.DataFrame, org: str, blend: tuple[float, float] | None = None,
             blend_from: str | None = None) -> dict:
    """scored は score/tau 列つきの評価対象。的中率・回収率・Log Loss・Brier・校正などを計算する。
    blend を渡すと、blend_from 以降のレースで市場との合成確率による期待値戦略も評価する (推定に使った期間は含めない)。"""
    scored = scored.sort_values(["race_date", "post_time", "race_id"], kind="stable")
    payouts: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    race_ids = set(scored["race_id"])
    for r in con.execute("SELECT race_id,bet_type,combo,amount FROM payout"):
        if r[0] in race_ids:
            payouts[(r[0], r[1])][r[2]] = r[3]
    final_odds, cutoff_odds = _final_odds(con, org), _cutoff_odds(con, org)
    win_counts: dict[str, int] = defaultdict(int)
    for (rid, bet, _c) in final_odds:
        if bet == "win":
            win_counts[rid] += 1
    ledger = Ledger()
    calib_win = []
    coverage = defaultdict(lambda: {"races": 0, "hit": 0})
    rows = []
    bands = {b: Ledger() for b in EV_BANDS}
    fav_hits = 0
    for rid, g in scored.groupby("race_id", sort=False):
        g = g.sort_values("umaban")
        pos = pd.to_numeric(g["finish_pos"], errors="coerce").to_numpy()
        if not (pos == 1).any() or len(g) < 3:
            continue
        taus = (g["tau1"].iloc[0], g["tau2"].iloc[0], g["tau3"].iloc[0])
        um, wk, s = g["umaban"].tolist(), [None if pd.isna(w) else w for w in g["waku"]], g["score"].to_numpy()
        probs = all_bet_probs(um, wk, s, taus, org)
        win = probs["win"]
        winners = g.loc[pos == 1, "umaban"].tolist()
        best = max(win, key=win.get)
        win_pay = payouts.get((rid, "win"), {})
        row = {"race_id": rid, "date": g["race_date"].iloc[0], "course": g["course_code"].iloc[0], "going": g["going"].iloc[0],
               "ll": -np.log(max(win[str(winners[0])], 1e-9)), "top1_hit": int(int(best) in winners),
               "top1_ret": win_pay.get(best, 0),
               "brier": sum((win[str(u)] - float(f == 1)) ** 2 for u, f in zip(um, pos))}
        for u, f in zip(um, pos):
            calib_win.append((win[str(u)], float(f == 1)))
        # 市場との比較 (確定オッズ由来。評価専用で特徴量には使わない)
        implied = {str(u): final_odds.get((rid, "win", str(u)), (None,))[0] for u in um}
        if all(implied.values()) and win_counts.get(rid) == len(um):  # 取消馬のオッズが残るレースは市場確率が歪む
            inv = {k: 1 / v for k, v in implied.items()}
            row["ll_market"] = -np.log(max(inv[str(winners[0])] / sum(inv.values()), 1e-9))
        use_blend = bool(blend and blend_from and row["date"] >= blend_from and "ll_market" in row)
        blended = None
        if use_blend:
            strength = blend_strength(s, [implied[str(u)] for u in um], blend[0], blend[1], taus[0])
            blended = all_bet_probs(um, wk, strength, (1.0, taus[1] / taus[0], taus[2] / taus[0]), org)
            row["ll_blend"] = -np.log(max(blended["win"][str(winners[0])], 1e-9))
        band_probs = blended["win"] if blended else win
        for u in um:
            o = final_odds.get((rid, "win", str(u)))
            if o and o[0] and win_pay and (blended or not blend):
                ev = band_probs[str(u)] * o[0]
                band = next(b for b in EV_BANDS if b[0] <= ev < b[1])
                bands[band].bet("x", rid, win_pay.get(str(u), 0))
        rows.append(row)
        fav = g.loc[pd.to_numeric(g["popularity_final"], errors="coerce") == 1, "umaban"].tolist()
        for bet_type, combos in probs.items():
            pay = payouts.get((rid, bet_type), {})
            if not pay:
                continue
            # 戦略1: 確率最上位を買う (判定にオッズを使わないので未来情報の問題がない)
            top = max(combos, key=combos.get)
            ledger.bet(("top1", bet_type, "none"), rid, pay.get(top, 0))
            # 戦略2: 期待値フィルタ
            for basis, table in (("final_optimistic", final_odds), ("cutoff", cutoff_odds)):
                for c, p in combos.items():
                    o = table.get((rid, bet_type, c))
                    if o is None or o[0] is None:
                        continue
                    for th in EV_THRESHOLDS:
                        if p * o[0] >= th:
                            ledger.bet((f"ev_{th}", bet_type, basis), rid, pay.get(c, 0))
            if blended and bet_type in blended:
                for c, p in blended[bet_type].items():
                    o = final_odds.get((rid, bet_type, c))
                    if o is None or o[0] is None:
                        continue
                    for th in EV_THRESHOLDS:
                        if p * o[0] >= th:
                            ledger.bet((f"blend_ev_{th}", bet_type, "final_optimistic"), rid, pay.get(c, 0))
        if fav:
            ledger.bet(("favorite_benchmark", "win", "benchmark"), rid, win_pay.get(str(fav[0]), 0))
            fav_hits += int(fav[0] in winners)
        # 候補馬絞り込みの検証: 実際の 1-3 着が上位 K 頭に入っていたか
        top3 = g.loc[pos <= 3, "umaban"].tolist()
        if len(top3) == 3:
            for k in (5, 6, 8, 10, 12):
                coverage[k]["races"] += 1
                coverage[k]["hit"] += int(set(top3) <= candidate_horses(um, s, k))
    if not rows:
        raise ValueError("評価できるレースがありません")
    df = pd.DataFrame(rows)
    df["month"] = df["date"].str[:7]
    arr = np.array(calib_win)
    edges = np.array([0, .02, .05, .1, .15, .2, .3, .5, 1.01])
    calib = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (arr[:, 0] >= a) & (arr[:, 0] < b)
        if m.sum():
            calib.append({"bin": f"{a:.0%}-{min(b, 1):.0%}", "n": int(m.sum()), "pred": float(arr[m, 0].mean()), "actual": float(arr[m, 1].mean())})
    monthly = _group_table(df, "month")
    if "ll_market" in df:
        mk = df.groupby("month")["ll_market"].mean()
        for m in monthly:
            v = mk.get(m["key"])
            m["logloss_market"] = None if v is None or pd.isna(v) else float(v)
    band_rows = []
    for b, led in bands.items():
        for _key, m in led.rows():
            band_rows.append({"band": f"{b[0]:.1f}以上" if b[1] > 100 else f"{b[0]:.1f}-{b[1]:.1f}", **m})
    curves = {}
    for key, name in ((("top1", "win", "none"), "AI本命 単勝"), (("favorite_benchmark", "win", "benchmark"), "確定1番人気 単勝")):
        if key in ledger.ret:
            curves[name] = ledger.curve(key)
    return {
        "ledger": ledger,
        "summary": {"n_races": len(df), "top1_hit": float(df["top1_hit"].mean()), "fav_hit": fav_hits / len(df),
                    "win_logloss_model": float(df["ll"].mean()), "brier": float(df["brier"].mean()),
                    "win_logloss_market": float(df["ll_market"].mean()) if "ll_market" in df and df["ll_market"].notna().any() else None,
                    "n_market_races": int(df["ll_market"].notna().sum()) if "ll_market" in df else 0,
                    "top1_win_roi": float(df["top1_ret"].sum() / (100 * len(df))),
                    "win_logloss_blend": float(df["ll_blend"].mean()) if "ll_blend" in df and df["ll_blend"].notna().any() else None,
                    "win_logloss_market_same_period": float(df.loc[df["ll_blend"].notna(), "ll_market"].mean()) if "ll_blend" in df and df["ll_blend"].notna().any() else None,
                    "win_logloss_model_same_period": float(df.loc[df["ll_blend"].notna(), "ll"].mean()) if "ll_blend" in df and df["ll_blend"].notna().any() else None},
        "calibration": calib,
        "candidate_coverage": {k: v["hit"] / v["races"] for k, v in coverage.items() if v["races"]},
        "monthly": monthly, "by_course": _group_table(df, "course"), "by_going": _group_table(df, "going"),
        "ev_bands": band_rows, "curves": curves,
    }


def run_backtest(con: sqlite3.Connection, org: str, test_start: str, step_months: int = 1, n_seeds: int = 1) -> dict:
    frame = load_frame(con, org)
    frame = frame[frame["status"].notna()].reset_index(drop=True)
    feats = build_features(add_market(frame, con))
    scored, info = walk_forward_scores(frame, feats, org, test_start, step_months, n_seeds)
    # 合成係数は評価期間の前半で推定し、後半だけで評価する。確定オッズ基準なので実運用より良く出る
    blend, blend_from = None, None
    odds_df = win_odds_frame(con, scored, org)
    if odds_df["race_id"].nunique() >= 200:
        dates = sorted(odds_df["race_date"].unique())
        blend_from = dates[len(dates) // 2]
        blend = fit_blend(odds_df[odds_df["race_date"] < blend_from])
    result = evaluate(con, scored, org, blend, blend_from)
    bid = f"{org}-{now_jst()}"
    rows = []
    for (strategy, bet_type, basis), m in result["ledger"].rows():
        con.execute(
            """INSERT OR REPLACE INTO backtest_result(backtest_id,created_at,org,strategy,bet_type,period_start,period_end,
               n_races,n_bets,n_hits,stake,payout,roi,roi_ci_low,roi_ci_high,odds_basis,detail)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (bid, now_jst(), org, f"{strategy}|{basis}", bet_type, scored["race_date"].min(), scored["race_date"].max(),
             m["n_races"], m["n_bets"], m["n_hits"], m["stake"], m["payout"], m["roi"], m["roi_ci_low"], m["roi_ci_high"], basis,
             json.dumps({"max_drawdown": m["max_drawdown"]})))
        rows.append({"strategy": strategy, "bet_type": bet_type, "odds_basis": basis, **m})
    con.commit()
    report = {"backtest_id": bid, "org": org, "period": [scored["race_date"].min(), scored["race_date"].max()],
              **{k: v for k, v in result.items() if k != "ledger"}, "folds": info["folds"], "rows": rows,
              "blend": {"alpha": blend[0], "beta": blend[1], "evaluated_from": blend_from} if blend else None}
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / f"backtest_{org}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def compare_models(con: sqlite3.Connection, org: str, test_start: str) -> list[dict]:
    """同じ時系列分割で複数手法を比較する。指標はレース内正規化した単勝 logloss と本命的中率。"""
    frame = load_frame(con, org)
    frame = frame[frame["status"].notna()].reset_index(drop=True)
    feats = build_features(add_market(frame, con))
    frame, feats = trainable(frame, feats)
    tr, te = (frame["race_date"] < test_start).to_numpy(), (frame["race_date"] >= test_start).to_numpy()
    assert_time_split(frame.loc[tr, "race_date"], frame.loc[te, "race_date"])
    spec = FeatureSpec.fit(feats[tr])
    te_frame = frame[te].reset_index(drop=True)
    y_tr = (pd.to_numeric(frame.loc[tr, "finish_pos"]) == 1).astype(int).to_numpy()
    y_te = (pd.to_numeric(te_frame["finish_pos"]) == 1).astype(int).to_numpy()
    Xt, Xv = spec.transform(feats[tr]), spec.transform(feats[te])
    out = []

    def score(name, s, taus=None):
        p = softmax_by_race(np.asarray(s), te_frame["race_id"], tau=1.0 if taus is None else taus)
        ll = -np.log(np.maximum(p[y_te == 1], 1e-9)).mean()
        pick = te_frame.assign(p=p).sort_values("p", ascending=False).groupby("race_id", sort=False).head(1)
        hit = (pd.to_numeric(pick["finish_pos"]) == 1).mean()
        out.append({"model": name, "win_logloss": float(ll), "top1_hit": float(hit), "n_test_races": te_frame["race_id"].nunique()})

    pl = train_race_model(frame[tr].reset_index(drop=True), feats[tr].reset_index(drop=True), org, n_seeds=1)
    score("LightGBM Plackett-Luce", pl.scores(feats[te].reset_index(drop=True)), pl.taus[0])

    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    import lightgbm as lgb
    b = lgb.train({"objective": "binary", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 60,
                   "feature_fraction": 0.7, "verbosity": -1}, lgb.Dataset(Xt, y_tr, categorical_feature=spec.categorical), 300)
    score("LightGBM 二値", logit(b.predict(Xv)))
    try:
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.7, tree_method="hist")
        m.fit(Xt.to_numpy(), y_tr)
        score("XGBoost 二値", logit(m.predict_proba(Xv.to_numpy())[:, 1]))
    except ImportError:
        log.info("xgboost 未導入のためスキップ")
    try:
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, verbose=False, allow_writing_files=False)
        m.fit(Xt.to_numpy(), y_tr)
        score("CatBoost 二値", logit(m.predict_proba(Xv.to_numpy())[:, 1]))
    except ImportError:
        log.info("catboost 未導入のためスキップ")
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    numeric = [c for c in spec.columns if c not in spec.vocab]
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=500, C=0.1))
    m.fit(Xt[numeric].to_numpy(), y_tr)
    score("ロジスティック回帰", logit(m.predict_proba(Xv[numeric].to_numpy())[:, 1]))
    # 比較基準: 確定人気 (評価専用)
    pop = pd.to_numeric(te_frame["popularity_final"], errors="coerce").fillna(99).to_numpy()
    fav = te_frame.assign(pop=pop).sort_values("pop").groupby("race_id", sort=False).head(1)
    out.append({"model": "基準: 確定1番人気 (事後情報・参考)", "win_logloss": None,
                "top1_hit": float((pd.to_numeric(fav["finish_pos"]) == 1).mean()), "n_test_races": te_frame["race_id"].nunique()})
    out.append({"model": "基準: 一様ランダム", "win_logloss": float(np.log(te_frame.groupby("race_id").size()).mean()),
                "top1_hit": float((1 / te_frame.groupby("race_id").size()).mean()), "n_test_races": te_frame["race_id"].nunique()})
    (config.REPORTS / f"compare_{org}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out
