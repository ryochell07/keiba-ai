"""封鎖テストでの最終評価。1回だけ実行でき、実行したことを記録する。ここでパラメータを選び直してはいけない。"""
from __future__ import annotations

import json
import logging
import sqlite3

import numpy as np
import pandas as pd

from . import backtest as bt
from . import config
from .features import add_market, build_features, load_frame
from .gbm import softmax_by_race
from .lab import (LOCKED_FROM, _train_residual, adopted_groups, best_params, best_since, race_metrics, record, summarize, win_payouts)
from .probs import fit_blend

log = logging.getLogger(__name__)
THRESHOLDS = (1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5)
MIN_BETS = 500


def stage_final(con: sqlite3.Connection, org: str) -> dict:
    if con.execute("SELECT 1 FROM experiment WHERE stage='final' AND org=? LIMIT 1", (org,)).fetchone():
        raise SystemExit("封鎖テストの最終評価は実行済みです。再実行すると過剰適合の検査にならないので止めます")
    groups, params, since = adopted_groups(org), best_params(org), best_since(con, org)
    frame = load_frame(con, org)
    frame = frame[frame["status"].notna() & (frame["race_date"] >= "2019-01-01")].reset_index(drop=True)
    feats = build_features(add_market(frame, con))
    pay = win_payouts(con, org)
    # 現行: 2024年以降の学習・追加特徴なし。新: 開発で決めた設定のまま。どちらも2か月ごとの再学習で同じレースを評価する
    base_mask = (frame["race_date"] >= "2024-01-01").to_numpy()
    scored_base, _ = bt.walk_forward_scores(frame[base_mask].reset_index(drop=True), feats[base_mask].reset_index(drop=True), org, LOCKED_FROM, 2, n_seeds=1, groups=())
    new_mask = (frame["race_date"] >= since).to_numpy()
    scored_new, info = bt.walk_forward_scores(frame[new_mask].reset_index(drop=True), feats[new_mask].reset_index(drop=True), org, LOCKED_FROM, 2, n_seeds=1, groups=groups, params=params)
    pb = softmax_by_race(scored_base["score"].to_numpy(), scored_base["race_id"], scored_base["tau1"].iloc[0])
    pn = softmax_by_race(scored_new["score"].to_numpy(), scored_new["race_id"], scored_new["tau1"].iloc[0])
    mb, mn = race_metrics(scored_base, pb, pay), race_metrics(scored_new, pn, pay)
    res_base, res_new = summarize(mb), summarize(mn, mb)
    # 市場: 確定オッズ。発走前の情報ではないので「予測時点が同じ比較」ではなく、市場の到達点としての参考値
    odds_df = bt.win_odds_frame(con, scored_new, org)
    inv = 1 / odds_df["odds"]
    market = inv / inv.groupby(odds_df["race_id"]).transform("sum")
    res_market_ref = summarize(race_metrics(odds_df, market.to_numpy(), pay))
    dates = sorted(odds_df["race_date"].unique())
    blend_from = dates[len(dates) // 2]
    fit_part = odds_df["race_date"] < blend_from
    a, b = fit_blend(odds_df[fit_part])
    ev_part = ~fit_part
    evd = odds_df[ev_part].reset_index(drop=True)
    mk_ev = market[ev_part].to_numpy()
    blend_p = softmax_by_race(a * (evd["score"] / evd["tau1"]).to_numpy() + b * np.log(mk_ev), evd["race_id"])
    # D: 市場の対数確率を起点に、AI の特徴量で差分だけを学習する (前半で学習、後半で評価)
    x_all = feats[new_mask].reset_index(drop=True)
    f_all = frame[new_mask].reset_index(drop=True)
    key_all = pd.MultiIndex.from_frame(f_all[["race_id", "umaban"]])
    pos = pd.Series(np.arange(len(f_all)), index=key_all)
    row_fit = pos.reindex(pd.MultiIndex.from_frame(odds_df.loc[fit_part, ["race_id", "umaban"]])).to_numpy()
    row_ev = pos.reindex(pd.MultiIndex.from_frame(evd[["race_id", "umaban"]])).to_numpy()
    res_fn = _train_residual(f_all.iloc[row_fit].reset_index(drop=True), x_all.iloc[row_fit].reset_index(drop=True), np.log(market[fit_part].to_numpy()), groups, rounds=150)
    resid_p = softmax_by_race(res_fn(x_all.iloc[row_ev].reset_index(drop=True)) + np.log(mk_ev), evd["race_id"])
    m_market_half = race_metrics(evd, mk_ev, pay)
    market_compare = {
        "A AI独自": summarize(race_metrics(evd, (evd["score"] * 0 + softmax_by_race(evd["score"].to_numpy(), evd["race_id"], evd["tau1"].iloc[0])).to_numpy(), pay), m_market_half),
        "B 市場 (確定オッズ)": summarize(m_market_half),
        f"C 単純合成 (AI {a:.3f} / 市場 {b:.3f})": summarize(race_metrics(evd, blend_p, pay), m_market_half),
        "D 市場起点でAIが補正": summarize(race_metrics(evd, resid_p, pay), m_market_half),
    }
    for name, r in market_compare.items():
        record(con, "final", name, org, f"{since}..", f"{blend_from}..", r, groups=groups, locked=True, decision="市場比較")
    # 買い目: 確定オッズ判定なので実運用より良く出る。合格基準は Bonferroni 補正後の下限 > 100% と 500点以上
    old_th = bt.EV_THRESHOLDS
    bt.EV_THRESHOLDS = THRESHOLDS
    try:
        result = bt.evaluate(con, scored_new, org, (a, b), blend_from)
    finally:
        bt.EV_THRESHOLDS = old_th
    rows = []
    for (strategy, bet, basis), m in result["ledger"].rows():
        rows.append({"strategy": strategy, "bet_type": bet, "odds_basis": basis, **m})
    candidates = [r for r in rows if r["strategy"].startswith("blend_ev_")]
    z_tests = max(len(candidates), 1)
    passed = []
    for r in candidates:
        # 95% 区間を検定回数で補正するのは粗いので、下限 > 100% と点数の両方を満たしたものだけ候補に挙げ、後で別期間の再現を要求する
        r["candidate"] = bool(r["n_bets"] >= MIN_BETS and (r["roi_ci_low"] or 0) > 1.0)
        passed += [r] if r["candidate"] else []
    adopted = res_new["delta_ci_high"] < 0
    record(con, "final", "現行 (封鎖テスト)", org, "2024-01-01..", f"{LOCKED_FROM}..", res_base, locked=True, decision="基準")
    record(con, "final", "新モデル (封鎖テスト)", org, f"{since}..", f"{LOCKED_FROM}..", res_new, groups=groups, params=params, locked=True, decision="採用" if adopted else "不採用")
    out = {"org": org, "note": "封鎖テストを最終評価に1回だけ使用。ただし前回の合成方式の判断時にこの期間の結果を見ている",
           "groups": groups, "params": params, "since": since, "baseline": res_base, "new": res_new, "market_reference_final_odds": res_market_ref,
           "adopted": bool(adopted), "market_compare": market_compare, "blend": {"alpha": a, "beta": b, "evaluated_from": blend_from}, "summary": result["summary"],
           "calibration": result["calibration"], "monthly": result["monthly"], "by_course": result["by_course"], "by_going": result["by_going"], "ev_bands": result["ev_bands"],
           "candidate_coverage": result["candidate_coverage"], "rows": rows, "n_tests": z_tests, "passed_candidates": passed, "folds": info["folds"]}
    (config.REPORTS / f"lab_final_{org}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return {k: out[k] for k in ("baseline", "new", "market_reference_final_odds", "adopted", "market_compare", "blend", "passed_candidates", "n_tests")}
