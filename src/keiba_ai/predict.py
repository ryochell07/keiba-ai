"""学習済みモデルで発走前のレースを予測し、台帳へ追記する。発走済みのレースには何も書かない。"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd

from . import config, ledger
from .bundles import active_bundle, columns_hash, git_commit
from .db import JST, now_jst
from .ev import STATUS_NO_ODDS, expected, latest_odds, make_flags, verified_bet_types
from .failures import ModelIntegrityError, is_fatal, record_failure
from .features import add_market, build_features, load_frame, serving_frame
from .gbm import RaceModel
from .probs import all_bet_probs, blend_strength, candidate_horses, joint_top3, prune_combos

log = logging.getLogger(__name__)
KEEP_TOP = {"win": 18, "place": 18, "bracket_quinella": 36, "bracket_exacta": 40, "quinella": 40, "wide": 40, "exacta": 40,
            "trio": 120, "trifecta": 200}
CANDIDATE_K = 10  # バックテストの candidate_coverage で検証する (地方の実データで1-3着を97%含む)
DEFAULT_BLEND = (0.22, 0.86)  # 係数の推定前に使う初期値。地方の封鎖テスト前半の推定値


def load_past_roi(org: str) -> dict[str, float]:
    """合成確率で期待値プラスと判定した買い目の、券種別バックテスト回収率。"""
    f = config.REPORTS / f"backtest_{org}.json"
    if not f.exists():
        return {}
    rows = json.loads(f.read_text(encoding="utf-8")).get("rows", [])
    return {r["bet_type"]: r["roi"] for r in rows if r["strategy"] == "blend_ev_1.0" and r["n_bets"] >= 100}


def latest_model(con, org: str) -> tuple[str, RaceModel] | None:
    """現在採用中のモデル。"""
    row = con.execute("SELECT model_version_id,path FROM model_version WHERE org=? AND status='active' "
                      "ORDER BY adopted_at DESC, created_at DESC LIMIT 1", (org,)).fetchone()
    if row is None:
        return None
    if not (config.ROOT / row["path"]).exists():
        raise ModelIntegrityError(f"{row['model_version_id']}: 採用中のモデルファイルがありません")
    return row["model_version_id"], RaceModel.load(config.ROOT / row["path"])


def _post_at(race_date: str, post_time) -> datetime | None:
    return datetime.fromisoformat(f"{race_date}T{post_time}:00+09:00") if isinstance(post_time, str) and post_time else None


def predict_races(con: sqlite3.Connection, org: str, from_date: str | None = None, at: str | None = None) -> dict:
    """from_date 以降で発走前のレースを予測する。特徴量は学習と同じ build_features を使う。"""
    found = latest_model(con, org)
    if found is None:
        return {"org": org, "skipped": "モデル未学習"}
    mid, model = found
    predicted_at = at or now_jst()
    now_dt = datetime.fromisoformat(predicted_at)
    today = from_date or now_dt.astimezone(JST).date().isoformat()
    frame = serving_frame(load_frame(con, org), today)
    if frame.empty:
        return {"org": org, "races": 0}
    feats = build_features(add_market(frame, con))
    target = (frame["race_date"] >= today) & frame["status"].isna()
    stats = {"org": org, "races": 0, "model": mid, "recorded": 0, "skipped_started": 0, "skipped_unchanged": 0, "failed": 0}
    if not target.any():
        return stats
    bundle = active_bundle(con, org, mid)
    tf, tx = frame[target].reset_index(drop=True), feats[target].reset_index(drop=True)
    X_all = model.spec.transform(tx)
    members = model.member_scores_x(X_all)
    verified = verified_bet_types(con, org)
    if bundle["blend_alpha"] is not None:
        alpha, beta = bundle["blend_alpha"], bundle["blend_beta"]
    else:
        alpha, beta = DEFAULT_BLEND
    enforce = bool(config.guards().get("blend_enforce")) and bundle["guard_recommendation"] == "market_only"
    past_roi = load_past_roi(org)
    t1, t2, t3 = model.taus
    commit = git_commit()
    cols_hash = columns_hash(model.spec.columns)
    for rid, g in tf.groupby("race_id", sort=False):
        try:
            pos = g.index.to_numpy()
            um, wk = g["umaban"].tolist(), [None if pd.isna(w) else int(w) for w in g["waku"]]
            race_date, post = g["race_date"].iloc[0], _post_at(g["race_date"].iloc[0], g["post_time"].iloc[0])
            if post is not None and now_dt >= post:
                stats["skipped_started"] += 1   # 発走後は予測しない。発走前の記録と表示はそのまま残す
                continue
            stats["races"] += 1
            tp, mins = ledger.time_point(now_dt, post, race_date)
            odds = latest_odds(con, rid, predicted_at, include_final=False)
            win_rec = [odds.get(("win", str(u))) for u in um]
            win_odds = [r["odds"] if r and r["odds"] else None for r in win_rec]   # 0 は発売前か票なし。値として使わない
            place_rec = [odds.get(("place", str(u))) for u in um]
            times = {r["captured_at"] for r in win_rec if r and r["odds"]}
            complete = all(win_odds)
            status = "complete" if complete else ("partial" if times else "none")
            fh, payload = ledger.encode_inputs(X_all.iloc[pos], um)
            odds_sig = [[r["odds"] if r else None, r["captured_at"] if r else None] for r in win_rec]
            ih = ledger.input_hash(mid, bundle["bundle_id"], fh, odds_sig)
            if not ledger.should_record(con, rid, tp, mid, ih, fh, predicted_at, now_dt):
                stats["skipped_unchanged"] += 1
                continue
            mean_s = members[pos].mean(axis=1)
            p1_ai, _, p3_ai = joint_top3(mean_s, model.taus)
            top3_ai = p3_ai.sum(axis=(1, 2)) + p3_ai.sum(axis=(0, 2)) + p3_ai.sum(axis=(0, 1))
            p_market, p1_bl, top3_bl = None, None, None
            taus_blend = (1.0, t2 / t1, t3 / t1)
            if complete:
                inv = 1 / np.asarray(win_odds, dtype=float)
                p_market = inv / inv.sum()
                b_s = blend_strength(mean_s, win_odds, alpha, beta, t1)
                p1_bl, _, p3_bl = joint_top3(b_s, taus_blend)
                top3_bl = p3_bl.sum(axis=(1, 2)) + p3_bl.sum(axis=(0, 2)) + p3_bl.sum(axis=(0, 1))
            if complete and enforce:
                # ガード有効時は市場単独。AI の寄与を確認できない合成は使わない
                taus, used = taus_blend, "market"
                strength = lambda s: blend_strength(s, win_odds, 0.0, 1.0, t1)
            elif complete:
                taus, used = taus_blend, "blend"
                strength = lambda s: blend_strength(s, win_odds, alpha, beta, t1)
            else:
                taus, used, strength = model.taus, "ai", (lambda s: s)
            raw_win = all_bet_probs(um, wk, mean_s, model.taus, org)["win"]
            per_member = [all_bet_probs(um, wk, strength(members[pos, m]), taus, org) for m in range(members.shape[1])]
            used_s = strength(mean_s)
            base = all_bet_probs(um, wk, used_s, taus, org)
            run = {"race_id": rid, "org": org, "predicted_at": predicted_at, "scheduled_post_at": post.isoformat() if post else None, "time_point": tp, "minutes_to_post": mins,
                   "model_version_id": mid, "bundle_id": bundle["bundle_id"], "git_commit": commit, "input_hash": ih,
                   "odds_captured_at": max(times) if times else None, "odds_status": status, "odds_same_time": int(complete and len(times) == 1), "used_model": used,
                   "blend_alpha": alpha if complete else None, "blend_beta": beta if complete else None, "taus": json.dumps(model.taus), "taus_used": json.dumps(taus)}
            horses = []
            for i, u in enumerate(um):
                horses.append({"umaban": int(u), "horse_id": g["horse_id"].iloc[i], "waku": wk[i], "p_win_ai": float(p1_ai[i]), "p_top3_ai": float(top3_ai[i]),
                               "p_win_market": float(p_market[i]) if p_market is not None else None, "p_win_blend": float(p1_bl[i]) if p1_bl is not None else None,
                               "p_top3_blend": float(top3_bl[i]) if top3_bl is not None else None, "p_win_used": base["win"][str(u)], "strength_ai": float(mean_s[i]), "strength_used": float(used_s[i]),
                               "win_odds": win_odds[i], "place_odds": place_rec[i]["odds"] if place_rec[i] and place_rec[i]["odds"] else None, "place_odds_max": place_rec[i]["odds_max"] if place_rec[i] and place_rec[i]["odds"] else None,
                               "odds_captured_at": win_rec[i]["captured_at"] if win_odds[i] else None})
            pid = ledger.record_run(con, run, horses, fh, payload, cols_hash, len(um))
            stats["recorded"] += 1
            # 表示用の最新値だけを入れ替える。記録は上の台帳にある
            cands = candidate_horses(um, used_s, CANDIDATE_K)
            avg_runs = float(tx.loc[pos, "runs_count"].mean())
            con.execute("DELETE FROM prediction_display WHERE race_id=?", (rid,))
            for bet_type, combos in base.items():
                full = combos if bet_type in ("win", "place") else prune_combos(combos, cands, KEEP_TOP[bet_type])
                for combo, prob in full.items():
                    std = float(np.std([pm[bet_type].get(combo, 0.0) for pm in per_member])) if len(per_member) > 1 else None
                    o = odds.get((bet_type, combo))
                    ev, ev_max = expected(prob, o["odds"] if o else None, o["odds_max"] if o else None)
                    flags, conf = make_flags(prob, std, avg_runs, o, ev, verified.get(bet_type), now_dt, past_roi.get(bet_type))
                    con.execute("""INSERT INTO prediction_display(race_id,bet_type,combo,model_version_id,prediction_id,predicted_at,prob,prob_std,
                                   odds,odds_max,odds_captured_at,ev,ev_max,status,flags) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                                (rid, bet_type, combo, mid, pid, predicted_at, prob, std, o["odds"] if o else None, o["odds_max"] if o else None,
                                 o["captured_at"] if o else None, ev, ev_max, "OK" if o else STATUS_NO_ODDS,
                                 json.dumps({"flags": flags, "confidence": conf, "avg_runs": round(avg_runs, 1), "blended": used == "blend",
                                             **({"raw_p": round(raw_win[combo], 5)} if bet_type == "win" else {})}, ensure_ascii=False)))
            con.commit()   # 1レースごとに確定する。失敗したレースだけを取り消すため
        except Exception as e:
            con.rollback()
            if is_fatal(e):
                raise
            stats["failed"] += 1
            record_failure(con, org, "predict_race", e, rid)
    con.commit()
    return stats
