"""学習済みモデルで未確定レースを予測し、全券種の確率・期待値を保存する。"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd

from . import config
from .db import JST, now_jst
from .ev import STATUS_NO_ODDS, expected, latest_odds, make_flags, verified_bet_types
from .features import add_market, build_features, load_frame
from .gbm import RaceModel
from .probs import all_bet_probs, blend_strength, candidate_horses, prune_combos

log = logging.getLogger(__name__)
KEEP_TOP = {"win": 18, "place": 18, "bracket_quinella": 36, "bracket_exacta": 40, "quinella": 40, "wide": 40, "exacta": 40,
            "trio": 120, "trifecta": 200}
CANDIDATE_K = 10  # バックテストの candidate_coverage で検証する (地方の実データで1-3着を97%含む)
DEFAULT_BLEND = (0.1, 0.95)  # 係数の推定前に使う初期値。地方の実データの推定値に近い


def load_blend(org: str) -> tuple[float, float] | None:
    """最新バックテストで推定した市場との合成係数。"""
    f = config.REPORTS / f"backtest_{org}.json"
    if not f.exists():
        return None
    b = json.loads(f.read_text(encoding="utf-8")).get("blend")
    return (b["alpha"], b["beta"]) if b else None


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
    if row is None or not (config.ROOT / row["path"]).exists():
        return None
    return row["model_version_id"], RaceModel.load(config.ROOT / row["path"])


def predict_races(con: sqlite3.Connection, org: str, from_date: str | None = None, at: str | None = None) -> dict:
    """from_date 以降で結果未確定のレースを予測する。特徴量は学習と同じ build_features を使う。"""
    found = latest_model(con, org)
    if found is None:
        return {"org": org, "skipped": "モデル未学習"}
    mid, model = found
    today = from_date or datetime.now(JST).date().isoformat()
    frame = load_frame(con, org)
    if frame.empty:
        return {"org": org, "races": 0}
    feats = build_features(add_market(frame, con))
    target = (frame["race_date"] >= today) & frame["status"].isna()
    if not target.any():
        return {"org": org, "races": 0, "model": mid}
    tf, tx = frame[target].reset_index(drop=True), feats[target].reset_index(drop=True)
    members = model.member_scores(tx)
    predicted_at = at or now_jst()
    verified = verified_bet_types(con, org)
    blend = load_blend(org) or DEFAULT_BLEND
    past_roi = load_past_roi(org)
    t1, t2, t3 = model.taus
    now_dt = datetime.fromisoformat(predicted_at)
    count = 0
    for rid, g in tf.groupby("race_id", sort=False):
        pos = g.index.to_numpy()
        um, wk = g["umaban"].tolist(), [None if pd.isna(w) else int(w) for w in g["waku"]]
        odds = latest_odds(con, rid, predicted_at)
        win_odds = [odds.get(("win", str(u)), {}).get("odds") for u in um]
        mean_s = members[pos].mean(axis=1)
        raw_win = all_bet_probs(um, wk, mean_s, model.taus, org)["win"]
        if all(win_odds):
            # 単勝オッズが全頭そろっていれば、市場と合成した確率で全券種を計算する
            taus = (1.0, t2 / t1, t3 / t1)
            strength = lambda s: blend_strength(s, win_odds, blend[0], blend[1], t1)
        else:
            taus, strength = model.taus, (lambda s: s)
        per_member = [all_bet_probs(um, wk, strength(members[pos, m]), taus, org) for m in range(members.shape[1])]
        base = all_bet_probs(um, wk, strength(mean_s), taus, org)
        cands = candidate_horses(um, strength(mean_s), CANDIDATE_K)
        avg_runs = float(tx.loc[pos, "runs_count"].mean())
        con.execute("DELETE FROM prediction WHERE race_id=? AND model_version_id=?", (rid, mid))
        for bet_type, combos in base.items():
            full = combos if bet_type in ("win", "place") else prune_combos(combos, cands, KEEP_TOP[bet_type])
            for combo, prob in full.items():
                std = float(np.std([pm[bet_type].get(combo, 0.0) for pm in per_member])) if len(per_member) > 1 else None
                o = odds.get((bet_type, combo))
                ev, ev_max = expected(prob, o["odds"] if o else None, o["odds_max"] if o else None)
                flags, conf = make_flags(prob, std, avg_runs, o, ev, verified.get(bet_type), now_dt, past_roi.get(bet_type))
                con.execute("""INSERT OR REPLACE INTO prediction(model_version_id,race_id,bet_type,combo,predicted_at,prob,prob_std,
                               odds,odds_max,odds_captured_at,ev,ev_max,status,flags) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (mid, rid, bet_type, combo, predicted_at, prob, std, o["odds"] if o else None, o["odds_max"] if o else None,
                             o["captured_at"] if o else None, ev, ev_max, "OK" if o else STATUS_NO_ODDS,
                             json.dumps({"flags": flags, "confidence": conf, "avg_runs": round(avg_runs, 1), "blended": all(win_odds),
                                         **({"raw_p": round(raw_win[combo], 5)} if bet_type == "win" else {})}, ensure_ascii=False)))
        count += 1
    con.commit()
    return {"org": org, "races": count, "model": mid}
