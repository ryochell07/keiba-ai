"""モデルの版管理。新モデルは旧モデルと同じ未学習期間で比較し、悪ければ採用しない。"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date, datetime

import pandas as pd

from . import config
from .backtest import assert_time_split, evaluate
from .db import JST, now_jst
from .features import add_market, build_features, load_frame
from .gbm import RaceModel, train_race_model

log = logging.getLogger(__name__)
MIN_HOLDOUT_RACES = 100
EARLY_STAGE_RACES = 3000    # これ未満はデータ追加の効果が明らかなので、比較なしで更新してよい
LOGLOSS_TOLERANCE = 0.001   # 誤差程度の悪化は同等とみなす
COMPARE_KEYS = ("win_logloss_model", "brier", "top1_hit", "top1_win_roi", "n_races")


def list_models(con: sqlite3.Connection, org: str | None = None) -> list[dict]:
    sql = "SELECT * FROM model_version"
    args: list = []
    if org:
        sql += " WHERE org=?"
        args.append(org)
    return [dict(r) for r in con.execute(sql + " ORDER BY created_at DESC", args)]


def active_row(con, org: str):
    return con.execute("SELECT * FROM model_version WHERE org=? AND status='active' ORDER BY adopted_at DESC LIMIT 1", (org,)).fetchone()


def _register(con, org: str, model: RaceModel, status: str, n_seeds: int, comparison: dict | None) -> str:
    mid = f"{org}-plackett-{datetime.now(JST).strftime('%Y%m%d%H%M%S%f')}"
    path = config.MODELS / f"{mid}.pkl"
    model.save(path)
    con.execute("""INSERT INTO model_version(model_version_id,org,algo,created_at,train_start,train_end,features,params,metrics,path,status,adopted_at,comparison)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (mid, org, "lightgbm-plackett-luce", now_jst(), model.train_start, model.train_end,
                 json.dumps(model.spec.columns, ensure_ascii=False), json.dumps({"n_seeds": n_seeds, "taus": model.taus}),
                 json.dumps(model.metrics), str(path.relative_to(config.ROOT)), status,
                 now_jst() if status == "active" else None, json.dumps(comparison, ensure_ascii=False) if comparison else None))
    from .bundles import register_bundle
    register_bundle(con, mid, adoption_reason=(comparison or {}).get("reason") or "比較なしで採用 (初回または --force)", trained_now=True)
    return mid


def _adopt(con, org: str, mid: str) -> None:
    con.execute("UPDATE model_version SET status='retired' WHERE org=? AND status='active' AND model_version_id<>?", (org, mid))
    con.execute("UPDATE model_version SET status='active', adopted_at=? WHERE model_version_id=?", (now_jst(), mid))


def rollback(con: sqlite3.Connection, org: str, to: str | None = None) -> str:
    """指定モデル (省略時は直前に採用していたモデル) に戻す。"""
    if to is None:
        row = con.execute("SELECT model_version_id FROM model_version WHERE org=? AND status='retired' AND adopted_at IS NOT NULL "
                          "ORDER BY adopted_at DESC LIMIT 1", (org,)).fetchone()
        if row is None:
            raise SystemExit("戻せる旧モデルがありません")
        to = row[0]
    row = con.execute("SELECT path FROM model_version WHERE model_version_id=? AND org=?", (to, org)).fetchone()
    if row is None or not (config.ROOT / row["path"]).exists():
        raise SystemExit(f"モデル {to} のファイルが見つかりません")
    _adopt(con, org, to)
    con.commit()
    return f"{org}: {to} に戻しました"


def _scored(frame, feats, model: RaceModel) -> pd.DataFrame:
    return frame.assign(score=model.scores(feats), tau1=model.taus[0], tau2=model.taus[1], tau3=model.taus[2])


def decide(new: dict, old: dict) -> tuple[bool, str]:
    """新モデルの採用判定。確率の質 (Log Loss) を主、Brier を副とする。回収率はばらつきが大きいので判定に使わない。"""
    if new["win_logloss_model"] > old["win_logloss_model"] + LOGLOSS_TOLERANCE:
        return False, f"Log Loss が悪化 ({old['win_logloss_model']:.4f} → {new['win_logloss_model']:.4f})"
    if new["brier"] > old["brier"] + LOGLOSS_TOLERANCE:
        return False, f"Brier Score が悪化 ({old['brier']:.4f} → {new['brier']:.4f})"
    return True, "旧モデル以上の確率精度"


def auto_retrain_needed(con: sqlite3.Connection, org: str, growth: float = 1.3) -> tuple[bool, bool]:
    """(再学習が必要か, 比較なしで採用してよい初期段階か)。過去データの追加で急に増えたときに再学習する。"""
    n = con.execute("SELECT COUNT(DISTINCT x.race_id) FROM result x JOIN race r USING(race_id) WHERE r.org=? AND x.finish_pos=1 AND r.course_code NOT LIKE 'N帯広%'", (org,)).fetchone()[0]
    champion = active_row(con, org)
    if champion is None:
        return n >= 100, True
    row = con.execute("SELECT n_races FROM training_run WHERE model_version_id=? ORDER BY run_id DESC LIMIT 1", (champion["model_version_id"],)).fetchone()
    trained = row["n_races"] if row and row["n_races"] else 0
    return n >= trained * growth, trained < EARLY_STAGE_RACES


def retrain_with_gate(con: sqlite3.Connection, org: str, n_seeds: int = 3, force: bool = False) -> dict:
    """手順: ① 旧モデルの学習末より後の期間を前半・後半に分ける
    ② 前半まで学習した挑戦者と旧モデルを後半で比較 ③ 合格なら全データで再学習して採用。"""
    started = now_jst()
    frame = load_frame(con, org)
    frame = frame[frame["status"].notna()].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{org}: 学習データがありません")
    feats = build_features(add_market(frame, con))
    champion = active_row(con, org)
    comparison = None
    if champion is not None and not force:
        unseen = sorted(frame.loc[frame["race_date"] > champion["train_end"], "race_date"].unique())
        if len(unseen) < 2:
            return {"org": org, "decision": "見送り", "reason": "旧モデル以降の新しいデータがありません"}
        hold_start = unseen[len(unseen) // 2]
        hold = (frame["race_date"] >= hold_start).to_numpy()
        n_hold = frame.loc[hold, "race_id"].nunique()
        if n_hold < MIN_HOLDOUT_RACES:
            return {"org": org, "decision": "見送り", "reason": f"比較用レース数が不足 ({n_hold}/{MIN_HOLDOUT_RACES})"}
        tr = ~hold
        assert_time_split(frame.loc[tr, "race_date"], frame.loc[hold, "race_date"])
        challenger = train_race_model(frame[tr].reset_index(drop=True), feats[tr].reset_index(drop=True), org, n_seeds=max(1, n_seeds - 1))
        hf, hx = frame[hold].reset_index(drop=True), feats[hold].reset_index(drop=True)
        new = evaluate(con, _scored(hf, hx, challenger), org)
        old = evaluate(con, _scored(hf, hx, RaceModel.load(config.ROOT / champion["path"])), org)
        ok, reason = decide(new["summary"], old["summary"])
        comparison = {"holdout": [hold_start, frame["race_date"].max()], "champion": champion["model_version_id"],
                      "new": {k: new["summary"][k] for k in COMPARE_KEYS}, "old": {k: old["summary"][k] for k in COMPARE_KEYS},
                      "new_by_going": new["by_going"], "old_by_going": old["by_going"], "new_by_course": new["by_course"],
                      "old_by_course": old["by_course"], "adopted": ok, "reason": reason}
        if not ok:
            mid = _register(con, org, challenger, "rejected", n_seeds, comparison)
            con.execute("INSERT INTO training_run(model_version_id,started_at,finished_at,n_races,n_rows,status,detail) VALUES (?,?,?,?,?,?,?)",
                        (mid, started, now_jst(), frame["race_id"].nunique(), len(frame), "rejected", reason))
            con.commit()
            return {"org": org, "decision": "旧モデルを維持", "reason": reason, "comparison": {k: comparison[k] for k in ("new", "old")}}
    model = train_race_model(frame, feats, org, n_seeds=n_seeds)
    mid = _register(con, org, model, "retired", n_seeds, comparison)
    _adopt(con, org, mid)
    con.execute("INSERT INTO training_run(model_version_id,started_at,finished_at,n_races,n_rows,status) VALUES (?,?,?,?,?,?)",
                (mid, started, now_jst(), frame["race_id"].nunique(), len(frame), "adopted"))
    con.commit()
    return {"org": org, "decision": "採用", "model": mid, "comparison": comparison and {k: comparison[k] for k in ("new", "old", "reason")}}
