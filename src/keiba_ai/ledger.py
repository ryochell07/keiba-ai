"""予測台帳。予測は追記のみで、発走後には作らない。入力も保存して後から同じ確率を再現できるようにする。"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
import zlib
from datetime import datetime

import numpy as np
import pandas as pd

from . import config
from .gbm import RaceModel, softmax_by_race

# 同じ時点区分の中でオッズだけが変わった場合に、次の記録まで空ける分数。容量を抑えるため
MIN_GAP_MIN = {"day_before": 720, "morning": 60, "t30": 10, "t10": 5, "last": 0, "unknown": 60}
TIME_POINTS = ("day_before", "morning", "t30", "t10", "last")


def time_point(now: datetime, post: datetime | None, race_date: str) -> tuple[str, float | None]:
    """発走までの残り時間から時点を決める。朝は当日で発走45分より前。"""
    if post is None:
        return "unknown", None
    mins = (post - now).total_seconds() / 60
    if now.date().isoformat() < race_date:
        return "day_before", mins
    if mins > 45:
        return "morning", mins
    if mins > 15:
        return "t30", mins
    if mins > 5:
        return "t10", mins
    return "last", mins


def encode_inputs(X: pd.DataFrame, umaban: list[int]) -> tuple[str, bytes]:
    rows = [[None if pd.isna(v) else float(v) for v in r] for r in X.to_numpy(dtype=float)]
    raw = json.dumps({"columns": list(X.columns), "umaban": [int(u) for u in umaban], "rows": rows}, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest(), zlib.compress(raw, 9)


def decode_inputs(payload: bytes) -> tuple[pd.DataFrame, list[int]]:
    d = json.loads(zlib.decompress(payload))
    return pd.DataFrame(d["rows"], columns=d["columns"], dtype=float), d["umaban"]


def input_hash(model_id: str, bundle_id: str, feature_hash: str, odds: list) -> str:
    raw = json.dumps({"m": model_id, "b": bundle_id, "f": feature_hash, "o": odds}, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def should_record(con: sqlite3.Connection, race_id: str, tp: str, model_id: str, ihash: str, feature_hash: str, predicted_at: str, now: datetime) -> bool:
    """入力が同じなら確率も同じなので記録しない。オッズだけの変化は時点ごとの間隔を空ける。
    前日は当日の結果が入るたびに特徴量が変わるので、特徴量の変化でも間隔を空ける。"""
    last = con.execute("SELECT * FROM prediction_run WHERE race_id=? AND predicted_at<=? ORDER BY predicted_at DESC, rowid DESC LIMIT 1", (race_id, predicted_at)).fetchone()
    if last is None or last["time_point"] != tp or last["model_version_id"] != model_id:
        return True
    if last["input_hash"] == ihash:
        return False
    if tp != "day_before" and last["feature_hash"] != feature_hash:
        return True
    return (now - datetime.fromisoformat(last["predicted_at"])).total_seconds() / 60 >= MIN_GAP_MIN.get(tp, 60)


def record_run(con: sqlite3.Connection, run: dict, horses: list[dict], X_hash: str, payload: bytes, cols_hash: str, n_rows: int) -> str:
    pid = uuid.uuid4().hex
    con.execute("INSERT OR IGNORE INTO prediction_input(feature_hash,columns_hash,n_rows,payload) VALUES (?,?,?,?)", (X_hash, cols_hash, n_rows, payload))
    r = {**run, "prediction_id": pid, "feature_hash": X_hash}
    cols = list(r)
    con.execute(f"INSERT INTO prediction_run({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", [r[c] for c in cols])
    for h in horses:
        row = {**h, "prediction_id": pid}
        cols = list(row)
        con.execute(f"INSERT INTO prediction_horse({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", [row[c] for c in cols])
    return pid


def replay(con: sqlite3.Connection, prediction_id: str) -> dict:
    """保存した入力とモデルから確率を計算し直し、記録と比べる。"""
    run = con.execute("SELECT * FROM prediction_run WHERE prediction_id=?", (prediction_id,)).fetchone()
    if run is None:
        raise ValueError(f"予測 {prediction_id} がありません")
    inp = con.execute("SELECT payload FROM prediction_input WHERE feature_hash=?", (run["feature_hash"],)).fetchone()
    X, umaban = decode_inputs(inp["payload"])
    path = con.execute("SELECT path FROM model_version WHERE model_version_id=?", (run["model_version_id"],)).fetchone()["path"]
    model = RaceModel.load(config.ROOT / path)
    s = model.member_scores_x(X[model.spec.columns]).mean(axis=1)
    ids = pd.Series([run["race_id"]] * len(s))
    p = softmax_by_race(s, ids, json.loads(run["taus"])[0])
    stored = {r["umaban"]: r["p_win_ai"] for r in con.execute("SELECT umaban,p_win_ai FROM prediction_horse WHERE prediction_id=?", (prediction_id,))}
    diff = max(abs(float(p[i]) - stored[int(u)]) for i, u in enumerate(umaban))
    return {"prediction_id": prediction_id, "race_id": run["race_id"], "predicted_at": run["predicted_at"], "n_horses": len(umaban), "n_columns": X.shape[1], "max_abs_diff": diff, "reproduced": bool(diff < 1e-9)}
