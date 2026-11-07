"""モデル本体・校正・市場との合成係数を1組の版として記録する。記録は追記のみ。"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import subprocess
from pathlib import Path

import yaml

from . import config
from .db import now_jst
from .failures import ModelIntegrityError
from .gbm import RaceModel

log = logging.getLogger(__name__)
FEATURE_VERSION = "2026-11-06-window-rate-fix"


def git_commit() -> str | None:
    """動いているコードのコミット。分からなければ NULL。"""
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.run(["git", "-C", str(config.ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def columns_hash(columns: list[str]) -> str:
    return hashlib.sha256(json.dumps(columns, ensure_ascii=False).encode()).hexdigest()


def guard_decision(alpha: float | None, ci: list[float] | None) -> tuple[str, str]:
    """AI と市場を合成してよいか。独立した検証で AI の寄与を確認できなければ市場単独を勧める。"""
    if alpha is None:
        return "market_only", "合成係数がない"
    if alpha <= 0:
        return "market_only", f"AI の係数が {alpha:.3f} で正でない"
    if ci is None:
        return "market_only", "係数の信頼区間が未算出"
    if ci[0] <= 0 <= ci[1]:
        return "market_only", f"AI の係数の95%信頼区間 [{ci[0]:.3f}, {ci[1]:.3f}] が0を含む"
    return "blend", f"AI の係数 {alpha:.3f}、95%信頼区間 [{ci[0]:.3f}, {ci[1]:.3f}]"


def _report_blend(org: str) -> tuple[dict | None, str | None, str | None]:
    f = config.REPORTS / f"backtest_{org}.json"
    if not f.exists():
        return None, None, None
    d = json.loads(f.read_text(encoding="utf-8"))
    b = d.get("blend")
    if not b:
        return None, None, None
    return b, f"backtest_{org}.json ({d.get('backtest_id')})", f"{d['period'][0]}..{b.get('evaluated_from')} (未満)"


def _known_ci(mid: str) -> dict | None:
    f = config.ROOT / "config" / "blend_ci.yaml"
    if not f.exists():
        return None
    return (yaml.safe_load(f.read_text(encoding="utf-8")) or {}).get(mid)


def _valid_period(con, row, n_valid: int) -> tuple[str | None, str | None, str | None]:
    """学習時に記録がないモデルについて、学習と同じ並びから検証区間を導く。"""
    rows = con.execute("""SELECT r.race_date FROM race r WHERE r.org=? AND r.course_code NOT LIKE 'N帯広%' AND r.race_date BETWEEN ? AND ?
                          AND EXISTS (SELECT 1 FROM result x WHERE x.race_id=r.race_id AND x.finish_pos=1)
                          ORDER BY r.race_date, r.post_time, r.race_id""", (row["org"], row["train_start"], row["train_end"])).fetchall()
    if not rows or n_valid >= len(rows):
        return None, None, None
    return rows[-n_valid - 1][0], rows[-n_valid][0], rows[-1][0]


def register_bundle(con: sqlite3.Connection, mid: str, adoption_reason: str | None = None, trained_now: bool = False, note: str | None = None) -> str:
    """現在の内容で版を記録する。同じ内容の版があれば何もしない。"""
    row = con.execute("SELECT * FROM model_version WHERE model_version_id=?", (mid,)).fetchone()
    path = config.ROOT / row["path"]
    model = RaceModel.load(path)
    m = model.metrics
    params = m.get("hyperparams")
    if not params:  # 学習時に記録していないモデルはモデルファイル自体から読む
        text = model.boosters[0].model_to_string()
        params = {k: v for k, v in re.findall(r"^\[(\w+): (.*)\]$", text, re.M)
                  if k in ("learning_rate", "num_leaves", "min_data_in_leaf", "feature_fraction", "bagging_fraction", "bagging_freq", "lambda_l2", "seed")}
    if m.get("valid_start"):
        tree_end, vs, ve, basis = m.get("tree_train_end"), m["valid_start"], m["valid_end"], "学習時の記録"
    else:
        tree_end, vs, ve = _valid_period(con, row, m.get("n_valid_races", 0))
        basis = "DBから学習と同じ並びで導出"
    blend, source, period = _report_blend(row["org"])
    alpha, beta = (blend["alpha"], blend["beta"]) if blend else (None, None)
    known = _known_ci(mid)
    ci = known.get("alpha_ci") if known else None
    rec, reason = guard_decision(alpha, ci)
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    content = {"model_version_id": mid, "org": row["org"], "git_commit": git_commit() if trained_now else None, "model_hash": h,
               "train_start": row["train_start"], "train_end": row["train_end"], "tree_train_end": tree_end, "valid_start": vs, "valid_end": ve,
               "valid_period_basis": basis, "n_train_races": m.get("n_train_races"), "n_valid_races": m.get("n_valid_races"), "feature_version": FEATURE_VERSION if trained_now else "学習時は未記録",
               "columns_hash": columns_hash(model.spec.columns), "features": json.dumps(model.spec.columns, ensure_ascii=False), "hyperparams": json.dumps(params, ensure_ascii=False),
               "n_seeds": len(model.boosters), "n_trees": json.dumps([b.num_trees() for b in model.boosters]), "taus": json.dumps(model.taus), "tau_period": f"{vs}..{ve}" if vs else None,
               "blend_alpha": alpha, "blend_beta": beta, "blend_source": source, "blend_fit_period": period, "blend_ci": json.dumps(known, ensure_ascii=False) if known else None,
               "guard_recommendation": rec, "guard_reason": reason, "adoption_reason": adoption_reason, "note": note}
    same = con.execute("SELECT * FROM model_bundle WHERE model_version_id=? ORDER BY created_at DESC, rowid DESC LIMIT 1", (mid,)).fetchone()
    if same and all(same[k] == content[k] for k in ("model_hash", "blend_alpha", "blend_beta", "blend_ci")):
        return same["bundle_id"]
    bid = f"{mid}:{hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]}"
    content.update(bundle_id=bid, created_at=now_jst())
    cols = list(content)
    con.execute(f"INSERT OR IGNORE INTO model_bundle({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", [content[c] for c in cols])
    return bid


def active_bundle(con: sqlite3.Connection, org: str, mid: str):
    """採用中モデルの最新の版。なければ今の内容で記録する。モデルファイルが記録と違えば止める。"""
    row = con.execute("SELECT * FROM model_bundle WHERE model_version_id=? ORDER BY created_at DESC, rowid DESC LIMIT 1", (mid,)).fetchone()
    if row is None:
        reg = con.execute("SELECT comparison FROM model_version WHERE model_version_id=?", (mid,)).fetchone()
        reason = json.loads(reg["comparison"]).get("reason") if reg and reg["comparison"] else "記録なし (Phase 1 以前に採用)"
        register_bundle(con, mid, adoption_reason=reason, note="Phase 1 で後付け登録。学習時のコミットと特徴量の版は記録がないため NULL または未記録")
        con.commit()
        row = con.execute("SELECT * FROM model_bundle WHERE model_version_id=? ORDER BY created_at DESC, rowid DESC LIMIT 1", (mid,)).fetchone()
    path = con.execute("SELECT path FROM model_version WHERE model_version_id=?", (mid,)).fetchone()["path"]
    if hashlib.sha256(Path(config.ROOT / path).read_bytes()).hexdigest() != row["model_hash"]:
        raise ModelIntegrityError(f"{mid}: モデルファイルが記録したハッシュと一致しません")
    return row
