"""評価の手順。基準 (main) と候補 (エージェントの作業コピー) のコードを別々のプロセスで読み込み、
同じデータ・同じ期間・同じ乱数で学習して採点する。

  score     指定したコードで学習し、評価期間の各馬の確率を CSV に書く
  features  指定したコードで短い期間の特徴量を作り、列ごとの要約と未来情報の検査結果を JSON に書く
  compare   基準と候補の CSV から指標の差と、開催日単位のブートストラップによる95%区間を計算する
  analyze   基準の CSV から弱点の集計を作る

score と features は候補のコードを実行するので、実行役 (runner) は通信と書き込みを制限した中で動かす。
プロセスの間は CSV と JSON だけで受け渡す (pickle は使わない)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import settings

# 候補の確率の関数に渡さない列 (発走後に分かる情報)
OUTCOME = ["won", "pay", "popularity_final"]
# 弱点の集計と、候補の確率の関数に渡す説明用の列
DESCRIBE = ["course_code", "surface", "distance", "going", "weather", "n_runners", "age", "sex", "class_ord", "runs_count",
            "rest_days", "last_finish", "jockey_changed", "dist_change", "class_change", "fin_ratio_m5"]
# 特徴量の検査に使う短い期間 (いずれも探索期間より前)
CHECK_SLICE = {"JRA": ("2024-01-01", "2024-10-01", "2024-12-31"), "NAR": ("2024-07-01", "2024-10-01", "2024-12-31")}
BOOT = 2000


# ---------------------------------------------------------------- コードとデータの読み込み

def import_code(code: Path):
    """code/keiba_ai を読み込む。インストール済みの keiba_ai が先に読まれていないことを確かめる。"""
    src = Path(code).resolve()
    if "keiba_ai" in sys.modules:
        raise RuntimeError("keiba_ai が先に読み込まれています")
    sys.path.insert(0, str(src))
    import keiba_ai
    if not Path(keiba_ai.__file__).resolve().is_relative_to(src):
        raise RuntimeError(f"別の keiba_ai が読み込まれました: {keiba_ai.__file__}")
    return keiba_ai


def code_hash(code: Path) -> str:
    h = hashlib.sha256()
    for p in sorted((Path(code) / "keiba_ai").rglob("*")):
        if p.is_file() and p.suffix in (".py", ".sql"):
            h.update(str(p.relative_to(code)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def db_for(cfg: dict, stage: str) -> Path:
    """確認期間だけは確認期間を含むDBを読む。それ以外は確認期間も含まない探索用DB。"""
    P = settings.paths(cfg)
    return P.db if stage == "confirm" else P.explore_db


def open_db(db: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _day_before(d: str) -> str:
    return (date.fromisoformat(d) - timedelta(days=1)).isoformat()


def assert_before(frame: pd.DataFrame, end: str, sealed_from: str) -> None:
    if end > sealed_from:
        raise RuntimeError(f"評価期間が封鎖期間に入っています: {end}")
    if len(frame) and frame["race_date"].max() >= end:
        raise RuntimeError(f"評価期間より後のデータが読み込まれました: {frame['race_date'].max()}")


def load(con, org: str, start: str, end: str, sealed_from: str):
    """start 以降 end 前の確定済みレース。end は封鎖日を超えない。"""
    from keiba_ai.features import add_market, build_features, load_frame
    from keiba_ai.gbm import trainable
    frame = load_frame(con, org, until=_day_before(end))
    frame = frame[frame["status"].notna() & (frame["race_date"] >= start)].reset_index(drop=True)
    assert_before(frame, end, sealed_from)
    feats = build_features(add_market(frame, con))
    return trainable(frame, feats)


# ---------------------------------------------------------------- 市場の代用と最終確率

def win_payouts(con, org: str) -> pd.DataFrame:
    return pd.read_sql_query("SELECT p.race_id, CAST(p.combo AS INTEGER) AS umaban, p.amount AS pay FROM payout p JOIN race r USING(race_id) "
                             "WHERE r.org=? AND p.bet_type='win'", con, params=[org])


def market_proxy(con, org: str, frame: pd.DataFrame, train_mask: np.ndarray, ev_mask: np.ndarray) -> pd.DataFrame:
    """評価用の市場の確率。中央は確定単勝オッズ、地方 (開発期間にオッズがない) は確定人気から作る代用。
    どちらも締切後の情報なので特徴量には使わず、合成と比較の基準にだけ使う。"""
    ev = frame.loc[ev_mask, ["race_id", "umaban", "n_runners", "popularity_final"]].reset_index(drop=True)
    if org == "JRA":
        o = pd.read_sql_query("SELECT o.race_id, CAST(o.combo AS INTEGER) AS umaban, o.odds FROM odds_snapshot o JOIN race r USING(race_id) "
                              "WHERE o.bet_type='win' AND o.is_final=1 AND r.org=?", con, params=[org])
        o = o.drop_duplicates(["race_id", "umaban"])
        d = ev.merge(o, how="left", on=["race_id", "umaban"])
        good = np.isfinite(d["odds"].to_numpy(dtype=float)) & (d["odds"].to_numpy(dtype=float) > 0)
        ok = pd.Series(good).groupby(d["race_id"]).transform("all").to_numpy(dtype=bool)
        # 取消馬のオッズが残るレースは市場の確率が歪むので除く
        n_odds = o.groupby("race_id").size()
        ok = ok & (d["race_id"].map(n_odds).to_numpy(dtype=float) == d.groupby("race_id")["umaban"].transform("count").to_numpy(dtype=float))
        inv = np.where(ok, 1 / np.where(good, d["odds"], np.nan), np.nan)
    else:
        tr = frame.loc[train_mask].copy()
        tr["pop"] = pd.to_numeric(tr["popularity_final"], errors="coerce")
        tr["y"] = (pd.to_numeric(tr["finish_pos"], errors="coerce") == 1).astype(float)
        tr["nb"] = pd.cut(tr["n_runners"], [0, 8, 10, 12, 99], labels=False)
        table = tr.groupby(["nb", "pop"])["y"].agg(["sum", "count"])
        rate = ((table["sum"] + 0.5) / (table["count"] + 5)).to_dict()
        pop = pd.to_numeric(ev["popularity_final"], errors="coerce")
        nb = pd.cut(ev["n_runners"], [0, 8, 10, 12, 99], labels=False)
        ok = pop.notna().groupby(ev["race_id"]).transform("all").to_numpy()
        inv = np.where(ok, [rate.get((b, q), 0.005) for b, q in zip(nb, pop.fillna(99))], np.nan)
    tot = pd.Series(inv).groupby(ev["race_id"]).transform("sum").to_numpy()
    p = inv / tot
    return pd.DataFrame({"p_mkt": p, "odds": 1 / p, "mkt_ok": ok})


def softmax(strength: np.ndarray, race_id: pd.Series) -> np.ndarray:
    s = pd.Series(strength, index=race_id.index)
    e = np.exp(s - s.groupby(race_id, sort=False).transform("max"))
    return (e / e.groupby(race_id, sort=False).transform("sum")).to_numpy()


def production_final(probs, hist: pd.DataFrame, cur: pd.DataFrame) -> np.ndarray:
    """本番と同じ合成: 過去の月で合成係数を推定し、当月に使う。市場のないレースは AI 単独。"""
    h = hist[hist["mkt_ok"]]
    if h["race_id"].nunique() < 100:
        return cur["p_ai"].to_numpy()
    a, b = probs.fit_blend(h.assign(win=h["won"].astype(int))[["race_id", "score", "tau1", "odds", "win"]])
    odds = cur["odds"].where(cur["mkt_ok"], 1.0).to_numpy(dtype=float)
    blended = probs.blend_strength(cur["score"].to_numpy(), odds, a, b, cur["tau1"].to_numpy())
    strength = np.where(cur["mkt_ok"], blended, cur["score"] / cur["tau1"])
    return softmax(strength, cur["race_id"])


def final_track(probs, rows: pd.DataFrame) -> np.ndarray:
    """月ごとに、それより前の月の結果だけを使って最終確率を作る。最初の月は採点しない。

    候補のコードの probs に win_probs_from_history(hist, cur) があればそれを使う。cur には結果の列を渡さない。"""
    fn = getattr(probs, "win_probs_from_history", None)
    month = rows["race_date"].str[:7]
    out = np.full(len(rows), np.nan)
    for m in sorted(month.unique())[1:]:
        hist = rows[(month < m).to_numpy()].reset_index(drop=True)
        cur_mask = (month == m).to_numpy()
        cur = rows[cur_mask].drop(columns=OUTCOME).reset_index(drop=True)
        q = np.asarray(fn(hist.copy(), cur.copy()) if fn else production_final(probs, hist, cur), dtype=float)
        if q.shape != (len(cur),) or not np.isfinite(q).all() or (q < 0).any():
            raise ValueError("最終確率の形か値が不正です")
        out[cur_mask] = q / pd.Series(q).groupby(cur["race_id"]).transform("sum").to_numpy()
    return out


# ---------------------------------------------------------------- score

def cmd_score(args, cfg) -> dict:
    t0 = time.time()
    per = cfg["periods"]
    start, end = per[args.stage]
    org = args.org
    import_code(args.code)
    from keiba_ai import probs
    from keiba_ai.gbm import softmax_by_race, train_race_model
    con = open_db(db_for(cfg, args.stage))
    frame, feats = load(con, org, cfg["protocol"]["history_from"][org], end, per["sealed_from"])
    d = frame["race_date"]
    tr = ((d >= cfg["protocol"]["train_since"][org]) & (d < start)).to_numpy()
    ev = ((d >= start) & (d < end)).to_numpy()
    model = train_race_model(frame[tr].reset_index(drop=True), feats[tr].reset_index(drop=True), org, n_seeds=cfg["protocol"]["n_seeds"])
    ef, ex = frame[ev].reset_index(drop=True), feats[ev].reset_index(drop=True)
    score = model.scores(ex)
    tau1 = float(model.taus[0])
    pay = win_payouts(con, org)
    rows = ef[["race_id", "umaban", "race_date", "popularity_final"]].copy()
    rows["won"] = (pd.to_numeric(ef["finish_pos"], errors="coerce") == 1).to_numpy()
    rows = rows.merge(pay.groupby(["race_id", "umaban"], as_index=False)["pay"].sum(), how="left", on=["race_id", "umaban"])
    rows["pay"] = rows["pay"].fillna(0.0)
    rows["score"], rows["tau1"] = score, tau1
    rows["p_ai"] = softmax_by_race(score, rows["race_id"], tau1)
    rows = pd.concat([rows, market_proxy(con, org, frame, tr, ev)], axis=1)
    for c in DESCRIBE:
        rows[c] = ex[c].to_numpy() if c in ex else (ef[c].to_numpy() if c in ef else np.nan)
    rows["p_final"] = final_track(probs, rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out, index=False, compression="gzip")
    meta = {"org": org, "stage": args.stage, "period": [start, end], "code_hash": code_hash(args.code),
            "train_since": cfg["protocol"]["train_since"][org], "n_train_races": int(frame.loc[tr, "race_id"].nunique()),
            "n_eval_races": int(ef["race_id"].nunique()), "taus": list(model.taus), "best_rounds": model.metrics["best_rounds"],
            "n_features": len(model.spec.columns), "features": model.spec.columns, "last_loaded_date": str(d.max()),
            "custom_final": hasattr(probs, "win_probs_from_history"), "seconds": round(time.time() - t0)}
    Path(str(out) + ".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


# ---------------------------------------------------------------- features (未来情報と既存列の検査)

def _digest(s: pd.Series) -> str:
    if s.dtype.kind in "fiub":
        v = np.round(s.to_numpy(dtype=float), 8)
        v = np.where(np.isnan(v), -1.2345e300, v)
        return hashlib.sha1(np.ascontiguousarray(v).tobytes()).hexdigest()[:16]
    return hashlib.sha1("\x1f".join(str(v) for v in s.tolist()).encode()).hexdigest()[:16]


def _differs(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    bad = []
    for c in b.columns:
        x, y = a[c].reset_index(drop=True), b[c].reset_index(drop=True)
        if x.dtype.kind in "fiub" and y.dtype.kind in "fiub":
            if not np.allclose(x.to_numpy(dtype=float), y.to_numpy(dtype=float), rtol=1e-7, atol=1e-9, equal_nan=True):
                bad.append(c)
        elif [str(v) for v in x.tolist()] != [str(v) for v in y.tolist()]:
            bad.append(c)
    return bad


def cmd_features(args, cfg) -> dict:
    import_code(args.code)
    from keiba_ai.features import add_market, build_features, load_frame
    org = args.org
    start, cut, end = CHECK_SLICE[org]
    con = open_db(db_for(cfg, "explore"))
    frame = load_frame(con, org, until=end)
    frame = frame[frame["status"].notna() & (frame["race_date"] >= start)].reset_index(drop=True)
    full = build_features(add_market(frame, con))
    # ① 未来のレースを消しても、過去の行の特徴量は変わらない
    before = (frame["race_date"] < cut).to_numpy()
    part = build_features(add_market(frame[before].reset_index(drop=True), con))
    future_leak = _differs(full[before].reset_index(drop=True)[part.columns], part)
    # ② そのレースの結果を入れ替えても、そのレースの特徴量は変わらない (各競馬場の最終レースで確かめる)
    day = frame.loc[frame["race_date"] >= cut, "race_date"].min()
    on_day = frame[frame["race_date"] == day]
    last = on_day.sort_values(["post_time", "race_no"]).groupby("course_code")["race_id"].last()
    target = frame["race_id"].isin(set(last)).to_numpy()
    mutated = frame.copy()
    for c in ("finish_pos", "time_sec", "margin", "last3f", "popularity_final", "corners"):
        mutated.loc[target, c] = mutated.loc[target, c].to_numpy()[::-1]
    after = build_features(add_market(mutated, con))
    own_leak = _differs(full[target].reset_index(drop=True), after[target].reset_index(drop=True))
    out = {"org": org, "slice": [start, cut, end], "n_rows": len(frame), "columns": list(full.columns),
           "digests": {c: _digest(full[c]) for c in full.columns}, "future_leak": future_leak, "own_result_leak": own_leak,
           "code_hash": code_hash(args.code)}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return {k: out[k] for k in ("org", "n_rows", "future_leak", "own_result_leak")}


def column_changes(base: dict, cand: dict) -> dict:
    """既存の列の値が変わっていないか。変わると、固定中の本番モデルに渡す入力が変わる。"""
    changed = [c for c in base["columns"] if cand["digests"].get(c) != base["digests"][c]]
    added = [c for c in cand["columns"] if c not in base["digests"]]
    return {"changed_existing": changed, "added": added}


# ---------------------------------------------------------------- compare

def read_rows(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"race_id": str, "race_date": str})


def per_race(rows: pd.DataFrame, col: str) -> pd.DataFrame:
    d = rows[["race_id", "race_date", "won", "pay", col]].rename(columns={col: "p"})
    d = d[~d["p"].isna().groupby(d["race_id"]).transform("any")]
    d["nl"] = np.where(d["won"], -np.log(np.maximum(d["p"], 1e-12)), np.nan)
    d["se"] = (d["p"] - d["won"].astype(float)) ** 2
    g = d.groupby("race_id", sort=False)
    top = d.loc[g["p"].idxmax()].set_index("race_id")
    out = pd.DataFrame({"race_date": g["race_date"].first(), "ll": g["nl"].mean(), "brier": g["se"].sum()})
    out["hit"] = top["won"].astype(float)
    out["ret"] = top["pay"] / 100.0
    return out.dropna(subset=["ll"])


def day_boot_ci(values: pd.Series, dates: pd.Series, n: int = BOOT, seed: int = 0) -> tuple[float, float]:
    """開催日をまとめて抜き出すブートストラップで平均の95%区間を出す。同じ日のレースの相関を考慮する。"""
    rng = np.random.default_rng(seed)
    g = pd.DataFrame({"v": values.to_numpy(), "d": dates.to_numpy()}).groupby("d")["v"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    pick = rng.integers(0, len(s), size=(n, len(s)))
    m = s[pick].sum(axis=1) / c[pick].sum(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def compare_track(base: pd.DataFrame, cand: pd.DataFrame, col: str) -> dict:
    b, c = per_race(base, col), per_race(cand, col)
    common = b.index.intersection(c.index)
    b, c = b.loc[common], c.loc[common]
    dll, dbr = c["ll"] - b["ll"], c["brier"] - b["brier"]
    return {"n_races": int(len(common)), "n_days": int(b["race_date"].nunique()),
            "ll_base": float(b["ll"].mean()), "ll_cand": float(c["ll"].mean()), "d_ll": float(dll.mean()), "d_ll_ci": day_boot_ci(dll, b["race_date"]),
            "brier_base": float(b["brier"].mean()), "brier_cand": float(c["brier"].mean()), "d_brier": float(dbr.mean()),
            "d_brier_ci": day_boot_ci(dbr, b["race_date"]), "hit_base": float(b["hit"].mean()), "hit_cand": float(c["hit"].mean()),
            "roi_base": float(b["ret"].mean()), "roi_cand": float(c["ret"].mean()), "roi_cand_ci": day_boot_ci(c["ret"], c["race_date"])}


def compare(base: pd.DataFrame, cand: pd.DataFrame) -> dict:
    key = ["race_id", "umaban"]
    if not base[key].sort_values(key).reset_index(drop=True).equals(cand[key].sort_values(key).reset_index(drop=True)):
        raise ValueError("基準と候補で評価した馬が違います")
    cand = base[key].merge(cand, on=key, how="left")
    out = {"ai": compare_track(base, cand, "p_ai"), "final": compare_track(base, cand, "p_final"),
           "ai_unchanged": bool(np.allclose(base["p_ai"], cand["p_ai"], rtol=0, atol=1e-12))}
    mk = per_race(base[base["mkt_ok"]], "p_mkt")
    fin = per_race(base, "p_final").reindex(mk.index).dropna()
    out["market_ref"] = {"n_races": int(len(fin)), "ll_market": float(mk.loc[fin.index, "ll"].mean()) if len(fin) else None,
                         "ll_final_base": float(fin["ll"].mean()) if len(fin) else None}
    return out


# ---------------------------------------------------------------- analyze

def analyze(rows: pd.DataFrame) -> dict:
    """基準の予測の弱点。予測した確率の平均と実際の勝率の差 (gap) と、区分ごとの Log Loss。"""
    d = rows.copy()
    d["y"] = d["won"].astype(float)
    p = np.clip(d["p_ai"].to_numpy(), 1e-9, 1 - 1e-9)
    d["bll"] = -(d["y"] * np.log(p) + (1 - d["y"]) * np.log(1 - p))
    pop = pd.to_numeric(d["popularity_final"], errors="coerce")
    dims = {
        "AIの勝率": pd.cut(d["p_ai"], [0, .02, .05, .1, .15, .2, .3, .5, 1.01]),
        "確定人気": pd.cut(pop, [0, 1, 2, 3, 6, 9, 99], labels=["1番人気", "2番人気", "3番人気", "4-6番人気", "7-9番人気", "10番人気以下"]),
        "AIと市場の差": pd.cut(d["p_ai"] - d["p_mkt"], [-1, -.1, -.05, -.02, .02, .05, .1, 1]),
        "競馬場": d["course_code"], "芝ダート": d["surface"], "馬場": d["going"], "天候": d["weather"],
        "距離": pd.cut(d["distance"], [0, 1000, 1300, 1500, 1700, 2100, 9999]), "頭数": pd.cut(d["n_runners"], [0, 7, 9, 11, 13, 99]),
        "クラス": d["class_ord"].round(), "馬齢": d["age"].clip(upper=9), "性別": d["sex"],
        "走数": pd.cut(d["runs_count"], [-1, 0, 2, 5, 10, 30, 999]), "休養日数": pd.cut(d["rest_days"], [0, 14, 30, 60, 120, 9999]),
        "前走着順": pd.cut(d["last_finish"], [0, 1, 3, 5, 9, 99]), "騎手変更": d["jockey_changed"], "距離変更": pd.cut(d["dist_change"], [-9999, -201, -1, 0, 200, 9999]),
        "クラス変更": d["class_change"], "月": d["race_date"].str[5:7],
    }
    out = {}
    for name, key in dims.items():
        t = d.assign(_k=np.asarray(key, dtype=object)).dropna(subset=["_k"]).groupby("_k", observed=True)
        tab = pd.DataFrame({"n_races": t["race_id"].nunique(), "n_horses": t.size(), "pred": t["p_ai"].mean(), "actual": t["y"].mean(),
                            "market_pred": t["p_mkt"].mean(), "binary_log_loss": t["bll"].mean()})
        tab["gap"] = tab["pred"] - tab["actual"]
        out[name] = [{"key": str(k), **{c: (round(float(v), 4) if c not in ("n_races", "n_horses") else int(v)) for c, v in r.items()}}
                     for k, r in tab[tab["n_horses"] >= 200].iterrows()]
    overall = per_race(rows, "p_ai")
    return {"overall": {"n_races": int(len(overall)), "log_loss": float(overall["ll"].mean()), "brier": float(overall["brier"].mean()),
                        "top1_hit": float(overall["hit"].mean())}, "dims": out}


# ---------------------------------------------------------------- 入口

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="autoresearch.harness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("score")
    p.add_argument("--code", type=Path, required=True)
    p.add_argument("--org", choices=["JRA", "NAR"], required=True)
    p.add_argument("--stage", choices=["explore", "confirm"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("features")
    p.add_argument("--code", type=Path, required=True)
    p.add_argument("--org", choices=["JRA", "NAR"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("compare")
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--cand", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("analyze")
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = settings.load()
    if args.cmd == "score":
        res = cmd_score(args, cfg)
    elif args.cmd == "features":
        res = cmd_features(args, cfg)
    elif args.cmd == "compare":
        res = compare(read_rows(args.base), read_rows(args.cand))
        args.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        res = analyze(read_rows(args.rows))
        args.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
