"""スマホ閲覧用の静的ページを生成する。予測データは暗号化し、合言葉を知る本人だけが読める。"""
from __future__ import annotations

import base64
import gzip
import json
import os
import shutil
import sqlite3
from datetime import date, datetime, timedelta

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from . import config
from .db import JST, now_jst
from .ev import TAKEOUT, label

PBKDF2_ITER = 310_000
TOP_N = 200
RESULT_DAYS = 7                     # レースタブで結果と答え合わせを見られる過去の日数
VOID = ("scratched", "excluded")    # 取消・除外の馬を含む買い目は返還される


def encrypt_payload(payload: dict, passphrase: str) -> dict:
    salt, iv = os.urandom(16), os.urandom(12)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=PBKDF2_ITER).derive(passphrase.encode())
    raw = gzip.compress(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(), 9)
    ct = AESGCM(key).encrypt(iv, raw, None)
    b64 = lambda b: base64.b64encode(b).decode()
    return {"v": 2, "iter": PBKDF2_ITER, "salt": b64(salt), "iv": b64(iv), "ct": b64(ct)}


def _lab_summary() -> dict | None:
    f = config.ROOT / "docs" / "lab_summary.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def ev_check(rows: list[dict], pay: dict[tuple[str, str], int], void: set[int]) -> dict:
    """発走前に期待値プラスだった買い目を、1点100円ずつ買った結果。取消・除外の馬を含む買い目は返還なので数えない。"""
    out = {"odds": any(p.get("ev") is not None for p in rows), "n": 0, "ret": 0, "void": 0, "hits": []}
    for p in rows:
        if p.get("ev") is None or p["ev"] <= 0:
            continue
        if not p["b"].startswith("bracket") and void & {int(u) for u in p["c"].split("-")}:
            out["void"] += 1
            continue
        out["n"] += 1
        amount = pay.get((p["b"], p["c"]))
        if amount:
            out["ret"] += amount
            out["hits"].append({"b": p["b"], "c": p["c"], "a": amount})
    return out


def collect(con: sqlite3.Connection, today: str | None = None) -> dict:
    today = today or datetime.now(JST).date().isoformat()
    since = (date.fromisoformat(today) - timedelta(days=RESULT_DAYS)).isoformat()
    races = {}
    for r in con.execute("""SELECT r.*, c.name AS course FROM race r JOIN racecourse c USING(course_code)
                            WHERE r.race_date>=? AND r.course_code NOT LIKE 'N帯広%' ORDER BY r.race_date, r.post_time, r.race_no""", (since,)):
        races[r["race_id"]] = {"id": r["race_id"], "org": r["org"], "date": r["race_date"], "course": r["course"], "no": r["race_no"],
                               "time": r["post_time"], "name": r["name"], "grade": r["grade"], "class": r["class_name"],
                               "surface": r["surface"], "distance": r["distance"], "going": r["going"], "weather": r["weather"],
                               "n": r["n_runners"], "horses": []}
    names: dict[tuple[str, int], dict] = {}
    if races:
        marks = ",".join("?" * len(races))
        ids = list(races)
        for e in con.execute(f"""SELECT e.race_id,e.umaban,e.waku,e.age,e.sex,e.carried_weight,e.body_weight,e.body_weight_diff,
                                 h.name,j.name AS jockey FROM entry e JOIN horse h USING(horse_id)
                                 LEFT JOIN jockey j USING(jockey_id) WHERE e.race_id IN ({marks}) ORDER BY e.umaban""", ids):
            row = {"u": e["umaban"], "w": e["waku"], "name": e["name"], "jockey": e["jockey"], "sex": e["sex"], "age": e["age"],
                   "cw": e["carried_weight"], "bw": e["body_weight"], "bd": e["body_weight_diff"]}
            races[e["race_id"]]["horses"].append(row)
            names[(e["race_id"], e["umaban"])] = row
        for x in con.execute(f"SELECT race_id,umaban,finish_pos,status,popularity_final FROM result WHERE race_id IN ({marks})", ids):
            row = {"u": x["umaban"], "f": x["finish_pos"], "st": None if x["status"] == "finished" else x["status"], "pop": x["popularity_final"]}
            races[x["race_id"]].setdefault("res", []).append({k: v for k, v in row.items() if v is not None})
        for x in con.execute(f"SELECT race_id,bet_type,combo,amount FROM payout WHERE race_id IN ({marks}) ORDER BY race_id,bet_type,amount", ids):
            races[x["race_id"]].setdefault("pay", []).append({"b": x["bet_type"], "c": x["combo"], "a": x["amount"]})
    models = {r["org"]: dict(r) for r in con.execute(
        "SELECT model_version_id,org,adopted_at,created_at,train_start,train_end,metrics,comparison FROM model_version WHERE status='active'")}
    by_race_all: dict[str, list[dict]] = {}
    # 払戻が出たレースは、発走前に表示した予測を版にかかわらずそのまま答え合わせに使う。発走後に作られた予測は使わない
    posts = {rid: datetime.fromisoformat(f"{r['date']}T{r['time']}:00+09:00") for rid, r in races.items() if "pay" in r and r["time"]}
    if races:
        for p in con.execute(f"""SELECT p.*, m.status AS model_status FROM prediction_display p JOIN model_version m USING(model_version_id)
                                 WHERE p.race_id IN ({marks})""", ids):
            rid = p["race_id"]
            if "pay" in races[rid]:
                if rid in posts and datetime.fromisoformat(p["predicted_at"]) >= posts[rid]:
                    continue
            elif p["model_status"] != "active":
                continue
            meta = json.loads(p["flags"] or "{}")
            row = {"r": p["race_id"], "b": p["bet_type"], "c": p["combo"], "p": round(p["prob"], 6), "s": p["prob_std"] and round(p["prob_std"], 6),
                   "o": p["odds"], "m": p["odds_max"], "t": p["odds_captured_at"], "ev": p["ev"] if p["ev"] is None else round(p["ev"], 4),
                   "evm": p["ev_max"] if p["ev_max"] is None else round(p["ev_max"], 4), "st": p["status"], "at": p["predicted_at"],
                   "f": meta.get("flags") or None, "cf": meta.get("confidence"), "n": meta.get("avg_runs"),
                   "rp": meta.get("raw_p"), "bl": 1 if meta.get("blended") else None}
            by_race_all.setdefault(p["race_id"], []).append({k: v for k, v in row.items() if v is not None})
    preds = []
    for rid, rows in by_race_all.items():
        r = races[rid]
        if "pay" in r:
            void = {x["u"] for x in r.get("res", []) if x.get("st") in VOID}
            r["chk"] = ev_check(rows, {(x["b"], x["c"]): x["a"] for x in r["pay"]}, void)
        if r["date"] < today:
            # 過去の日は、印に使う単勝・複勝と、券種ごとに最も当たりやすい買い目だけを載せてデータ量を抑える
            top: dict[str, dict] = {}
            for x in rows:
                if x["b"] not in top or x["p"] > top[x["b"]]["p"]:
                    top[x["b"]] = x
            rows = [x for x in rows if x["b"] in ("win", "place") or top[x["b"]] is x]
        preds.extend(rows)
    # 市場との乖離は単勝について計算する: AI勝率 − オッズ逆数を正規化した市場確率
    by_race: dict[str, list[dict]] = {}
    for p in preds:
        if p["b"] == "win":
            by_race.setdefault(p["r"], []).append(p)
    for rows in by_race.values():
        if all(x.get("o") for x in rows):
            tot = sum(1 / x["o"] for x in rows)
            for rank, x in enumerate(sorted(rows, key=lambda y: y["o"]), 1):
                x["pop"] = rank
                x["mk"] = round((1 / x["o"]) / tot, 5)
                x["gap"] = round(x.get("rp", x["p"]) - x["mk"], 5)  # 合成前の AI 単独評価と市場の差
    def reports(prefix: str) -> dict:
        out = {}
        for org in ("JRA", "NAR"):
            f = config.REPORTS / f"{prefix}_{org}.json"
            if f.exists():
                out[org] = json.loads(f.read_text(encoding="utf-8"))
        return out
    return {
        "generated_at": now_jst(),
        "bet_labels": {b: label(b) for b in TAKEOUT},
        "takeout": TAKEOUT,
        "races": list(races.values()),
        "predictions": preds,
        "models": models,
        "market_only_orgs": config.guards().get("market_only_orgs") or [],
        "backtests": reports("backtest"),
        "compare": reports("compare"),
        "forward": reports("forward"),
        "lab": _lab_summary(),
        "today": today,
        "data_status": {
            "counts": [dict(r) for r in con.execute("SELECT org,COUNT(*) AS races,MIN(race_date) AS first,MAX(race_date) AS last FROM race GROUP BY org")],
            "sources": [dict(r) for r in con.execute("SELECT source,enabled,automated_fetch,license_status,last_success_at,last_error FROM source_status")],
            "errors_7d": con.execute("SELECT COUNT(*) FROM fetch_log WHERE ok=0 AND fetched_at>=date('now','-7 day')").fetchone()[0],
            "odds_quality": [dict(r) for r in con.execute("SELECT * FROM odds_capture_daily ORDER BY race_date DESC, org LIMIT 14")],
        },
    }


def build_site(con: sqlite3.Connection, sources: dict | None = None, out_dir=None, today: str | None = None) -> str:
    passphrase = os.environ.get("KEIBA_SITE_PASSPHRASE", "")
    if len(passphrase) < 8:
        return "KEIBA_SITE_PASSPHRASE (8文字以上) が未設定のためページ生成を省略しました"
    out = out_dir or config.PUBLIC
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(config.ROOT / "web" / "index.html", out / "index.html")
    (out / ".nojekyll").write_text("")
    (out / "robots.txt").write_text("User-agent: *\nDisallow: /\n")
    payload = collect(con, today)
    (out / "data.enc.json").write_text(json.dumps(encrypt_payload(payload, passphrase)), encoding="utf-8")
    return f"site: races={len(payload['races'])} predictions={len(payload['predictions'])}"
