"""発走前に保存した予測と確定結果を照合する。これが一番信頼できる実運用の成績。

時点ごとに AI・市場・AI+市場を、同じ時刻に取得したオッズだけで比べる。
確定オッズとの比較は別枠で、購入可能な予測としては扱わない。
"""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict

import numpy as np

from . import config
from .backtest import Ledger
from .db import now_jst
from .ledger import TIME_POINTS
from .probs import all_bet_probs

MODELS = {"ai": "p_win_ai", "market": "p_win_market", "blend": "p_win_blend"}
SELECTION_RULE = "同じ時点に複数あれば morning と day_before は最初の予測、t30・t10・last は発走に最も近い予測を使う"


def _select(runs: list[sqlite3.Row]) -> dict[tuple[str, str], sqlite3.Row]:
    out: dict[tuple[str, str], sqlite3.Row] = {}
    for r in runs:  # predicted_at の昇順
        key = (r["race_id"], r["time_point"])
        if key not in out or r["time_point"] not in ("day_before", "morning"):
            out[key] = r
    return out


def _stats(rows: list[dict]) -> dict:
    """rows はレースごとの ll/brier/hit/ret/ev_ret/ev_n。"""
    if not rows:
        return {"n_races": 0}
    ll = np.array([x["ll"] for x in rows])
    n_ev = sum(x["ev_n"] for x in rows)
    return {"n_races": len(rows), "log_loss": float(ll.mean()), "brier": float(np.mean([x["brier"] for x in rows])), "top1_hit": float(np.mean([x["hit"] for x in rows])),
            "top1_roi": float(sum(x["ret"] for x in rows) / (100 * len(rows))), "ev100_bets": n_ev, "ev100_roi": float(sum(x["ev_ret"] for x in rows) / (100 * n_ev)) if n_ev else None}


def _score(horses: list[sqlite3.Row], col: str, fin: dict[int, int | None], pay: dict[str, int], odds_col: str = "win_odds", odds: dict[int, float] | None = None) -> dict | None:
    ps = {h["umaban"]: h[col] for h in horses}
    if any(v is None for v in ps.values()):
        return None
    winners = [u for u in ps if fin.get(u) == 1]
    if not winners:
        return None
    top = max(ps, key=ps.get)
    get_o = (lambda h: odds.get(h["umaban"])) if odds is not None else (lambda h: h[odds_col])
    ev = [h["umaban"] for h in horses if get_o(h) and ps[h["umaban"]] * get_o(h) >= 1.0]
    return {"ll": -np.log(max(ps[winners[0]], 1e-9)), "brier": sum((p - float(fin.get(u) == 1)) ** 2 for u, p in ps.items()), "hit": int(top in winners), "ret": pay.get(str(top), 0),
            "ev_n": len(ev), "ev_ret": sum(pay.get(str(u), 0) for u in ev)}


def reconcile(con: sqlite3.Connection, org: str) -> dict:
    """台帳の発走前の予測だけを集計する。結果は別テーブルから紐付け、台帳は変更しない。"""
    runs = con.execute(
        """SELECT p.* FROM prediction_run p
           WHERE p.org=? AND (p.scheduled_post_at IS NULL OR p.predicted_at<p.scheduled_post_at)
             AND EXISTS (SELECT 1 FROM result x WHERE x.race_id=p.race_id AND x.finish_pos=1)
           ORDER BY p.race_id, p.predicted_at, p.rowid""", (org,)).fetchall()
    payouts: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    if runs:
        for r in con.execute("SELECT p.race_id,p.bet_type,p.combo,p.amount FROM payout p JOIN race r USING(race_id) WHERE r.org=?", (org,)):
            payouts[(r[0], r[1])][r[2]] = r[3]
    finals: dict[str, dict[int, float]] = defaultdict(dict)
    for r in con.execute("SELECT o.race_id,CAST(o.combo AS INTEGER),o.odds FROM odds_snapshot o JOIN race r USING(race_id) WHERE r.org=? AND o.is_final=1 AND o.bet_type='win'", (org,)):
        finals[r[0]][r[1]] = r[2]
    selected = _select(runs)
    tp_rows: dict[str, dict[str, list]] = {tp: defaultdict(list) for tp in TIME_POINTS}
    final_rows: dict[str, dict[str, list]] = {tp: defaultdict(list) for tp in TIME_POINTS}
    for (rid, tp), run in selected.items():
        if tp not in tp_rows:
            continue
        horses = con.execute("SELECT * FROM prediction_horse WHERE prediction_id=?", (run["prediction_id"],)).fetchall()
        fin = {r[0]: r[1] for r in con.execute("SELECT umaban, finish_pos FROM result WHERE race_id=?", (rid,))}
        pay = payouts.get((rid, "win"), {})
        ai_all = _score(horses, "p_win_ai", fin, pay)
        if ai_all:
            tp_rows[tp]["ai_all_races"].append(ai_all)
        if run["odds_status"] == "complete" and run["odds_same_time"]:
            scores = {m: _score(horses, c, fin, pay) for m, c in MODELS.items()}
            if all(scores.values()):
                for m, s in scores.items():
                    tp_rows[tp][m].append(s)
        # 確定オッズは別枠。全頭の確定オッズがそろうレースだけ
        fo = finals.get(rid, {})
        um = [h["umaban"] for h in horses]
        if ai_all and fo and all(u in fo for u in um):
            tot = sum(1 / fo[u] for u in um)
            mk = {u: (1 / fo[u]) / tot for u in um}
            winners = [u for u in um if fin.get(u) == 1]
            if winners:
                top = max(mk, key=mk.get)
                final_rows[tp]["market_final"].append({"ll": -np.log(max(mk[winners[0]], 1e-9)), "brier": sum((p - float(fin.get(u) == 1)) ** 2 for u, p in mk.items()), "hit": int(top in winners),
                                                       "ret": pay.get(str(top), 0), "ev_n": 0, "ev_ret": 0})
                final_rows[tp]["ai_same_races"].append(ai_all)
    # 券種別の互換集計: レースごとに発走に最も近い予測
    ledger = Ledger()
    groups: set[str] = set()
    last: dict[str, sqlite3.Row] = {}
    for r in runs:
        last[r["race_id"]] = r
    order = con.execute("SELECT race_id FROM race WHERE org=? ORDER BY race_date, post_time, race_id", (org,)).fetchall()
    for row in order:
        run = last.get(row[0])
        if run is None:
            continue
        rid = run["race_id"]
        groups.add(rid)
        horses = con.execute("SELECT * FROM prediction_horse WHERE prediction_id=? ORDER BY umaban", (run["prediction_id"],)).fetchall()
        um, wk = [h["umaban"] for h in horses], [h["waku"] for h in horses]
        probs = all_bet_probs(um, wk, np.array([h["strength_used"] for h in horses]), tuple(json.loads(run["taus_used"])), org)
        for bet, combos in probs.items():
            pay = payouts.get((rid, bet))
            if not pay:
                continue
            top = max(combos, key=combos.get)
            ledger.bet(("top1", bet, "none"), rid, pay.get(top, 0))
        for bet, col in (("win", "win_odds"), ("place", "place_odds")):
            pay = payouts.get((rid, bet))
            if not pay or bet not in probs:
                continue
            for h in horses:
                o, p = h[col], probs[bet].get(str(h["umaban"]))
                if o and p is not None:
                    if p * o >= 1.0:
                        ledger.bet(("ev_1.0", bet, "cutoff"), rid, pay.get(str(h["umaban"]), 0))
                    if p * o >= 1.2:
                        ledger.bet(("ev_1.2", bet, "cutoff"), rid, pay.get(str(h["umaban"]), 0))
    out_rows = []
    bid = f"forward-{org}"
    created = now_jst()
    con.execute("DELETE FROM backtest_result WHERE backtest_id=?", (bid,))
    for (strategy, bet, basis), m in ledger.rows():
        con.execute("""INSERT INTO backtest_result(backtest_id,created_at,org,strategy,bet_type,n_races,n_bets,n_hits,stake,payout,roi,roi_ci_low,roi_ci_high,odds_basis,detail)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (bid, created, org, f"{strategy}|{basis}", bet, m["n_races"], m["n_bets"], m["n_hits"], m["stake"], m["payout"],
                     m["roi"], m["roi_ci_low"], m["roi_ci_high"], basis, json.dumps({"max_drawdown": m["max_drawdown"]})))
        out_rows.append({"strategy": strategy, "bet_type": bet, "odds_basis": basis, **m})
    con.commit()
    report = {"org": org, "n_races": len(groups), "rows": out_rows, "updated_at": created, "selection_rule": SELECTION_RULE,
              "by_time_point": {tp: {k: _stats(v) for k, v in d.items()} for tp, d in tp_rows.items() if d},
              "final_odds_reference": {"purchasable": False, "note": "確定オッズは発走後に確定する値で、購入時点の比較ではない",
                                       "by_time_point": {tp: {k: _stats(v) for k, v in d.items()} for tp, d in final_rows.items() if d}}}
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / f"forward_{org}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report
