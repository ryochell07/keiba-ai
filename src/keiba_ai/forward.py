"""発走前に保存した予測と確定結果を照合する。これが一番信頼できる実運用の成績。"""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict

from . import config
from .backtest import Ledger
from .db import now_jst


def reconcile(con: sqlite3.Connection, org: str) -> dict:
    """発走前の予測・発走前のオッズだけを使って回収率を集計する。"""
    rows = con.execute(
        """SELECT p.race_id, p.bet_type, p.combo, p.prob, p.ev, p.odds_captured_at, r.race_date, r.post_time, p.predicted_at
           FROM prediction p JOIN race r USING(race_id)
           WHERE r.org=? AND EXISTS (SELECT 1 FROM result x WHERE x.race_id=p.race_id AND x.finish_pos=1)
           ORDER BY r.race_date, r.post_time, p.race_id""", (org,)).fetchall()
    payouts: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    if rows:
        for r in con.execute("SELECT p.race_id,p.bet_type,p.combo,p.amount FROM payout p JOIN race r USING(race_id) WHERE r.org=?", (org,)):
            payouts[(r[0], r[1])][r[2]] = r[3]
    ledger = Ledger()
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for r in rows:
        if r["post_time"] and r["predicted_at"][:16] >= f"{r['race_date']}T{r['post_time']}":
            continue  # 発走後に作られた予測は集計しない
        groups[(r["race_id"], r["bet_type"])].append(r)
    for (rid, bet), combos in groups.items():
        pay = payouts.get((rid, bet))
        if not pay:
            continue
        top = max(combos, key=lambda x: x["prob"])
        ledger.bet(("top1", bet, "none"), rid, pay.get(top["combo"], 0))
        for c in combos:
            if c["ev"] is None or c["odds_captured_at"] in (None, "final"):
                continue
            if c["ev"] >= 0:
                ledger.bet(("ev_1.0", bet, "cutoff"), rid, pay.get(c["combo"], 0))
            if c["ev"] >= 0.2:
                ledger.bet(("ev_1.2", bet, "cutoff"), rid, pay.get(c["combo"], 0))
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
    report = {"org": org, "n_races": len({k[0] for k in groups}), "rows": out_rows, "updated_at": created}
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / f"forward_{org}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report
