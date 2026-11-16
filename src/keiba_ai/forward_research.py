"""未来の結果を使わず、開催日を進めながら市場とAIの重みを評価する。

採用モデルは変更しない。復号済みの読み取り専用DBから検証対象のみを
一時的に抽出し、出力には集計値しか含めない（馬番・レースIDは出さない）。
"""
from __future__ import annotations

from collections import defaultdict
import sqlite3

import numpy as np

from .forward import (
    _chronological_prestart_runs, _date_block_ci95, _fresh_prestart_odds,
    _result_integrity_reason, _score, _select, _stats, _win_odds_same_time,
)

POINTS = ("t30", "t10", "last")
ALPHAS = tuple(round(i * 0.05, 2) for i in range(21))


def _best_alpha(history: list[dict], regularization: float = 0.002) -> float:
    """市場pとAI pの線形混合。過去の勝ち馬の確率のみで重みを推定する。"""
    if not history:
        return 0.5
    ai = np.array([x["ai"][x["winner"]] for x in history], dtype=float)
    market = np.array([x["market"][x["winner"]] for x in history], dtype=float)
    values = []
    for alpha in ALPHAS:
        p = np.clip(alpha * ai + (1 - alpha) * market, 1e-12, 1)
        values.append(float(-np.log(p).mean()) + regularization * (alpha - 0.5) ** 2)
    return float(ALPHAS[int(np.argmin(values))])


def _score_record(record: dict, probabilities: dict[int, float]) -> dict:
    """勝ち馬は1頭かつ馬番一致済みの検証レコードのみを採点する。"""
    winner = record["winner"]
    top = max(probabilities, key=probabilities.get)
    payout = record["payout"]
    verified = (str(winner) in payout and payout[str(winner)] is not None
                and np.isfinite(payout[str(winner)]) and payout[str(winner)] > 0)
    return {
        "race_id": record["race_id"], "race_date": record["race_date"],
        "ll": float(-np.log(max(probabilities[winner], 1e-12))),
        "brier": float(sum((p - (u == winner)) ** 2 for u, p in probabilities.items())),
        "hit": int(top == winner),
        "ret": float(payout.get(str(top), 0)) if verified else None,
        "ev_n": 0, "ev_ret": 0,
    }


def walk_forward_blend(records: list[dict], *, min_history: int = 50,
                       min_history_dates: int = 2) -> dict:
    """同一開催日の結果を見て同日レースを調整しない、逐次外部検証。"""
    days: dict[str, list[dict]] = defaultdict(list)
    race_ids = set()
    for record in records:
        if not record.get("race_date") or not record.get("race_id"):
            raise ValueError("race date and race ID are required")
        if record["race_id"] in race_ids:
            raise ValueError("one forward snapshot per race and timepoint is required")
        race_ids.add(record["race_id"])
        days[record["race_date"]].append(record)
    history: list[dict] = []
    selected: dict[str, list[dict]] = {m: [] for m in ("candidate", "ai", "market", "blend", "deployed")}
    selected_alphas: list[float] = []
    evaluation_days = 0

    for day in sorted(days):
        batch = days[day]
        # 当日の勝敗を評価してから重みを選択することは絶対にしない。
        if len(history) >= min_history and len({r["race_date"] for r in history}) >= min_history_dates:
            alpha = _best_alpha(history)
            evaluation_days += 1
            selected_alphas.extend([alpha] * len(batch))
            for rec in batch:
                probabilities = {
                    "candidate": {u: alpha * rec["ai"][u] + (1 - alpha) * rec["market"][u]
                                  for u in rec["ai"]},
                    "ai": rec["ai"], "market": rec["market"],
                    "blend": rec["blend"], "deployed": rec["used"],
                }
                for model, probs in probabilities.items():
                    selected[model].append(_score_record(rec, probs))
        history.extend(batch)

    n = len(selected["candidate"])
    scored = {k: _stats(v) for k, v in selected.items()} if n else {}
    reference = selected["deployed"]
    candidate = selected["candidate"]
    delta = (_date_block_ci95(candidate, reference) if n else None)
    candidate_mean = scored.get("candidate", {}).get("log_loss")
    deployed_mean = scored.get("deployed", {}).get("log_loss")
    improvement = (
        (deployed_mean - candidate_mean) / deployed_mean * 100
        if candidate_mean is not None and deployed_mean is not None and deployed_mean > 0 else None
    )
    return {
        "valid_races": len(records), "valid_days": len(days),
        "minimum_training_races": min_history, "minimum_training_days": min_history_dates,
        "forward_test_races": n, "forward_test_days": evaluation_days,
        "mean_ai_weight": float(np.mean(selected_alphas)) if selected_alphas else None,
        "min_ai_weight": min(selected_alphas) if selected_alphas else None,
        "max_ai_weight": max(selected_alphas) if selected_alphas else None,
        "scores_on_same_forward_test_races": scored,
        "relative_log_loss_improvement_vs_deployed_pct": float(improvement) if improvement is not None else None,
        "candidate_minus_deployed_log_loss": (
            float(candidate_mean - deployed_mean)
            if candidate_mean is not None and deployed_mean is not None else None
        ),
        "candidate_minus_deployed_ci95": delta,
        "note": "学習データは必ず以前の開催日のみ。調整を行う研究候補で、本番採用や収益改善を意味しない",
    }


def _probabilities_valid(rows: list[sqlite3.Row], col: str) -> bool:
    ps = [h[col] for h in rows]
    return (bool(ps) and all(v is not None and np.isfinite(v) and 0 <= v <= 1 for v in ps)
            and abs(sum(ps) - 1) <= 1e-4)


def evaluate_shadow(con: sqlite3.Connection, org: str, *, min_history: int = 50) -> dict:
    """厳格な同一時刻・全頭オッズがある保存済み発走前予測だけで影の候補を評価。"""
    candidates = con.execute(
        """SELECT p.*, r.race_date FROM prediction_run p JOIN race r USING (race_id)
           WHERE p.org=? AND EXISTS (SELECT 1 FROM result x WHERE x.race_id=p.race_id)
           ORDER BY p.race_id,p.predicted_at,p.rowid""", (org,)
    ).fetchall()
    selected = _select(_chronological_prestart_runs(candidates))
    items: dict[str, list[dict]] = {tp: [] for tp in POINTS}
    columns = ("p_win_ai", "p_win_market", "p_win_blend", "p_win_used")
    for (race_id, tp), run in selected.items():
        if tp not in items or run["odds_status"] != "complete" or not run["odds_same_time"]:
            continue
        rows = con.execute(
            "SELECT * FROM prediction_horse WHERE prediction_id=? ORDER BY umaban",
            (run["prediction_id"],)
        ).fetchall()
        if (_result_integrity_reason(con, race_id, rows)
                or not _fresh_prestart_odds(run)
                or not _win_odds_same_time(con, race_id, rows, run["odds_captured_at"])
                or not all(_probabilities_valid(rows, col) for col in columns)):
            continue
        fin = dict(con.execute("SELECT umaban, finish_pos FROM result WHERE race_id=?", (race_id,)))
        winners = [u for u, pos in fin.items() if pos == 1]
        if len(winners) != 1:
            continue
        payout = {str(r[0]): r[1] for r in con.execute(
            "SELECT combo, amount FROM payout WHERE race_id=? AND bet_type='win'", (race_id,)
        )}
        items[tp].append({
            "race_id": race_id, "race_date": run["race_date"],
            "winner": winners[0], "payout": payout,
            "ai": {h["umaban"]: h["p_win_ai"] for h in rows},
            "market": {h["umaban"]: h["p_win_market"] for h in rows},
            "blend": {h["umaban"]: h["p_win_blend"] for h in rows},
            "used": {h["umaban"]: h["p_win_used"] for h in rows},
        })
    return {
        "org": org,
        "method": "AI確率と同時刻市場確率の凸結合。過去の開催日だけで0〜1の重みを0.05刻みで選び、その翌日以降を逐次検証。市場オッズが不完全な場合は評価対象外",
        "by_time_point": {
            tp: walk_forward_blend(items[tp], min_history=min_history)
            for tp in POINTS
        },
    }
