"""未来の結果を使わず、開催日を進めながら市場とAIの重みを評価する。

採用モデルは変更しない。復号済みの読み取り専用DBから検証対象のみを
一時的に抽出し、出力には集計値しか含めない（馬番・レースIDは出さない）。
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import secrets
import sqlite3

import numpy as np

from .forward import (
    _chronological_prestart_runs, _date_block_ci95, _fresh_prestart_odds,
    _result_integrity_reason, _score, _select, _stats, _win_odds_same_time,
)

POINTS = ("t30", "t10", "last")
ALPHAS = tuple(round(i * 0.05, 2) for i in range(21))
# プロセス内だけで共有する比較用秘密値。公開レポートに出力しない。
_COMPARISON_KEY = secrets.token_bytes(32)


def _evaluation_token(rows: list[dict]) -> str:
    """同一レース集合の照合用。IDそのものや推測可能な無鍵ハッシュを公開しない。"""
    hasher = hashlib.blake2b(key=_COMPARISON_KEY, digest_size=32)
    for race_id, race_date in sorted((str(r["race_id"]), str(r["race_date"])) for r in rows):
        payload = (race_id + "\\0" + race_date).encode("utf-8")
        hasher.update(len(payload).to_bytes(8, "big"))
        hasher.update(payload)
    return hasher.hexdigest()


def _best_alpha(history: list[dict], regularization: float = 0.002,
                brier_weight: float = 0.0) -> float:
    """過去の開催日だけでAI・市場の混合比率を推定する。

    brier_weight=0 は従来のLogLoss最適化と完全互換。
    正の値は研究用の複合損失であり、本番採用を意味しない。
    """
    if not history:
        return 0.5
    if not np.isfinite(brier_weight) or brier_weight < 0:
        raise ValueError("brier_weight must be a finite nonnegative number")
    ai = np.array([x["ai"][x["winner"]] for x in history], dtype=float)
    market = np.array([x["market"][x["winner"]] for x in history], dtype=float)
    # Brierは混合比率alphaの二次関数。係数を一度だけ計算することで
    # 21候補のたびに全頭データを走査する処理を避ける（計算式は同一）。
    brier_constant = brier_linear = brier_quadratic = 0.0
    if brier_weight:
        for rec in history:
            if set(rec["ai"]) != set(rec["market"]):
                raise ValueError("AI and market runner sets must match")
            for umaban, a in rec["ai"].items():
                m = rec["market"][umaban]
                diff = a - m
                residual = m - float(umaban == rec["winner"])
                brier_constant += residual * residual
                brier_linear += 2 * residual * diff
                brier_quadratic += diff * diff
        scale = 1.0 / len(history)
        brier_constant *= scale
        brier_linear *= scale
        brier_quadratic *= scale

    values = []
    for alpha in ALPHAS:
        p = np.clip(alpha * ai + (1 - alpha) * market, 1e-12, 1)
        loss = float(-np.log(p).mean())
        if brier_weight:
            brier = (brier_constant + alpha * brier_linear
                     + alpha * alpha * brier_quadratic)
            loss += brier_weight * brier
        values.append(loss + regularization * (alpha - 0.5) ** 2)
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
        # 校正の集計用。全出走馬の (予測確率, 1着か) と本命の予測確率。公開集計では区間ごとの平均だけを出す
        "cal": [(float(p), int(u == winner)) for u, p in probabilities.items()],
        "p_top": float(probabilities[top]),
    }


def _calibration(rows: list[dict]) -> dict:
    """予測確率の信頼性。全出走馬の確率を0.1刻みの10区間に分けた期待較正誤差 (ECE) と、本命の見込みと実際の1着率。"""
    pairs = np.array([c for r in rows for c in r["cal"]], dtype=float)
    bins = np.minimum((pairs[:, 0] * 10).astype(int), 9)
    ece = sum(abs(pairs[bins == b, 0].mean() - pairs[bins == b, 1].mean()) * int((bins == b).sum())
              for b in range(10) if (bins == b).any()) / len(pairs)
    return {"ece_10bin": float(ece), "n_horses": int(len(pairs)),
            "top1_mean_prob": float(np.mean([r["p_top"] for r in rows])),
            "top1_hit": float(np.mean([r["hit"] for r in rows]))}


def _paired_roi_difference_ci95(candidate: list[dict], deployed: list[dict],
                                draws: int = 2000) -> list[float] | None:
    """候補−採用モデルのTOP1 ROI差。開催日単位の探索的95%区間。

    全将来レースの払戻が検証済みで、同一レースに対応する場合だけ計算。
    ROIは回収額/100円ベットの倍率で表現する。
    """
    if len(candidate) != len(deployed) or not candidate or draws < 1:
        return None
    by_day: dict[str, list[float]] = defaultdict(list)
    for left, right in zip(candidate, deployed):
        if (not left.get("race_id") or left["race_id"] != right.get("race_id")
                or not left.get("race_date") or left["race_date"] != right.get("race_date")
                or left["ret"] is None or right["ret"] is None
                or not np.isfinite(left["ret"]) or not np.isfinite(right["ret"])):
            return None
        by_day[left["race_date"]].append((left["ret"] - right["ret"]) / 100.0)
    if len(by_day) < 10:
        return None
    days = sorted(by_day)
    sums = np.array([sum(by_day[d]) for d in days], dtype=float)
    counts = np.array([len(by_day[d]) for d in days], dtype=float)
    rng = np.random.default_rng(20261116)
    picks = rng.integers(0, len(days), size=(draws, len(days)))
    sampled = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return [float(np.quantile(sampled, q)) for q in (0.025, 0.975)]


def walk_forward_blend(records: list[dict], *, min_history: int = 50,
                       min_history_dates: int = 2, brier_weight: float = 0.0,
                       include_private: bool = False) -> dict:
    """同一開催日の結果を見て同日レースを調整しない、逐次外部検証。"""
    if not np.isfinite(brier_weight) or brier_weight < 0:
        raise ValueError("brier_weight must be a finite nonnegative number")
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
            alpha = _best_alpha(history, brier_weight=brier_weight)
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
    scored = {k: {**_stats(v), "calibration": _calibration(v)} for k, v in selected.items()} if n else {}
    reference = selected["deployed"]
    candidate = selected["candidate"]
    delta = (_date_block_ci95(candidate, reference) if n else None)
    roi_ci = _paired_roi_difference_ci95(candidate, reference) if n else None
    candidate_mean = scored.get("candidate", {}).get("log_loss")
    deployed_mean = scored.get("deployed", {}).get("log_loss")
    improvement = (
        (deployed_mean - candidate_mean) / deployed_mean * 100
        if candidate_mean is not None and deployed_mean is not None and deployed_mean > 0 else None
    )
    # 研究用に見つかった小幅改善を、そのまま本番採用しないための多重ゲート。
    # 連続する異なる開催日と同一レース比較の十分な規模が必要。
    candidate_stats = scored.get("candidate", {})
    deployed_stats = scored.get("deployed", {})
    ci_ll = (delta or {}).get("log_loss")
    checks = {
        "at_least_300_future_races": n >= 300,
        "at_least_15_future_race_days": evaluation_days >= 15,
        "cluster_95pct_interval_below_zero": bool(ci_ll and ci_ll[1] < 0),
        "brier_not_worse": bool(
            candidate_stats and deployed_stats
            and candidate_stats["brier"] <= deployed_stats["brier"]
        ),
        # 平均Brierだけでは開催日ごとの変動を隠す。レビュー資格は
        # 開催日単位の95%区間でも悪化が否定できる場合だけ与える。
        "brier_cluster_ci_not_worse": bool(
            delta and delta.get("brier")
            and delta["brier"][1] <= 0
        ),
        # 払戻が欠損したレースだけを除いてROIを比較すると選択バイアスが
        # 起きうるため、レビュー資格には未来評価集合の払戻完備を要求する。
        "complete_forward_payout_coverage": bool(
            n > 0
            and candidate_stats.get("top1_roi_races") == n
            and deployed_stats.get("top1_roi_races") == n
        ),
        "top1_roi_not_worse_on_verified_results": bool(
            candidate_stats.get("top1_roi_races", 0) >= 300
            and candidate_stats.get("top1_roi") is not None
            and deployed_stats.get("top1_roi") is not None
            and candidate_stats["top1_roi"] >= deployed_stats["top1_roi"]
        ),
        # 平均ROIが同等でも、開催日単位の不確実性が大きければ保留。
        # 0以上の下限が必要（差は候補ROI−現行ROI）。
        "roi_cluster_ci_not_worse": bool(roi_ci and roi_ci[0] >= 0),
    }
    summary = {
        "promotion_gate": {
            "eligible_for_review": all(checks.values()),
            "applied_to_production": False,
            "checks": checks,
        },
        "optimization_brier_weight": brier_weight,
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
        "candidate_minus_deployed_roi_ci95": roi_ci,
        "note": "学習データは必ず以前の開催日のみ。調整を行う研究候補で、本番採用や収益改善を意味しない",
    }
    # デフォルト返却値は集計値のみ。個別レースの結果は明示的な
    # 同一プロセス内研究比較に限り内部フィールドとして提供する。
    if include_private:
        summary["_comparison_token"] = _evaluation_token(selected["candidate"])
        summary["_candidate_scored_rows"] = selected["candidate"]
    return summary



def compare_shadow_summaries(logloss: dict, brier: dict) -> dict:
    """同じ未来評価母集団に限り、Brier対応候補と従来研究候補を比較する。

    集計差は探索用の記述統計。日付ブロックCI・本番採用の根拠ではない。
    """
    result = {}
    for tp in POINTS:
        left, right = logloss[tp], brier[tp]
        n = left["forward_test_races"]
        if (n != right["forward_test_races"]
                or left["forward_test_days"] != right["forward_test_days"]
                or not left.get("_comparison_token")
                or left["_comparison_token"] != right.get("_comparison_token")):
            raise ValueError("shadow candidates must share the same forward test set")
        old = left["scores_on_same_forward_test_races"].get("candidate", {})
        new = right["scores_on_same_forward_test_races"].get("candidate", {})
        if old.get("n_races", 0) != n or new.get("n_races", 0) != n:
            raise ValueError("shadow comparison has inconsistent scored race counts")
        if not n:
            result[tp] = {"n_races": 0, "n_days": 0,
                          "log_loss_delta": None, "brier_delta": None,
                          "top1_roi_delta": None,
                          "candidate_minus_legacy_roi_ci95": None,
                          "candidate_minus_legacy_ci95": None}
            continue
        # 負値ならBrier対応候補が改善。ROIは双方の払戻確認数が一致する場合のみ比較。
        old_rows = left.get("_candidate_scored_rows", [])
        new_rows = right.get("_candidate_scored_rows", [])
        if (len(old_rows) != n or len(new_rows) != n
                or [(x["race_id"], x["race_date"]) for x in old_rows]
                   != [(x["race_id"], x["race_date"]) for x in new_rows]):
            raise ValueError("shadow candidates must share the same forward test set")
        ci95 = _date_block_ci95(new_rows, old_rows)
        # 部分払戻集合のROIを比較すると選択バイアスが生じる。
        # 将来評価の全レースで払戻確認済みの場合だけ差と区間を表示する。
        roi_ok = (old.get("top1_roi_races") == n
                  and new.get("top1_roi_races") == n
                  and old.get("top1_roi") is not None
                  and new.get("top1_roi") is not None)
        roi_ci95 = (_paired_roi_difference_ci95(new_rows, old_rows)
                    if roi_ok else None)
        result[tp] = {
            "n_races": n,
            "n_days": left["forward_test_days"],
            "log_loss_delta": float(new["log_loss"] - old["log_loss"]),
            "brier_delta": float(new["brier"] - old["brier"]),
            "candidate_minus_legacy_ci95": ci95,
            "candidate_minus_legacy_roi_ci95": roi_ci95,
            "top1_roi_delta": (
                float(new["top1_roi"] - old["top1_roi"]) if roi_ok else None
            ),
            "note": "負の損失差は新候補が優位。探索的比較であり採用判定や将来収益の証明ではない",
        }
    return result


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
            "model_version_id": run["model_version_id"],
            "winner": winners[0], "payout": payout,
            "ai": {h["umaban"]: h["p_win_ai"] for h in rows},
            "market": {h["umaban"]: h["p_win_market"] for h in rows},
            "blend": {h["umaban"]: h["p_win_blend"] for h in rows},
            "used": {h["umaban"]: h["p_win_used"] for h in rows},
        })
    legacy, challenger, comparison = _shadow_block(items, min_history)
    # 再学習でAIの版が変わると、古い版の成績で選んだ重みを新しい版に当てることになる。
    # 採用中の版だけで同じ評価をやり直し、版の混在による見かけの差と区別する。
    active = con.execute(
        "SELECT model_version_id FROM model_version WHERE org=? AND status='active' "
        "ORDER BY adopted_at DESC LIMIT 1", (org,)
    ).fetchone()
    current = None
    if active is not None:
        same = {tp: [x for x in items[tp] if x["model_version_id"] == active[0]] for tp in POINTS}
        c_legacy, c_challenger, c_comparison = _shadow_block(same, min_history)
        current = {
            "by_time_point": c_legacy,
            "brier_regularized_challenger": {"brier_weight": 0.25, "by_time_point": c_challenger},
            "challenger_minus_legacy": c_comparison,
        }
    return {
        "org": org,
        "method": "AI確率と同時刻市場確率の凸結合。過去の開催日だけで0〜1の重みを0.05刻みで選び、その翌日以降を逐次検証。市場オッズが不完全な場合は評価対象外",
        "model_versions_in_valid_races": {
            tp: len({x["model_version_id"] for x in items[tp]}) for tp in POINTS
        },
        "by_time_point": legacy,
        "brier_regularized_challenger": {
            "brier_weight": 0.25,
            "by_time_point": challenger,
        },
        "challenger_minus_legacy": comparison,
        "current_model_only": current,
    }


def _shadow_block(items: dict[str, list[dict]], min_history: int) -> tuple[dict, dict, dict]:
    """従来の研究候補とBrier対応候補を同じ将来評価集合で比べ、内部の照合値を除いて返す。"""
    legacy = {tp: walk_forward_blend(items[tp], min_history=min_history,
                                     include_private=True)
              for tp in POINTS}
    challenger = {tp: walk_forward_blend(items[tp], min_history=min_history,
                                       brier_weight=0.25, include_private=True)
                  for tp in POINTS}
    comparison = compare_shadow_summaries(legacy, challenger)
    # ID照合トークンは内部のみ。暗号化レポートにも書き出さない。
    for summary in (*legacy.values(), *challenger.values()):
        summary.pop("_comparison_token", None)
        summary.pop("_candidate_scored_rows", None)
    return legacy, challenger, comparison
