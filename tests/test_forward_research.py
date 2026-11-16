"""開催日をまたぐウォークフォワード混合モデルの厳格な先読み防止テスト。"""
import math

import pytest

from keiba_ai.forward_research import _best_alpha, walk_forward_blend, compare_shadow_summaries


def _record(day, race_no, *, ai_winner=0.9, market_winner=0.2):
    probs_ai = {1: ai_winner, 2: 1 - ai_winner}
    probs_market = {1: market_winner, 2: 1 - market_winner}
    return {
        "race_id": f"race-{day:02d}-{race_no:02d}",
        "race_date": f"2030-04-{day:02d}",
        "winner": 1,
        "ai": probs_ai,
        "market": probs_market,
        "blend": {1: 0.5, 2: 0.5},
        "used": {1: 0.5, 2: 0.5},
        "payout": {"1": 250},
    }


def test_walk_forward_only_scores_after_minimum_previous_dates():
    rows = [_record(d, r) for d in range(1, 6) for r in range(25)]
    out = walk_forward_blend(rows, min_history=40, min_history_dates=2)
    assert out["valid_races"] == 125
    assert out["valid_days"] == 5
    assert out["forward_test_races"] == 75
    assert out["forward_test_days"] == 3
    assert out["min_ai_weight"] >= 0.95
    assert out["max_ai_weight"] <= 1
    metrics = out["scores_on_same_forward_test_races"]
    assert all(s["n_races"] == 75 for s in metrics.values())
    assert metrics["candidate"]["log_loss"] < metrics["deployed"]["log_loss"]
    assert out["relative_log_loss_improvement_vs_deployed_pct"] > 0
    assert out["candidate_minus_deployed_ci95"] is None  # 3日ではCIを出さない
    assert out["promotion_gate"]["eligible_for_review"] is False
    assert out["promotion_gate"]["applied_to_production"] is False
    assert out["promotion_gate"]["checks"]["at_least_15_future_race_days"] is False


def test_current_day_results_never_leak_into_weight_tuning():
    training = [_record(d, r) for d in (1, 2) for r in range(25)]
    # 直後の開催日では市場が正確・AIが不正確。しかし当日の答えを重み決定に使用できない。
    opposite = [_record(3, r, ai_winner=0.1, market_winner=0.9) for r in range(25)]
    out = walk_forward_blend(training + opposite, min_history=50, min_history_dates=2)
    assert out["forward_test_races"] == 25
    assert out["mean_ai_weight"] == pytest.approx(1.0)
    assert out["scores_on_same_forward_test_races"]["candidate"]["log_loss"] > 2
    assert out["scores_on_same_forward_test_races"]["market"]["log_loss"] < 0.2
    assert out["relative_log_loss_improvement_vs_deployed_pct"] < 0


def test_walk_forward_rejects_duplicate_ids_and_short_history():
    a = _record(1, 1)
    with pytest.raises(ValueError, match="one forward snapshot"):
        walk_forward_blend([a, a])
    out = walk_forward_blend([a], min_history=10)
    assert out["forward_test_races"] == 0
    assert out["mean_ai_weight"] is None
    assert out["scores_on_same_forward_test_races"] == {}


def test_best_alpha_uses_prior_labels_not_future_dates():
    history = [_record(1, i) for i in range(10)]
    assert _best_alpha(history) == pytest.approx(1.0)
    opposite = [_record(2, i, ai_winner=0.1, market_winner=0.9) for i in range(10)]
    assert _best_alpha(opposite) == pytest.approx(0.0)


def test_promotion_requires_day_cluster_brier_noninferiority():
    # 18日×25レース。最初の2日を訓練に使い、残り16日を未来評価する。
    # 十分な独立日数があり、候補がLL/Brierとも全日で優位な場合は
    # 採用ではなく「レビュー対象」にのみ到達する。
    rows = [_record(d, r) for d in range(1, 19) for r in range(25)]
    out = walk_forward_blend(rows, min_history=50, min_history_dates=2)
    assert out["forward_test_races"] == 400
    assert out["forward_test_days"] == 16
    assert out["promotion_gate"]["checks"]["brier_cluster_ci_not_worse"] is True
    assert out["promotion_gate"]["eligible_for_review"] is True
    assert out["promotion_gate"]["applied_to_production"] is False
    assert out["candidate_minus_deployed_ci95"]["brier"][1] < 0


def test_promotion_rejects_insufficient_brier_cluster_evidence():
    rows = [_record(d, r) for d in range(1, 11) for r in range(40)]
    out = walk_forward_blend(rows, min_history=80, min_history_dates=2)
    # 件数は足りても、独立した将来開催日が8日しかなくCIは算出しない。
    assert out["forward_test_races"] == 320
    assert out["candidate_minus_deployed_ci95"] is None
    assert out["promotion_gate"]["checks"]["brier_cluster_ci_not_worse"] is False
    assert out["promotion_gate"]["eligible_for_review"] is False


def test_brier_aware_shadow_keeps_training_strictly_before_test_day():
    training = [_record(d, r) for d in (1, 2) for r in range(25)]
    test_day = [_record(3, r, ai_winner=0.1, market_winner=0.9)
                for r in range(25)]
    old = walk_forward_blend(training, min_history=50)
    new = walk_forward_blend(training + test_day, min_history=50,
                             brier_weight=0.25)
    assert old["forward_test_races"] == 0
    assert new["forward_test_races"] == 25
    assert new["optimization_brier_weight"] == 0.25
    assert new["mean_ai_weight"] == pytest.approx(1.0)
    assert new["scores_on_same_forward_test_races"]["candidate"]["log_loss"] > 2


def test_brier_aware_shadow_rejects_invalid_objective_weight():
    rows = [_record(1, 1)]
    for invalid in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="brier_weight"):
            walk_forward_blend(rows, brier_weight=invalid)
        with pytest.raises(ValueError, match="brier_weight"):
            _best_alpha(rows, brier_weight=invalid)


def test_challenger_comparison_reports_same_sample_deltas():
    rows = [_record(d, r) for d in range(1, 6) for r in range(25)]
    legacy = walk_forward_blend(rows, min_history=50)
    challenger = walk_forward_blend(rows, min_history=50, brier_weight=0.25)
    summary = compare_shadow_summaries(
        {"t30": legacy, "t10": legacy, "last": legacy},
        {"t30": challenger, "t10": challenger, "last": challenger},
    )
    assert summary["t30"]["n_races"] == 75
    assert summary["t30"]["n_days"] == 3
    assert summary["t30"]["log_loss_delta"] == pytest.approx(
        challenger["scores_on_same_forward_test_races"]["candidate"]["log_loss"]
        - legacy["scores_on_same_forward_test_races"]["candidate"]["log_loss"]
    )
    assert summary["t30"]["brier_delta"] == pytest.approx(
        challenger["scores_on_same_forward_test_races"]["candidate"]["brier"]
        - legacy["scores_on_same_forward_test_races"]["candidate"]["brier"]
    )


def test_challenger_comparison_rejects_different_evaluation_sizes():
    rows = [_record(d, r) for d in range(1, 6) for r in range(25)]
    legacy = walk_forward_blend(rows, min_history=50)
    mismatch = walk_forward_blend(rows[:-1], min_history=50, brier_weight=0.25)
    with pytest.raises(ValueError, match="same forward test set"):
        compare_shadow_summaries(
            {"t30": legacy, "t10": legacy, "last": legacy},
            {"t30": mismatch, "t10": mismatch, "last": mismatch},
        )


def test_challenger_comparison_rejects_different_race_ids_at_same_size():
    # レース数・開催日数が同じでも、異なるレースは比較しない。
    original = [_record(d, r) for d in range(1, 6) for r in range(25)]
    changed = [_record(d, r) for d in range(1, 6) for r in range(25)]
    changed[-1] = {**changed[-1], "race_id": "different-race-at-same-date"}
    legacy = walk_forward_blend(original, min_history=50)
    challenger = walk_forward_blend(changed, min_history=50, brier_weight=0.25)
    assert legacy["forward_test_races"] == challenger["forward_test_races"]
    assert legacy["forward_test_days"] == challenger["forward_test_days"]
    with pytest.raises(ValueError, match="same forward test set"):
        compare_shadow_summaries(
            {"t30": legacy, "t10": legacy, "last": legacy},
            {"t30": challenger, "t10": challenger, "last": challenger},
        )


def test_private_comparison_token_is_not_plain_race_id():
    rows = [_record(d, r) for d in range(1, 4) for r in range(25)]
    result = walk_forward_blend(rows, min_history=50)
    token = result["_comparison_token"]
    assert len(token) == 64
    assert all(r["race_id"] not in token for r in rows)


def test_challenger_difference_has_paired_cluster_ci_for_sufficient_days():
    rows = [_record(d, r) for d in range(1, 19) for r in range(25)]
    old = walk_forward_blend(rows, min_history=50)
    new = walk_forward_blend(rows, min_history=50, brier_weight=0.25)
    comparison = compare_shadow_summaries(
        {tp: old for tp in ("t30", "t10", "last")},
        {tp: new for tp in ("t30", "t10", "last")},
    )
    ci = comparison["t30"]["candidate_minus_legacy_ci95"]
    assert ci is not None
    assert len(ci["log_loss"]) == 2
    assert len(ci["brier"]) == 2
    assert ci["log_loss"][0] <= comparison["t30"]["log_loss_delta"] <= ci["log_loss"][1]


def test_challenger_difference_rejects_missing_private_paired_rows():
    rows = [_record(d, r) for d in range(1, 5) for r in range(25)]
    old = walk_forward_blend(rows, min_history=50)
    new = walk_forward_blend(rows, min_history=50, brier_weight=0.25)
    new.pop("_candidate_scored_rows")
    with pytest.raises(ValueError, match="same forward test set"):
        compare_shadow_summaries(
            {tp: old for tp in ("t30", "t10", "last")},
            {tp: new for tp in ("t30", "t10", "last")},
        )


def test_brier_quadratic_optimizer_matches_bruteforce_objective():
    import numpy as np

    history = []
    for index in range(30):
        ai = {1: .1 + (index % 5) * .04, 2: .3, 3: .6 - (index % 5) * .04}
        market = {1: .5, 2: .2 + (index % 3) * .05,
                  3: .3 - (index % 3) * .05}
        history.append({"ai": ai, "market": market,
                        "winner": (index % 3) + 1})
    for weight in (0.0, 0.25, 1.0, 3.0):
        scores = []
        for alpha in (i * .05 for i in range(21)):
            ll, briers = [], []
            for rec in history:
                combined = {
                    u: alpha * rec["ai"][u] + (1 - alpha) * rec["market"][u]
                    for u in rec["ai"]
                }
                ll.append(-math.log(max(combined[rec["winner"]], 1e-12)))
                briers.append(sum(
                    (prob - (u == rec["winner"])) ** 2
                    for u, prob in combined.items()
                ))
            scores.append(np.mean(ll) + weight * np.mean(briers)
                          + .002 * (alpha - .5) ** 2)
        expected = round(.05 * int(np.argmin(scores)), 2)
        assert _best_alpha(history, brier_weight=weight) == pytest.approx(expected)


def test_brier_optimizer_rejects_mismatching_runner_sets():
    record = _record(1, 1)
    record["market"] = {1: .2, 3: .8}
    with pytest.raises(ValueError, match="runner sets"):
        _best_alpha([record], brier_weight=.25)


def test_promotion_requires_complete_payout_coverage():
    # 未来側400レースあり予測は極めて良好。それでも勝ち馬の
    # 払戻が1レース欠けたらROIによる採用判定は通さない。
    rows = [_record(d, r) for d in range(1, 19) for r in range(25)]
    rows[-1] = {**rows[-1], "payout": {}}
    out = walk_forward_blend(rows, min_history=50, min_history_dates=2)
    assert out["forward_test_races"] == 400
    assert out["scores_on_same_forward_test_races"]["candidate"]["top1_roi_races"] == 399
    assert out["promotion_gate"]["checks"]["complete_forward_payout_coverage"] is False
    assert out["promotion_gate"]["eligible_for_review"] is False


def test_promotion_full_payout_coverage_on_consistent_reference():
    rows = [_record(d, r) for d in range(1, 19) for r in range(25)]
    out = walk_forward_blend(rows, min_history=50, min_history_dates=2)
    assert out["promotion_gate"]["checks"]["complete_forward_payout_coverage"] is True
