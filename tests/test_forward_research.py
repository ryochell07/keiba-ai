"""開催日をまたぐウォークフォワード混合モデルの厳格な先読み防止テスト。"""
import math

import pytest

from keiba_ai.forward_research import _best_alpha, walk_forward_blend


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
