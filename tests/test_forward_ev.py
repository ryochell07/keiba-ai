"""発走前評価で欠損オッズを購入可能な期待値と誤認しないための回帰テスト。"""
from keiba_ai.forward import _score


def test_ai_all_races_does_not_evaluate_ev_without_complete_odds():
    horses = [
        {"umaban": 1, "p_win_ai": 0.7, "win_odds": 2.0},
        {"umaban": 2, "p_win_ai": 0.3, "win_odds": None},
    ]
    result = _score(horses, "p_win_ai", {1: 1, 2: 2}, {"1": 200}, evaluate_ev=False)
    assert result["hit"] == 1
    assert result["ev_n"] == 0
    assert result["ev_ret"] == 0


def test_ev_still_available_for_eligible_same_time_odds():
    horses = [
        {"umaban": 1, "p_win_ai": 0.7, "win_odds": 2.0},
        {"umaban": 2, "p_win_ai": 0.3, "win_odds": 4.0},
    ]
    result = _score(horses, "p_win_ai", {1: 1, 2: 2}, {"1": 200})
    assert result["ev_n"] == 2
    assert result["ev_ret"] == 200
