"""公開してよい集計値だけを出す要約のテスト。レポート内の文字列やIDは決して出さない。"""
import pytest

from keiba_ai.eval_summary import summarize
from keiba_ai.forward_research import walk_forward_blend

SECRET = "J202611160511"


def _record(day, race_no):
    return {"race_id": f"{SECRET}-{day:02d}-{race_no:02d}", "race_date": f"2030-04-{day:02d}",
            "winner": 1, "ai": {1: .9, 2: .1}, "market": {1: .2, 2: .8},
            "blend": {1: .5, 2: .5}, "used": {1: .5, 2: .5}, "payout": {"1": 250}}


def _report(org="NAR"):
    rows = [_record(d, r) for d in range(1, 6) for r in range(25)]
    block = {tp: walk_forward_blend(rows, min_history=40) for tp in ("t30", "t10", "last")}
    brier = {tp: walk_forward_blend(rows, min_history=40, brier_weight=.25) for tp in ("t30", "t10", "last")}
    stats = {"n_races": 75, "log_loss": 1.5, "brier": .7, "top1_hit": .3}
    return {
        "org": org,
        "method": f"説明文 {SECRET}",
        "by_time_point": {"t30": {"ai": stats, "market": stats, "blend": stats,
                                  "used_all_races": {**stats, "note": SECRET}}},
        "paired_model_comparison": {"by_time_point": {"t30": {"n_race_dates": 5}}},
        "shadow_research": {
            "method": SECRET,
            "by_time_point": block,
            "brier_regularized_challenger": {"brier_weight": .25, "by_time_point": brier},
            "current_model_only": {"by_time_point": block,
                                   "brier_regularized_challenger": {"by_time_point": brier}},
            "model_versions_in_valid_races": {"t30": 1, "t10": 1, "last": 1},
        },
    }


def test_summary_shows_numbers_but_no_strings_from_the_report():
    out = summarize([_report("JRA"), _report("NAR")])
    assert SECRET not in out and "説明文" not in out
    assert "### JRA" in out and "### NAR" in out
    assert "| 30分前 | 75 | 3 |" in out           # 検証レース・検証日
    assert "1.5000" in out                       # 保存済み予測の LL
    assert "ECE" in out and "審査資格" in out


def test_summary_hides_non_numeric_values():
    report = _report()
    report["by_time_point"]["t30"]["ai"]["log_loss"] = SECRET
    report["shadow_research"]["model_versions_in_valid_races"]["t30"] = {"id": SECRET}
    out = summarize(report)
    assert SECRET not in out
    assert "?" in out


def test_summary_rejects_unknown_org():
    with pytest.raises(ValueError):
        summarize({"org": SECRET})
