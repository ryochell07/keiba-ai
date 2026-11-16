"""評価レポートから、公開してよい集計値だけを Markdown の表にする。

鍵を持たない利用者が結果を読めるよう、手動評価で明示的に指定したときだけ Actions の画面に出す。
決めた項目の数値 (件数・平均・割合・区間) だけを取り出し、レポート内の文字列は一切出さない。
レースID・馬番・時刻・オッズ・モデルの版は表に含まれない。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ORGS = ("JRA", "NAR")
POINTS = (("t30", "30分前"), ("t10", "10分前"), ("last", "直前"))


def _num(value, kind: str = "f4") -> str:
    """数値と真偽値だけを整形する。それ以外 (文字列・辞書など) は内容を出さずに '?' にする。"""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "○" if value else "×"
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "?"
    if kind == "int":
        return str(int(value))
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "pp":
        return f"{value:+.3f}%"
    return f"{value:.4f}"


def _get(d, *path):
    for key in path:
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d


def _ci(pair) -> str:
    if not isinstance(pair, list) or len(pair) != 2:
        return "-"
    return f"[{_num(pair[0])}, {_num(pair[1])}]"


def _improve(base, new):
    """現行より何%良いか (損失が下がれば正)。"""
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in (base, new)) or base <= 0:
        return None
    return (base - new) / base * 100


def _research_tables(block: dict) -> list[str]:
    legacy = _get(block, "by_time_point") or {}
    brier = _get(block, "brier_regularized_challenger", "by_time_point") or {}
    out = ["| 時点 | 検証レース | 検証日 | 現行LL | LL候補 | 改善率 | Brier候補 | 改善率 | 現行Brier | LL候補Brier | 改善率 | Brier候補Brier | 改善率 | LL候補−現行 LL差95%区間 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for tp, label in POINTS:
        s_old = _get(legacy, tp, "scores_on_same_forward_test_races") or {}
        s_new = _get(brier, tp, "scores_on_same_forward_test_races") or {}
        dep, cand, chal = s_old.get("deployed") or {}, s_old.get("candidate") or {}, s_new.get("candidate") or {}
        out.append("| " + " | ".join([
            label, _num(_get(legacy, tp, "forward_test_races"), "int"), _num(_get(legacy, tp, "forward_test_days"), "int"),
            _num(dep.get("log_loss")), _num(cand.get("log_loss")), _num(_improve(dep.get("log_loss"), cand.get("log_loss")), "pp"),
            _num(chal.get("log_loss")), _num(_improve(dep.get("log_loss"), chal.get("log_loss")), "pp"),
            _num(dep.get("brier")), _num(cand.get("brier")), _num(_improve(dep.get("brier"), cand.get("brier")), "pp"),
            _num(chal.get("brier")), _num(_improve(dep.get("brier"), chal.get("brier")), "pp"),
            _ci(_get(legacy, tp, "candidate_minus_deployed_ci95", "log_loss")),
        ]) + " |")
    out += ["", "| 時点 | 現行的中 | LL候補的中 | Brier候補的中 | 現行ROI | LL候補ROI | Brier候補ROI | 払戻カバー | 現行ECE | LL候補ECE | Brier候補ECE | 現行の本命見込み | AI重み(LL/Brier) | 審査資格(LL/Brier) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for tp, label in POINTS:
        s_old = _get(legacy, tp, "scores_on_same_forward_test_races") or {}
        s_new = _get(brier, tp, "scores_on_same_forward_test_races") or {}
        dep, cand, chal = s_old.get("deployed") or {}, s_old.get("candidate") or {}, s_new.get("candidate") or {}
        out.append("| " + " | ".join([
            label, _num(dep.get("top1_hit"), "pct"), _num(cand.get("top1_hit"), "pct"), _num(chal.get("top1_hit"), "pct"),
            _num(dep.get("top1_roi"), "pct"), _num(cand.get("top1_roi"), "pct"), _num(chal.get("top1_roi"), "pct"),
            _num(dep.get("payout_coverage_rate"), "pct"),
            _num(_get(dep, "calibration", "ece_10bin")), _num(_get(cand, "calibration", "ece_10bin")),
            _num(_get(chal, "calibration", "ece_10bin")), _num(_get(dep, "calibration", "top1_mean_prob"), "pct"),
            f"{_num(_get(legacy, tp, 'mean_ai_weight'))} / {_num(_get(brier, tp, 'mean_ai_weight'))}",
            f"{_num(_get(legacy, tp, 'promotion_gate', 'eligible_for_review'))} / "
            f"{_num(_get(brier, tp, 'promotion_gate', 'eligible_for_review'))}",
        ]) + " |")
    out += ["", "| 時点 | AI単独LL | 市場単独LL | 従来合成LL | AI単独的中 | 市場単独的中 |", "|---|---:|---:|---:|---:|---:|"]
    for tp, label in POINTS:
        s = _get(legacy, tp, "scores_on_same_forward_test_races") or {}
        out.append("| " + " | ".join([
            label, _num(_get(s, "ai", "log_loss")), _num(_get(s, "market", "log_loss")), _num(_get(s, "blend", "log_loss")),
            _num(_get(s, "ai", "top1_hit"), "pct"), _num(_get(s, "market", "top1_hit"), "pct"),
        ]) + " |")
    return out


def summarize(reports) -> str:
    if isinstance(reports, dict):
        reports = [reports]
    lines = ["## 発走前予測の評価(集計値のみ)", "",
             "LL・Brier・ECE は低いほど良い。改善率は現行(実際に使った確率)比で、正なら候補が良い。"
             "ROI は本命の単勝を100円ずつ買ったとする参考値。区間は開催日単位の95%区間(10日未満は -)。", ""]
    for report in reports:
        org = report.get("org") if isinstance(report, dict) else None
        if org not in ORGS:
            raise ValueError("主催が不正です")
        lines += [f"### {org}", "", "#### 保存済み予測の比較(同じレース・同じ時刻のオッズ)", "",
                  "| 時点 | レース | 開催日 | AI単独LL | 市場LL | 合成LL | AI単独Brier | 市場Brier | 合成Brier | 実採用LL(全件) | 実採用レース |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for tp, label in POINTS:
            b = _get(report, "by_time_point", tp) or {}
            lines.append("| " + " | ".join([
                label, _num(_get(b, "ai", "n_races"), "int"),
                _num(_get(report, "paired_model_comparison", "by_time_point", tp, "n_race_dates"), "int"),
                _num(_get(b, "ai", "log_loss")), _num(_get(b, "market", "log_loss")), _num(_get(b, "blend", "log_loss")),
                _num(_get(b, "ai", "brier")), _num(_get(b, "market", "brier")), _num(_get(b, "blend", "brier")),
                _num(_get(b, "used_all_races", "log_loss")), _num(_get(b, "used_all_races", "n_races"), "int"),
            ]) + " |")
        shadow = _get(report, "shadow_research") or {}
        lines += ["", "#### 研究モデル(全期間のウォークフォワード)", ""] + _research_tables(shadow)
        current = shadow.get("current_model_only") if isinstance(shadow, dict) else None
        lines += ["", "#### 研究モデル(採用中のモデルの版だけ)", ""]
        lines += _research_tables(current) if isinstance(current, dict) else ["採用中の版の記録がありません。"]
        versions = [_num(_get(shadow, "model_versions_in_valid_races", tp), "int") for tp, _ in POINTS]
        lines += ["", f"全期間に含まれるモデルの版の数(30分前/10分前/直前): {' / '.join(versions)}", ""]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="評価レポートの集計値だけを Markdown で出す")
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args(argv)
    print(summarize(json.loads(args.input.expanduser().read_text(encoding="utf-8"))), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
