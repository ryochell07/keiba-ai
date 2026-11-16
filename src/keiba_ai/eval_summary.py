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


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _research_tables(block: dict) -> list[str]:
    """現行と3つの研究候補 (線形・Brier対応・対数線形) を、同じ将来評価レースで並べる。"""
    runs = {
        "linear": _get(block, "by_time_point") or {},
        "brier": _get(block, "brier_regularized_challenger", "by_time_point") or {},
        "loglinear": _get(block, "loglinear_challenger", "by_time_point") or {},
    }
    names = ("linear", "brier", "loglinear")

    def scores(tp):
        dep = _get(runs["linear"], tp, "scores_on_same_forward_test_races", "deployed") or {}
        return dep, [_get(runs[k], tp, "scores_on_same_forward_test_races", "candidate") or {} for k in names]

    out = ["LogLoss", "", "| 時点 | 検証レース | 検証日 | 現行 | 線形 | 改善率 | Brier対応 | 改善率 | 対数線形 | 改善率 "
           "| 線形−現行の95%区間 | 対数線形−現行の95%区間 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for tp, label in POINTS:
        dep, cands = scores(tp)
        cells = [label, _num(_get(runs["linear"], tp, "forward_test_races"), "int"),
                 _num(_get(runs["linear"], tp, "forward_test_days"), "int"), _num(dep.get("log_loss"))]
        for c in cands:
            cells += [_num(c.get("log_loss")), _num(_improve(dep.get("log_loss"), c.get("log_loss")), "pp")]
        cells += [_ci(_get(runs["linear"], tp, "candidate_minus_deployed_ci95", "log_loss")),
                  _ci(_get(runs["loglinear"], tp, "candidate_minus_deployed_ci95", "log_loss"))]
        out.append(_row(cells))
    out += ["", "Brier", "", "| 時点 | 現行 | 線形 | 改善率 | Brier対応 | 改善率 | 対数線形 | 改善率 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for tp, label in POINTS:
        dep, cands = scores(tp)
        cells = [label, _num(dep.get("brier"))]
        for c in cands:
            cells += [_num(c.get("brier")), _num(_improve(dep.get("brier"), c.get("brier")), "pp")]
        out.append(_row(cells))
    out += ["", "的中率・参考ROI", "", "| 時点 | 現行的中 | 線形 | Brier対応 | 対数線形 | 現行ROI | 線形 | Brier対応 | 対数線形 | 払戻カバー |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for tp, label in POINTS:
        dep, cands = scores(tp)
        out.append(_row([label, _num(dep.get("top1_hit"), "pct")] + [_num(c.get("top1_hit"), "pct") for c in cands]
                        + [_num(dep.get("top1_roi"), "pct")] + [_num(c.get("top1_roi"), "pct") for c in cands]
                        + [_num(dep.get("payout_coverage_rate"), "pct")]))
    out += ["", "校正と選ばれた比率", "", "| 時点 | 現行ECE | 線形 | Brier対応 | 対数線形 | 現行の本命見込み | 線形のAI重み "
            "| Brier対応のAI重み | 対数線形の指数(AI/市場) | 審査資格(線形/Brier対応/対数線形) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for tp, label in POINTS:
        dep, cands = scores(tp)
        out.append(_row([label, _num(_get(dep, "calibration", "ece_10bin"))]
                        + [_num(_get(c, "calibration", "ece_10bin")) for c in cands]
                        + [_num(_get(dep, "calibration", "top1_mean_prob"), "pct"),
                           _num(_get(runs["linear"], tp, "mean_ai_weight")), _num(_get(runs["brier"], tp, "mean_ai_weight")),
                           f"{_num(_get(runs['loglinear'], tp, 'mean_ai_exponent'))} / "
                           f"{_num(_get(runs['loglinear'], tp, 'mean_market_exponent'))}",
                           " / ".join(_num(_get(runs[k], tp, "promotion_gate", "eligible_for_review")) for k in names)]))
    out += ["", "単独モデル(同じ将来評価レース)", "", "| 時点 | AI単独LL | 市場単独LL | 従来合成LL | AI単独的中 | 市場単独的中 |",
            "|---|---:|---:|---:|---:|---:|"]
    for tp, label in POINTS:
        s = _get(runs["linear"], tp, "scores_on_same_forward_test_races") or {}
        out.append(_row([label, _num(_get(s, "ai", "log_loss")), _num(_get(s, "market", "log_loss")),
                         _num(_get(s, "blend", "log_loss")), _num(_get(s, "ai", "top1_hit"), "pct"),
                         _num(_get(s, "market", "top1_hit"), "pct")]))
    return out


def summarize(reports) -> str:
    if isinstance(reports, dict):
        reports = [reports]
    lines = ["## 発走前予測の評価(集計値のみ)", "",
             "LL・Brier・ECE は低いほど良い。改善率は現行(実際に使った確率)比で、正なら候補が良い。"
             "ROI は本命の単勝を100円ずつ買ったとする参考値。区間は開催日単位の95%区間(10日未満は -)。", "",
             "研究候補: 線形 = α×AI+(1−α)×市場 / Brier対応 = 線形で LogLoss+0.25×Brier を最小化 / "
             "対数線形 = AI^a×市場^b を正規化(本番の合成と同じ形)。どれも比率は前日までの結果だけで選ぶ。", ""]
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
