"""学習・予測・バックテスト・公開ページ生成のコマンド。"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta

from . import config
from .db import JST

log = logging.getLogger(__name__)


def _orgs(con, args) -> list[str]:
    if getattr(args, "org", None):
        return [args.org]
    return [r[0] for r in con.execute("SELECT DISTINCT org FROM race")]


def cmd_train(args, con, sources):
    """新モデルを学習し、旧モデルと比較して良いときだけ採用する。"""
    from .registry import auto_retrain_needed, retrain_with_gate
    for org in _orgs(con, args):
        print(json.dumps(retrain_with_gate(con, org, n_seeds=args.seeds, force=args.force), ensure_ascii=False, indent=1))


def cmd_models(args, con, sources):
    from .registry import list_models
    for m in list_models(con, args.org):
        print(m["model_version_id"], m["status"], m["train_end"], m["adopted_at"] or "-")


def cmd_rollback(args, con, sources):
    from .registry import rollback
    print(rollback(con, args.org, args.to))


def cmd_reconcile(args, con, sources):
    from .forward import reconcile
    for org in _orgs(con, args):
        print(json.dumps(reconcile(con, org), ensure_ascii=False, indent=1))


def cmd_predict(args, con, sources):
    from .predict import predict_races
    for org in _orgs(con, args):
        print(predict_races(con, org, args.from_date))


def cmd_backtest(args, con, sources):
    from .backtest import run_backtest
    r = run_backtest(con, args.org, args.start, args.step, args.seeds)
    print(json.dumps({k: r[k] for k in ("period", "summary", "calibration", "candidate_coverage")}, ensure_ascii=False, indent=1))
    for row in sorted(r["rows"], key=lambda x: (x["bet_type"], x["strategy"], x["odds_basis"])):
        print(row["bet_type"], row["strategy"], row["odds_basis"], row["n_bets"], f"hit={row['n_hits']}",
              f"roi={row['roi']:.3f}", f"ci=[{row['roi_ci_low']:.3f},{row['roi_ci_high']:.3f}]")


def cmd_compare(args, con, sources):
    from .backtest import compare_models
    for r in compare_models(con, args.org, args.start):
        print(r)


def cmd_site(args, con, sources):
    from .site import build_site
    print(build_site(con, sources))


def cmd_daily(args, con, sources):
    """日次処理。取得の失敗は記録して続行し、最後に終了コードで知らせる。"""
    from .cli import _jra, _nar, _record
    from .http import FetchBlocked, FetchFailed, FetchStop
    from .backtest import run_backtest
    from .forward import reconcile
    from .predict import predict_races
    from .providers.nar_csv import NarCsvProvider
    from .registry import auto_retrain_needed, retrain_with_gate
    from .site import build_site
    failures = []
    if not args.no_fetch:
        jra = _jra(con, sources)
        steps = [("results", lambda: jra.update_results(con, args.days)), ("cards", lambda: jra.fetch_racecards(con))]
        if args.odds:
            steps.append(("odds", lambda: jra.capture_odds(con)))
        for name, fn in steps:
            try:
                log.info("%s: %s", name, fn())
                _record(con, "jra", True)
            except (FetchFailed, FetchStop, FetchBlocked) as e:
                failures.append(f"jra {name}: {e}")
                _record(con, "jra", False, str(e))
        nar = _nar(con, sources)
        try:
            log.info("nar: %s", [{k: v for k, v in x.items() if k != "warnings"} for x in nar.update(con)])
            _record(con, "nar_csv", True)
        except (FetchFailed, FetchStop, FetchBlocked) as e:
            failures.append(f"nar: {e}")
            _record(con, "nar_csv", False, str(e))
    log.info("nar inbox: %s", NarCsvProvider(sources["sources"]["nar_csv"]).import_inbox(con))
    for org in _orgs(con, args):
        needed, early = auto_retrain_needed(con, org)
        if args.retrain or needed:
            try:
                log.info("train %s", retrain_with_gate(con, org, force=early))
                start = (datetime.now(JST) - timedelta(days=240)).strftime("%Y-%m-01")
                run_backtest(con, org, start, step_months=2)
            except ValueError as e:
                log.warning("%s", e)
    for org in _orgs(con, args):
        log.info("reconcile %s", reconcile(con, org).get("n_races"))
        log.info("predict %s", predict_races(con, org))
    log.info("site %s", build_site(con, sources))
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 2
    return 0


def cmd_state(args, con, sources):
    from .state import pack, unpack
    print(pack(args.path) if args.action == "pack" else unpack(args.path))


def register(sub):
    p = sub.add_parser("train", help="新モデルを学習し、旧モデルより良ければ採用")
    p.add_argument("--org", choices=["JRA", "NAR"])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--force", action="store_true", help="比較結果によらず採用する")
    p.set_defaults(func=cmd_train)
    p = sub.add_parser("models", help="モデルの版を一覧")
    p.add_argument("--org", choices=["JRA", "NAR"])
    p.set_defaults(func=cmd_models)
    p = sub.add_parser("rollback", help="以前のモデルへ戻す")
    p.add_argument("--org", choices=["JRA", "NAR"], required=True)
    p.add_argument("--to", help="モデルID。省略時は直前の採用モデル")
    p.set_defaults(func=cmd_rollback)
    p = sub.add_parser("reconcile", help="発走前の予測と確定結果を照合")
    p.add_argument("--org", choices=["JRA", "NAR"])
    p.set_defaults(func=cmd_reconcile)
    p = sub.add_parser("predict", help="未確定レースを予測")
    p.add_argument("--org", choices=["JRA", "NAR"])
    p.add_argument("--from-date")
    p.set_defaults(func=cmd_predict)
    p = sub.add_parser("backtest", help="ウォークフォワードのバックテスト")
    p.add_argument("--org", choices=["JRA", "NAR"], default="JRA")
    p.add_argument("--start", required=True, help="評価開始 YYYY-MM-01")
    p.add_argument("--step", type=int, default=1, help="再学習間隔 (月)")
    p.add_argument("--seeds", type=int, default=1)
    p.set_defaults(func=cmd_backtest)
    p = sub.add_parser("compare", help="モデル手法の比較")
    p.add_argument("--org", choices=["JRA", "NAR"], default="JRA")
    p.add_argument("--start", required=True)
    p.set_defaults(func=cmd_compare)
    p = sub.add_parser("site", help="スマホ閲覧ページを生成")
    p.set_defaults(func=cmd_site)
    p = sub.add_parser("daily", help="日次処理を一括実行")
    p.add_argument("--days", type=int, default=10)
    p.add_argument("--odds", action="store_true")
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--no-fetch", action="store_true")
    p.set_defaults(func=cmd_daily)
    p = sub.add_parser("state", help="DBとモデルの暗号化バックアップ")
    p.add_argument("action", choices=["pack", "unpack"])
    p.add_argument("path")
    p.set_defaults(func=cmd_state)
