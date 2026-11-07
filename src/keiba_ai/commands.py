"""学習・予測・バックテスト・公開ページ生成のコマンド。"""
from __future__ import annotations

import json
import logging
import sys
import time
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
    from .http import FetchBlocked, FetchStop
    from .backtest import run_backtest
    from .bundles import register_bundle
    from .forward import reconcile
    from .monitor import odds_quality
    from .predict import predict_races
    from .providers.nar_csv import NarCsvProvider
    from .registry import auto_retrain_needed, retrain_with_gate
    from .site import build_site
    failures = []
    if not args.no_fetch:
        from .failures import is_fatal, record_failure
        jra = _jra(con, sources)
        steps = [("results", lambda: jra.update_results(con, args.days)), ("cards", lambda: jra.fetch_racecards(con))]
        if args.odds:
            steps.append(("odds", lambda: jra.capture_odds(con)))
        for name, fn in steps:
            try:
                log.info("%s: %s", name, fn())
                _record(con, "jra", True)
            except Exception as e:
                con.rollback()
                if is_fatal(e):
                    raise
                rec = record_failure(con, "JRA", name, e)
                failures.append(f"jra {name}: {rec['type']}: {rec['message']}")
                _record(con, "jra", False, rec["message"])
                if isinstance(e, (FetchStop, FetchBlocked)):
                    break   # 拒否されたら、この実行ではそれ以上要求しない
        nar = _nar(con, sources)
        try:
            log.info("nar: %s", [{k: v for k, v in x.items() if k != "warnings"} for x in nar.update(con)])
            _record(con, "nar_csv", True)
        except Exception as e:
            con.rollback()
            if is_fatal(e):
                raise
            rec = record_failure(con, "NAR", "update", e)
            failures.append(f"nar: {rec['type']}: {rec['message']}")
            _record(con, "nar_csv", False, rec["message"])
    log.info("nar inbox: %s", NarCsvProvider(sources["sources"]["nar_csv"]).import_inbox(con))
    frozen = bool(config.guards().get("freeze_models"))
    for org in _orgs(con, args):
        needed, early = auto_retrain_needed(con, org)
        if frozen:
            if args.retrain or needed:
                log.info("train %s: 再学習は freeze_models で停止中", org)
            continue
        if args.retrain or needed:
            try:
                log.info("train %s", retrain_with_gate(con, org, force=early))
                start = (datetime.now(JST) - timedelta(days=240)).strftime("%Y-%m-01")
                run_backtest(con, org, start, step_months=2)
                row = con.execute("SELECT model_version_id FROM model_version WHERE org=? AND status='active'", (org,)).fetchone()
                if row:
                    register_bundle(con, row[0], adoption_reason="再学習後のバックテストで合成係数を更新")
                    con.commit()
            except ValueError as e:
                log.warning("%s", e)
    from .failures import is_fatal, record_failure

    def guarded(org, stage, fn):
        """片方の主催者の失敗で、もう片方の照合や予測を止めない。"""
        try:
            return fn()
        except Exception as e:
            con.rollback()
            if is_fatal(e):
                raise
            rec = record_failure(con, org, stage, e)
            failures.append(f"{org} {stage}: {rec['type']}: {rec['message']}")
            return None

    for org in _orgs(con, args):
        rep = guarded(org, "reconcile", lambda: reconcile(con, org))
        log.info("reconcile %s", rep.get("n_races") if rep else None)
        stats = guarded(org, "predict", lambda: predict_races(con, org))
        log.info("predict %s", stats)
        if stats and stats.get("failed"):
            failures.append(f"{org} predict: {stats['failed']} レース失敗")
        for offset in (1, 0):
            day = (datetime.now(JST) - timedelta(days=offset)).date().isoformat()
            log.info("odds quality %s", guarded(org, "odds_quality", lambda: odds_quality(con, day, org)))
    log.info("site %s", build_site(con, sources))
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 2
    return 0


def cmd_race_loop(args, con, sources):
    """開催中に短い間隔でオッズを取得して予測を記録する。発走が近いレースがなければ終わる。
    主催者ごとに分離するので、片方の失敗でもう片方は止まらない。整合性に関わるエラーだけは全体を止める。"""
    from .cli import _jra, _nar, _record
    from .failures import SOURCE, is_fatal, record_failure
    from .http import FetchBlocked, FetchStop
    from .predict import predict_races
    started = time.monotonic()
    failures, race_failures, cycles = 0, 0, 0
    stopped: set[str] = set()   # 提供元に拒否された主催者。この実行ではもう要求しない
    while True:
        begin = time.monotonic()
        now = datetime.now(JST)
        cycles += 1
        posts = con.execute("""SELECT r.org, r.post_time FROM race r WHERE r.race_date=? AND r.post_time IS NOT NULL AND r.course_code NOT LIKE 'N帯広%'
                               AND NOT EXISTS (SELECT 1 FROM result x WHERE x.race_id=r.race_id AND x.finish_pos=1)""", (now.date().isoformat(),)).fetchall()
        soon: dict[str, float] = {}
        for r in posts:
            mins = (datetime.fromisoformat(f"{now.date().isoformat()}T{r['post_time']}:00+09:00") - now).total_seconds() / 60
            if mins > 0:
                soon[r["org"]] = min(mins, soon.get(r["org"], mins))
        active = [o for o, m in soon.items() if m <= args.window]
        log.info("race-loop %s: 次の発走まで %s", now.isoformat(timespec="seconds"), soon)
        for org in active:
            if org not in stopped:
                try:
                    out = _jra(con, sources).capture_odds(con) if org == "JRA" else _nar(con, sources).capture_today(con)
                    log.info("odds %s: %s", org, out)
                    race_failures += out.get("failed", 0)
                    _record(con, SOURCE[org], True)
                except Exception as e:
                    con.rollback()
                    if is_fatal(e):
                        raise
                    failures += 1
                    rec = record_failure(con, org, "capture", e)
                    _record(con, SOURCE[org], False, rec["message"])
                    if isinstance(e, (FetchStop, FetchBlocked)):
                        stopped.add(org)
            try:
                stats = predict_races(con, org)
                log.info("predict %s", stats)
                race_failures += stats.get("failed", 0)
            except Exception as e:
                con.rollback()
                if is_fatal(e):
                    raise
                failures += 1
                record_failure(con, org, "predict", e)
        if not active or time.monotonic() - started >= args.minutes * 60:
            break
        time.sleep(max(0.0, args.interval - (time.monotonic() - begin)))
    print(json.dumps({"race_loop": "finished", "cycles": cycles, "org_failures": failures, "race_failures": race_failures, "stopped": sorted(stopped)}, ensure_ascii=False))
    return 1 if failures or race_failures else 0


def cmd_env_check(args, con, sources):
    from .envcheck import EXIT_ENV, env_mismatches
    bad = env_mismatches()
    print("\n".join(bad) if bad else "環境は uv.lock と一致しています")
    return EXIT_ENV if bad else 0


def cmd_replay(args, con, sources):
    from .ledger import replay
    print(json.dumps(replay(con, args.prediction_id), ensure_ascii=False, indent=1))


def cmd_odds_report(args, con, sources):
    from .monitor import odds_quality
    day = args.date or datetime.now(JST).date().isoformat()
    for org in _orgs(con, args):
        print(json.dumps(odds_quality(con, day, org), ensure_ascii=False))


def cmd_bundle(args, con, sources):
    from .bundles import active_bundle
    for r in con.execute("SELECT org, model_version_id FROM model_version WHERE status='active'").fetchall():
        b = dict(active_bundle(con, r["org"], r["model_version_id"]))
        b["features"] = f"{len(json.loads(b['features']))} 列"
        print(json.dumps(b, ensure_ascii=False, indent=1))


def cmd_lab(args, con, sources):
    from .lab import run_stage
    print(json.dumps(run_stage(con, args.org, args.stage, confirm=args.confirm_locked_test), ensure_ascii=False, indent=1, default=str))


def cmd_state(args, con, sources):
    from .state import pack, unpack
    print(pack(args.path) if args.action == "pack" else unpack(args.path))


def register(sub):
    p = sub.add_parser("lab", help="精度改善の実験 (封鎖テスト期間は既定で使わない)")
    p.add_argument("stage", choices=["baseline", "analyze", "history", "groups", "models", "calibrate", "split", "market", "final", "list", "summary"])
    p.add_argument("--org", choices=["JRA", "NAR"], default="NAR")
    p.add_argument("--confirm-locked-test", action="store_true", help="final で封鎖テストを使うことを明示する")
    p.set_defaults(func=cmd_lab)
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
    p = sub.add_parser("race-loop", help="開催中に短い間隔でオッズ取得と予測記録を繰り返す")
    p.add_argument("--minutes", type=float, default=20)
    p.add_argument("--interval", type=float, default=300, help="周期 (秒)")
    p.add_argument("--window", type=float, default=65, help="この分数以内に発走があるときだけ動く")
    p.set_defaults(func=cmd_race_loop)
    p = sub.add_parser("env-check", help="動いている環境が uv.lock と同じか確かめる")
    p.set_defaults(func=cmd_env_check)
    p = sub.add_parser("replay", help="保存した入力から予測を再計算して記録と比べる")
    p.add_argument("prediction_id")
    p.set_defaults(func=cmd_replay)
    p = sub.add_parser("odds-report", help="発走前オッズの取得状況を集計")
    p.add_argument("--date")
    p.add_argument("--org", choices=["JRA", "NAR"])
    p.set_defaults(func=cmd_odds_report)
    p = sub.add_parser("bundle", help="採用中モデルの版情報を表示")
    p.set_defaults(func=cmd_bundle)
    p = sub.add_parser("state", help="DBとモデルの暗号化バックアップ")
    p.add_argument("action", choices=["pack", "unpack"])
    p.add_argument("path")
    p.set_defaults(func=cmd_state)
