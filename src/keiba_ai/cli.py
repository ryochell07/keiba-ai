"""コマンドライン入口。"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from . import config
from .db import connect, now_jst
from .http import FetchBlocked, FetchFailed, FetchLimit, FetchStop, PoliteClient


def _sync_sources(con, sources: dict) -> None:
    for name, s in sources["sources"].items():
        con.execute(
            "INSERT INTO source_status(source,enabled,automated_fetch,license_status,license_note,permission)"
            " VALUES (?,?,?,?,?,?) ON CONFLICT(source) DO UPDATE SET enabled=excluded.enabled,"
            " automated_fetch=excluded.automated_fetch, license_status=excluded.license_status,"
            " license_note=excluded.license_note, permission=excluded.permission",
            (name, int(s.get("enabled", False)), int(s.get("automated_fetch", False)), s.get("license_status"),
             s.get("license_note"), json.dumps(s.get("permission", {}), ensure_ascii=False)))
    con.commit()


def _record(con, source: str, ok: bool, error: str | None = None) -> None:
    if ok:
        con.execute("UPDATE source_status SET last_success_at=?, last_error=NULL WHERE source=?", (now_jst(), source))
    else:
        con.execute("UPDATE source_status SET last_error=? WHERE source=?", (f"{now_jst()} {error}", source))
    con.commit()


def _jra(con, sources):
    from .providers.jra import JraProvider
    return JraProvider(PoliteClient("jra", sources["sources"]["jra"], con,
                                    max_requests=sources["limits"]["max_requests_per_run"]))


def _nar(con, sources):
    from .providers.nar_csv import NarCsvProvider
    conf = sources["sources"]["nar_csv"]
    return NarCsvProvider(conf, PoliteClient("nar_csv", conf, con, max_requests=sources["limits"]["max_requests_per_run"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="keiba")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db")
    p = sub.add_parser("jra-backfill", help="過去のレース結果を取得 (新しい月から)")
    p.add_argument("--from", dest="start", required=True, help="YYYY-MM")
    p.add_argument("--to", dest="end", required=True, help="YYYY-MM")
    p.add_argument("--all-odds-since", help="この日以降は全券種の最終オッズも取得 YYYY-MM-DD")
    p = sub.add_parser("jra-update", help="直近の結果を取得")
    p.add_argument("--days", type=int, default=10)
    p.add_argument("--all-odds-days", type=int, default=0)
    sub.add_parser("jra-cards", help="公開中の出馬表を取得")
    sub.add_parser("jra-odds", help="発売中のオッズを取得時刻付きで保存")
    p = sub.add_parser("nar-import", help="手動DLした公式ZIPを取り込む")
    p.add_argument("paths", nargs="*", help="ZIPファイル。省略時は data/inbox/nar/")
    p = sub.add_parser("nar-backfill", help="公式の月次ZIPを自動取得 (許諾がなければ拒否される)")
    p.add_argument("--from", dest="start", required=True, help="YYYY-MM")
    p.add_argument("--to", dest="end", required=True, help="YYYY-MM")
    p.add_argument("--odds-from", default="2026-03", help="オッズを取得する開始月")
    p.add_argument("--full-odds-from", help="全券種のオッズを保存する開始月")
    sub.add_parser("nar-update", help="前月・当月の月次ZIPを更新 (1日1回まで)")
    sub.add_parser("nar-today", help="当日ZIPで出馬表と発売中オッズを更新")
    sub.add_parser("status", help="DB件数と取得元の状態を表示")
    from .commands import register
    register(sub)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config.ensure_dirs()
    con = connect()
    sources = config.load_sources()
    _sync_sources(con, sources)
    try:
        if args.cmd == "init-db":
            print("OK", config.DB_PATH)
        elif args.cmd == "jra-backfill":
            from .providers.jra import month_range
            print(_jra(con, sources).backfill(con, month_range(args.start, args.end), args.all_odds_since))
            _record(con, "jra", True)
        elif args.cmd == "jra-update":
            print(_jra(con, sources).update_results(con, args.days, args.all_odds_days))
            _record(con, "jra", True)
        elif args.cmd == "jra-cards":
            print(_jra(con, sources).fetch_racecards(con))
            _record(con, "jra", True)
        elif args.cmd == "jra-odds":
            print(_jra(con, sources).capture_odds(con))
            _record(con, "jra", True)
        elif args.cmd == "nar-import":
            from pathlib import Path
            from .providers.nar_csv import NarCsvProvider, import_zip
            if args.paths:
                out = [import_zip(con, Path(x)) for x in sorted(args.paths, key=lambda n: ("odds" in n, n))]
            else:
                out = NarCsvProvider(sources["sources"]["nar_csv"]).import_inbox(con)
            print(json.dumps(out, ensure_ascii=False, indent=1))
            _record(con, "nar_csv", True)
        elif args.cmd in ("nar-backfill", "nar-update", "nar-today"):
            prov = _nar(con, sources)
            if args.cmd == "nar-backfill":
                out = prov.backfill(con, args.start, args.end, args.odds_from, args.full_odds_from)
                print(json.dumps([{k: v for k, v in x.items() if k != "warnings"} for x in out], ensure_ascii=False))
            elif args.cmd == "nar-update":
                print(prov.update(con))
            else:
                print(prov.capture_today(con))
            _record(con, "nar_csv", True)
        elif args.cmd == "status":
            for t in ("race", "entry", "result", "payout", "odds_snapshot", "horse", "prediction", "model_version", "fetch_log"):
                print(t, con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
            for r in con.execute("SELECT org,MIN(race_date),MAX(race_date),COUNT(*) FROM race GROUP BY org"):
                print(tuple(r))
            for r in con.execute("SELECT source,enabled,automated_fetch,license_status,last_success_at,last_error FROM source_status"):
                print(tuple(r))
        else:
            return args.func(args, con, sources) or 0
    except FetchBlocked as e:
        print(f"取得を中止しました (規約/robots/設定): {e}", file=sys.stderr)
        _record(con, args.cmd.split("-")[0] if args.cmd.startswith("jra") else "nar_csv", False, str(e))
        return 3
    except FetchLimit as e:
        print(f"{e}。次回の実行で続きから取得します")
        return 0
    except (FetchStop, FetchFailed) as e:
        print(f"取得エラー: {e}", file=sys.stderr)
        _record(con, "jra" if args.cmd.startswith("jra") else "nar_csv", False, str(e))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
