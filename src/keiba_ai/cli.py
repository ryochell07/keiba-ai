"""コマンドライン入口。"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings

from . import config
from .db import connect, now_jst
from .failures import EXIT_FATAL, clean_message, is_fatal, record_failure
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


def _public_excepthook(exc_type, exc, tb) -> None:
    """Actions のログは公開される。想定外の例外でも traceback と文面を出さない。終了コードは変わらず 1。"""
    print(f"想定外のエラーで停止しました: {exc_type.__name__}", file=sys.stderr)


def _public_warning(message, category, filename, lineno, file=None, line=None) -> None:
    print(f"警告: {category.__name__}", file=sys.stderr)


def _command_org(cmd: str) -> str | None:
    return "JRA" if cmd.startswith("jra") else "NAR" if cmd.startswith("nar") else None


_NEEDS_LOCKED_ENV = ("daily", "race-loop", "predict", "train")


def _jra(con, sources):
    from .providers.jra import JraProvider
    return JraProvider(PoliteClient("jra", sources["sources"]["jra"], con,
                                    max_requests=sources["limits"]["max_requests_per_run"]))


def _nar(con, sources):
    from .providers.nar_csv import NarCsvProvider
    conf = sources["sources"]["nar_csv"]
    return NarCsvProvider(conf, PoliteClient("nar_csv", conf, con, max_requests=sources["limits"]["max_requests_per_run"]))


def _readonly_reconcile_database(args, path, *, immutable: bool = False) -> int:
    """評価指標のみを取り出す。暗号化バックアップの場合は必ず安全なファイルへ保存。"""
    import os
    import sqlite3
    from pathlib import Path
    from .forward import reconcile

    uri = path.resolve().as_uri() + ("?mode=ro&immutable=1" if immutable else "?mode=ro")
    with sqlite3.connect(uri, uri=True, timeout=60) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only = ON")
        if immutable:
            # 暗号の認証成功はSQLiteファイルの論理的な整合性の保証ではない。
            # スナップショットを更新せず、DB破損なら数値評価を拒否する。
            try:
                if con.execute("PRAGMA quick_check(1)").fetchone()[0] != "ok":
                    raise ValueError("評価用データベースの整合性に問題があります")
            except sqlite3.DatabaseError as exc:
                raise ValueError("評価用データベースを安全に読み取れません") from exc
        orgs = [args.org] if args.org else [
            r[0] for r in con.execute("SELECT DISTINCT org FROM race ORDER BY org")
        ]
        summaries = []
        for org in orgs:
            report = reconcile(con, org, persist=False, write_report=False)
            fields = ("org", "n_races", "n_races_with_selected_prestart_run",
                      "excluded_runs_without_verified_prestart_time",
                      "evaluation_coverage", "paired_model_comparison", "by_time_point")
            summaries.append({k: report[k] for k in fields})
    payload = json.dumps(summaries[0] if len(summaries) == 1 else summaries,
                         ensure_ascii=False, indent=1) + "\n"
    if args.output:
        dest = Path(args.output).expanduser()
        # repo内に秘密の性能集計を誤コミットしない。シンボリックリンクを含む
        # 保存先の実パスで検査し、既存ファイルは上書き不可。
        resolved = dest.resolve()
        if resolved.is_relative_to(config.ROOT.resolve()):
            raise ValueError("評価結果を公開リポジトリ内へ保存することはできません")
        fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                out.write(payload)
        except BaseException:
            dest.unlink(missing_ok=True)
            raise
    else:
        # 非暗号化DBでの従来動作だけ標準出力を許可。
        print(payload, end="")
    return 0


def _readonly_reconcile(args) -> int:
    """既存DBまたは隔離復号した状態アーカイブを一切永続的に変更せず評価する。"""
    if not args.encrypted_state:
        return _readonly_reconcile_database(args, config.DB_PATH)

    from pathlib import Path
    from tempfile import TemporaryDirectory
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    import io
    import shutil
    import tarfile

    from .state import MAGIC, _key

    path = Path(args.encrypted_state).expanduser()
    raw = path.read_bytes()
    if not raw.startswith(MAGIC) or len(raw) <= len(MAGIC) + 12 + 16:
        raise ValueError("状態ファイルの形式が不正です")
    nonce = raw[len(MAGIC):len(MAGIC) + 12]
    try:
        plaintext = AESGCM(_key()).decrypt(nonce, raw[len(MAGIC) + 12:], MAGIC)
    except InvalidTag as exc:
        raise ValueError("状態ファイルの復号に失敗しました。鍵またはファイルを確認してください") from exc
    # 本番の config.DB_PATH / state.unpack / config.ensure_dirs は一切呼ばない。
    # 復元先は他ユーザーから閲覧できない一時フォルダ。終了時に完全削除する。
    with TemporaryDirectory(prefix="keiba-readonly-eval-") as tmp:
        target = Path(tmp) / "keiba.db"
        with tarfile.open(fileobj=io.BytesIO(plaintext), mode="r:gz") as archive:
            member = archive.getmember("keiba.db")
            if not member.isfile() or member.size <= 0 or member.size > 8 * 1024**3:
                raise ValueError("状態内のDBサイズが不正です")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("状態内にDBがありません")
            with source, target.open("xb") as dest:
                shutil.copyfileobj(source, dest)
        target.chmod(0o600)
        return _readonly_reconcile_database(args, target, immutable=True)


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
    sub.add_parser("jra-results", help="当日の確定したレースの結果を取得 (未取得のレースがなければ要求しない)")
    sub.add_parser("jra-cards", help="公開中の出馬表を取得")
    sub.add_parser("jra-odds", help="発売中のオッズを取得時刻付きで保存")
    p = sub.add_parser("nar-import", help="手動DLした公式ZIPを取り込む")
    p.add_argument("paths", nargs="*", help="ZIPファイル。省略時は data/inbox/nar/")
    p = sub.add_parser("nar-backfill", help="公式の月次ZIPを自動取得 (許諾がなければ拒否される)")
    p.add_argument("--from", dest="start", required=True, help="YYYY-MM")
    p.add_argument("--to", dest="end", required=True, help="YYYY-MM")
    p.add_argument("--odds-from", default="2026-03", help="オッズを取得する開始月")
    p.add_argument("--full-odds-from", help="全券種のオッズを保存する開始月")
    p.add_argument("--refresh", action="store_true", help="取り込み処理の修正を反映するため再取得 (同じ月は1日1回まで)")
    sub.add_parser("nar-update", help="前月・当月の月次ZIPを更新 (1日1回まで)")
    sub.add_parser("nar-today", help="当日ZIPで出馬表と発売中オッズを更新")
    sub.add_parser("status", help="DB件数と取得元の状態を表示")
    from .commands import register
    register(sub)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.excepthook, warnings.showwarning = _public_excepthook, _public_warning
    if args.cmd == "reconcile" and args.encrypted_state and not args.read_only:
        ap.error("--encrypted-state は --read-only と併用してください")
    if args.cmd == "reconcile" and args.encrypted_state and not args.output:
        ap.error("--encrypted-state では --output に非公開ファイルの保存先を指定してください")
    if args.cmd == "reconcile" and args.output and not args.read_only:
        ap.error("--output は --read-only と併用してください")
    if args.cmd == "reconcile" and args.read_only:
        return _readonly_reconcile(args)
    config.ensure_dirs()
    if args.cmd == "state":  # 復元先の DB を開いてはいけない (WAL が残って復元した DB を壊す)
        return args.func(args, None, None) or 0
    if args.cmd == "env-check":  # 環境の確認で DB を開いたり移行したりしない
        return args.func(args, None, None) or 0
    if args.cmd in _NEEDS_LOCKED_ENV:
        from .envcheck import EXIT_ENV, env_mismatches
        bad = env_mismatches()
        if bad:
            print("環境が uv.lock と違うため停止しました:\n" + "\n".join(bad), file=sys.stderr)
            return EXIT_ENV
    con = None
    try:
        con = connect()
        sources = config.load_sources()
        _sync_sources(con, sources)
        if args.cmd == "init-db":
            print("OK", config.DB_PATH)
        elif args.cmd == "jra-backfill":
            from .providers.jra import month_range
            print(_jra(con, sources).backfill(con, month_range(args.start, args.end), args.all_odds_since))
            _record(con, "jra", True)
        elif args.cmd == "jra-update":
            print(_jra(con, sources).update_results(con, args.days, args.all_odds_days))
            _record(con, "jra", True)
        elif args.cmd == "jra-results":
            out = _jra(con, sources).update_today_results(con)
            print(out)
            if out["pending"]:
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
                out = prov.backfill(con, args.start, args.end, args.odds_from, args.full_odds_from, args.refresh)
                print(json.dumps([{k: v for k, v in x.items() if k != "warnings"} for x in out], ensure_ascii=False))
            elif args.cmd == "nar-update":
                print(prov.update(con))
            else:
                print(prov.capture_today(con))
            _record(con, "nar_csv", True)
        elif args.cmd == "status":
            for t in ("race", "entry", "result", "payout", "odds_snapshot", "horse", "prediction", "prediction_run", "prediction_horse", "field_observation", "model_bundle", "model_version", "fetch_log"):
                print(t, con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
            for r in con.execute("SELECT org,MIN(race_date),MAX(race_date),COUNT(*) FROM race GROUP BY org"):
                print(tuple(r))
            # 公開ログに出るので、失敗の内容は表示せず有無だけを出す
            for r in con.execute("SELECT source,enabled,automated_fetch,license_status,last_success_at,last_error IS NOT NULL FROM source_status"):
                print(tuple(r))
        else:
            return args.func(args, con, sources) or 0
    except FetchBlocked as e:
        print(f"取得を中止しました (規約/robots/設定): {type(e).__name__}", file=sys.stderr)
        _record(con, args.cmd.split("-")[0] if args.cmd.startswith("jra") else "nar_csv", False, clean_message(e))
        return 3
    except FetchLimit:
        print("1回の実行の要求数上限に達しました。次回の実行で続きから取得します")
        return 0
    except (FetchStop, FetchFailed) as e:
        print(f"取得エラー: {type(e).__name__}", file=sys.stderr)
        _record(con, "jra" if args.cmd.startswith("jra") else "nar_csv", False, clean_message(e))
        return 2
    except Exception as e:
        if con is not None:
            try:   # DB が壊れていて残せなくても、詳細を画面には出さない
                con.rollback()
                record_failure(con, _command_org(args.cmd), f"command:{args.cmd}", e)
            except Exception:
                pass
        if not is_fatal(e):
            raise
        print(f"整合性に関わるエラーのため停止しました: {type(e).__name__}", file=sys.stderr)
        return EXIT_FATAL
    return 0


if __name__ == "__main__":
    sys.exit(main())
