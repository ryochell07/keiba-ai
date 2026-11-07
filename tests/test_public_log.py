"""公開される標準出力、標準エラー、ログに詳細が出ないこと、詳細は DB に残ることを確かめる。値はすべて架空。"""
import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import textwrap

import pytest

from keiba_ai import cli, config
from keiba_ai.commands import cmd_daily
from keiba_ai.db import connect
from keiba_ai.envcheck import PROJECT
from keiba_ai.failures import EXIT_FATAL, public_fields, record_failure
from keiba_ai.http import FetchBlocked, FetchFailed, FetchLimit, FetchStop
from keiba_ai.providers import jra as jra_mod
from keiba_ai.providers import nar_csv as nar_mod

SECRET = "fake-secret-value-123"
BODY = "<html>fake-response-body</html>"
HEADER = "Authorization: Bearer fake-header-token-456"
QUERY = "https://example.test/data?type=daily&token=fake-query-789"
FRAGMENT = "架空馬ハナコ 1:23.4"
PRIVATE = (SECRET, BODY, "fake-header-token-456", "token=fake-query-789", "fake-query-789", FRAGMENT, "Traceback")
MESSAGE = f"key={SECRET} body={BODY} {HEADER} url={QUERY} data={FRAGMENT}"
RACE = "J202611070501"


def assert_public(text: str) -> None:
    for word in PRIVATE:
        assert word not in text, word


def raised(cls=ValueError):
    try:
        raise cls(MESSAGE)
    except cls as e:
        return e


@pytest.fixture
def root(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    shutil.copy(PROJECT / "config" / "sources.yaml", tmp_path / "config" / "sources.yaml")
    for name, val in (("ROOT", tmp_path), ("DATA", tmp_path / "data"), ("RAW", tmp_path / "data/raw"), ("INBOX_NAR", tmp_path / "data/inbox/nar"),
                      ("DB_PATH", tmp_path / "data/keiba.db"), ("MODELS", tmp_path / "data/models"), ("REPORTS", tmp_path / "data/reports")):
        monkeypatch.setattr(config, name, val)
    config.ensure_dirs()
    return tmp_path


@pytest.mark.parametrize("cls,kind", [(ValueError, "parse"), (FetchFailed, "fetch")])
def test_例外の詳細は公開ログに出ず識別情報は残る(con, caplog, capsys, cls, kind):
    with caplog.at_level("INFO"):
        record_failure(con, "JRA", "odds_race", raised(cls), RACE)
    out = capsys.readouterr()
    public = caplog.text + out.out + out.err
    assert_public(public)                                # 秘密、本文、ヘッダー、クエリ、取得データ、traceback
    for word in ("JRA", "odds_race", kind, RACE, cls.__name__):   # 調査に必要な識別情報
        assert word in public


def test_想定と違う形式のレースIDは公開しない():
    assert public_fields("NAR", "odds_race", ValueError(), f"N20261107{FRAGMENT}")["race_id"] is None
    assert public_fields("NAR", "odds_race", ValueError(), "N20261107N高知01")["race_id"] == "N20261107N高知01"
    assert public_fields("JRA", "odds_race", ValueError(), RACE)["race_id"] == RACE


def test_詳細は暗号化対象のDBに残り認証情報は残らない(con, monkeypatch):
    monkeypatch.setenv("KEIBA_STATE_KEY", SECRET)
    record_failure(con, "NAR", "card_race", raised(), "N20261107N高知01")
    row = con.execute("SELECT source,url,ref,ok,error FROM fetch_log ORDER BY id DESC LIMIT 1").fetchone()
    assert (row["source"], row["url"], row["ref"], row["ok"]) == ("nar_csv", "stage:card_race", "N20261107N高知01", 0)
    rec = json.loads(row["error"])
    assert (rec["org"], rec["stage"], rec["kind"], rec["type"]) == ("NAR", "card_race", "parse", "ValueError")
    assert FRAGMENT in rec["message"] and BODY in rec["message"]    # 調査に使う詳細は残る
    assert "Traceback" in rec["traceback"] and "raised" in rec["traceback"]
    for word in (SECRET, "fake-header-token-456", "fake-query-789"):       # 認証情報は DB にも残さない
        assert word not in row["error"]
    assert "https://example.test/data?***" in rec["message"]


class BrokenDb:
    def execute(self, *a, **k):
        raise sqlite3.OperationalError(f"disk failure {MESSAGE}")


def test_DBへの記録に失敗しても詳細を出さない(caplog, capsys):
    with caplog.at_level("INFO"), pytest.raises(sqlite3.OperationalError):
        record_failure(BrokenDb(), "JRA", "odds_race", raised(), RACE)
    out = capsys.readouterr()
    public = caplog.text + out.out + out.err
    assert_public(public)
    assert "失敗の記録に失敗" in public and "OperationalError" in public and "odds_race" in public


def _main(argv, caplog, capsys):
    with caplog.at_level("INFO"):
        rc = cli.main(argv)
    out = capsys.readouterr()
    return rc, caplog.text + out.out + out.err


@pytest.mark.parametrize("err,rc", [(FetchBlocked, 3), (FetchLimit, 0), (FetchStop, 2), (FetchFailed, 2)])
def test_コマンドの取得エラーは詳細を出さず終了コードも変わらない(root, monkeypatch, caplog, capsys, err, rc):
    class Provider:
        def fetch_racecards(self, con):
            raise err(MESSAGE)
    monkeypatch.setattr(cli, "_jra", lambda con, s: Provider())
    got, public = _main(["jra-cards"], caplog, capsys)
    assert got == rc
    assert_public(public)


def test_statusは失敗の内容を表示しない(root, caplog, capsys):
    con = connect()
    cli._sync_sources(con, config.load_sources())
    cli._record(con, "jra", False, MESSAGE)
    con.close()
    rc, public = _main(["status"], caplog, capsys)
    assert rc == 0
    assert_public(public)
    assert "'jra'" in public    # 取得元の行は残る


def test_日次処理の失敗一覧に詳細を出さない(root, monkeypatch, caplog, capsys):
    class Provider:
        def update_results(self, con, days):
            raise ValueError(MESSAGE)

        def fetch_racecards(self, con):
            raise AttributeError(MESSAGE)

        def update(self, con):
            raise KeyError(MESSAGE)
    monkeypatch.setattr(cli, "_jra", lambda con, s: Provider())
    monkeypatch.setattr(cli, "_nar", lambda con, s: Provider())
    con = connect()
    args = argparse.Namespace(no_fetch=False, retrain=False, org=None, days=1, odds=False)
    with caplog.at_level("INFO"):
        assert cmd_daily(args, con, config.load_sources()) == 2
    out = capsys.readouterr()
    public = caplog.text + out.out + out.err
    assert_public(public)
    assert "jra results: ValueError" in out.err and "jra cards: AttributeError" in out.err and "nar: KeyError" in out.err
    assert con.execute("SELECT COUNT(*) FROM fetch_log WHERE url LIKE 'stage:%'").fetchone()[0] == 3
    assert FRAGMENT in con.execute("SELECT error FROM fetch_log WHERE url='stage:results'").fetchone()[0]


def test_手動取り込みの失敗に詳細を出さない(root, monkeypatch, caplog, capsys):
    (config.INBOX_NAR / "bad.zip").write_bytes(b"x")
    monkeypatch.setattr(nar_mod, "import_zip", lambda con, p: (_ for _ in ()).throw(ValueError(MESSAGE)))
    con = connect()
    with caplog.at_level("INFO"):
        res = nar_mod.NarCsvProvider({}).import_inbox(con)
    out = capsys.readouterr()
    assert_public(json.dumps(res, ensure_ascii=False) + caplog.text + out.out + out.err)
    assert res == [{"file": "bad.zip", "error": "ValueError"}]
    assert FRAGMENT in con.execute("SELECT error FROM fetch_log WHERE url='stage:inbox'").fetchone()[0]


def test_中央の過去データの解析失敗に詳細を出さない(root, monkeypatch, caplog, capsys):
    class Parser:
        @staticmethod
        def parse_result(content, cname):
            raise ValueError(MESSAGE)

    class Client:
        def fetch(self, url, data=None, ref=None):
            return b""
    monkeypatch.setattr(jra_mod, "P", Parser)
    con = connect()
    stats = {"races": 0, "skipped": 0, "failed": 0, "odds": 0, "warnings": 0}
    with caplog.at_level("INFO"):
        jra_mod.JraProvider(Client())._ingest_past_race(con, "pw01sde1005202605010120261107Z/00", None, stats)
    out = capsys.readouterr()
    assert stats["failed"] == 1
    assert_public(caplog.text + out.out + out.err)
    assert "ValueError" in caplog.text and RACE in caplog.text
    assert FRAGMENT in con.execute("SELECT error FROM fetch_log WHERE url='parse'").fetchone()[0]


def _run(tmp_path, body):
    """本物の例外処理を確かめるため、別のプロセスで実行する。"""
    script = "import sys\nfrom keiba_ai import cli, config\n" + f"MESSAGE = {MESSAGE!r}\n" + textwrap.dedent(body)
    (tmp_path / "config").mkdir(exist_ok=True)
    shutil.copy(PROJECT / "config" / "sources.yaml", tmp_path / "config" / "sources.yaml")
    env = {**os.environ, "KEIBA_ROOT": str(tmp_path), "KEIBA_DB": str(tmp_path / "data" / "keiba.db")}
    return subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=120, cwd=tmp_path)


def test_想定外の例外はtracebackを出さず詳細をDBに残して終了コード1(tmp_path):
    r = _run(tmp_path, """
        def broken():
            raise ValueError(MESSAGE)
        config.load_sources = broken
        sys.exit(cli.main(["status"]))
    """)
    assert r.returncode == 1
    assert_public(r.stdout + r.stderr)
    assert "ValueError" in r.stderr
    con = sqlite3.connect(tmp_path / "data" / "keiba.db")
    rec = json.loads(con.execute("SELECT error FROM fetch_log WHERE url='stage:command:status'").fetchone()[0])
    assert FRAGMENT in rec["message"] and "Traceback" in rec["traceback"]


def test_DBに記録できない重大エラーでも詳細を出さず終了コード4(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "keiba.db").write_bytes(b"this is not a database" * 500)
    r = _run(tmp_path, """
        sys.exit(cli.main(["status"]))
    """)
    assert r.returncode == EXIT_FATAL
    assert_public(r.stdout + r.stderr)
    assert "DatabaseError" in r.stderr


def test_状態ファイルの復号失敗もtracebackを出さない(tmp_path):
    r = _run(tmp_path, f"""
        import os
        os.environ["KEIBA_STATE_KEY"] = "ab" * 32
        open({str(tmp_path / 'bad.enc')!r}, "wb").write(b"KEIBA1" + b"0" * 64)
        sys.exit(cli.main(["state", "unpack", {str(tmp_path / 'bad.enc')!r}]))
    """)
    assert r.returncode == 1
    assert_public(r.stdout + r.stderr)
    assert "InvalidTag" in r.stderr


def test_警告の文面とパスを出さない(tmp_path):
    r = _run(tmp_path, """
        import warnings
        cli.main(["env-check"])
        warnings.warn(f"internal {MESSAGE}", RuntimeWarning)
    """)
    assert_public(r.stdout + r.stderr)
    assert "RuntimeWarning" in r.stderr and str(tmp_path) not in r.stderr


def test_取得記録のURLとエラーを伏せる(con, monkeypatch):
    import requests
    from keiba_ai.http import PoliteClient
    from keiba_ai.redact import mask_url
    monkeypatch.setenv("KEIBA_SITE_PASSPHRASE", SECRET)
    assert mask_url("https://x.test/a?CNAME=pw01/AB&type=daily&k_year=2026&k_month=11") == "https://x.test/a?CNAME=pw01/AB&type=daily&k_year=2026&k_month=11"
    assert mask_url("https://x.test/a?type=daily&token=fake-query-789") == "https://x.test/a?type=daily&token=***"
    assert mask_url("https://x.test/a") == "https://x.test/a"

    class Session:
        headers = {}

        def get(self, url, **k):
            raise requests.ConnectionError(f"{QUERY} {HEADER} key={SECRET}")
    conf = {"enabled": True, "automated_fetch": True, "min_interval_sec": 0, "permission": {"status": "granted"}}
    client = PoliteClient("jra", conf, con, session=Session(), retries=1, sleep=lambda s: None)
    with pytest.raises(FetchFailed):
        client.fetch("https://example.test/data?type=daily&token=fake-query-789", ref="r1")
    row = con.execute("SELECT url,error,ok FROM fetch_log ORDER BY id DESC LIMIT 1").fetchone()
    assert row["ok"] == 0 and row["url"] == "https://example.test/data?type=daily&token=***"
    for word in (SECRET, "fake-query-789", "fake-header-token-456"):
        assert word not in row["url"] + row["error"]
    assert "ConnectionError" in row["error"]   # 調査に必要な内容は残す
