"""主催者やレースの失敗を分離できること、整合性に関わる失敗では止まること、環境の固定を確かめる。データはすべて架空。"""
import argparse
import json
import shutil
import sqlite3
from datetime import datetime, timedelta

import pytest

from keiba_ai import cli, commands, config, ledger, predict
from keiba_ai.commands import cmd_race_loop
from keiba_ai.db import JST, connect
from keiba_ai.envcheck import PROJECT, env_mismatches
from keiba_ai.failures import EXIT_FATAL, ModelIntegrityError, clean_message, failure_kind, is_fatal, record_failure
from keiba_ai.http import FetchFailed, FetchStop
from keiba_ai.ingest import save_race
from keiba_ai.predict import predict_races
from keiba_ai.providers import jra as jra_mod
from keiba_ai.providers.base import EntryRec, OddsRec, RaceRec
from keiba_ai.providers.jra_parser import OddsPage
from keiba_ai.providers.nar_csv import _per_race
from test_ledger import DAY, world  # noqa: F401  pytest のフィクスチャとして使う

SOURCES = {"sources": {"jra": {}, "nar_csv": {}}}
ARGS = argparse.Namespace(minutes=0, interval=0, window=65)


def _race(org, course, no, minutes):
    post = datetime.now(JST) + timedelta(minutes=minutes)
    r = RaceRec(org=org, course_code=course, course_name=course, race_date=post.date().isoformat(), race_no=no, source="t", post_time=post.strftime("%H:%M"), surface="ダ", distance=1400)
    for u in range(1, 4):
        r.entries.append(EntryRec(umaban=u, horse_id=f"{course}{no}H{u}", horse_name=f"架空{u}"))
    return r


@pytest.fixture
def today(con):
    for org, course in (("JRA", "05"), ("NAR", "N大井")):
        save_race(con, _race(org, course, 1, 30))
    con.commit()
    return con


class Calls:
    """捕まえた呼び出しを数える偽の取得処理。"""
    def __init__(self, error=None):
        self.error, self.n = error, 0

    def capture(self, con, *a, **k):
        self.n += 1
        if self.error:
            raise self.error
        return {"races": 1, "odds": 3}


def _patch(monkeypatch, jra, nar, predicted=None):
    monkeypatch.setattr(cli, "_jra", lambda con, s: type("P", (), {"capture_odds": jra.capture})())
    monkeypatch.setattr(cli, "_nar", lambda con, s: type("P", (), {"capture_today": nar.capture})())
    monkeypatch.setattr(predict, "predict_races", predicted or (lambda con, org: {"org": org, "failed": 0}))


def _failures(con):
    return [json.loads(r[0]) for r in con.execute("SELECT error FROM fetch_log WHERE ok=0 AND url LIKE 'stage:%' ORDER BY id")]


def test_中央の解析失敗でも地方を取得する(today, monkeypatch, capsys):
    jra, nar = Calls(AttributeError("想定外の表")), Calls()
    _patch(monkeypatch, jra, nar)
    assert cmd_race_loop(ARGS, today, SOURCES) == 1
    assert jra.n == 1 and nar.n == 1
    rec = _failures(today)
    assert len(rec) == 1
    assert rec[0]["org"] == "JRA" and rec[0]["stage"] == "capture" and rec[0]["kind"] == "parse" and rec[0]["type"] == "AttributeError" and "想定外" in rec[0]["message"]
    row = today.execute("SELECT source,fetched_at,ok FROM fetch_log WHERE url='stage:capture'").fetchone()
    assert row["source"] == "jra" and row["ok"] == 0 and row["fetched_at"]
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["org_failures"] == 1


def test_地方の失敗でも中央を取得し取得失敗として区別する(today, monkeypatch):
    jra, nar = Calls(), Calls(FetchFailed("接続できません"))
    _patch(monkeypatch, jra, nar)
    assert cmd_race_loop(ARGS, today, SOURCES) == 1
    assert jra.n == 1 and nar.n == 1
    assert _failures(today)[0]["org"] == "NAR" and _failures(today)[0]["kind"] == "fetch"


def test_中央の予測が失敗しても地方の予測を続ける(today, monkeypatch):
    done = []

    def predicted(con, org):
        if org == "JRA":
            raise KeyError("列がない")
        done.append(org)
        return {"org": org, "failed": 0}
    _patch(monkeypatch, Calls(), Calls(), predicted)
    assert cmd_race_loop(ARGS, today, SOURCES) == 1
    assert done == ["NAR"]
    assert _failures(today)[0]["kind"] == "predict"


def test_拒否された主催者にはその実行で再び要求しない(today, monkeypatch):
    jra, nar = Calls(FetchStop("HTTP 403")), Calls()
    _patch(monkeypatch, jra, nar)
    clock = iter(range(0, 10000, 100))
    monkeypatch.setattr(commands.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(commands.time, "sleep", lambda s: None)
    # 900 秒まで続けるので複数周する
    assert cmd_race_loop(argparse.Namespace(minutes=15, interval=0, window=65), today, SOURCES) == 1
    assert jra.n == 1 and nar.n > 1


def test_全て成功なら終了コード0(today, monkeypatch):
    _patch(monkeypatch, Calls(), Calls())
    assert cmd_race_loop(ARGS, today, SOURCES) == 0
    assert _failures(today) == []


class FakeClient:
    def fetch(self, url, data=None, ref=None):
        return b""


def test_中央の1レースの解析失敗で別のレースは保存する(con, monkeypatch):
    a, b = _race("JRA", "05", 1, 30), _race("JRA", "05", 2, 30)
    save_race(con, a)
    save_race(con, b)
    con.commit()
    ca, cb = _real_cname(a), _real_cname(b)
    assert jra_mod.race_id_from_cname(ca) == a.race_id and jra_mod.race_id_from_cname(cb) == b.race_id

    class Parser:
        links = staticmethod(lambda content, prefix: [ca, cb] if prefix == "pw151ou" else [])
        calls = 0

        @staticmethod
        def parse_odds(content):
            Parser.calls += 1
            if Parser.calls == 1:
                raise AttributeError("このレースだけ壊れた表")
            return OddsPage(False, [OddsRec("win", str(u), 2.0 + u) for u in range(1, 4)], {})
    monkeypatch.setattr(jra_mod, "P", Parser)
    stats = jra_mod.JraProvider(FakeClient()).capture_odds(con)
    assert stats["failed"] == 1 and stats["races"] == 1
    assert con.execute("SELECT COUNT(*) FROM odds_snapshot WHERE race_id=?", (b.race_id,)).fetchone()[0] == 3
    assert con.execute("SELECT COUNT(*) FROM odds_snapshot WHERE race_id=?", (a.race_id,)).fetchone()[0] == 0
    rec = _failures(con)[0]
    assert rec["race_id"] == a.race_id and rec["stage"] == "odds_race" and rec["kind"] == "parse" and rec["org"] == "JRA"


def _real_cname(r):
    # JRA の CNAME の並び: 場 年 回 日 R 日付
    return f"pw151ou10{r.course_code}{r.race_date[:4]}05{r.race_no:02d}{r.race_no:02d}{r.race_date.replace('-', '')}Z/00"


def test_中央の拒否はレース単位で続けず止める(con, monkeypatch):
    a = _race("JRA", "05", 1, 30)
    save_race(con, a)
    con.commit()
    ca = _real_cname(a)

    class Parser:
        links = staticmethod(lambda content, prefix: [ca] if prefix == "pw151ou" else [])
        parse_odds = staticmethod(lambda content: pytest.fail("拒否の後に解析してはいけない"))
    monkeypatch.setattr(jra_mod, "P", Parser)

    class Refuse(FakeClient):
        n = 0

        def fetch(self, url, data=None, ref=None):
            Refuse.n += 1
            if Refuse.n > 1:
                raise FetchStop("HTTP 429")
            return b""
    with pytest.raises(FetchStop):
        jra_mod.JraProvider(Refuse()).capture_odds(con)


def test_地方の1レースの失敗はそのレースだけ取り消す(con):
    a, b = _race("NAR", "N大井", 1, 30), _race("NAR", "N大井", 2, 30)
    save_race(con, a)
    save_race(con, b)
    con.commit()
    seen = []
    on_error = lambda stage, rid, e: seen.append((stage, rid, type(e).__name__))

    def broken():
        con.execute("INSERT INTO odds_snapshot(race_id,bet_type,combo,captured_at,odds,is_final,source) VALUES (?,?,?,?,?,0,'t')", (a.race_id, "win", "1", "t1", 2.0))
        raise ValueError("このレースだけ壊れた行")

    def fine():
        con.execute("INSERT INTO odds_snapshot(race_id,bet_type,combo,captured_at,odds,is_final,source) VALUES (?,?,?,?,?,0,'t')", (b.race_id, "win", "1", "t1", 2.0))
    _per_race(con, on_error, "odds_race", a.race_id, broken)
    _per_race(con, on_error, "odds_race", b.race_id, fine)
    con.commit()
    assert seen == [("odds_race", a.race_id, "ValueError")]
    assert con.execute("SELECT race_id FROM odds_snapshot").fetchall()[0][0] == b.race_id
    assert con.execute("SELECT COUNT(*) FROM odds_snapshot").fetchone()[0] == 1


def test_地方の1レースのDBエラーは止める(con):
    def broken():
        raise sqlite3.DatabaseError("database disk image is malformed")
    with pytest.raises(sqlite3.DatabaseError):
        _per_race(con, lambda *a: pytest.fail("記録して続けてはいけない"), "odds_race", "r", broken)


@pytest.mark.parametrize("err", [sqlite3.DatabaseError("database disk image is malformed"), sqlite3.OperationalError("no such table: race"), ModelIntegrityError("ハッシュ不一致")])
def test_整合性のエラーは続行せず止める(today, monkeypatch, err):
    jra, nar = Calls(err), Calls()
    _patch(monkeypatch, jra, nar)
    with pytest.raises(type(err)):
        cmd_race_loop(ARGS, today, SOURCES)
    assert nar.n == 0
    assert today.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_致命的かどうかの判定(con):
    assert is_fatal(sqlite3.DatabaseError("x")) and is_fatal(ModelIntegrityError("x")) and is_fatal(MemoryError())
    assert not is_fatal(AttributeError("x")) and not is_fatal(FetchFailed("x")) and not is_fatal(ValueError("x"))
    con.execute("CREATE TABLE t(a UNIQUE)")
    con.execute("INSERT INTO t VALUES (1)")
    with pytest.raises(sqlite3.IntegrityError) as row_conflict:
        con.execute("INSERT INTO t VALUES (1)")
    assert not is_fatal(row_conflict.value)   # 1行の衝突はそのレースだけの問題
    with pytest.raises(sqlite3.IntegrityError) as guard:
        con.execute("INSERT INTO prediction(model_version_id,race_id,bet_type,combo,predicted_at,prob) VALUES ('m','r','win','1','t',0.1)")
    assert is_fatal(guard.value)              # 台帳の保護に当たったら止める


def test_記録には秘密の値を残さない(con, monkeypatch):
    monkeypatch.setenv("KEIBA_SITE_PASSPHRASE", "dummy-site-word")
    monkeypatch.setenv("KEIBA_STATE_KEY", "ab" * 32)
    rec = record_failure(con, "NAR", "capture", ValueError(f"失敗 dummy-site-word {'ab' * 32}"))
    stored = con.execute("SELECT error FROM fetch_log ORDER BY id DESC LIMIT 1").fetchone()[0]
    assert "dummy-site-word" not in stored and "ab" * 32 not in stored and "dummy-site-word" not in rec["message"]
    assert clean_message(ValueError("x" * 900)) == "x" * 500
    assert failure_kind(ValueError(), "predict_race") == "predict" and failure_kind(ValueError(), "odds_race") == "parse"


# --- 予測は1レースごとに分離し、台帳を傷めない -------------------------------------------------
def _dump(con):
    return [tuple(r) for r in con.execute("SELECT * FROM prediction_run ORDER BY prediction_id")], [tuple(r) for r in con.execute("SELECT * FROM prediction_horse ORDER BY prediction_id, umaban")]


def test_1レースの予測失敗で別のレースは記録し台帳は壊れない(world, monkeypatch):  # noqa: F811
    con, races = world
    before = _dump(con)
    real, calls, failed_race = ledger.record_run, [], []

    def flaky(c, run, *a, **k):
        calls.append(run["race_id"])
        real(c, run, *a, **k)
        if len(calls) == 1:
            failed_race.append(run["race_id"])
            raise ValueError("記録の直後に失敗")
        return "pid"
    monkeypatch.setattr(ledger, "record_run", flaky)
    at = f"{DAY}T09:10:00+09:00"
    stats = predict_races(con, "JRA", DAY, at=at)
    assert stats["failed"] == 1 and stats["recorded"] == len(calls) - 1
    assert con.execute("SELECT COUNT(*) FROM prediction_run WHERE race_id=? AND predicted_at=?", (failed_race[0], at)).fetchone()[0] == 0
    rec = _failures(con)[-1]
    assert rec["race_id"] == failed_race[0] and rec["stage"] == "predict_race"
    after = _dump(con)
    assert set(before[0]) <= set(after[0]) and set(before[1]) <= set(after[1])   # 既存の行は変わらない
    assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_予測中のDBエラーは止まり確定済みの台帳は残る(world, monkeypatch):  # noqa: F811
    con, _ = world
    before = _dump(con)

    def broken(*a, **k):
        raise sqlite3.DatabaseError("database disk image is malformed")
    monkeypatch.setattr(ledger, "record_run", broken)
    with pytest.raises(sqlite3.DatabaseError):
        predict_races(con, "JRA", DAY, at=f"{DAY}T09:20:00+09:00")
    assert _dump(con) == before


def test_採用中のモデルファイルがなければ止める(world):  # noqa: F811
    con, _ = world
    row = con.execute("SELECT path FROM model_version WHERE org='JRA' AND status='active'").fetchone()
    path = config.ROOT / row["path"]
    moved = path.with_suffix(".moved")
    shutil.move(path, moved)
    try:
        with pytest.raises(ModelIntegrityError):
            predict_races(con, "JRA", DAY, at=f"{DAY}T09:30:00+09:00")
    finally:
        shutil.move(moved, path)


# --- コマンド全体 -----------------------------------------------------------------------------
@pytest.fixture
def root(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    shutil.copy(PROJECT / "config" / "sources.yaml", tmp_path / "config" / "sources.yaml")
    for name, val in (("ROOT", tmp_path), ("DATA", tmp_path / "data"), ("RAW", tmp_path / "data/raw"), ("INBOX_NAR", tmp_path / "data/inbox/nar"),
                      ("DB_PATH", tmp_path / "data/keiba.db"), ("MODELS", tmp_path / "data/models"), ("REPORTS", tmp_path / "data/reports")):
        monkeypatch.setattr(config, name, val)
    config.ensure_dirs()
    return tmp_path


def test_壊れたDBは終了コード4で止まり中身を上書きしない(root, capsys):
    db = config.DB_PATH
    db.write_bytes(b"this is not a database" * 500)
    original = db.read_bytes()
    assert cli.main(["race-loop", "--minutes", "0"]) == EXIT_FATAL
    assert "整合性" in capsys.readouterr().err
    assert db.read_bytes() == original


def test_壊れたDBは状態ファイルへ保存しない(root, monkeypatch):
    from keiba_ai import state
    monkeypatch.setenv("KEIBA_STATE_KEY", "ab" * 32)
    con = connect()
    con.close()
    good = root / "good.enc"
    state.pack(str(good))
    kept = good.read_bytes()
    config.DB_PATH.write_bytes(b"this is not a database" * 500)
    for suffix in ("-wal", "-shm"):
        (config.DB_PATH.parent / f"{config.DB_PATH.name}{suffix}").unlink(missing_ok=True)
    with pytest.raises((sqlite3.DatabaseError, SystemExit)):
        state.pack(str(good))
    assert good.read_bytes() == kept


def test_環境の確認はDBを開かない(root, capsys):
    assert not config.DB_PATH.exists()
    cli.main(["env-check"])
    assert not config.DB_PATH.exists()


# --- 依存の固定 -------------------------------------------------------------------------------
def test_環境はロックと一致する():
    assert env_mismatches() == []


def test_ロックと違う環境を検知する(tmp_path):
    (tmp_path / "uv.lock").write_text('version = 1\n[[package]]\nname = "lightgbm"\nversion = "0.0.1"\n', encoding="utf-8")
    (tmp_path / ".python-version").write_text("3.0.0\n")
    bad = env_mismatches(tmp_path)
    assert any(x.startswith("lightgbm") for x in bad) and any(x.startswith("python") for x in bad)
    assert env_mismatches(tmp_path / "nothing") != []


def test_環境が違えば予測のコマンドを止める(root, monkeypatch, capsys):
    import keiba_ai.envcheck as env
    monkeypatch.setattr(env, "env_mismatches", lambda: ["lightgbm: 環境 0 / 固定 1"])
    assert cli.main(["race-loop", "--minutes", "0"]) == env.EXIT_ENV
    assert not config.DB_PATH.exists()


def test_ワークフローはロックファイルを使う():
    setup = (PROJECT / ".github/actions/setup/action.yml").read_text(encoding="utf-8")
    assert "uv sync --locked" in setup and "env-check" in setup and "uv pip install" not in setup
    raceday = (PROJECT / ".github/workflows/raceday.yml").read_text(encoding="utf-8")
    assert "--minutes 20" in raceday and "timeout-minutes: 55" in raceday
    assert (PROJECT / "uv.lock").exists() and (PROJECT / ".python-version").read_text().strip().startswith("3.12.")


def test_再学習は手動でだけ動き保存は明示したときだけ():
    import yaml
    flow = yaml.safe_load((PROJECT / ".github/workflows/retrain.yml").read_text(encoding="utf-8"))
    on = flow[True]   # YAML は on を真偽値として読む
    assert list(on) == ["workflow_dispatch"]
    assert on["workflow_dispatch"]["inputs"]["save"]["default"] is False
    assert on["workflow_dispatch"]["inputs"]["force"]["default"] is False
    steps = flow["jobs"]["run"]["steps"]
    assert [s.get("uses") for s in steps][:4] == ["actions/checkout@v4", "./.github/actions/guard", "./.github/actions/setup", "./.github/actions/pull-state"]
    assert steps[-1]["uses"] == "./.github/actions/push-state" and steps[-1]["if"] == "inputs.save && success()"
    assert flow["jobs"]["run"]["timeout-minutes"] <= 350   # 1 ジョブ 6 時間の上限に収める
    # 定期実行の再学習は止めたまま。中央だけ市場単独にする
    guards = yaml.safe_load((PROJECT / "config/sources.yaml").read_text(encoding="utf-8"))["guards"]
    assert guards["freeze_models"] is True and guards["blend_enforce"] is False and guards["market_only_orgs"] == ["JRA"]


# --- 中央の発売前ページ -----------------------------------------------------------------------
@pytest.mark.parametrize("cell", ["発売前", "----", "取消", ""])
def test_発売前や取消の表記はオッズとして登録しない(con, cell):
    from keiba_ai.ingest import save_odds
    from keiba_ai.providers import jra_parser as P
    html = f"""<div id="contentsBody"><div class="race_header">発売中</div><table class="tanpuku">
    <tr><td class="num">1</td><td class="odds_tan">{cell}</td><td class="odds_fuku">{cell}</td></tr>
    <tr><td class="num">2</td><td class="odds_tan">3.4</td><td class="odds_fuku"><span class="min">1.2</span><span class="max">1.5</span></td></tr></table></div>"""
    page = P.parse_odds(html.encode("cp932"))
    assert page.is_final is False
    assert sorted((o.bet_type, o.combo) for o in page.odds) == [("place", "2"), ("win", "2")]
    r = _race("JRA", "05", 1, 30)
    save_race(con, r)
    save_odds(con, r.race_id, page.odds, "2030-01-01T09:00:00+09:00", page.is_final, "t")
    assert con.execute("SELECT COUNT(*) FROM odds_snapshot WHERE race_id=? AND combo='1'", (r.race_id,)).fetchone()[0] == 0
