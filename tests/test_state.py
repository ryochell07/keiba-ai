"""暗号化バックアップの実手順を再現する。CLI を別プロセスの代わりに main() で通す。"""
import shutil

import pytest

from conftest import make_race
from keiba_ai import cli, config
from keiba_ai.db import connect
from keiba_ai.ingest import save_race

KEY = "ab" * 32


@pytest.fixture
def world(tmp_path, monkeypatch):
    src_config = config.ROOT / "config"
    (tmp_path / "config").mkdir()
    shutil.copy(src_config / "sources.yaml", tmp_path / "config" / "sources.yaml")
    monkeypatch.setenv("KEIBA_STATE_KEY", KEY)
    for name, val in (("ROOT", tmp_path), ("DATA", tmp_path / "data"), ("RAW", tmp_path / "data/raw"), ("INBOX_NAR", tmp_path / "data/inbox/nar"),
                      ("DB_PATH", tmp_path / "data/keiba.db"), ("MODELS", tmp_path / "data/models"), ("REPORTS", tmp_path / "data/reports")):
        monkeypatch.setattr(config, name, val)
    config.ensure_dirs()
    return tmp_path


def _fill(n=3):
    con = connect()
    for i in range(n):
        save_race(con, make_race(f"2024-01-{i + 1:02d}", seed=i))
    con.commit()
    con.close()


def _count():
    con = connect()
    try:
        return con.execute("SELECT COUNT(*) FROM race").fetchone()[0]
    finally:
        con.close()


def test_新しい環境で復元してもデータが残る(world):
    _fill()
    (config.REPORTS / "backtest_JRA.json").write_text("{}")
    enc = world / "s.enc"
    assert cli.main(["state", "pack", str(enc)]) == 0
    shutil.rmtree(config.DATA)
    config.ensure_dirs()
    assert cli.main(["state", "unpack", str(enc)]) == 0
    assert (config.REPORTS / "backtest_JRA.json").exists()
    assert _count() == 3


def test_古いWALが残っていても復元したDBを壊さない(world):
    _fill()
    enc = world / "s.enc"
    cli.main(["state", "pack", str(enc)])
    shutil.rmtree(config.DATA)
    config.ensure_dirs()
    keep = connect()          # 空の DB を開いたまま (WAL あり) で復元する
    keep.execute("INSERT INTO racecourse VALUES ('X','x','JRA')")
    keep.commit()
    cli.main(["state", "unpack", str(enc)])
    assert _count() == 3
    keep.close()


def test_復元後にレースが減ったら保存を拒否する(world):
    _fill()
    enc = world / "s.enc"
    cli.main(["state", "pack", str(enc)])
    cli.main(["state", "unpack", str(enc)])
    con = connect()
    con.execute("PRAGMA foreign_keys=OFF")
    for t in ("result", "payout", "entry", "race"):
        con.execute(f"DELETE FROM {t}")
    con.commit()
    con.close()
    with pytest.raises(SystemExit, match="減って"):
        cli.main(["state", "pack", str(enc)])


def test_鍵が違うと復元できない(world, monkeypatch):
    from cryptography.exceptions import InvalidTag
    _fill()
    enc = world / "s.enc"
    cli.main(["state", "pack", str(enc)])
    monkeypatch.setenv("KEIBA_STATE_KEY", "cd" * 32)
    with pytest.raises(InvalidTag):
        cli.main(["state", "unpack", str(enc)])
