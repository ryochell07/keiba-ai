"""暗号化ReleaseのDBを本番に復元せず、隔離一時領域のみで評価する。架空データを使用。"""
import io
import json
import os
import sqlite3
import tarfile
import tempfile
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from keiba_ai import config
from keiba_ai.cli import main
from keiba_ai.state import MAGIC


def _make_encrypted_state(tmp_path: Path, key: bytes) -> Path:
    source = tmp_path / "fixture.sqlite"
    with sqlite3.connect(source) as con:
        con.executescript("""
            CREATE TABLE race (race_id TEXT, org TEXT, race_date TEXT, post_time TEXT);
            CREATE TABLE prediction_run (
                race_id TEXT, org TEXT, predicted_at TEXT, scheduled_post_at TEXT);
            CREATE TABLE result (race_id TEXT, umaban INTEGER, finish_pos INTEGER);
            CREATE TABLE odds_snapshot (race_id TEXT, bet_type TEXT, combo TEXT,
                                        captured_at TEXT, odds REAL, is_final INTEGER);
            INSERT INTO race VALUES ('r','NAR','2030-03-02','10:30');
        """)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        archive.add(source, arcname="keiba.db")
    nonce = bytes.fromhex("12" * 12)
    state = tmp_path / "keiba-state.enc"
    state.write_bytes(MAGIC + nonce + AESGCM(key).encrypt(nonce, buf.getvalue(), MAGIC))
    source.unlink()
    return state


def test_encrypted_release_is_evaluated_without_touching_production_or_leaving_plain_db(
        tmp_path, monkeypatch, capsys):
    key = bytes.fromhex("11" * 32)  # 架空テスト鍵
    state = _make_encrypted_state(tmp_path, key)
    raw_before = state.read_bytes()
    monkeypatch.setenv("KEIBA_STATE_KEY", key.hex())
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "production-must-not-be-created.db")
    monkeypatch.setattr(config, "REPORTS", tmp_path / "reports-must-not-be-created")
    original = tempfile.TemporaryDirectory
    paths = []

    def tmp_factory(*args, **kwargs):
        kwargs["dir"] = str(tmp_path)
        instance = original(*args, **kwargs)
        paths.append(Path(instance.name))
        return instance

    monkeypatch.setattr(tempfile, "TemporaryDirectory", tmp_factory)
    output_path = tmp_path / "private-results.json"
    assert main(["reconcile", "--org", "NAR", "--read-only",
                 "--encrypted-state", str(state), "--output", str(output_path)]) == 0
    assert capsys.readouterr().out == ""  # 公開Actionsログに集計を流さない
    output = output_path.read_text(encoding="utf-8")
    summary = json.loads(output)
    assert summary["org"] == "NAR"
    assert summary["n_races"] == 0
    assert key.hex() not in output
    assert (output_path.stat().st_mode & 0o777) == 0o600
    assert len(paths) == 1 and not paths[0].exists()
    assert not config.DB_PATH.exists()
    assert not config.REPORTS.exists()
    assert state.read_bytes() == raw_before


def test_encrypted_release_wrong_key_fails_closed_without_writing_database(
        tmp_path, monkeypatch):
    state = _make_encrypted_state(tmp_path, bytes.fromhex("11" * 32))
    monkeypatch.setenv("KEIBA_STATE_KEY", "22" * 32)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "must-not-exist.sqlite")
    with pytest.raises(ValueError, match="復号に失敗"):
        main(["reconcile", "--org", "NAR", "--read-only",
              "--encrypted-state", str(state), "--output", str(tmp_path / "private.json")])
    assert not config.DB_PATH.exists()


def test_encrypted_release_option_requires_readonly(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "must-not-exist.sqlite")
    with pytest.raises(SystemExit) as exc:
        main(["reconcile", "--encrypted-state", str(tmp_path / "any.enc")])
    assert exc.value.code == 2
    assert not config.DB_PATH.exists()


def test_encrypted_release_requires_explicit_private_output_path(tmp_path, monkeypatch):
    state = _make_encrypted_state(tmp_path, bytes.fromhex("11" * 32))
    monkeypatch.setenv("KEIBA_STATE_KEY", "11" * 32)
    with pytest.raises(SystemExit) as exc:
        main(["reconcile", "--read-only", "--org", "NAR",
              "--encrypted-state", str(state)])
    assert exc.value.code == 2
    assert not list(tmp_path.glob("*.json"))


def test_encrypted_release_refuses_repo_output_and_overwriting(tmp_path, monkeypatch):
    state = _make_encrypted_state(tmp_path, bytes.fromhex("11" * 32))
    monkeypatch.setenv("KEIBA_STATE_KEY", "11" * 32)
    private = tmp_path / "existing.json"
    private.write_text("ORIGINAL", encoding="utf-8")
    with pytest.raises(FileExistsError):
        main(["reconcile", "--read-only", "--org", "NAR", "--encrypted-state",
              str(state), "--output", str(private)])
    assert private.read_text(encoding="utf-8") == "ORIGINAL"
    unsafe = config.ROOT / "evaluation-result-must-not-be-created.json"
    with pytest.raises(ValueError, match="公開リポジトリ"):
        main(["reconcile", "--read-only", "--org", "NAR", "--encrypted-state",
              str(state), "--output", str(unsafe)])
    assert not unsafe.exists()
