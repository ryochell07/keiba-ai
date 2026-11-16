"""公開GitHub Actionsへ暗号化された評価集計だけを持ち出す仕組みの単体テスト。"""
import os

import pytest
from cryptography.exceptions import InvalidTag

from keiba_ai import config
from keiba_ai.eval_artifact import MAGIC, decrypt_report, encrypt_report, main


@pytest.fixture
def artifact_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_STATE_KEY", "ab" * 32)  # 架空のテスト鍵
    src = tmp_path / "confidential.json"
    src.write_text('{"org":"NAR","paired_scored":7}\n', encoding="utf-8")
    return src, tmp_path / "payload.enc", tmp_path / "result.json"


def test_artifact_encryption_roundtrip_with_restricted_permissions(artifact_fixture):
    src, cipher, result = artifact_fixture
    encrypt_report(src, cipher)
    assert cipher.read_bytes().startswith(MAGIC)
    assert b'paired_scored' not in cipher.read_bytes()
    assert cipher.stat().st_mode & 0o777 == 0o600
    decrypt_report(cipher, result)
    assert result.read_bytes() == src.read_bytes()
    assert result.stat().st_mode & 0o777 == 0o600


def test_artifact_encryption_nonce_is_randomized(artifact_fixture, tmp_path):
    src, cipher, _ = artifact_fixture
    second = tmp_path / "second.enc"
    encrypt_report(src, cipher)
    encrypt_report(src, second)
    assert cipher.read_bytes() != second.read_bytes()


def test_wrong_key_and_modified_ciphertext_fail_without_plaintext_file(artifact_fixture, monkeypatch):
    src, cipher, result = artifact_fixture
    encrypt_report(src, cipher)
    monkeypatch.setenv("KEIBA_STATE_KEY", "cd" * 32)
    with pytest.raises(InvalidTag):
        decrypt_report(cipher, result)
    assert not result.exists()

    monkeypatch.setenv("KEIBA_STATE_KEY", "ab" * 32)
    corrupt = cipher.parent / "corrupt.enc"
    blob = bytearray(cipher.read_bytes())
    blob[-1] ^= 1
    corrupt.write_bytes(blob)
    with pytest.raises(InvalidTag):
        decrypt_report(corrupt, result)
    assert not result.exists()


def test_cannot_overwrite_or_write_into_public_repo(artifact_fixture):
    src, cipher, result = artifact_fixture
    result.write_text("NEVER OVERWRITE", encoding="utf-8")
    encrypt_report(src, cipher)
    with pytest.raises(FileExistsError):
        decrypt_report(cipher, result)
    assert result.read_text() == "NEVER OVERWRITE"

    unsafe = config.ROOT / "never-create-secret-report.json"
    with pytest.raises(ValueError, match="リポジトリ内"):
        decrypt_report(cipher, unsafe)
    assert not unsafe.exists()


def test_eval_artifact_cli_does_not_echo_json_or_key(artifact_fixture, capsys):
    src, cipher, result = artifact_fixture
    assert main(["encrypt", "--input", str(src), "--output", str(cipher)]) == 0
    assert main(["decrypt", "--input", str(cipher), "--output", str(result)]) == 0
    output = capsys.readouterr().out
    assert "paired_scored" not in output
    assert "ab" * 32 not in output
    assert result.read_bytes() == src.read_bytes()
