"""機密の評価レポートを暗号化してGitHub Actionsから安全に受け渡す。

状態バックアップと異なる鍵（HKDFでドメイン分離）とランダムnonceを使用する。
暗号化済みファイルのみActionsのartifactとしてアップロードする。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from . import config
from .state import _key

MAGIC = b"KEIBA-EVAL1"
MAX_REPORT_BYTES = 20 * 1024 * 1024


def _report_key() -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"keiba-evaluation-artifact-v1",
        info=b"confidential-forward-evaluation-report",
    ).derive(_key())


def _private_new_file(path: Path, data: bytes) -> None:
    target = path.expanduser()
    if target.resolve().is_relative_to(config.ROOT.resolve()):
        raise ValueError("評価ファイルをリポジトリ内に作成できません")
    # ファイルがすでに存在する場合は失敗。symlinkも上書きしない。
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def encrypt_report(input_path: Path, output_path: Path) -> None:
    source = input_path.expanduser()
    if source.stat().st_size > MAX_REPORT_BYTES:
        raise ValueError("評価ファイルが上限を超えています")
    plaintext = source.read_bytes()
    nonce = os.urandom(12)
    encrypted = MAGIC + nonce + AESGCM(_report_key()).encrypt(nonce, plaintext, MAGIC)
    _private_new_file(output_path, encrypted)


def decrypt_report(input_path: Path, output_path: Path) -> None:
    encrypted = input_path.expanduser().read_bytes()
    if not encrypted.startswith(MAGIC) or len(encrypted) <= len(MAGIC) + 12 + 16:
        raise ValueError("評価ファイルの暗号形式が不正です")
    if len(encrypted) > MAX_REPORT_BYTES + 128:
        raise ValueError("暗号化された評価ファイルが上限を超えています")
    nonce = encrypted[len(MAGIC):len(MAGIC) + 12]
    # 誤鍵や改ざんはInvalidTagとして終了し、復号済みファイルは作成しない。
    plaintext = AESGCM(_report_key()).decrypt(nonce, encrypted[len(MAGIC) + 12:], MAGIC)
    _private_new_file(output_path, plaintext)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="評価結果のローカル暗号化・復号")
    parser.add_argument("task", choices=("encrypt", "decrypt"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.task == "encrypt":
        encrypt_report(args.input, args.output)
    else:
        decrypt_report(args.input, args.output)
    # レース成績・鍵・保存先のフルパスは公開ログに出さない。
    print("OK: confidential evaluation artifact processed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
