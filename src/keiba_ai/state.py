"""DB とモデルを暗号化して持ち運ぶ。公開リポジトリのリリースに置いても中身は読めない。"""
from __future__ import annotations

import io
import os
import sqlite3
import tarfile
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from . import config

MAGIC = b"KEIBA1"


def _key() -> bytes:
    key = os.environ.get("KEIBA_STATE_KEY", "")
    if len(key) != 64:
        raise SystemExit("KEIBA_STATE_KEY に 64 桁の16進数を設定してください (openssl rand -hex 32)")
    return bytes.fromhex(key)


def _models_to_keep() -> list[Path]:
    """採用中と直前のモデルは日付に関係なく必ず残す (ロールバックのため)。"""
    con = sqlite3.connect(config.DB_PATH)
    keep = {r[0] for r in con.execute(
        "SELECT path FROM (SELECT path, ROW_NUMBER() OVER (PARTITION BY org ORDER BY adopted_at DESC) AS n "
        "FROM model_version WHERE adopted_at IS NOT NULL AND path IS NOT NULL) WHERE n<=3")}
    con.close()
    return [config.ROOT / p for p in keep if (config.ROOT / p).exists()]


def pack(path: str) -> str:
    snap = config.DB_PATH.with_suffix(".snapshot")
    src = sqlite3.connect(config.DB_PATH)
    dst = sqlite3.connect(snap)
    src.backup(dst)
    if dst.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise SystemExit("DB の整合性チェックに失敗したため保存を中止しました")
    dst.close()
    src.close()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        t.add(snap, arcname="keiba.db")
        if config.MODELS.exists():
            for m in _models_to_keep():   # 古いモデルは持ち運ばない
                t.add(m, arcname=f"models/{m.name}")
        if config.REPORTS.exists():
            for r in config.REPORTS.glob("*.json"):
                t.add(r, arcname=f"reports/{r.name}")
    snap.unlink()
    size_mb = buf.tell() / 1e6
    limit = config.load_sources()["limits"]["max_state_size_mb"]
    if size_mb > limit:  # 無料枠を守るガード
        raise SystemExit(f"状態ファイルが {size_mb:.0f}MB で上限 {limit}MB を超えました。無料枠保護のため停止します")
    nonce = os.urandom(12)
    Path(path).write_bytes(MAGIC + nonce + AESGCM(_key()).encrypt(nonce, buf.getvalue(), MAGIC))
    return f"packed {size_mb:.1f}MB -> {path}"


def unpack(path: str) -> str:
    raw = Path(path).read_bytes()
    if not raw.startswith(MAGIC):
        raise SystemExit("状態ファイルの形式が不正です")
    nonce, body = raw[6:18], raw[18:]
    plain = AESGCM(_key()).decrypt(nonce, body, MAGIC)
    config.ensure_dirs()
    with tarfile.open(fileobj=io.BytesIO(plain), mode="r:gz") as t:
        t.extractall(config.DATA / "_state", filter="data")
    (config.DATA / "_state" / "keiba.db").replace(config.DB_PATH)
    for m in (config.DATA / "_state" / "models").glob("*.pkl"):
        m.replace(config.MODELS / m.name)
    for r in (config.DATA / "_state" / "reports").glob("*.json"):
        r.replace(config.REPORTS / r.name)
    return f"unpacked {path}"
