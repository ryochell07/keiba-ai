"""取得と予測の失敗を記録する。主催者やレースごとに分離して続行するが、データの整合性に関わるものは止める。"""
from __future__ import annotations

import json
import logging
import os
import sqlite3

from .db import now_jst
from .http import FetchBlocked, FetchFailed, FetchStop

log = logging.getLogger(__name__)
EXIT_FATAL = 4
SOURCE = {"JRA": "jra", "NAR": "nar_csv"}
_SECRET_ENV = ("KEIBA_STATE_KEY", "KEIBA_SITE_PASSPHRASE", "GH_TOKEN", "GITHUB_TOKEN")
_LEDGER_GUARDS = ("予測台帳", "入力の記録", "取得履歴", "モデル版", "旧予測")


class ModelIntegrityError(ValueError):
    """採用中のモデルファイルがない、または記録と一致しない。"""


def is_fatal(e: BaseException) -> bool:
    """続行するとデータを傷めるおそれがあるか。行単位の制約違反はそのレースだけの問題として扱う。"""
    if isinstance(e, sqlite3.IntegrityError):
        return any(g in str(e) for g in _LEDGER_GUARDS)
    return isinstance(e, (sqlite3.DatabaseError, ModelIntegrityError, MemoryError))


def failure_kind(e: BaseException, stage: str) -> str:
    if isinstance(e, (FetchFailed, FetchStop, FetchBlocked)):
        return "fetch"
    return "predict" if stage.startswith("predict") else "parse"


def clean_message(e: BaseException) -> str:
    """秘密の値が例外の文面に混ざっていても記録しない。"""
    text = str(e)
    for name in _SECRET_ENV:
        value = os.environ.get(name)
        if value:
            text = text.replace(value, "***")
    return text[:500]


def record_failure(con: sqlite3.Connection, org: str, stage: str, e: BaseException, race_id: str | None = None) -> dict:
    """呼び出し側は先に失敗した分を取り消しておくこと。ここで確定する。"""
    rec = {"org": org, "stage": stage, "kind": failure_kind(e, stage), "race_id": race_id, "type": type(e).__name__, "message": clean_message(e)}
    con.execute("INSERT INTO fetch_log(source,url,ref,fetched_at,http_status,bytes,attempts,ok,error) VALUES (?,?,?,?,NULL,0,1,0,?)",
                (SOURCE[org], f"stage:{stage}", race_id, now_jst(), json.dumps(rec, ensure_ascii=False)))
    con.commit()
    log.warning("失敗 %s", json.dumps(rec, ensure_ascii=False))
    return rec
