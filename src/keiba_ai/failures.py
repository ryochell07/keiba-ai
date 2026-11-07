"""取得と予測の失敗を記録する。主催者やレースごとに分離して続行するが、データの整合性に関わるものは止める。"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import traceback

from .db import now_jst
from .http import FetchBlocked, FetchFailed, FetchStop
from .redact import mask_text

log = logging.getLogger(__name__)
EXIT_FATAL = 4
SOURCE = {"JRA": "jra", "NAR": "nar_csv"}
_LEDGER_GUARDS = ("予測台帳", "入力の記録", "取得履歴", "モデル版", "旧予測")
_PUBLIC_RACE_ID = re.compile(r"[JN]\d{8}[0-9A-Za-z\u3040-\u30ff\u4e00-\u9fff]{1,16}")
_TRACE_CHARS = 2000


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
    return mask_text(str(e))[:500]


def public_fields(org: str | None, stage: str, e: BaseException, race_id: str | None = None) -> dict:
    """公開ログに出してよい項目だけ。例外の文面は含めない。"""
    return {"org": org or "-", "stage": stage, "kind": failure_kind(e, stage), "race_id": race_id if race_id and _PUBLIC_RACE_ID.fullmatch(race_id) else None, "type": type(e).__name__}


def record_failure(con: sqlite3.Connection, org: str | None, stage: str, e: BaseException, race_id: str | None = None) -> dict:
    """公開ログには識別情報だけを出し、詳細は暗号化される DB に残す。
    呼び出し側は先に失敗した分を取り消しておくこと。ここで確定する。"""
    public = public_fields(org, stage, e, race_id)
    log.warning("失敗 %s", json.dumps(public, ensure_ascii=False))
    trace = mask_text("".join(traceback.format_exception(type(e), e, e.__traceback__)))[-_TRACE_CHARS:]
    rec = {**public, "race_id": race_id, "message": clean_message(e), "traceback": trace}
    try:
        con.execute("INSERT INTO fetch_log(source,url,ref,fetched_at,http_status,bytes,attempts,ok,error) VALUES (?,?,?,?,NULL,0,1,0,?)",
                    (SOURCE.get(org, "keiba"), f"stage:{stage}", race_id, now_jst(), json.dumps(rec, ensure_ascii=False)))
        con.commit()
    except Exception as db_error:
        log.warning("失敗の記録に失敗 %s", json.dumps({"org": org or "-", "stage": stage, "type": type(db_error).__name__}, ensure_ascii=False))
        raise
    return rec
