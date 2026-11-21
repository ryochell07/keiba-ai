"""封鎖期間を含まない開発用DBを作る。

研究用スナップショットを読み取り専用で開き、封鎖日より前のレースとその出走・結果・オッズ・払戻だけを
新しいDBに写す。予測台帳・モデル・実験記録などは写さない。エージェントが読めるのはこのDBだけなので、
どんなコードを書いても封鎖期間のデータには届かない。
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

# 写すテーブル。レースに付く表はレースの絞り込みに従う
STATIC = ("racecourse", "horse", "jockey", "trainer")
BY_DATE = ("meeting", "race")
BY_RACE = ("entry", "result", "odds_snapshot", "payout")


def _columns(con: sqlite3.Connection, schema: str, table: str) -> list[str]:
    return [r[1] for r in con.execute(f"PRAGMA {schema}.table_info({table})")]


def build(snapshot: Path, dest: Path, sealed_from: str) -> dict:
    from keiba_ai.db import connect   # 本番と同じスキーマで作る

    if not snapshot.exists():
        raise FileNotFoundError(f"スナップショットがありません: {snapshot}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".building")
    for p in (tmp, Path(f"{tmp}-wal"), Path(f"{tmp}-shm")):
        p.unlink(missing_ok=True)
    connect(tmp).close()
    con = sqlite3.connect(f"file:{tmp}", uri=True)
    con.execute("PRAGMA foreign_keys = OFF")
    con.execute("ATTACH DATABASE ? AS src", (f"file:{snapshot}?immutable=1",))
    for t in STATIC + BY_DATE + BY_RACE:
        cols = [c for c in _columns(con, "main", t) if c in set(_columns(con, "src", t))]
        names = ",".join(cols)
        if t in STATIC:
            where = ""
        elif t in BY_DATE:
            where = " WHERE race_date < :sealed"
        else:
            where = " WHERE race_id IN (SELECT race_id FROM main.race)"
        con.execute(f"INSERT INTO main.{t} ({names}) SELECT {names} FROM src.{t}{where}", {"sealed": sealed_from})
    con.commit()
    con.execute("DETACH DATABASE src")
    manifest = verify(con, sealed_from)
    con.execute("PRAGMA journal_mode = DELETE")
    con.execute("VACUUM")
    con.close()
    st = snapshot.stat()
    manifest.update(snapshot=str(snapshot), snapshot_bytes=st.st_size, snapshot_mtime=st.st_mtime, sealed_from=sealed_from,
                    created_at=datetime.now().isoformat(timespec="seconds"))
    os.replace(tmp, dest)
    dest.chmod(0o444)
    manifest_path(dest).write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


def verify(con: sqlite3.Connection, sealed_from: str) -> dict:
    """封鎖期間のレースがなく、台帳などが空であることを確かめる。"""
    last = con.execute("SELECT MAX(race_date) FROM race").fetchone()[0]
    if last is None or last >= sealed_from:
        raise RuntimeError(f"開発用DBに封鎖期間のレースがあります: {last}")
    for t in BY_RACE:
        orphan = con.execute(f"SELECT COUNT(*) FROM {t} WHERE race_id NOT IN (SELECT race_id FROM race)").fetchone()[0]
        if orphan:
            raise RuntimeError(f"{t} にレースのない行があります: {orphan}")
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    kept = set(STATIC + BY_DATE + BY_RACE)
    filled = [t for t in tables if t not in kept and not t.startswith("sqlite_") and con.execute(f"SELECT 1 FROM {t} LIMIT 1").fetchone()]
    if filled:
        raise RuntimeError(f"写さないはずの表に行があります: {filled}")
    counts = {org: n for org, n in con.execute("SELECT org, COUNT(*) FROM race GROUP BY org")}
    return {"last_race_date": last, "races": counts}


def manifest_path(db: Path) -> Path:
    return db.with_name(db.name + ".manifest.json")


def ensure(snapshot: Path, dest: Path, sealed_from: str) -> dict:
    """なければ作る。あれば中身が封鎖日より前だけかを確かめる。"""
    m = manifest_path(dest)
    if not dest.exists() or not m.exists():
        return build(snapshot, dest, sealed_from)
    con = sqlite3.connect(f"file:{dest}?mode=ro", uri=True)
    try:
        verify(con, sealed_from)
    finally:
        con.close()
    return json.loads(m.read_text(encoding="utf-8"))
