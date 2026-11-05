"""正規化レコードを DB に保存する (重複は主キーで排除)。整合性チェックもここに置く。"""
from __future__ import annotations

import sqlite3

from .db import now_jst, upsert
from .providers.base import OddsRec, RaceRec


class IntegrityError(ValueError):
    pass


def validate_race(r: RaceRec) -> list[str]:
    """保存前の整合性チェック。致命的なら IntegrityError、軽微なら警告を返す。"""
    warn: list[str] = []
    nums = [e.umaban for e in r.entries]
    if len(nums) != len(set(nums)):
        raise IntegrityError(f"{r.race_id}: 馬番が重複 {nums}")
    if not 1 <= r.race_no <= 12:
        raise IntegrityError(f"{r.race_id}: レース番号が不正")
    fin = sorted(e.finish_pos for e in r.entries if e.finish_pos)
    if fin and fin[0] != 1:
        warn.append(f"{r.race_id}: 1着馬なし")
    if r.distance is not None and not 100 <= r.distance <= 4500:
        raise IntegrityError(f"{r.race_id}: 距離が不正 {r.distance}")
    for e in r.entries:
        if e.waku is not None and not 1 <= e.waku <= 8:
            raise IntegrityError(f"{r.race_id}: 枠番が不正 {e.waku}")
    return warn


def save_race(con: sqlite3.Connection, r: RaceRec) -> list[str]:
    warn = validate_race(r)
    meeting_id = f"{r.org[0]}{r.race_date.replace('-', '')}{r.course_code}"
    upsert(con, "racecourse", {"course_code": r.course_code, "name": r.course_name, "org": r.org}, ["course_code"])
    upsert(con, "meeting", {"meeting_id": meeting_id, "org": r.org, "course_code": r.course_code,
                            "race_date": r.race_date, "kai": r.kai, "nichi": r.nichi}, ["meeting_id"])
    upsert(con, "race", {
        "race_id": r.race_id, "meeting_id": meeting_id, "org": r.org, "course_code": r.course_code,
        "race_date": r.race_date, "race_no": r.race_no, "post_time": r.post_time, "name": r.name,
        "grade": r.grade, "race_kind": r.race_kind, "category": r.category, "class_name": r.class_name,
        "rule": r.rule, "weight_rule": r.weight_rule, "conditions": r.conditions, "surface": r.surface,
        "distance": r.distance, "direction": r.direction, "weather": r.weather, "going": r.going,
        "n_runners": r.n_runners or len(r.entries) or None, "source": r.source, "source_ref": r.source_ref,
        "updated_at": now_jst()}, ["race_id"])
    for e in r.entries:
        upsert(con, "horse", {"horse_id": e.horse_id, "name": e.horse_name, "sex": e.sex, "birth_date": e.birth_date,
                              "coat": e.coat, "sire": e.sire, "dam": e.dam, "damsire": e.damsire,
                              "owner": e.owner, "breeder": e.breeder}, ["horse_id"])
        if e.jockey_id:
            upsert(con, "jockey", {"jockey_id": e.jockey_id, "name": e.jockey_name or e.jockey_id,
                                   "affiliation": e.jockey_affiliation}, ["jockey_id"])
        if e.trainer_id:
            upsert(con, "trainer", {"trainer_id": e.trainer_id, "name": e.trainer_name or e.trainer_id,
                                    "affiliation": e.trainer_affiliation}, ["trainer_id"])
        upsert(con, "entry", {"race_id": r.race_id, "umaban": e.umaban, "waku": e.waku, "horse_id": e.horse_id,
                              "sex": e.sex, "age": e.age, "carried_weight": e.carried_weight,
                              "jockey_id": e.jockey_id, "trainer_id": e.trainer_id,
                              "body_weight": e.body_weight, "body_weight_diff": e.body_weight_diff},
               ["race_id", "umaban"])
        if e.status:
            upsert(con, "result", {"race_id": r.race_id, "umaban": e.umaban, "finish_pos": e.finish_pos,
                                   "status": e.status, "time_sec": e.time_sec, "margin": e.margin,
                                   "last3f": e.last3f, "corners": e.corners,
                                   "popularity_final": e.popularity_final}, ["race_id", "umaban"])
    for p in r.payouts:
        upsert(con, "payout", {"race_id": r.race_id, "bet_type": p.bet_type, "combo": p.combo,
                               "amount": p.amount, "popularity": p.popularity}, ["race_id", "bet_type", "combo"])
    return warn


def save_odds(con: sqlite3.Connection, race_id: str, odds: list[OddsRec], captured_at: str,
              is_final: bool, source: str) -> int:
    rows = [(race_id, o.bet_type, o.combo, captured_at, o.odds, o.odds_max, o.popularity, int(is_final), source)
            for o in odds if o.odds is not None]
    con.executemany("INSERT OR IGNORE INTO odds_snapshot"
                    "(race_id,bet_type,combo,captured_at,odds,odds_max,popularity,is_final,source)"
                    " VALUES (?,?,?,?,?,?,?,?,?)", rows)
    return len(rows)
