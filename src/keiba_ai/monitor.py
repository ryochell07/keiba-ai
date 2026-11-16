"""発走前オッズの取得状況を日次で集計する。取れなかった分を別の値で埋めることはしない。"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import numpy as np

from .db import now_jst


def odds_quality(con: sqlite3.Connection, day: str, org: str, now: datetime | None = None) -> dict:
    """発走済みのレースだけを母数にする。オッズ 0 は発売前か票なしなので取得できた数に含めない。"""
    now = now or datetime.fromisoformat(now_jst())
    races = con.execute("""SELECT r.race_id, r.post_time, (SELECT COUNT(*) FROM entry e LEFT JOIN result x ON x.race_id=e.race_id AND x.umaban=e.umaban
                                  WHERE e.race_id=r.race_id AND COALESCE(x.status,'') NOT IN ('scratched','excluded')) n
                           FROM race r WHERE r.org=? AND r.race_date=? AND r.post_time IS NOT NULL AND r.course_code NOT LIKE 'N帯広%'""", (org, day)).fetchall()
    any_, t30, t10, last_, delays, missing, snaps = 0, 0, 0, 0, [], [], 0
    races = [r for r in races if r["n"] > 0 and datetime.fromisoformat(f"{day}T{r['post_time']}:00+09:00") <= now]
    for r in races:
        post = datetime.fromisoformat(f"{day}T{r['post_time']}:00+09:00")
        # 取消・除外を除いた馬番を集合で検証する。単なる件数一致では、
        # 一部の馬が欠け、取消馬のオッズが残るケースを検出できない。
        active = {
            str(row[0]) for row in con.execute(
                """SELECT e.umaban FROM entry e LEFT JOIN result x
                   ON x.race_id=e.race_id AND x.umaban=e.umaban
                   WHERE e.race_id=? AND COALESCE(x.status,'') NOT IN ('scratched','excluded')""",
                (r["race_id"],),
            )
        }
        snapshots: dict[datetime, set[str]] = {}
        for row in con.execute(
            """SELECT captured_at, combo, odds FROM odds_snapshot
               WHERE race_id=? AND bet_type='win' AND is_final=0
                 AND captured_at<>'final'""",
            (r["race_id"],),
        ):
            try:
                stamp = datetime.fromisoformat(row["captured_at"])
                if stamp.tzinfo is None or stamp.utcoffset() is None or stamp >= post:
                    continue
            except (TypeError, ValueError, OverflowError):
                continue
            positive = snapshots.setdefault(stamp, set())
            if row["odds"] is not None and np.isfinite(row["odds"]) and row["odds"] > 0:
                positive.add(str(row["combo"]))
        if not active:
            missing.append(1.0)
            continue
        times = sorted(stamp for stamp, positive in snapshots.items() if active == positive)
        snaps += len(times)
        latest = max(snapshots) if snapshots else None
        # 欠損率は最後の発走前スナップショットにおける現役出走馬の欠損割合。
        missing.append(1 - len(active & snapshots[latest]) / len(active) if latest else 1.0)
        if not times:
            continue
        any_ += 1
        # その時点で全出走馬のオッズがそろう最新値が、許容の古さに収まっているか
        def fresh(minutes: int, max_age: int) -> tuple[bool, float | None]:
            limit = post - timedelta(minutes=minutes)
            ok = [t for t in times if t <= limit]
            if not ok:
                return False, None
            age = (limit - ok[-1]).total_seconds() / 60
            return age <= max_age, age
        t30 += fresh(30, 15)[0]
        hit, age = fresh(10, 10)
        t10 += hit
        if age is not None:
            delays.append(age)
        last_ += fresh(0, 5)[0]
    n = len(races)
    src = "jra" if org == "JRA" else "nar_csv"
    errors = con.execute("SELECT COUNT(*) FROM fetch_log WHERE source=? AND ok=0 AND substr(fetched_at,1,10)=?", (src, day)).fetchone()[0]
    res = {"race_date": day, "org": org, "n_races": n, "any_rate": any_ / n if n else None, "t30_rate": t30 / n if n else None, "t10_rate": t10 / n if n else None, "last_rate": last_ / n if n else None,
           "median_delay_t10_min": float(np.median(delays)) if delays else None, "p90_delay_t10_min": float(np.percentile(delays, 90)) if delays else None,
           "missing_rate": float(np.mean(missing)) if missing else None, "n_errors": errors, "n_snapshots": snaps, "updated_at": now_jst()}
    cols = list(res)
    con.execute(f"INSERT OR REPLACE INTO odds_capture_daily({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", [res[c] for c in cols])
    con.commit()
    return res
