import numpy as np
import pytest

from keiba_ai.db import connect
from keiba_ai.ingest import save_race
from keiba_ai.providers.base import EntryRec, PayoutRec, RaceRec


def make_race(date, course="05", no=1, n=8, seed=0, finished=True, org="JRA"):
    """検証用の架空レース。実在データではない。"""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n) + 1
    race = RaceRec(org=org, course_code=course, course_name={"05": "東京", "06": "中山"}.get(course, course), race_date=date, race_no=no, source="test",
                   post_time="15:00", surface="芝", distance=1600, direction="左", weather="晴", going="良", class_name="オープン")
    for u in range(1, n + 1):
        pos = int(order[u - 1]) if finished else None
        race.entries.append(EntryRec(
            umaban=u, horse_id=f"T{u:04d}", horse_name=f"テスト馬{u}", waku=(u + 1) // 2, sex="牡", age=4, carried_weight=56.0,
            jockey_id=f"JK{u % 4}", jockey_name=f"騎手{u % 4}", trainer_id=f"TR{u % 3}", trainer_name=f"調教師{u % 3}",
            body_weight=470.0 + u, body_weight_diff=0.0, finish_pos=pos, status="finished" if finished else None,
            time_sec=96.0 + (pos or 0) * 0.2 if finished else None, last3f=34.0 + (pos or 0) * 0.1 if finished else None,
            corners=f"{u},{u}" if finished else None, popularity_final=u if finished else None))
    if finished:
        w = int(np.argmin(order)) + 1
        race.payouts = [PayoutRec("win", str(w), 500, 3)]
    return race


@pytest.fixture
def con():
    c = connect(":memory:")
    yield c
    c.close()


@pytest.fixture
def filled(con):
    for i in range(12):
        save_race(con, make_race(f"2024-{i % 12 + 1:02d}-15", seed=i))
    save_race(con, make_race("2025-01-15", seed=99, finished=False))
    con.commit()
    return con
