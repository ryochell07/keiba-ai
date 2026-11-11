"""架空データで 学習→バックテスト→予測→期待値→暗号化ページ までを通す統合テスト。"""
import base64
import gzip
import json

import numpy as np
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from keiba_ai import config
from keiba_ai.backtest import run_backtest
from keiba_ai.ingest import save_odds, save_race
from keiba_ai.predict import predict_races
from keiba_ai.registry import retrain_with_gate
from keiba_ai.providers.base import EntryRec, OddsRec, PayoutRec, RaceRec
from keiba_ai.site import build_site, collect, encrypt_payload


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """馬ごとの潜在能力で着順が決まる世界。モデルが能力を過去走から学べるかを検証する。"""
    from keiba_ai.db import connect
    tmp = tmp_path_factory.mktemp("w")
    old = config.MODELS, config.REPORTS, config.ROOT
    config.MODELS, config.REPORTS = tmp / "models", tmp / "reports"
    config.ROOT = tmp
    (tmp / "reports").mkdir()
    (tmp / "web").mkdir()
    (tmp / "web" / "index.html").write_text("<html></html>")
    con = connect(tmp / "k.db")
    rng = np.random.default_rng(7)
    ability = rng.normal(0, 1.0, 400)
    day = np.datetime64("2023-01-07")
    for r in range(900):
        date = str(day + np.timedelta64(r // 6, "D"))
        n = int(rng.integers(8, 15))
        hs = rng.choice(400, n, replace=False)
        perf = ability[hs] + rng.gumbel(0, 1.0, n)
        order = np.argsort(-perf)
        rank = np.empty(n, int)
        rank[order] = np.arange(1, n + 1)
        race = RaceRec(org="JRA", course_code="05", course_name="東京", race_date=date, race_no=r % 6 + 1, source="t", post_time="15:00",
                       surface="芝", distance=1600, direction="左", weather="晴", going="良", class_name="オープン")
        for u, h in enumerate(hs, 1):
            race.entries.append(EntryRec(umaban=u, horse_id=f"H{h}", horse_name=f"架空{h}", waku=min(8, (u + 1) // 2), sex="牡", age=4, carried_weight=56.0,
                                         jockey_id=f"J{h % 9}", jockey_name="騎手", trainer_id=f"T{h % 7}", trainer_name="調教師", body_weight=470.0,
                                         finish_pos=int(rank[u - 1]), status="finished", time_sec=96 + rank[u - 1] * .2, last3f=34 + rank[u - 1] * .1,
                                         corners=f"{u},{u}", popularity_final=u))
        w = int(order[0]) + 1
        race.payouts = [PayoutRec("win", str(w), 500, 2), PayoutRec("quinella", "-".join(map(str, sorted((order[:2] + 1).tolist()))), 1500, 3)]
        save_race(con, race)
        save_odds(con, race.race_id, [OddsRec("win", str(u), float(n)) for u in range(1, n + 1)], "final", True, "t")
    future = RaceRec(org="JRA", course_code="05", course_name="東京", race_date="2030-01-05", race_no=1, source="t", post_time="15:00", surface="芝", distance=1600, direction="左", class_name="オープン")
    top = np.argsort(-ability)[:6]
    for u, h in enumerate(top, 1):
        future.entries.append(EntryRec(umaban=u, horse_id=f"H{h}", horse_name=f"架空{h}", waku=u, sex="牡", age=4, carried_weight=56.0, jockey_id=f"J{h % 9}", trainer_id=f"T{h % 7}"))
    save_race(con, future)
    save_odds(con, future.race_id, [OddsRec("win", "1", 3.0)], "2030-01-05T14:00:00+09:00", False, "t")
    con.commit()
    yield con, ability
    con.close()
    config.MODELS, config.REPORTS, config.ROOT = old


def test_バックテストと予測(world, monkeypatch):
    con, _ = world
    r = run_backtest(con, "JRA", "2023-05-01", step_months=2)
    s = r["summary"]
    uniform = float(np.log(11))
    assert s["win_logloss_model"] < uniform - 0.05, s    # 能力を学習できている
    assert s["top1_hit"] > 0.15
    assert {"top1", "ev_1.0"} <= {x["strategy"] for x in r["rows"]}
    assert all(x["odds_basis"] != "cutoff" for x in r["rows"])  # 発走前オッズがなければ cutoff 集計は出ない

    first = retrain_with_gate(con, "JRA", n_seeds=2)
    assert first["decision"] == "採用"
    assert retrain_with_gate(con, "JRA", n_seeds=2)["decision"] == "見送り"  # 新データなし
    out = predict_races(con, "JRA", "2030-01-01", at="2030-01-05T14:30:00+09:00")
    assert out["races"] == 1
    rows = con.execute("SELECT * FROM prediction_display WHERE race_id='J203001050501'").fetchall()
    bets = {x["bet_type"] for x in rows}
    assert {"win", "place", "quinella", "exacta", "wide", "trio", "trifecta"} <= bets
    assert "bracket_quinella" not in bets  # 6頭立て
    win1 = next(x for x in rows if x["bet_type"] == "win" and x["combo"] == "1")
    assert win1["ev"] == pytest.approx(win1["prob"] * 3.0 - 1)
    assert win1["status"] == "OK" and win1["odds_captured_at"].startswith("2030")
    win2 = next(x for x in rows if x["bet_type"] == "win" and x["combo"] == "2")
    assert win2["ev"] is None and win2["status"] == "オッズ未取得"
    flags = json.loads(win1["flags"])
    assert flags["confidence"] in "ABC"
    if win1["ev"] > 0:
        assert "長期回収率は未検証" in flags["flags"]
    # 取得時刻より前の時点では、そのオッズを使わない
    predict_races(con, "JRA", "2030-01-01", at="2030-01-05T13:00:00+09:00")
    row = con.execute("SELECT odds,status FROM prediction_display WHERE race_id='J203001050501' AND bet_type='win' AND combo='1' AND predicted_at LIKE '%13:00%'").fetchone()
    assert row["odds"] is None and row["status"] == "オッズ未取得"


def test_暗号化ページ(world, monkeypatch, tmp_path):
    con, _ = world
    monkeypatch.delenv("KEIBA_SITE_PASSPHRASE", raising=False)
    assert "省略" in build_site(con, out_dir=tmp_path)
    monkeypatch.setenv("KEIBA_SITE_PASSPHRASE", "test-passphrase")
    build_site(con, out_dir=tmp_path, today="2030-01-01")
    raw = (tmp_path / "data.enc.json").read_text()
    assert "架空" not in raw and "predictions" not in raw
    e = json.loads(raw)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=base64.b64decode(e["salt"]), iterations=e["iter"]).derive(b"test-passphrase")
    plain = json.loads(gzip.decompress(AESGCM(key).decrypt(base64.b64decode(e["iv"]), base64.b64decode(e["ct"]), None)))
    assert plain["races"][0]["id"] == "J203001050501" and plain["predictions"]
    # 市場単独にする主催はページの注意書きに使う
    assert plain["market_only_orgs"] == []
    monkeypatch.setattr(config, "guards", lambda: {"market_only_orgs": ["JRA"]})
    assert collect(con, "2030-01-01")["market_only_orgs"] == ["JRA"]


def test_暗号文は合言葉なしで読めない():
    from cryptography.exceptions import InvalidTag
    e = encrypt_payload({"a": 1}, "right-pass")
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=base64.b64decode(e["salt"]), iterations=e["iter"]).derive(b"wrong-pass")
    with pytest.raises(InvalidTag):
        AESGCM(key).decrypt(base64.b64decode(e["iv"]), base64.b64decode(e["ct"]), None)
