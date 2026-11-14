"""架空データで 学習→バックテスト→予測→期待値→暗号化ページ までを通す統合テスト。"""
import base64
import gzip
import json
from datetime import date, timedelta

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


def test_結果と答え合わせ(world):
    """過去の日は結果と払戻を載せ、予測は単勝・複勝と券種ごとの最上位だけに絞る。期待値プラスの買い目は返還を除いて集計する。"""
    con, _ = world
    rid, day = con.execute("SELECT race_id,race_date FROM race WHERE race_date<'2030-01-01' ORDER BY race_date DESC, race_no LIMIT 1").fetchone()
    fin = dict(con.execute("SELECT umaban,finish_pos FROM result WHERE race_id=?", (rid,)).fetchall())
    first, second, last = (next(u for u, f in fin.items() if f == k) for k in (1, 2, max(fin.values())))
    quinella = "-".join(map(str, sorted((first, second))))
    mid = con.execute("SELECT model_version_id FROM model_version WHERE status='active'").fetchone()[0]
    rows = [("win", str(first), .40, .2), ("win", str(second), .30, .5), ("win", str(last), .01, 1.0), ("place", str(first), .70, None),
            ("quinella", quinella, .10, .3), ("quinella", f"{min(first, last)}-{max(first, last)}", .02, 2.0)]
    try:
        con.execute("UPDATE result SET status='scratched', finish_pos=NULL WHERE race_id=? AND umaban=?", (rid, last))
        con.executemany("INSERT INTO prediction_display(race_id,bet_type,combo,model_version_id,predicted_at,prob,ev) VALUES (?,?,?,?,?,?,?)",
                        [(rid, b, c, mid, f"{day}T14:50:00+09:00", p, ev) for b, c, p, ev in rows])
        # 発走 (15:00) 後に作られた予測は、答え合わせにもページにも使わない
        con.execute("INSERT INTO prediction_display(race_id,bet_type,combo,model_version_id,predicted_at,prob,ev) VALUES (?,?,?,?,?,?,?)",
                    (rid, "exacta", f"{first}-{second}", mid, f"{day}T15:30:00+09:00", .5, 3.0))
        next_day = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
        out = collect(con, next_day)
        race = next(r for r in out["races"] if r["id"] == rid)
        assert {"u": first, "f": 1, "pop": first} in race["res"] and {"b": "win", "c": str(first), "a": 500} in race["pay"]
        assert next(x for x in race["res"] if x["u"] == last)["st"] == "scratched"
        # 単勝の勝ち馬と馬連が的中。取り消した馬を含む2点は返還
        chk = race["chk"]
        assert (chk["odds"], chk["n"], chk["ret"], chk["void"]) == (True, 3, 2000, 2)
        assert sorted(chk["hits"], key=lambda x: x["b"]) == [{"b": "quinella", "c": quinella, "a": 1500}, {"b": "win", "c": str(first), "a": 500}]
        kept = {(p["b"], p["c"]) for p in out["predictions"] if p["r"] == rid}
        assert kept == {("win", str(first)), ("win", str(second)), ("win", str(last)), ("place", str(first)), ("quinella", quinella)}
        assert all(r["date"] >= (date.fromisoformat(next_day) - timedelta(days=7)).isoformat() for r in out["races"])
        # 当日のレースは絞らない
        assert len([p for p in collect(con, day)["predictions"] if p["r"] == rid]) == len(rows)
    finally:
        con.rollback()


def test_暗号文は合言葉なしで読めない():
    from cryptography.exceptions import InvalidTag
    e = encrypt_payload({"a": 1}, "right-pass")
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=base64.b64decode(e["salt"]), iterations=e["iter"]).derive(b"wrong-pass")
    with pytest.raises(InvalidTag):
        AESGCM(key).decrypt(base64.b64decode(e["iv"]), base64.b64decode(e["ct"]), None)
