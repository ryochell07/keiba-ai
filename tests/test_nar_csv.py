"""説明書 (2026/9/16版) の列定義と例から作った架空データで検証する。実物 ZIP での検証は別途必要。"""
import csv
import io
import zipfile

import pytest

from keiba_ai.providers.nar_csv import (SpecChanged, decode_csv, import_zip, load_columns, parse_nar_time, parse_odds_file)

RACE_HEAD = ["競馬場", "競走年月日", "レース番号", "発走時刻", "競走種類名称", "レース名"] + [f"副賞名{i}" for i in range(1, 16)] + [
    "芝ダート区分", "回り", "距離", "天候", "馬場", "頭数", "条件", "1着賞金(円)", "2着賞金(円)", "3着賞金(円)", "4着賞金(円)", "5着賞金(円)", "上がり4F", "上がり3F"
] + [f"ハロンタイム{i}" for i in range(1, 16)] + [f"コーナー名称{i}" for i in range(1, 9)] + [f"コーナー通過順{i}" for i in range(1, 9)]
HORSE_HEAD = ["競馬場", "競走年月日", "レース番号", "枠番", "帽色", "馬番", "馬名", "性", "齢", "毛色", "生年月日", "父馬名", "母馬名", "母父馬名", "騎手名", "騎手所属", "負担重量", "騎手成績", "調教師", "調教師所属", "馬主氏名", "生産牧場名", "馬体重", "馬体重増減", "全成績", "ダート左成績", "ダート右成績", "当競馬場成績", "うち当距離成績", "最高タイム", "最高タイム良馬場", "着順", "タイム", "着差", "上がり3F", "人気"]
ODDS_HEAD = ["競馬場", "競走年月日", "レース番号", "賭式", "番号1", "番号2", "番号3", "オッズ", "オッズ（最大）", "人気"]
PAY = ["競馬場", "競走年月日", "レース番号", "レース名", "単勝組番", "単勝払戻金（円）", "単勝人気", "複勝組番1", "複勝払戻金1（円）", "複勝人気1", "複勝組番2", "複勝払戻金2（円）", "複勝人気2", "複勝組番3", "複勝払戻金3（円）", "複勝人気3", "枠複組番1", "枠複組番2", "枠複払戻金（円）", "枠複人気", "枠単組番1", "枠単組番2", "枠単払戻金（円）", "枠単人気", "馬複組番1", "馬複組番2", "馬複払戻金（円）", "馬複人気1", "馬単組番1", "馬単組番2", "馬単払戻金（円）", "馬単人気1", "ワイド組番1馬番1", "ワイド組番1馬番2", "ワイド払戻金1（円）", "ワイド人気1", "ワイド組番2馬番1", "ワイド組番2馬番2", "ワイド払戻金2（円）", "ワイド人気2", "ワイド組番3馬番1", "ワイド組番3馬番2", "ワイド払戻金3（円）", "ワイド人気3", "３連複組番馬番1", "３連複組番馬番2", "３連複組番馬番3", "３連複払戻金（円）", "３連複人気", "３連単組番馬番1", "３連単組番馬番2", "３連単組番馬番3", "３連単払戻金（円）", "３連単人気"]


def csv_text(head, rows):
    b = io.StringIO()
    w = csv.writer(b)
    w.writerow(head)
    w.writerows(rows)
    return b.getvalue()


def horse(u, finish):
    d = dict.fromkeys(HORSE_HEAD, "")
    d.update({"競馬場": "大井", "競走年月日": "20251229", "レース番号": "9", "枠番": str((u + 1) // 2), "馬番": str(u), "馬名": f"テストウマ{u}", "性": "牡", "齢": "5", "生年月日": "20200310",
              "騎手名": f"騎手{u}", "騎手所属": "大井", "負担重量": "57", "調教師": "調教師A", "馬体重": "496", "馬体重増減": "-8", "着順": finish, "タイム": "2043", "上がり3F": "38.2", "人気": str(u)})
    return [d[h] for h in HORSE_HEAD]


def race_row():
    d = dict.fromkeys(RACE_HEAD, "")
    d.update({"競馬場": "大井", "競走年月日": "20251229", "レース番号": "9", "発走時刻": "1540", "競走種類名称": "重賞", "レース名": "テスト大賞典", "芝ダート区分": "ダート", "回り": "右", "距離": "2000", "天候": "晴", "馬場": "重", "頭数": "4", "条件": "サラブレッド系 3歳以上 定量"})
    return [d[h] for h in RACE_HEAD]


def pay_row(win="1"):
    d = dict.fromkeys(PAY, "")
    d.update({"競馬場": "大井", "競走年月日": "20251229", "レース番号": "9", "単勝組番": win, "単勝払戻金（円）": "7730", "単勝人気": "7", "馬複組番1": "2", "馬複組番2": "1", "馬複払戻金（円）": "2870", "馬複人気1": "7",
              "３連単組番馬番1": "1", "３連単組番馬番2": "2", "３連単組番馬番3": "3", "３連単払戻金（円）": "81520", "３連単人気": "123"})
    return [d[h] for h in PAY]


def make_zip(path, encoding="utf-8-sig", racelist=None):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("202512_racelist.csv", (racelist or csv_text(RACE_HEAD, [race_row()])).encode(encoding))
        rows = [horse(1, "1"), horse(2, "2"), horse(3, "3"), horse(4, "中止")]
        z.writestr("202512_horselist.csv", csv_text(HORSE_HEAD, rows).encode(encoding))
        z.writestr("202512_payback.csv", csv_text(PAY, [pay_row(), pay_row("2")]).encode(encoding))
    return path


def test_ZIP取り込みと差分(con, tmp_path):
    z = make_zip(tmp_path / "202512_1766000000_race.zip")
    s = import_zip(con, z)
    assert s["races"] == 1
    rid = "N20251229N大井09"
    r = con.execute("SELECT * FROM race WHERE race_id=?", (rid,)).fetchone()
    assert (r["post_time"], r["distance"], r["surface"], r["going"]) == ("15:40", 2000, "ダート", "重")
    e = con.execute("SELECT * FROM entry JOIN result USING(race_id,umaban) WHERE race_id=? AND umaban=1", (rid,)).fetchone()
    assert e["time_sec"] == pytest.approx(124.3) and e["finish_pos"] == 1 and e["body_weight_diff"] == -8
    assert con.execute("SELECT status FROM result WHERE race_id=? AND umaban=4", (rid,)).fetchone()[0] == "dnf"
    assert import_zip(con, z)["skipped"] is True


def test_同着の払戻行と組番の正規化(con, tmp_path):
    import_zip(con, make_zip(tmp_path / "a.zip"))
    rows = {(r[0], r[1]) for r in con.execute("SELECT bet_type,combo FROM payout")}
    assert {("win", "1"), ("win", "2"), ("quinella", "1-2"), ("trifecta", "1-2-3")} <= rows


def test_文字コード判定():
    text = "競馬場,馬名\n大井,テスト\n"
    assert decode_csv(text.encode("utf-8-sig")) == text
    assert decode_csv(text.encode("cp932")) == text


def test_ShiftJISのZIPも読める(con, tmp_path):
    assert import_zip(con, make_zip(tmp_path / "b.zip", encoding="cp932"))["races"] == 1


def test_仕様変更を検出(con, tmp_path):
    bad = csv_text([h for h in RACE_HEAD if h != "距離"], [race_row()[:RACE_HEAD.index("距離")] + race_row()[RACE_HEAD.index("距離") + 1:]])
    with pytest.raises(SpecChanged, match="distance"):
        import_zip(con, make_zip(tmp_path / "c.zip", racelist=bad))


def test_タイム変換():
    assert parse_nar_time("2043") == pytest.approx(124.3)
    assert parse_nar_time("583") == pytest.approx(58.3)
    assert parse_nar_time("2:03.9") == pytest.approx(123.9)
    assert parse_nar_time("") is None


def test_オッズ変換():
    text = csv_text(ODDS_HEAD, [["川崎", "20260304", "7", "３連単", "10", "5", "1", "66.7", "", "12"], ["川崎", "20260304", "7", "馬複", "5", "2", "", "3.4", "", "1"], ["川崎", "20260304", "7", "複勝", "5", "", "", "1.2", "1.8", "1"]])
    r = parse_odds_file(text, load_columns())["N20260304N川崎07"]
    assert [(o.bet_type, o.combo, o.odds, o.odds_max) for o in r] == [("trifecta", "10-5-1", 66.7, None), ("quinella", "2-5", 3.4, None), ("place", "5", 1.2, 1.8)]


def test_実ファイルの帯広はばんえいとして学習対象外(con, tmp_path):
    from keiba_ai.features import load_frame
    text = csv_text(RACE_HEAD, [race_row()]).replace("大井", "帯広ば")
    z = make_zip(tmp_path / "d.zip", racelist=text)
    import zipfile as zf
    assert load_frame(con, "NAR").empty
    assert zf.is_zipfile(z)


def test_自動取得は許諾がなければ拒否(con):
    from keiba_ai.http import FetchBlocked, PoliteClient
    from keiba_ai.providers.nar_csv import NarCsvProvider

    class S:
        headers = {}

        def get(self, url, **kw):
            raise AssertionError("アクセスしてはいけない")

    conf = {"enabled": True, "automated_fetch": False, "permission": {"status": "none"}}
    with pytest.raises(FetchBlocked):
        NarCsvProvider(conf, PoliteClient("nar_csv", conf, con, session=S())).fetch_month(con, 2026, 10)


def test_確定済みの月は再取得しない(con, tmp_path):
    from keiba_ai.providers.nar_csv import NarCsvProvider
    raw = make_zip(tmp_path / "e.zip").read_bytes()
    calls = []

    class Client:
        def fetch(self, url, ref=None):
            calls.append(url)
            con.execute("INSERT INTO fetch_log(source,url,ref,fetched_at,ok) VALUES ('nar_csv',?,?,'2026-01-01T00:00:00',1)", (url, ref))
            return raw

    p = NarCsvProvider({}, Client())
    p.fetch_month(con, 2025, 12)
    p.fetch_month(con, 2025, 12)
    assert len(calls) == 1
    assert "type=monthly&k_year=2025&k_month=12" in calls[0]
