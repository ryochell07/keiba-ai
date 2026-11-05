"""JRA ページ構造の解析テスト。権利の都合で実ページは保存せず、同じ構造の架空 HTML を使う。"""
from keiba_ai.providers import jra_parser as P
from keiba_ai.providers.jra import month_range, race_id_from_cname

HEADER = """<div class="race_header"><div class="date">2026年1月10日（土曜） 1回中山3日</div><div class="time">発走時刻：<strong>15時25分</strong></div>
<div class="cell baba"><ul><li class="weather"><span class="txt">晴</span></li><li class="durt"><span class="txt">稍重</span></li></ul></div>
<div class="race_number"><img alt="11レース"></div><span class="race_name">テストステークス<span class="grade_icon"><img alt="GⅢ"></span></span>
<div class="cell category">3歳以上</div><div class="cell class">オープン</div><div class="cell rule">（混合）</div><div class="cell weight">定量</div>
<div class="cell course">1,800<span class="detail">（ダート・右）</span></div></div>"""
ROW = """<tr><td class="place">{p}</td><td class="waku"><img alt="枠2黒"></td><td class="num">{u}</td>
<td class="horse"><a href="/JRADB/accessU.html?CNAME=pw01dud102020100001/30">テスト馬</a></td><td class="age">牡5</td><td class="weight">58.0</td>
<td class="jockey"><a href="#" onclick="return doAction('/JRADB/accessK.html', 'pw04kmk001141/B0');">騎手</a></td><td class="time">1:50.6</td><td class="margin"></td>
<td class="corner"><ul><li>5</li><li>3</li></ul></td><td class="f_time">36.5</td><td class="h_weight">458<span>(+2)</span></td>
<td class="trainer"><a href="#" onclick="return doAction('/JRADB/accessC.html', 'pw05cmk001121/1D');">調教師</a></td><td class="pop">2</td></tr>"""
RESULT = f"""<div id="race_result">{HEADER}<table><thead></thead><tbody>{ROW.format(p=1, u=3)}{ROW.format(p="取消", u=4).replace("00001", "00002")}</tbody></table>
<div class="refund_area"><ul><li class="win"><dl><dd><div class="line"><div class="num">3</div><div class="yen">1,080円</div><div class="pop">4番人気</div></div></dd></dl></li>
<li class="umaren"><dl><dd><div class="line"><div class="num">10-3</div><div class="yen">12,970円</div><div class="pop">24番人気</div></div></dd></dl></li>
<li class="tierce"><dl><dd><div class="line"><div class="num">17-10-3</div><div class="yen">297,820円</div><div class="pop">455番人気</div></div></dd></dl></li></ul></div></div>"""


def test_レース結果の解析():
    r = P.parse_result(RESULT.encode("cp932"))
    assert (r.race_id, r.grade, r.surface, r.distance, r.direction, r.going, r.post_time) == ("J202601100611", "GⅢ", "ダート", 1800, "右", "稍重", "15:25")
    a, b = r.entries
    assert (a.horse_id, a.waku, a.time_sec, a.body_weight, a.body_weight_diff, a.corners, a.finish_pos) == ("J2020100001", 2, 110.6, 458.0, 2.0, "5,3", 1)
    assert (b.finish_pos, b.status) == (None, "scratched")
    assert r.n_runners == 1
    assert {(p.bet_type, p.combo) for p in r.payouts} == {("win", "3"), ("quinella", "3-10"), ("trifecta", "17-10-3")}


def test_オッズの解析():
    html = f"""<div id="contentsBody"><div class="race_header">最終オッズ</div>
    <table class="tanpuku"><tr><td class="num">1</td><td class="odds_tan"><strong>2.3</strong></td><td class="odds_fuku"><span class="min">1.1</span><span class="max">1.3</span></td></tr>
    <tr><td class="num">2</td><td class="odds_tan">取消</td><td class="odds_fuku">-</td></tr></table>
    <table class="waku"><caption><img alt="枠1白"></caption><tr><th>2</th><td>11.8</td></tr></table>
    <table class="umatan"><caption>3</caption><tr><th>3</th><td> </td></tr><tr><th>1</th><td>700.1</td></tr></table>
    <table class="wide"><caption>1</caption><tr><th>2</th><td class="odds"><span class="min">4.3</span><span class="max">5.6</span></td></tr></table>
    <table class="fuku3"><caption>1-2</caption><tr><th>3</th><td>1463.1</td></tr></table>
    <div class="tan3_unit"><div class="p_line">1着 5</div><div class="p_line">2着 2</div><ul><li><table class="tan3"><caption><div class="p_line">3着</div></caption><tr><th>2</th><td> </td></tr><tr><th>7</th><td>7372.6</td></tr></table></li></ul></div></div>"""
    o = P.parse_odds(html.encode("cp932"))
    got = {(x.bet_type, x.combo): (x.odds, x.odds_max) for x in o.odds}
    assert o.is_final
    assert got == {("win", "1"): (2.3, None), ("place", "1"): (1.1, 1.3), ("bracket_quinella", "1-2"): (11.8, None), ("exacta", "3-1"): (700.1, None),
                   ("wide", "1-2"): (4.3, 5.6), ("trio", "1-2-3"): (1463.1, None), ("trifecta", "5-2-7"): (7372.6, None)}


def test_cnameからレースID():
    assert race_id_from_cname("pw151ou1005202604111120261101Z/83") == "J202611010511"
    assert race_id_from_cname("pw01sde1005202604111120261101/41") == "J202611010511"
    assert race_id_from_cname("pw01dde0103202603011120261107/3F") == "J202611070311"


def test_月範囲は新しい順():
    assert month_range("2025-11", "2026-02") == [(2026, 2), (2026, 1), (2025, 12), (2025, 11)]


def test_未知の構造はエラーにする():
    import pytest
    with pytest.raises(ValueError):
        P.parse_result(b"<html></html>")
