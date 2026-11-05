"""JRA 公式ページの HTML 解析。取得処理とは完全に分離している。"""
from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag

from .base import EntryRec, OddsRec, PayoutRec, RaceRec

COURSES = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
           "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
CNAME_RE = r"pw[0-9a-z]+?[0-9A-Za-z]*/[0-9A-F]{2}"
_RACE_CNAME = re.compile(r"(\d{2})(\d{4})(\d{2})(\d{2})(\d{2})(\d{8})(?=Z|/)")
BET_FROM_CLASS = {"win": "win", "place": "place", "wakuren": "bracket_quinella", "wide": "wide",
                  "umaren": "quinella", "umatan": "exacta", "trio": "trio", "tierce": "trifecta"}
ORDERED = {"exacta", "trifecta"}


def decode(content: bytes) -> str:
    return content.decode("cp932", errors="replace")


def soup(content: bytes | str) -> BeautifulSoup:
    return BeautifulSoup(decode(content) if isinstance(content, bytes) else content, "lxml")


def _num(text: str | None, cast=float):
    if text is None:
        return None
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", text)
    return cast(m.group(0).replace(",", "")) if m else None


def _text(node: Tag | None) -> str | None:
    return node.get_text(" ", strip=True) if node else None


def parse_time(text: str | None) -> float | None:
    if not text:
        return None
    m = re.fullmatch(r"(?:(\d+):)?(\d{1,2}\.\d)", text.strip())
    return int(m.group(1) or 0) * 60 + float(m.group(2)) if m else None


def combo_key(nums: list[int], bet_type: str) -> str:
    """組番の正規形。順序なし券種は昇順にそろえる。"""
    if bet_type not in ORDERED and bet_type != "bracket_exacta":
        nums = sorted(nums)
    return "-".join(str(n) for n in nums)


# --- 一覧ページ ---------------------------------------------------------------------
def parse_month_table(content: bytes) -> dict[str, str]:
    """過去レース検索ページの JS にある年月→チェック値の表を読む (サイト自身が公開している値)。"""
    return dict(re.findall(r'objParam\["(\d{4})"\]="([0-9A-F]{2})"', decode(content)))


def links(content: bytes, prefix: str) -> list[str]:
    found = re.findall(rf"{prefix}[0-9A-Za-z]*/[0-9A-F]{{2}}", decode(content))
    return sorted(set(found))


# --- 共通ヘッダ -----------------------------------------------------------------------
def _parse_header(root: Tag, cname: str | None) -> dict:
    h = root.find(class_="race_header")
    if h is None:
        raise ValueError("race_header が見つかりません (ページ構造の変更の可能性)")
    date_line = _text(h.find(class_="date")) or ""
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日.*?(\d+)回(\S+?)(\d+)日", date_line)
    if not m:
        raise ValueError(f"開催日を解析できません: {date_line!r}")
    y, mo, d, kai, course_name, nichi = m.groups()
    code = next((k for k, v in COURSES.items() if v == course_name), None)
    if code is None:
        raise ValueError(f"未知の競馬場: {course_name}")
    t = re.search(r"(\d{1,2})時(\d{2})分", _text(h.find(class_="time")) or "")
    alt = h.find(class_="race_number")
    no = _num(alt.find("img")["alt"], int) if alt is not None and alt.find("img") else None
    if no is None and cname:
        c = _RACE_CNAME.search(cname)
        no = int(c.group(5)) if c else None
    if no is None:
        raise ValueError("レース番号を解析できません")
    name_node = h.find(class_="race_name")
    grade_img = name_node.find("img") if name_node else None
    going, surface_cap = None, None
    baba = h.find(class_="baba")
    if baba:
        for cls in ("turf", "durt"):
            li = baba.find("li", class_=cls)
            if li:
                going, surface_cap = _text(li.find(class_="txt")), cls
    course = h.find(class_="course")
    detail = _text(course.find(class_="detail")) if course else ""
    surface = "芝" if "芝" in (detail or "") else ("ダート" if "ダート" in (detail or "") else ("障害" if "障害" in (detail or "") else None))
    if "障害" in (date_line + _text(h)):
        surface = "障害" if "障害" in (_text(h.find(class_="category")) or "") or "障害" in (detail or "") else surface
    direction = next((x for x in ("右", "左", "直") if x in (detail or "")), None)
    return dict(org="JRA", course_code=code, course_name=course_name,
                race_date=f"{y}-{int(mo):02d}-{int(d):02d}", race_no=no, source="jra", source_ref=cname,
                post_time=f"{int(t.group(1)):02d}:{t.group(2)}" if t else None,
                name=_clean_name(name_node), grade=grade_img.get("alt") if grade_img else None,
                category=_text(h.find(class_="category")), class_name=_text(h.find(class_="class")),
                rule=_text(h.find(class_="rule")), weight_rule=_text(h.find(class_="weight")),
                surface=surface, distance=_num(_text(course), int) if course else None, direction=direction,
                weather=_text(baba.find("li", class_="weather").find(class_="txt")) if baba and baba.find("li", class_="weather") else None,
                going=going, kai=int(kai), nichi=int(nichi))


def _clean_name(node: Tag | None) -> str | None:
    if node is None:
        return None
    return "".join(s for s in node.find_all(string=True, recursive=False)).strip() or _text(node)


def _id_after(tag: Tag | None, pattern: str) -> str | None:
    if tag is None:
        return None
    a = tag.find("a")
    if a is None:
        return None
    m = re.search(pattern, (a.get("href") or "") + (a.get("onclick") or ""))
    return m.group(1) if m else None


_HORSE = r"pw01dud\d(\d{10,11})/"
_JOCKEY = r"pw04kmk(\d+)/"
_TRAINER = r"pw05cmk(\d+)/"
_STATUS = {"中止": "dnf", "除外": "excluded", "取消": "scratched", "失格": "disq", "降着": "finished"}


def _horse_id(tag: Tag | None) -> str | None:
    h = _id_after(tag, _HORSE)
    return f"J{h[-10:]}" if h else None


def _body_weight(text: str | None) -> tuple[float | None, float | None]:
    m = re.search(r"(\d{3})\s*(?:kg)?\s*\(\s*([+\-−]?\d+|前計不|初出走)?\s*\)", text or "")
    if not m:
        m2 = re.search(r"\b(\d{3})\b", text or "")
        return (float(m2.group(1)), None) if m2 else (None, None)
    diff = m.group(2)
    return float(m.group(1)), float(diff.replace("−", "-")) if diff and re.fullmatch(r"[+\-−]?\d+", diff) else None


# --- レース結果 -----------------------------------------------------------------------
def parse_result(content: bytes, cname: str | None = None) -> RaceRec:
    root = soup(content).find(id="race_result")
    if root is None:
        raise ValueError("race_result が見つかりません")
    race = RaceRec(**_parse_header(root, cname))
    table = root.find("table")
    for tr in table.find("tbody").find_all("tr", recursive=False) if table.find("tbody") else table.find_all("tr"):
        place = tr.find("td", class_="place")
        num = tr.find("td", class_="num")
        if place is None or num is None:
            continue
        umaban = _num(_text(num), int)
        hid = _horse_id(tr.find("td", class_="horse"))
        if umaban is None or hid is None:
            continue
        ptext = _text(place) or ""
        finish = int(ptext) if ptext.isdigit() else None
        status = "finished" if finish else next((v for k, v in _STATUS.items() if k in ptext), "dnf")
        waku = tr.find("td", class_="waku")
        wimg = waku.find("img") if waku else None
        age = _text(tr.find("td", class_="age")) or ""
        bw, diff = _body_weight(_text(tr.find("td", class_="h_weight")))
        corner = [li.get_text(strip=True) for li in tr.select("td.corner li")]
        jockey, trainer = tr.find("td", class_="jockey"), tr.find("td", class_="trainer")
        race.entries.append(EntryRec(
            umaban=umaban, horse_id=hid, horse_name=_text(tr.find("td", class_="horse")),
            waku=_num(wimg.get("alt"), int) if wimg else None,
            sex=age[:1] or None, age=_num(age, int), carried_weight=_num(_text(tr.find("td", class_="weight"))),
            jockey_id=(f"J{i}" if (i := _id_after(jockey, _JOCKEY)) else None), jockey_name=_text(jockey),
            trainer_id=(f"J{i}" if (i := _id_after(trainer, _TRAINER)) else None), trainer_name=_text(trainer),
            body_weight=bw, body_weight_diff=diff, finish_pos=finish, status=status,
            time_sec=parse_time(_text(tr.find("td", class_="time"))), margin=_text(tr.find("td", class_="margin")) or None,
            last3f=_num(_text(tr.find("td", class_="f_time"))), corners=",".join(corner) or None,
            popularity_final=_num(_text(tr.find("td", class_="pop")), int)))
    race.n_runners = sum(1 for e in race.entries if e.status not in ("scratched", "excluded"))
    race.payouts = _parse_payouts(root)
    return race


def _parse_payouts(root: Tag) -> list[PayoutRec]:
    out: list[PayoutRec] = []
    area = root.find(class_="refund_area")
    if area is None:
        return out
    for li in area.select("ul > li"):
        cls = next((c for c in li.get("class", []) if c in BET_FROM_CLASS), None)
        if cls is None:
            continue
        bet = BET_FROM_CLASS[cls]
        for line in li.find_all("div", class_="line"):
            nums = [int(x) for x in re.findall(r"\d+", _text(line.find(class_="num")) or "")]
            amount = _num(_text(line.find(class_="yen")), int)
            if nums and amount is not None:
                out.append(PayoutRec(bet, combo_key(nums, bet), amount, _num(_text(line.find(class_="pop")), int)))
    return out


# --- 出馬表 ---------------------------------------------------------------------------
def parse_racecard(content: bytes, cname: str | None = None) -> RaceRec:
    """出馬表。枠順確定前は馬番が空なので entries が空になる。"""
    s = soup(content)
    root = s.find(id="contentsBody") or s
    race = RaceRec(**_parse_header(root, cname))
    for table in root.find_all("table"):
        head = _text(table.find("thead")) or ""
        if "馬名" not in head:
            continue
        for tr in table.find("tbody").find_all("tr", recursive=False):
            cells = tr.find_all("td", recursive=False)
            by = {c: next((td for td in cells if c in td.get("class", [])), None)
                  for c in ("waku", "num", "horse", "jockey", "odds", "weight")}
            umaban = _num(_text(by["num"]), int)
            horse = by["horse"]
            hid = _horse_id(horse.find(class_="name")) if horse else None
            if umaban is None or hid is None:
                continue  # 枠順未確定または取消行
            jockey = by["jockey"]
            age = _text(jockey.find(class_="age")) if jockey else ""
            wimg = by["waku"].find("img") if by["waku"] else None
            trainer = horse.find(class_="trainer")
            bw, diff = _body_weight(_text(horse.find(class_="weight")) or _text(by["weight"]))
            race.entries.append(EntryRec(
                umaban=umaban, horse_id=hid, horse_name=_text(horse.find(class_="name")),
                waku=_num(wimg.get("alt"), int) if wimg else _num(_text(by["waku"]), int),
                sex=(age or "")[:1] or None, age=_num(age, int),
                carried_weight=_num(_text(jockey.find(class_="weight"))) if jockey else None,
                jockey_id=(f"J{i}" if (i := _id_after(jockey.find(class_="jockey"), _JOCKEY)) else None) if jockey else None,
                jockey_name=_text(jockey.find(class_="jockey")) if jockey else None,
                trainer_id=(f"J{i}" if (i := _id_after(trainer, _TRAINER)) else None),
                trainer_name=(_text(trainer.find("a")) if trainer else None),
                trainer_affiliation=_text(trainer.find(class_="division")) if trainer else None,
                sire=_family(horse, "sire"), dam=_family(horse, "mare"),
                body_weight=bw, body_weight_diff=diff))
        break
    race.n_runners = len(race.entries) or None
    return race


def _family(horse: Tag, cls: str) -> str | None:
    li = horse.find("li", class_=cls)
    if li is None:
        return None
    return next((str(x).strip() for x in li.find_all(string=True, recursive=False) if str(x).strip()), None)


# --- オッズ ---------------------------------------------------------------------------
@dataclass
class OddsPage:
    is_final: bool
    odds: list[OddsRec]
    tabs: dict[str, str]        # bet page -> cname
    n_runners_note: str | None = None


def _cell_value(td: Tag) -> tuple[float | None, float | None]:
    mn, mx = td.find(class_="min"), td.find(class_="max")
    if mn is not None:
        return _odds_num(_text(mn)), _odds_num(_text(mx))
    return _odds_num(_text(td)), None


def _odds_num(text: str | None) -> float | None:
    if not text or not re.fullmatch(r"[\d,]+(\.\d+)?", text.strip()):
        return None
    return float(text.replace(",", ""))


def parse_odds(content: bytes) -> OddsPage:
    s = soup(content)
    body = s.find(id="contentsBody") or s
    text_head = _text(body.find(class_="race_header")) or ""
    is_final = "最終オッズ" in text_head
    html = decode(content)
    tabs = {}
    for prefix, name in (("pw151ou", "win_place"), ("pw153ou", "bracket_quinella"), ("pw154ou", "quinella"),
                         ("pw155ou", "wide"), ("pw156ou", "exacta"), ("pw157ou", "trio"), ("pw158ou", "trifecta")):
        m = re.search(rf"'({prefix}[0-9A-Za-z]*/[0-9A-F]{{2}})'\);\">[^<]*(?:単勝|枠連|馬連|ワイド|馬単|3連複|3連単)", html)
        if m:
            tabs[name] = m.group(1)
    out: list[OddsRec] = []
    t = body.find("table", class_="tanpuku")
    if t is not None:
        for tr in t.find_all("tr"):
            num, tan, fuku = tr.find("td", class_="num"), tr.find("td", class_="odds_tan"), tr.find("td", class_="odds_fuku")
            n = _num(_text(num), int)
            if n is None:
                continue
            if tan is not None and (v := _cell_value(tan)[0]) is not None:
                out.append(OddsRec("win", str(n), v))
            if fuku is not None:
                lo, hi = _cell_value(fuku)
                if lo is not None:
                    out.append(OddsRec("place", str(n), lo, hi))
    for cls, bet in (("waku", "bracket_quinella"), ("umaren", "quinella"), ("wide", "wide"), ("umatan", "exacta")):
        for tb in body.find_all("table", class_=cls):
            first = _num(_text(tb.find("caption")) or (tb.find("caption").find("img")["alt"] if tb.find("caption") and tb.find("caption").find("img") else None), int)
            if first is None and tb.find("caption") and tb.find("caption").find("img"):
                first = _num(tb.find("caption").find("img")["alt"], int)
            for tr in tb.find_all("tr"):
                th, td = tr.find("th"), tr.find("td")
                second = _num(_text(th), int)
                if first is None or second is None or td is None:
                    continue
                lo, hi = _cell_value(td)
                if lo is not None:
                    out.append(OddsRec(bet, combo_key([first, second], bet), lo, hi))
    for tb in body.find_all("table", class_="fuku3"):
        pair = re.findall(r"\d+", _text(tb.find("caption")) or "")
        if len(pair) != 2:
            continue
        for tr in tb.find_all("tr"):
            third, td = _num(_text(tr.find("th")), int), tr.find("td")
            if third is None or td is None:
                continue
            v = _cell_value(td)[0]
            if v is not None:
                out.append(OddsRec("trio", combo_key([int(pair[0]), int(pair[1]), third], "trio"), v))
    for unit in body.select("div.tan3_unit"):
        out.extend(_parse_tan3_unit(unit))
    return OddsPage(is_final=is_final, odds=out, tabs=tabs)


def _parse_tan3_unit(unit: Tag) -> list[OddsRec]:
    out: list[OddsRec] = []
    first = second = None
    for node in unit.descendants:
        if not isinstance(node, Tag):
            continue
        cls = node.get("class") or []
        if "p_line" in cls:
            txt = _text(node) or ""
            m1 = re.search(r"1着\s*(\d+)", txt)
            m2 = re.search(r"2着\s*(\d+)", txt)
            first = int(m1.group(1)) if m1 else first
            second = int(m2.group(1)) if m2 else second
        elif node.name == "table" and "tan3" in cls and first and second:
            for tr in node.find_all("tr"):
                third, td = _num(_text(tr.find("th")), int), tr.find("td")
                if third is None or td is None:
                    continue
                v = _cell_value(td)[0]
                if v is not None:
                    out.append(OddsRec("trifecta", f"{first}-{second}-{third}", v))
    return out
