"""地方競馬 公式CSV Provider。ZIP 展開・文字コード判定・カラム統一・差分取り込みを行う。

変換処理はカラム定義 (config/nar_columns.yaml) に分離しているので、CSV 仕様変更時は定義の修正で済む。
"""
from __future__ import annotations

import csv
import hashlib
import io
import logging
import re
import unicodedata
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml
from charset_normalizer import from_bytes

from .. import config
from ..db import JST, now_jst
from ..http import FetchBlocked, FetchFailed, FetchStop, PoliteClient
from ..capture import exotics_due, win_place_due
from ..failures import is_fatal, record_failure
from ..ingest import save_odds, save_race
from .base import EntryRec, OddsRec, PayoutRec, Provider, RaceRec
from .jra import month_range
from .jra_parser import combo_key

log = logging.getLogger(__name__)
BASE = "https://www.keiba.go.jp/KeibaWeb/DataDownload"


class SpecChanged(ValueError):
    """CSV の必須カラムが見つからない = 仕様変更の可能性。"""


def load_columns() -> dict:
    with open(config.ROOT / "config" / "nar_columns.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def decode_csv(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp932"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    best = from_bytes(raw).best()
    if best is None:
        raise ValueError("文字コードを判定できません")
    return str(best)


def clean(value: str | None) -> str | None:
    """全角英数の統一と空白除去。馬名などの表記ゆれ対策。"""
    if value is None:
        return None
    value = unicodedata.normalize("NFKC", value).strip()
    return re.sub(r"\s+", " ", value) or None


def to_int(v: str | None) -> int | None:
    v = clean(v)
    if v is None:
        return None
    m = re.fullmatch(r"[+\-]?\d+", v.replace(",", ""))
    return int(m.group(0)) if m else None


def to_float(v: str | None) -> float | None:
    v = clean(v)
    if v is None:
        return None
    try:
        return float(v.replace(",", ""))
    except ValueError:
        return None


def to_date(v: str | None) -> str | None:
    v = clean(v)
    if not v or not re.fullmatch(r"\d{8}", v):
        return None
    return f"{v[:4]}-{v[4:6]}-{v[6:]}"


def parse_nar_time(v: str | None) -> float | None:
    """タイム列は 2043 = 2分04秒3。"""
    v = clean(v)
    if not v:
        return None
    if ":" in v:
        m = re.fullmatch(r"(\d+):(\d+\.\d)", v)
        return int(m.group(1)) * 60 + float(m.group(2)) if m else None
    if not v.isdigit() or len(v) < 3:
        return None
    return int(v[:-3] or 0) * 60 + int(v[-3:-1]) + int(v[-1]) / 10


def parse_corner_order(text: str | None) -> dict[int, int]:
    """コーナー通過順 "1,3,13,(2,7),4-5" を {馬番: 順位} にする。括弧内は同順位。"""
    text = clean(text)
    out: dict[int, int] = {}
    if not text:
        return out
    pos = 1
    for group, single in re.findall(r"\(([^)]*)\)|(\d+)", text):
        nums = [int(x) for x in re.findall(r"\d+", group)] if group else [int(single)]
        for n in nums:
            out.setdefault(n, pos)
        pos += len(nums)
    return out


def course_code(name: str) -> str:
    return f"N{clean(name)}"


def _reader(text: str, section: dict, filename: str) -> tuple[csv.DictReader, dict]:
    reader = csv.DictReader(io.StringIO(text))
    headers = {clean(h): h for h in reader.fieldnames or []}
    mapping, missing = {}, []
    for key, candidates in section.items():
        if key == "groups":
            continue
        hit = next((headers[clean(c)] for c in candidates if clean(c) in headers), None)
        if hit is None:
            missing.append(key)
        mapping[key] = hit
    if missing:
        raise SpecChanged(f"{filename}: カラムなし {missing}。config/nar_columns.yaml を確認してください")
    return reader, mapping


def parse_race_files(racelist: str, horselist: str, payback: str | None, columns: dict) -> list[RaceRec]:
    races: dict[tuple, RaceRec] = {}
    corner_orders: dict[tuple, list[dict[int, int]]] = {}
    reader, c = _reader(racelist, columns["racelist"], "racelist")
    for row in reader:
        rdate = to_date(row[c["date"]])
        no = to_int(row[c["race_no"]])
        course = clean(row[c["course"]])
        if not (rdate and no and course):
            continue
        t = clean(row[c["post_time"]]) or ""
        races[(course, rdate, no)] = RaceRec(
            org="NAR", course_code=course_code(course), course_name=course, race_date=rdate, race_no=no,
            source="nar_csv", post_time=f"{t[:-2].zfill(2)}:{t[-2:]}" if t.isdigit() and len(t) >= 3 else None,
            name=clean(row[c["name"]]), race_kind=clean(row[c["race_kind"]]), conditions=clean(row[c["conditions"]]),
            surface=clean(row[c["surface"]]), distance=to_int(row[c["distance"]]), direction=clean(row[c["direction"]]),
            weather=clean(row[c["weather"]]), going=clean(row[c["going"]]), n_runners=to_int(row[c["n_runners"]]))
        orders = [parse_corner_order(row.get(f"コーナー通過順{i}")) for i in range(1, 9)]
        corner_orders[(course, rdate, no)] = [o for o in orders if o]
    reader, c = _reader(horselist, columns["horselist"], "horselist")
    for row in reader:
        key = (clean(row[c["course"]]), to_date(row[c["date"]]), to_int(row[c["race_no"]]))
        race = races.get(key)
        umaban, name = to_int(row[c["umaban"]]), clean(row[c["horse_name"]])
        if race is None or umaban is None or not name:
            continue
        birth = to_date(row[c["birth_date"]])
        finish_txt = clean(row[c["finish"]])
        finish = to_int(finish_txt)
        status = None
        if finish:
            status = "finished"
        elif finish_txt:
            status = {"中": "dnf", "止": "dnf", "除": "excluded", "取": "scratched", "失": "disq"}.get(finish_txt[0], "dnf")
        jockey, trainer = clean(row[c["jockey"]]), clean(row[c["trainer"]])
        race.entries.append(EntryRec(
            umaban=umaban, horse_id=f"N{name}|{birth or ''}", horse_name=name, waku=to_int(row[c["waku"]]),
            sex=clean(row[c["sex"]]), age=to_int(row[c["age"]]), carried_weight=to_float(row[c["carried_weight"]]),
            jockey_id=f"N{jockey}" if jockey else None, jockey_name=jockey, jockey_affiliation=clean(row[c["jockey_aff"]]),
            trainer_id=f"N{trainer}" if trainer else None, trainer_name=trainer, trainer_affiliation=clean(row[c["trainer_aff"]]),
            body_weight=to_float(row[c["body_weight"]]), body_weight_diff=to_float(row[c["body_weight_diff"]]),
            birth_date=birth, coat=clean(row[c["coat"]]), sire=clean(row[c["sire"]]), dam=clean(row[c["dam"]]),
            damsire=clean(row[c["damsire"]]), owner=clean(row[c["owner"]]), breeder=clean(row[c["breeder"]]),
            finish_pos=finish, status=status, time_sec=parse_nar_time(row[c["time"]]), margin=clean(row[c["margin"]]),
            last3f=to_float(row[c["last3f"]]), popularity_final=to_int(row[c["popularity"]]),
            corners=",".join(str(o.get(umaban, "")) for o in corner_orders.get(key, [])) or None))
    for race in races.values():
        # 結果が出ているレースで着順が空欄の馬は、取消・除外とみなす (出走頭数に数えない)
        if any(e.finish_pos for e in race.entries):
            for e in race.entries:
                e.status = e.status or "scratched"
    if payback:
        reader, c = _reader(payback, {k: v for k, v in columns["payback"].items() if k != "groups"}, "payback")
        for row in reader:
            race = races.get((clean(row[c["course"]]), to_date(row[c["date"]]), to_int(row[c["race_no"]])))
            if race is None:
                continue
            row = {clean(k): v for k, v in row.items() if k}   # 説明書と実ファイルで全角半角が違うことがある
            for g in columns["payback"]["groups"]:
                nums = [to_int(row.get(clean(n))) for n in g["nums"]]
                amount = to_int(row.get(clean(g["amount"])))
                if amount is None or any(n is None for n in nums):
                    continue
                combo = combo_key(nums, g["bet"])
                if not any(p.bet_type == g["bet"] and p.combo == combo for p in race.payouts):  # 同着時の重複行
                    race.payouts.append(PayoutRec(g["bet"], combo, amount, to_int(row.get(clean(g["pop"])))))
    return list(races.values())


def parse_odds_file(text: str, columns: dict) -> dict[str, list[OddsRec]]:
    reader, c = _reader(text, columns["odds"], "odds")
    bet_map = {clean(k): v for k, v in columns["bet_types"].items()}
    out: dict[str, list[OddsRec]] = defaultdict(list)
    for row in reader:
        rdate, no, course = to_date(row[c["date"]]), to_int(row[c["race_no"]]), clean(row[c["course"]])
        bet = bet_map.get(clean(row[c["bet_type"]]) or "")
        odds = to_float(row[c["odds"]])
        if not (rdate and no and course and bet) or odds is None:
            continue
        nums = [n for n in (to_int(row[c[k]]) for k in ("n1", "n2", "n3")) if n is not None]
        if not nums:
            continue
        rid = f"N{rdate.replace('-', '')}{course_code(course)}{no:02d}"
        out[rid].append(OddsRec(bet, combo_key(nums, bet), odds, to_float(row[c["odds_max"]]), to_int(row[c["popularity"]])))
    return out


_ZIP_NAME = re.compile(r"(\d{6}|\d{8})_(\d{9,11})_(race|odds)")


def import_zip(con, path: Path, columns: dict | None = None, force: bool = False, odds_filter=None) -> dict:
    """公式ZIPのファイルを取り込む。"""
    return import_bytes(con, path.read_bytes(), path.name, columns, force, odds_filter)


def _per_race(con, on_error, stage: str, race_id: str, fn) -> None:
    """on_error があれば、1レースの失敗はそのレースだけ取り消して続ける。"""
    if on_error is None:
        fn()
        return
    con.execute("SAVEPOINT nar_race")
    try:
        fn()
    except Exception as e:
        con.execute("ROLLBACK TO nar_race")
        con.execute("RELEASE nar_race")
        if is_fatal(e):
            raise
        on_error(stage, race_id, e)
    else:
        con.execute("RELEASE nar_race")


def import_bytes(con, raw: bytes, name: str, columns: dict | None = None, force: bool = False, odds_filter=None, on_error=None) -> dict:
    """公式ZIPを取り込む。同一内容は再取り込みしない。odds_filter(race_id, rec) で保存するオッズを絞れる。"""
    columns = columns or load_columns()
    sha = hashlib.sha256(raw).hexdigest()
    if not force and con.execute("SELECT 1 FROM ingested_file WHERE sha256=?", (sha,)).fetchone():
        return {"file": name, "skipped": True}
    stats = {"file": name, "races": 0, "odds": 0, "warnings": []}
    m = _ZIP_NAME.search(name)
    timestamp = int(m.group(2)) if m else None
    is_daily = bool(m and len(m.group(1)) == 8)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = {n.rsplit("/", 1)[-1]: n for n in z.namelist() if n.lower().endswith(".csv")}
        read = lambda suffix: next((decode_csv(z.read(full)) for n, full in names.items() if n.endswith(suffix)), None)
        racelist, horselist = read("racelist.csv"), read("horselist.csv")
        if racelist and horselist:
            for race in parse_race_files(racelist, horselist, read("payback.csv"), columns):
                def save_one(race=race):
                    stats["warnings"] += save_race(con, race)
                    stats["races"] += 1
                _per_race(con, on_error, "card_race", race.race_id, save_one)
        known = {r[0] for r in con.execute("SELECT race_id FROM race WHERE org='NAR'")}
        missing = 0
        for n, full in names.items():
            if not n.endswith("odds.csv"):
                continue
            # 月次ファイルは確定オッズのみ。当日ファイルのみ取得時刻(ZIP名のUNIX時刻)を持つ中間オッズとして扱う。
            captured = datetime.fromtimestamp(timestamp, JST).isoformat(timespec="seconds") if is_daily and timestamp else "final"
            for rid, recs in parse_odds_file(decode_csv(z.read(full)), columns).items():
                if rid not in known:
                    missing += 1
                    continue
                def save_one(rid=rid, recs=recs):
                    kept = [o for o in recs if odds_filter(rid, o)] if odds_filter else recs
                    stats["odds"] += save_odds(con, rid, kept, captured, captured == "final", "nar_csv")
                _per_race(con, on_error, "odds_race", rid, save_one)
        if missing:
            stats["warnings"].append(f"レース情報のないオッズ {missing} レース分を保留")
    con.execute("INSERT OR REPLACE INTO ingested_file(sha256,source,filename,imported_at,rows,detail)"
                " VALUES (?,?,?,?,?,?)", (sha, "nar_csv", name, now_jst(), stats["races"] + stats["odds"],
                                          f"warnings={len(stats['warnings'])}"))
    con.commit()
    return stats


def _win_place(_rid: str, rec: OddsRec) -> bool:
    return rec.bet_type in ("win", "place")


class NarCsvProvider(Provider):
    name = "nar_csv"

    def __init__(self, conf: dict, client: PoliteClient | None = None):
        self.conf, self.client = conf, client

    def import_inbox(self, con, inbox: Path | None = None) -> list[dict]:
        """手動で置かれた ZIP を inbox から取り込む。レース情報を先に処理する。"""
        inbox = inbox or config.INBOX_NAR
        files = sorted(inbox.glob("*.zip"), key=lambda p: ("odds" in p.name, p.name))
        out = []
        for p in files:
            try:
                out.append(import_zip(con, p))
            except (SpecChanged, zipfile.BadZipFile, ValueError) as e:
                con.rollback()
                record_failure(con, "NAR", "inbox", e)
                out.append({"file": p.name, "error": type(e).__name__})
        return out

    def _download(self, kind: str, query: str, ref: str) -> bytes:
        """kind は Race または Odds。ダウンロード先は公式ページのボタンと同じ。"""
        if self.client is None:
            raise FetchBlocked("nar_csv: クライアント未設定")
        content = self.client.fetch(f"{BASE}/{kind}DataDownload?{query}", ref=ref)
        if content[:2] != b"PK":
            raise FetchFailed(f"{ref}: ZIP ではない応答を受信しました")
        return content

    def _fetched_ok(self, con, ref: str, on_day: str | None = None) -> bool:
        sql = "SELECT 1 FROM fetch_log WHERE source='nar_csv' AND ref=? AND ok=1"
        args = [ref]
        if on_day:
            sql += " AND fetched_at>=?"
            args.append(on_day)
        return con.execute(sql + " LIMIT 1", args).fetchone() is not None

    def fetch_month(self, con, year: int, month: int, kind: str = "Race", odds_filter=None, once_per_day_only: bool = False, refresh: bool = False) -> dict:
        """月次ファイル。確定済みの月は1回だけ、更新中の月は1日1回まで取得する (説明書FAQの目安)。"""
        ref = f"{year}{month:02d}{kind}"
        today = datetime.now(JST).date().isoformat()
        if self._fetched_ok(con, ref, today if (once_per_day_only or refresh) else None):
            return {"file": ref, "skipped": True}
        raw = self._download(kind, f"type=monthly&k_year={year}&k_month={month}", ref)
        name = f"{year}{month:02d}_{int(datetime.now(JST).timestamp())}_{kind.lower()}.zip"
        return import_bytes(con, raw, name, odds_filter=odds_filter, force=refresh)

    def backfill(self, con, start: str, end: str, odds_from: str | None = "2026-03", full_odds_from: str | None = None, refresh: bool = False) -> list[dict]:
        """過去の月次ファイルを古い順に取り込む。オッズは容量が大きいので、古い月は単勝・複勝だけ保存する。"""
        out = []
        for y, m in reversed(month_range(start, end)):
            ym = f"{y}-{m:02d}"
            this_month = ym == datetime.now(JST).strftime("%Y-%m")
            out.append(self.fetch_month(con, y, m, "Race", once_per_day_only=this_month, refresh=refresh))
            if odds_from and ym >= odds_from:
                full = bool(full_odds_from and ym >= full_odds_from)
                out.append(self.fetch_month(con, y, m, "Odds", odds_filter=None if full else _win_place,
                                            once_per_day_only=this_month))
            log.info("%s 完了: %s", ym, {k: v for k, v in out[-1].items() if k != "warnings"})
        return out

    def update(self, con) -> list[dict]:
        """前月と当月の確定分を更新する。翌日以降の出馬表も当月ファイルに含まれる。"""
        now = datetime.now(JST)
        prev_y, prev_m = (now.year - 1, 12) if now.month == 1 else (now.year, now.month - 1)
        out = [self.fetch_month(con, prev_y, prev_m, "Race", once_per_day_only=True),
               self.fetch_month(con, now.year, now.month, "Race", once_per_day_only=True)]
        out.append(self.fetch_month(con, now.year, now.month, "Odds", odds_filter=_win_place, once_per_day_only=True))
        return out

    def capture_today(self, con, exotics_within_min: int = 15) -> dict:
        """当日ファイルで最新の出馬表と発売中オッズを保存する。発走前のレースのオッズだけを残す。"""
        now = datetime.now(JST)
        ts = int(now.timestamp())
        day = now.strftime("%Y%m%d")
        failed = 0

        def on_error(stage: str, race_id: str | None, e: Exception) -> None:
            nonlocal failed
            failed += 1
            record_failure(con, "NAR", stage, e, race_id)

        def fetch_file(kind: str, stage: str, suffix: str, odds_filter=None) -> dict:
            """片方のファイルが失敗しても、もう片方は取り込む。"""
            try:
                return import_bytes(con, self._download(kind, "type=daily", f"daily-{suffix}-{ts}"), f"{day}_{ts}_{suffix}.zip", odds_filter=odds_filter, on_error=on_error)
            except (FetchStop, FetchBlocked):
                con.rollback()
                raise
            except Exception as e:
                con.rollback()
                if is_fatal(e):
                    raise
                on_error(stage, None, e)
                return {}

        race = fetch_file("Race", "card_file", "race")
        posts = {r[0]: r[1] for r in con.execute("SELECT race_id,post_time FROM race WHERE org='NAR' AND race_date=?", (now.date().isoformat(),))}

        due = {}

        def keep(rid: str, rec: OddsRec) -> bool:
            post = posts.get(rid)
            if not post:
                return False
            if rid not in due:
                mins = (datetime.fromisoformat(f"{now.date().isoformat()}T{post}:00+09:00") - now).total_seconds() / 60
                due[rid] = (win_place_due(con, rid, mins, now), exotics_due(con, rid, mins, now, exotics_within_min))
            return due[rid][0] if rec.bet_type in ("win", "place") else due[rid][1]

        odds = fetch_file("Odds", "odds_file", "odds", keep)
        return {"races": race.get("races", 0), "odds": odds.get("odds", 0), "skipped": bool(odds.get("skipped")), "failed": failed}

    def run(self, con, **kw) -> dict:
        return {"imported": self.import_inbox(con)}
