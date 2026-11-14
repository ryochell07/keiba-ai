"""JRA 公式サイト Provider。サイト内リンクを順にたどるだけで、チェック値の総当たりはしない。"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta

from ..db import JST, now_jst
from ..failures import clean_message, is_fatal, public_fields, record_failure
from ..http import FetchBlocked, FetchFailed, FetchStop, PoliteClient, load_raw, save_raw
from ..capture import exotics_due, win_place_due
from ..ingest import IntegrityError, save_odds, save_race
from . import jra_parser as P
from .base import Provider, RaceRec

log = logging.getLogger(__name__)
BASE = "https://www.jra.go.jp/JRADB"
PAST_TOP = "pw01skl00999999/B3"
CARD_TOP = "pw01dli00/F3"
ODDS_TOP = "pw15oli00/6D"
RESULT_TOP = "pw01sli00/AF"
_ID = re.compile(r"(\d{2})(\d{4})(\d{2})(\d{2})(\d{2})(\d{8})(?=Z|/)")


def race_id_from_cname(cname: str) -> str | None:
    m = _ID.search(cname)
    return f"J{m.group(6)}{m.group(1)}{m.group(5)}" if m else None


def race_date_from_cname(cname: str) -> str | None:
    m = _ID.search(cname)
    return f"{m.group(6)[:4]}-{m.group(6)[4:6]}-{m.group(6)[6:]}" if m else None


def race_date_from_srl(cname: str) -> str | None:
    m = re.search(r"(\d{8})/", cname)
    return f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}" if m else None


class JraProvider(Provider):
    name = "jra"

    def __init__(self, client: PoliteClient):
        self.client = client
        self.recent_days: list[str] = []

    def _load_top(self) -> dict[str, str]:
        top = self._post("accessS.html", PAST_TOP)
        self.recent_days = P.links(top, "pw01srl10")
        return P.parse_month_table(top)

    def _post(self, page: str, cname: str) -> bytes:
        return self.client.fetch(f"{BASE}/{page}", data={"cname": cname}, ref=cname)

    def _get_result(self, cname: str, use_cache: bool = True) -> bytes:
        cached = load_raw("jra", cname) if use_cache else None
        if cached:
            return cached
        content = self.client.fetch(f"{BASE}/accessS.html?CNAME={cname}", ref=cname)
        save_raw("jra", cname, content)
        return content

    # --- 過去データ ----------------------------------------------------------------
    def month_race_cnames(self, table: dict[str, str], ym: tuple[int, int]) -> list[str]:
        y, m = ym
        check = table.get(f"{y % 100:02d}{m:02d}")
        if not check:
            log.warning("%d-%02d は検索対象外です", y, m)
            return []
        month_page = self._post("accessS.html", f"pw01skl10{y}{m:02d}/{check}")
        # 当月分は月別ページに載らないことがあるので、検索トップの最新リンクも併用する
        days = set(P.links(month_page, "pw01srl10"))
        days.update(c for c in self.recent_days if (race_date_from_srl(c) or "").startswith(f"{y}-{m:02d}"))
        cnames: list[str] = []
        for day in sorted(days):
            cnames.extend(P.links(self._post("accessS.html", day), "pw01sde10"))
        return sorted(set(cnames))

    def backfill(self, con, months: list[tuple[int, int]], all_odds_since: str | None = None, stop_on_empty=False) -> dict:
        """指定月のレース結果と最終オッズを取得。処理済みレースは飛ばすので中断・再開できる。"""
        table = self._load_top()
        stats = {"races": 0, "skipped": 0, "failed": 0, "odds": 0, "warnings": 0}
        this_month = datetime.now(JST).strftime("%Y%m")
        for ym in months:
            marker = f"jra-month-{ym[0]}{ym[1]:02d}"
            # 確定済みの月は完了の記録があれば一覧ページも取りに行かない
            if con.execute("SELECT 1 FROM ingested_file WHERE sha256=?", (marker,)).fetchone():
                continue
            failed_before = stats["failed"]
            cnames = self.month_race_cnames(table, ym)
            for cname in sorted(cnames, reverse=True):
                self._ingest_past_race(con, cname, all_odds_since, stats)
            if cnames and stats["failed"] == failed_before and f"{ym[0]}{ym[1]:02d}" < this_month:
                con.execute("INSERT OR REPLACE INTO ingested_file(sha256,source,filename,imported_at,rows) VALUES (?,?,?,?,?)",
                            (marker, "jra", marker, now_jst(), len(cnames)))
                con.commit()
            log.info("%d-%02d 完了 %s", *ym, stats)
        return stats

    def _ingest_past_race(self, con, cname: str, all_odds_since: str | None, stats: dict, same_day: bool = False) -> None:
        rid = race_id_from_cname(cname)
        rdate = race_date_from_cname(cname)
        want_all = bool(all_odds_since and rdate and rdate >= all_odds_since)
        done_res = con.execute("SELECT 1 FROM result WHERE race_id=? LIMIT 1", (rid,)).fetchone()
        have = {r[0] for r in con.execute(
            "SELECT DISTINCT bet_type FROM odds_snapshot WHERE race_id=? AND is_final=1", (rid,))}
        if done_res and "win" in have and (not want_all or "trifecta" in have):
            stats["skipped"] += 1
            return
        try:
            # 当日は保存済みのページを使わず、単勝の払戻が出ていなければ取り込まずに次の実行で取り直す
            content = self._get_result(cname, use_cache=not same_day)
            race = P.parse_result(content, cname)
            if same_day and not any(p.bet_type == "win" for p in race.payouts):
                raise ValueError("払戻が未確定")
            stats["warnings"] += len(save_race(con, race))
            links = P.links(content, "pw151ou")
            own = [c for c in links if race_id_from_cname(c) == race.race_id]
            if own:
                self._capture_all(con, race.race_id, own[0], want_all, stats, final_only=True)
            con.commit()
            stats["races"] += 1
        except (FetchFailed, ValueError, IntegrityError) as e:
            con.rollback()
            stats["failed"] += 1
            log.warning("失敗 %s", public_fields("JRA", "past_race", e, rid))
            con.execute("INSERT INTO fetch_log(source,url,ref,fetched_at,http_status,bytes,attempts,ok,error)"
                        " VALUES ('jra','parse',?,?,NULL,0,1,0,?)", (cname, now_jst(), clean_message(e)))
            con.commit()

    def _capture_all(self, con, race_id: str, win_cname: str, exotics: bool, stats: dict, final_only=False) -> None:
        page = P.parse_odds(self._post("accessO.html", win_cname))
        captured = "final" if page.is_final else now_jst()
        if final_only and not page.is_final:
            return
        stats["odds"] += save_odds(con, race_id, page.odds, captured, page.is_final, "jra")
        if exotics:
            for name, c in page.tabs.items():
                if name != "win_place":
                    sub = P.parse_odds(self._post("accessO.html", c))
                    stats["odds"] += save_odds(con, race_id, sub.odds, captured, page.is_final, "jra")

    # --- 最新データ ----------------------------------------------------------------
    def update_results(self, con, days: int = 10, all_odds_days: int = 0) -> dict:
        """直近の確定結果を取り込む。"""
        today = datetime.now(JST).date()
        months = sorted({(d.year, d.month) for d in (today, today - timedelta(days=days))}, reverse=True)
        since = (today - timedelta(days=days)).isoformat()
        odds_since = (today - timedelta(days=all_odds_days)).isoformat() if all_odds_days else None
        table = self._load_top()
        stats = {"races": 0, "skipped": 0, "failed": 0, "odds": 0, "warnings": 0}
        for ym in months:
            for cname in self.month_race_cnames(table, ym):
                d = race_date_from_cname(cname)
                if d and d >= since:
                    self._ingest_past_race(con, cname, odds_since, stats)
        return stats

    def update_today_results(self, con, settle_min: int = 10, now: datetime | None = None) -> dict:
        """当日のレースで、発走から settle_min 分たっても結果がないものを「レース結果」ページからたどって取り込む。
        該当するレースがなければ公式サイトへ要求しない。取り込んだレースは翌朝の update_results で飛ばされる。"""
        stats = {"races": 0, "skipped": 0, "failed": 0, "odds": 0, "warnings": 0, "pending": 0}
        now = now or datetime.now(JST)
        today = now.date().isoformat()
        pending = {r[0] for r in con.execute(
            """SELECT race_id FROM race r WHERE org='JRA' AND race_date=? AND post_time IS NOT NULL AND post_time<=?
               AND NOT EXISTS (SELECT 1 FROM result x WHERE x.race_id=r.race_id AND x.finish_pos=1)""",
            (today, (now - timedelta(minutes=settle_min)).strftime("%H:%M")))}
        stats["pending"] = len(pending)
        if not pending:
            return stats
        top = self._post("accessS.html", RESULT_TOP)
        found = P.links(top, "pw01sde")
        for day in P.links(top, "pw01srl"):
            if race_date_from_srl(day) == today:
                found += P.links(self._post("accessS.html", day), "pw01sde")
        # 同じレースへのリンクが複数あれば1つだけ使う
        cnames = {race_id_from_cname(c): c for c in sorted(found) if race_id_from_cname(c) in pending}
        for cname in cnames.values():
            self._ingest_past_race(con, cname, None, stats, same_day=True)
        return stats

    def fetch_racecards(self, con) -> dict:
        """公開中の出馬表を取り込む。枠順確定前のレースは馬番がないので出走馬は未登録になる。"""
        stats = {"races": 0, "entries": 0, "failed": 0}
        top = self._post("accessD.html", CARD_TOP)
        days = P.links(top, "pw01drl")
        cards = set(P.links(top, "pw01dde01"))
        for d in days:
            cards.update(P.links(self._post("accessD.html", d), "pw01dde01"))
        today = datetime.now(JST).date().isoformat()
        for cname in sorted(c for c in cards if (race_date_from_cname(c) or "") >= today):
            try:
                race = P.parse_racecard(self.client.fetch(f"{BASE}/accessD.html?CNAME={cname}", ref=cname), cname)
                save_race(con, race)
                con.commit()
                stats["races"] += 1
                stats["entries"] += len(race.entries)
            except (FetchStop, FetchBlocked):
                con.rollback()
                raise
            except Exception as e:
                con.rollback()
                if is_fatal(e):
                    raise
                stats["failed"] += 1
                record_failure(con, "JRA", "card_race", e, race_id_from_cname(cname))
        return stats

    def capture_odds(self, con, exotics_within_min: int = 15) -> dict:
        """発売中オッズを取得時刻付きで保存。発走前のレースだけが対象で、確定前の値を上書きしない。"""
        stats = {"races": 0, "odds": 0, "failed": 0}
        now = datetime.now(JST)
        today = now.date().isoformat()
        pending = con.execute("SELECT race_id, post_time FROM race WHERE org='JRA' AND race_date=? AND post_time IS NOT NULL", (today,)).fetchall()
        if not any(win_place_due(con, r[0], (datetime.fromisoformat(f"{today}T{r[1]}:00+09:00") - now).total_seconds() / 60, now) for r in pending):
            return stats  # 取得すべきレースがなければ公式サイトへ要求しない
        top = self._post("accessO.html", ODDS_TOP)
        day_pages = [c for c in P.links(top, "pw15orl") if race_date_from_cname(c) == today]
        wins = {c for c in P.links(top, "pw151ou") if race_date_from_cname(c) == today}
        for d in day_pages:
            wins.update(c for c in P.links(self._post("accessO.html", d), "pw151ou") if race_date_from_cname(c) == today)
        seen: set[str] = set()
        for cname in sorted(wins):
            rid = race_id_from_cname(cname)
            row = con.execute("SELECT post_time FROM race WHERE race_id=?", (rid,)).fetchone()
            if rid in seen or row is None or not row["post_time"]:
                continue
            seen.add(rid)
            post = datetime.fromisoformat(f"{today}T{row['post_time']}:00+09:00")
            mins = (post - now).total_seconds() / 60
            if mins < 0 or not win_place_due(con, rid, mins, now):
                continue
            try:
                self._capture_all(con, rid, cname, exotics_due(con, rid, mins, now, exotics_within_min), stats)
                con.commit()
                stats["races"] += 1
            except (FetchStop, FetchBlocked):
                con.rollback()
                raise
            except Exception as e:
                con.rollback()
                if is_fatal(e):
                    raise
                stats["failed"] += 1
                record_failure(con, "JRA", "odds_race", e, rid)
        return stats

    def run(self, con, **kw) -> dict:
        return self.update_results(con, **kw)


def month_range(start: str, end: str) -> list[tuple[int, int]]:
    """YYYY-MM の範囲を新しい順に返す。"""
    y, m = map(int, end.split("-"))
    sy, sm = map(int, start.split("-"))
    out = []
    while (y, m) >= (sy, sm):
        out.append((y, m))
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)
    return out
