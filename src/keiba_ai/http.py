"""礼儀正しい HTTP クライアント。robots 遵守・レート制限・再試行・取得履歴を一元管理する。"""
from __future__ import annotations

import gzip
import hashlib
import logging
import sqlite3
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

from . import config
from .db import now_jst
from .redact import mask_text, mask_url

log = logging.getLogger(__name__)


class FetchBlocked(RuntimeError):
    """robots/規約/設定により取得してはいけない。回避せず呼び出し側で停止・代替する。"""


class FetchStop(RuntimeError):
    """403/429 等、サイト側が拒否している、または要求数上限に達したので実行全体を止める。"""


class FetchLimit(FetchStop):
    """1回の実行の要求数上限に到達。異常ではなく、次回の実行で続きを取得する。"""


class FetchFailed(RuntimeError):
    """再試行しても取得できなかった。"""


class PoliteClient:
    def __init__(self, source: str, conf: dict, con: sqlite3.Connection | None = None,
                 session: requests.Session | None = None, max_requests: int = 6000, retries: int = 3,
                 sleep=time.sleep):
        self.source, self.conf, self.con = source, conf, con
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = config.USER_AGENT
        self.min_interval = float(conf.get("min_interval_sec", 3.0))
        self.max_requests, self.retries, self.sleep = max_requests, retries, sleep
        self.requests_made = 0
        self._last = 0.0
        self._robots: dict[str, RobotFileParser] = {}

    # --- 許可判定 -----------------------------------------------------------------
    def _robots_for(self, url: str) -> RobotFileParser:
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        if base not in self._robots:
            rp = RobotFileParser()
            self._throttle()
            r = self.session.get(base + "/robots.txt", timeout=30)
            if r.status_code >= 500:
                raise FetchFailed(f"robots.txt {r.status_code}")  # 不明なら取得しない
            rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            delay = rp.crawl_delay(config.USER_AGENT)
            if delay:
                self.min_interval = max(self.min_interval, float(delay))
            self._robots[base] = rp
        return self._robots[base]

    def check_allowed(self, url: str) -> None:
        if not self.conf.get("enabled"):
            raise FetchBlocked(f"{self.source}: 無効化されています")
        granted = self.conf.get("permission", {}).get("status") == "granted"
        if not (self.conf.get("automated_fetch") or granted):
            raise FetchBlocked(f"{self.source}: 自動取得は無効です (config/sources.yaml)")
        if not granted and not self._robots_for(url).can_fetch(config.USER_AGENT, url):
            raise FetchBlocked(f"{self.source}: robots.txt で禁止: {url}")

    # --- 取得 ---------------------------------------------------------------------
    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            self.sleep(wait)
        self._last = time.monotonic()

    def _log(self, url, ref, status, size, attempts, ok, error=None) -> None:
        if self.con is None:
            return
        self.con.execute(
            "INSERT INTO fetch_log(source,url,ref,fetched_at,http_status,bytes,attempts,ok,error)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (self.source, mask_url(url), ref, now_jst(), status, size, attempts, int(ok), mask_text(error) if error else error))
        self.con.commit()

    def fetch(self, url: str, data: dict | None = None, ref: str | None = None) -> bytes:
        self.check_allowed(url)
        last_err, status = None, None
        for attempt in range(1, self.retries + 1):
            if self.requests_made >= self.max_requests:
                raise FetchLimit(f"1回の実行の要求数上限 {self.max_requests} に達しました")
            self._throttle()
            self.requests_made += 1
            try:
                r = (self.session.post(url, data=data, timeout=60, allow_redirects=False) if data
                     else self.session.get(url, timeout=60, allow_redirects=False))
                status = r.status_code
                if status == 200:
                    self._log(url, ref, status, len(r.content), attempt, True)
                    return r.content
                if status in (401, 403, 429):
                    self._log(url, ref, status, 0, attempt, False, "拒否されたため停止")
                    raise FetchStop(f"{url} が HTTP {status} を返したので停止します")
                last_err = f"HTTP {status}"
                if status < 500:
                    break  # 3xx/4xx は再試行しても変わらない
            except requests.RequestException as e:
                last_err = repr(e)
            self.sleep(min(60.0, 5.0 * 2 ** (attempt - 1)))
        self._log(url, ref, status, 0, attempt, False, last_err)
        raise FetchFailed(f"{url}: {last_err}")


def raw_path(source: str, key: str) -> Path:
    h = hashlib.sha1(key.encode()).hexdigest()
    return config.RAW / source / h[:2] / f"{key.replace('/', '_')}.gz"


def save_raw(source: str, key: str, content: bytes) -> None:
    p = raw_path(source, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(gzip.compress(content, 6))


def load_raw(source: str, key: str) -> bytes | None:
    p = raw_path(source, key)
    return gzip.decompress(p.read_bytes()) if p.exists() else None
