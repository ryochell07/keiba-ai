import pytest

from keiba_ai.http import FetchBlocked, FetchFailed, FetchStop, PoliteClient


class Resp:
    def __init__(self, status=200, text="", content=b"ok"):
        self.status_code, self.text, self.content = status, text, content


class Session:
    def __init__(self, robots="User-agent: *\nDisallow: /private/\n", statuses=()):
        self.headers, self.robots, self.statuses, self.calls = {}, robots, list(statuses), []

    def get(self, url, **kw):
        self.calls.append(url)
        if url.endswith("/robots.txt"):
            return Resp(200, self.robots)
        return Resp(self.statuses.pop(0) if self.statuses else 200)

    post = get


CONF = dict(enabled=True, automated_fetch=True, min_interval_sec=0, permission={"status": "none"})
nosleep = lambda s: None


def client(conf=CONF, **kw):
    s = Session(**kw)
    return PoliteClient("x", conf, session=s, sleep=nosleep), s


def test_robots禁止パスには要求を送らない():
    c, s = client()
    with pytest.raises(FetchBlocked):
        c.fetch("https://e.test/private/a")
    assert s.calls == ["https://e.test/robots.txt"]


def test_自動取得オフなら一切アクセスしない():
    c, s = client({**CONF, "automated_fetch": False})
    with pytest.raises(FetchBlocked):
        c.fetch("https://e.test/a")
    assert s.calls == []


def test_無効な取得元は拒否():
    c, s = client({**CONF, "enabled": False})
    with pytest.raises(FetchBlocked):
        c.fetch("https://e.test/a")


def test_書面許諾があればrobotsより優先():
    c, s = client({**CONF, "automated_fetch": False, "permission": {"status": "granted", "evidence": "2026-12-01 文書"}})
    assert c.fetch("https://e.test/private/a") == b"ok"


def test_403は再試行せず停止():
    c, s = client(statuses=[403])
    with pytest.raises(FetchStop):
        c.fetch("https://e.test/a")
    assert len(s.calls) == 2


def test_500は再試行して回復():
    c, s = client(statuses=[500, 503, 200])
    assert c.fetch("https://e.test/a") == b"ok"


def test_再試行の上限():
    c, s = client(statuses=[500, 500, 500])
    with pytest.raises(FetchFailed):
        c.fetch("https://e.test/a")


def test_要求数の上限で停止():
    c, s = client()
    c.max_requests = 1
    c.fetch("https://e.test/a")
    with pytest.raises(FetchStop):
        c.fetch("https://e.test/b")
