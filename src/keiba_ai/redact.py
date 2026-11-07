"""秘密の値と認証情報を伏せる。暗号化される DB に残す文面にも通す共通の処理。"""
from __future__ import annotations

import os
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_SECRET_ENV = ("KEIBA_STATE_KEY", "KEIBA_SITE_PASSPHRASE", "GH_TOKEN", "GITHUB_TOKEN")
_URL_QUERY = re.compile(r"(https?://[^\s?'\"]+)\?[^\s'\"]*")
_BEARER = re.compile(r"(?i)\b(bearer)\s+\S+")
_CREDENTIAL = re.compile(r"(?i)\b(proxy-authorization|authorization|set-cookie|cookie|x-api-key|api[_-]?key|token|password|passphrase|secret)\s*[:=]\s*\S+")
SAFE_QUERY_KEYS = frozenset({"CNAME", "type", "k_year", "k_month"})   # 取得先の識別に使うだけで、秘密を含まない


def mask_text(text: str) -> str:
    for name in _SECRET_ENV:
        value = os.environ.get(name)
        if value:
            text = text.replace(value, "***")
    text = _URL_QUERY.sub(r"\1?***", text)
    text = _BEARER.sub(r"\1 ***", text)
    return _CREDENTIAL.sub(r"\1=***", text)


def mask_url(url: str) -> str:
    """取得記録に残す URL。既知のキーだけ値を残す。"""
    parts = urlsplit(url)
    if not parts.query:
        return url
    query = [(k, v if k in SAFE_QUERY_KEYS else "***") for k, v in parse_qsl(parts.query, keep_blank_values=True)]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query, safe="*/"), ""))
