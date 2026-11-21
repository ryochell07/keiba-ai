"""GitHub Issues からの研究の指示。ChatGPT などで書いた指示を、決まったラベルの Issue で受け取る。

公開リポジトリでは誰でも Issue やコメントを書けるので、設定した作成者のものだけを読む。
実行役が書き込んだ結果のコメントは目印で見分け、指示としては読まない。
読んだ内容は「データ」として Claude に渡す。規則や権限を変える力は持たない。
"""
from __future__ import annotations

import json
import subprocess

MARK = "<!-- keiba-autoresearch -->"


def _gh(args: list[str], stdin: str | None = None, timeout: int = 60) -> str:
    r = subprocess.run(["gh", *args], input=stdin, capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f"gh {args[0]} {args[1] if len(args) > 1 else ''} が失敗しました")
    return r.stdout


def select(items: list[dict], gh_cfg: dict) -> list[dict]:
    trusted = {a.lower() for a in gh_cfg["trusted_authors"]}
    limit = gh_cfg["max_issue_chars"]
    out = []
    for it in items:
        author = (it.get("author") or {}).get("login", "")
        if author.lower() not in trusted:
            continue
        comments = [c for c in it.get("comments") or []
                    if (c.get("author") or {}).get("login", "").lower() in trusted and MARK not in (c.get("body") or "")]
        text = (it.get("body") or "").strip()
        for c in comments[-5:]:
            text += f"\n\n[追記 {str(c.get('createdAt', ''))[:10]}]\n{(c.get('body') or '').strip()}"
        out.append({"number": int(it["number"]), "title": (it.get("title") or "")[:200], "body": text[:limit], "url": it.get("url"),
                    "author": author})
    return sorted(out, key=lambda x: x["number"])


def fetch(gh_cfg: dict) -> list[dict]:
    raw = _gh(["issue", "list", "--repo", gh_cfg["repo"], "--label", gh_cfg["task_label"], "--state", "open", "--limit", "20",
               "--json", "number,title,body,author,url,comments"])
    return select(json.loads(raw), gh_cfg)


def comment(gh_cfg: dict, number: int, body: str) -> None:
    _gh(["issue", "comment", str(number), "--repo", gh_cfg["repo"], "--body-file", "-"], stdin=f"{MARK}\n{body}")
