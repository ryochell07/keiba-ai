"""実験の記録と、実行回数の上限。

記録 (ledger.jsonl) は追記だけで、成功・失敗・規則違反・重複も全部残す。
同じ変更 (差分の指紋が同じ) は二度評価しない。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

STATUSES = ("improved", "explore_failed", "confirm_failed", "duplicate", "rule_violation", "tests_failed", "leak",
            "no_change", "invalid_proposal", "claude_error", "error")


def now() -> datetime:
    return datetime.now().astimezone()


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append(path: Path, entry: dict) -> None:
    if entry.get("status") not in STATUSES:
        raise ValueError(f"不明な状態: {entry.get('status')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def fingerprint(diff: str) -> str:
    """変更の指紋。空白だけの違いは同じ変更とみなす。"""
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in diff.splitlines()
             if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:20]


def normalize_key(key: str) -> str:
    return re.sub(r"[^0-9a-z_/-]+", "-", (key or "").lower()).strip("-")


def seen(entries: list[dict], fp: str) -> dict | None:
    for e in entries:
        if e.get("fingerprint") == fp and e.get("status") not in ("error", "claude_error"):
            return e
    return None


def same_key(entries: list[dict], key: str, org: str) -> list[dict]:
    k = normalize_key(key)
    return [e for e in entries if normalize_key(e.get("key", "")) == k and e.get("org") == org]


def for_agent(entries: list[dict], limit: int = 40) -> list[dict]:
    """エージェントに渡す要約。確認期間の数値は渡さず、合否だけにする (確認期間に合わせた仮説を避けるため)。"""
    out = []
    for e in entries[-limit:]:
        ex = (e.get("explore") or {}).get(e.get("primary_track") or "ai") or {}
        out.append({"run": e.get("run_id"), "org": e.get("org"), "category": e.get("category"), "key": e.get("key"), "title": e.get("title"),
                    "hypothesis": (e.get("hypothesis") or "")[:300], "status": e.get("status"), "reason": e.get("reason"),
                    "explore_d_log_loss": ex.get("d_ll"), "explore_d_log_loss_ci": ex.get("d_ll_ci"), "explore_d_brier": ex.get("d_brier"),
                    "confirm_passed": None if e.get("confirm") is None else e.get("status") == "improved", "issue": e.get("issue")})
    return out


@dataclass
class State:
    """起動の記録。回数の上限と、利用上限に当たったあとの停止に使う。"""
    cycles: list[str] = field(default_factory=list)
    paused_until: str | None = None
    notes: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "State":
        if not path.exists():
            return cls()
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(d.get("cycles", []), d.get("paused_until"), d.get("notes", {}))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.__dict__, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)

    def budget(self, limits: dict, at: datetime) -> tuple[bool, str]:
        """この時刻に1サイクル始めてよいか。Claude を呼ぶサイクルだけを数える。"""
        if self.paused_until and at < datetime.fromisoformat(self.paused_until):
            return False, f"利用上限のため {self.paused_until} まで停止中"
        times = [datetime.fromisoformat(t) for t in self.cycles]
        day = [t for t in times if t.date() == at.date()]
        week = [t for t in times if at - t < timedelta(days=7)]
        if len(day) >= limits["max_cycles_per_day"]:
            return False, f"今日は上限の {limits['max_cycles_per_day']} 回に達しました"
        if len(week) >= limits["max_cycles_per_week"]:
            return False, f"直近7日で上限の {limits['max_cycles_per_week']} 回に達しました"
        if times and at - max(times) < timedelta(hours=limits["min_hours_between_cycles"]):
            return False, f"前回から {limits['min_hours_between_cycles']} 時間たっていません"
        return True, "ok"

    def record_cycle(self, at: datetime) -> None:
        self.cycles = [t for t in self.cycles if at - datetime.fromisoformat(t) < timedelta(days=30)] + [at.isoformat(timespec="seconds")]

    def pause(self, at: datetime, hours: float) -> None:
        self.paused_until = (at + timedelta(hours=hours)).isoformat(timespec="seconds")
