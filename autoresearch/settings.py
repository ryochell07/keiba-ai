"""設定とパス。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

TOOLS = Path(__file__).resolve().parents[1]
CONFIG = TOOLS / "config" / "autoresearch.yaml"


def load(path: Path | None = None) -> dict:
    with open(path or CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Paths:
    home: Path

    @property
    def tools(self) -> Path:      # 評価と実行の手順。エージェントは書き換えられない
        return TOOLS

    @property
    def work(self) -> Path:       # エージェントが編集する作業コピー
        return self.home / "work"

    @property
    def base(self) -> Path:       # 基準にする main の作業コピー
        return self.home / "base"

    @property
    def data(self) -> Path:       # 評価の KEIBA_ROOT
        return self.home / "data"

    @property
    def db(self) -> Path:         # 封鎖期間を含まない開発用DB。確認期間を含むので実行役だけが読む
        return self.data / "keiba.db"

    @property
    def explore_db(self) -> Path:  # 確認期間も含まない探索用DB。エージェントが読めるのはこれだけ
        return self.data / "explore.db"

    @property
    def cache(self) -> Path:
        return self.home / "cache"

    @property
    def runs(self) -> Path:
        return self.home / "runs"

    @property
    def ledger(self) -> Path:
        return self.home / "ledger.jsonl"

    @property
    def state(self) -> Path:
        return self.home / "state.json"

    @property
    def logs(self) -> Path:
        return self.home / "logs"

    @property
    def python(self) -> Path:
        return TOOLS / ".venv" / "bin" / "python"


def paths(cfg: dict) -> Paths:
    return Paths(Path(os.environ.get("KEIBA_AGENT_HOME") or cfg["home"]).expanduser())
