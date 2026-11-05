"""パスと設定の読み込み。"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("KEIBA_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
RAW = DATA / "raw"
INBOX_NAR = DATA / "inbox" / "nar"
DB_PATH = Path(os.environ.get("KEIBA_DB", DATA / "keiba.db"))
MODELS = DATA / "models"
REPORTS = DATA / "reports"
PUBLIC = ROOT / "public"
USER_AGENT = "keiba-ai-personal-research/0.1 (personal non-commercial; low rate)"


def load_sources(path: Path | None = None) -> dict:
    with open(path or ROOT / "config" / "sources.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    for p in (DATA, RAW, INBOX_NAR, MODELS, REPORTS):
        p.mkdir(parents=True, exist_ok=True)
