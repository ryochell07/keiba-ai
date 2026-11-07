"""動いている環境が uv.lock と同じかを確かめる。違っていても更新はせず、知らせて止める。"""
from __future__ import annotations

import sys
import tomllib
from importlib import metadata
from pathlib import Path

EXIT_ENV = 5
PROJECT = Path(__file__).resolve().parents[2]
CHECKED = ("beautifulsoup4", "charset-normalizer", "cryptography", "lightgbm", "lxml", "numpy", "pandas", "pyyaml", "requests", "scikit-learn", "scipy")


def env_mismatches(root: Path = PROJECT) -> list[str]:
    lock, pin = root / "uv.lock", root / ".python-version"
    if not lock.exists() or not pin.exists():
        return [f"{f.name} がありません" for f in (lock, pin) if not f.exists()]
    out = []
    python = ".".join(map(str, sys.version_info[:3]))
    if python != pin.read_text().strip():
        out.append(f"python: 環境 {python} / 固定 {pin.read_text().strip()}")
    locked = {p["name"]: p["version"] for p in tomllib.loads(lock.read_text(encoding="utf-8"))["package"] if "version" in p}
    for name in CHECKED:
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = None
        if have != locked.get(name):
            out.append(f"{name}: 環境 {have} / 固定 {locked.get(name)}")
    return out
