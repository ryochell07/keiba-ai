"""候補のコードを動かすときの制限。macOS の sandbox-exec で通信を止め、書き込み先と読み取り先を絞る。

エージェントの Claude 自身の Bash は Claude Code の sandbox 設定で同じように制限する (runner.claude_settings)。
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

SANDBOX_EXEC = "/usr/bin/sandbox-exec"


def secret_paths(home: Path | None = None) -> list[Path]:
    """読ませない場所。鍵・認証情報、本番の作業コピー (.env と事故調査用DBがある)、研究用の元データ。"""
    h = home or Path.home()
    return [h / ".ssh", h / ".gnupg", h / ".aws", h / ".config" / "gh", h / ".netrc", h / ".git-credentials",
            h / "Library" / "Keychains", h / ".claude", h / ".claude.json", h / "keiba-ai", h / "keiba-research"]


def _q(p: Path | str) -> str:
    s = str(Path(p).resolve()) if Path(p).exists() else str(p)
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def profile(write_dirs: list[Path], deny_read: list[Path]) -> str:
    writes = " ".join(f"(subpath {_q(d)})" for d in write_dirs)
    reads = " ".join(f"(subpath {_q(d)})" for d in deny_read)
    return f"""(version 1)
(allow default)
(deny network*)
(allow network* (remote unix-socket))
(deny file-write*)
(allow file-write* {writes} (subpath "/private/var/folders") (subpath "/private/tmp") (subpath "/dev"))
(deny file-read* {reads})
"""


def minimal_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """引き継ぐ環境変数を絞る。API キーやトークンは渡さない。"""
    keep = ("HOME", "USER", "LOGNAME", "LANG", "TMPDIR", "SHELL")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.setdefault("LANG", "ja_JP.UTF-8")
    env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"
    env.update(extra or {})
    return env


def run(cmd: list[str], *, write_dirs: list[Path], env: dict[str, str], cwd: Path, timeout: int, log: Path,
        deny_read: list[Path] | None = None) -> int:
    """制限した中でコマンドを動かし、出力を log に書く。戻り値は終了コード (時間切れは 124)。"""
    prof = profile(write_dirs, secret_paths() if deny_read is None else deny_read)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as f:
        f.write(f"$ {' '.join(cmd)}\n".encode())
        f.flush()
        try:
            return subprocess.run([SANDBOX_EXEC, "-p", prof, *cmd], cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                  timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            f.write(b"\n[timeout]\n")
            return 124
