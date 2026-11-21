#!/bin/bash
# 自動研究エージェントを Mac に入れる (何度実行してもよい)。docs/AUTORESEARCH.md
#   install.sh [版]      版を省くと origin/main。定期実行も登録する
#   install.sh --no-schedule [版]   定期実行は登録しない
set -euo pipefail
SCHEDULE=1
if [ "${1:-}" = "--no-schedule" ]; then SCHEDULE=0; shift; fi
REF="${1:-origin/main}"
AGENT_HOME="${KEIBA_AGENT_HOME:-$HOME/keiba-agent}"
REPO_URL=https://github.com/ryochell07/keiba-ai.git
LABEL=com.ryochell07.keiba-autoresearch
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

mkdir -p "$AGENT_HOME/logs" "$AGENT_HOME/cache" "$AGENT_HOME/runs" "$AGENT_HOME/data"
# エージェント専用の複製。本番の作業コピー (~/keiba-ai) とは別にする
[ -d "$AGENT_HOME/repo/.git" ] || git clone --quiet "$REPO_URL" "$AGENT_HOME/repo"
git -C "$AGENT_HOME/repo" fetch --quiet --prune origin
if [ -d "$AGENT_HOME/tools" ]; then
  git -C "$AGENT_HOME/tools" checkout --quiet --detach --force "$REF"
else
  git -C "$AGENT_HOME/repo" worktree add --quiet --detach "$AGENT_HOME/tools" "$REF"
fi
(cd "$AGENT_HOME/tools" && uv sync --locked --extra dev --extra compare --quiet)
echo "tools: $(git -C "$AGENT_HOME/tools" log --oneline -1)"

if [ "$SCHEDULE" = 1 ]; then
  mkdir -p "$HOME/Library/LaunchAgents"
  sed -e "s#__HOME__#$AGENT_HOME#g" "$AGENT_HOME/tools/scripts/autoresearch/launchd.plist" > "$PLIST"
  launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$PLIST"
  echo "定期実行を登録しました: $LABEL (1時間ごとに起動し、上限の範囲で動きます)"
fi
