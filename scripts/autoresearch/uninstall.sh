#!/bin/bash
# 定期実行を止める。記録 (~/keiba-agent) は消さない
set -euo pipefail
PLIST="$HOME/Library/LaunchAgents/com.ryochell07.keiba-autoresearch.plist"
launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null || true
rm -f "$PLIST"
echo "定期実行を止めました"
