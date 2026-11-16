"""機密評価のワークフローが公開PRや定期実行で秘密鍵を使わないことを監査。"""
from pathlib import Path

import yaml

from keiba_ai import config


def test_readonly_evaluation_workflow_requires_main_owner_and_manual_dispatch():
    path = config.ROOT / ".github/workflows/forward-evaluate-readonly.yml"
    flow = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert set(flow["on"]) == {"workflow_dispatch"}
    assert flow["permissions"] == {"contents": "read"}
    assert flow["concurrency"]["group"] != "keiba-state"

    job = flow["jobs"]["evaluate"]
    condition = job["if"]
    assert "github.ref == 'refs/heads/main'" in condition
    assert "github.ref_protected == true" in condition
    assert "github.actor == github.repository_owner" in condition
    assert "github.event_name == 'workflow_dispatch'" in condition
    assert "KEIBA_STATE_KEY" not in job.get("env", {})
    assert "KEIBA_STATE_KEY" not in flow.get("env", {})

    steps = job["steps"]
    joined = "\n".join(str(step) for step in steps)
    assert "push-state" not in joined
    assert "pull-state" not in joined
    assert "state unpack" not in joined
    assert "upload-pages-artifact" not in joined
    assert "deploy-pages" not in joined
    assert ".venv/bin/keiba" in joined
    assert "--read-only" in joined
    # 実データ評価で研究候補の比較が必ず含まれること。
    assert "--research" in joined
    assert "--encrypted-state" in joined
    assert "--output" in joined
    assert "keiba_ai.eval_artifact encrypt" in joined
    assert "actions/upload-artifact@" in joined

    secret_steps = [step for step in steps if "KEIBA_STATE_KEY" in step.get("env", {})]
    assert len(secret_steps) == 1
    assert secret_steps[0]["name"] == "Evaluate read-only and encrypt confidential results"
    upload = next(step for step in steps if "actions/upload-artifact@" in step.get("uses", ""))
    assert upload["with"]["path"] == "${{ runner.temp }}/forward-evaluation.enc"
    assert upload["with"]["retention-days"] == "3"
    secret_script = secret_steps[0]["run"]
    # tracebackやstdoutに個別の機密情報が含まれる場合もActionsへ出さない
    assert '2> "$workdir/evaluation.stderr"' in secret_script
    assert '> "$workdir/evaluation.stdout"' in secret_script
    assert '2> "$workdir/encryption.stderr"' in secret_script
    assert '> "$workdir/encryption.stdout"' in secret_script
    assert "if [ \"$rc\" -ne 0 ]" in secret_script
    assert "set +x" in secret_script
    assert "\\n          eval " not in secret_script
    assert "cat \"$workdir/metrics.json\"" not in joined
    assert "evaluation.stderr" in steps[-1]["run"]
    assert "encryption.stderr" in steps[-1]["run"]
    assert "summary.stderr" in steps[-1]["run"]


def test_public_summary_is_opt_in_and_prints_only_the_whitelisted_table():
    path = config.ROOT / ".github/workflows/forward-evaluate-readonly.yml"
    flow = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    option = flow["on"]["workflow_dispatch"]["inputs"]["public_summary"]
    assert option["type"] == "boolean" and option["default"] == "false"
    step = next(s for s in flow["jobs"]["evaluate"]["steps"] if "KEIBA_STATE_KEY" in s.get("env", {}))
    script = step["run"]
    # 既定では成績を出さずに終わり、表は要約モジュールの出力だけを表示する
    stop = script.index('if [ "$EVAL_PUBLIC_SUMMARY" != "true" ]')
    assert script.index("keiba_ai.eval_summary") > stop
    assert '2> "$workdir/summary.stderr"' in script
    assert 'cat "$workdir/summary.md"' in script
    assert script.count("cat ") == 2
