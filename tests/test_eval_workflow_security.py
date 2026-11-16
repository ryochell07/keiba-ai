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
    assert "--encrypted-state" in joined
    assert "--output" in joined
    assert "keiba_ai.eval_artifact encrypt" in joined
    assert "upload-artifact@v4" in joined

    secret_steps = [step for step in steps if "KEIBA_STATE_KEY" in step.get("env", {})]
    assert len(secret_steps) == 1
    assert secret_steps[0]["name"] == "Evaluate read-only and encrypt confidential results"
    upload = next(step for step in steps if "upload-artifact@v4" in step.get("uses", ""))
    assert upload["with"]["path"] == "${{ runner.temp }}/forward-evaluation.enc"
    assert upload["with"]["retention-days"] == "3"
