"""鍵と合言葉を使うステップだけに渡し、外部アクションをコミットのハッシュで固定していることを監査。
タグが書き換えられた外部アクションや、鍵を使わないステップから秘密情報が読まれないようにする。"""
import re

import yaml

from keiba_ai import config

GITHUB = config.ROOT / ".github"
PRODUCTION = ("daily", "raceday", "backfill", "retrain")
SECRET_NAMES = ("KEIBA_STATE_KEY", "KEIBA_SITE_PASSPHRASE")
STATE_ACTIONS = ("./.github/actions/pull-state", "./.github/actions/push-state")
PINNED = re.compile(r"^[\w.-]+/[\w.-]+@[0-9a-f]{40}$")


def _load(path):
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _definition_files():
    return sorted((GITHUB / "workflows").glob("*.yml")) + sorted((GITHUB / "actions").glob("*/action.yml"))


def _steps(path):
    data = _load(path)
    if "runs" in data:
        return data["runs"].get("steps", [])
    return [step for job in data["jobs"].values() for step in job.get("steps", [])]


def _needs(step):
    """そのステップが本当に使う秘密情報。鍵は状態の復号・暗号化、合言葉はページの生成で読む。"""
    need = set()
    if step.get("uses") in STATE_ACTIONS:
        need.add("KEIBA_STATE_KEY")
    if re.search(r"keiba (daily|site)\b", step.get("run", "")):
        need.add("KEIBA_SITE_PASSPHRASE")
    return need


def test_external_actions_are_pinned_to_a_commit_with_its_version():
    external = [(path.name, step["uses"]) for path in _definition_files() for step in _steps(path)
                if "uses" in step and not step["uses"].startswith("./")]
    assert len(external) >= 5   # 何も見つからないまま通らないように
    for name, uses in external:
        assert PINNED.match(uses), f"{name}: {uses} がコミットのハッシュで固定されていない"
    # 人が版を確かめられるように、行末に版を書く
    for path in _definition_files():
        for line in path.read_text(encoding="utf-8").splitlines():
            if re.search(r"uses: [^.\s]", line):
                assert re.search(r"@[0-9a-f]{40}  # v\d+\.\d+\.\d+$", line), f"{path.name}: {line.strip()}"


def test_secrets_are_never_set_for_a_whole_workflow_or_job():
    for path in sorted((GITHUB / "workflows").glob("*.yml")):
        flow = _load(path)
        assert "secrets." not in str(flow.get("env", {})), path.name
        for job in flow["jobs"].values():
            assert "secrets." not in str(job.get("env", {})), path.name


def test_production_steps_get_exactly_the_secrets_they_use():
    for name in PRODUCTION:
        steps = _load(GITHUB / f"workflows/{name}.yml")["jobs"]["run"]["steps"]
        given_count = {key: 0 for key in SECRET_NAMES}
        for step in steps:
            env = step.get("env", {})
            label = f"{name}: {step.get('name') or step.get('uses')}"
            given = {key for key in SECRET_NAMES if key in env}
            # 使うのに渡し忘れると、ページの生成は黙って省略され、状態は保存できない
            assert given == _needs(step), label
            for key in given:
                assert env[key] == f"${{{{ secrets.{key} }}}}", label
                given_count[key] += 1
            assert all(k in SECRET_NAMES for k, v in env.items() if "secrets." in v), label
            if not step.get("uses", "./").startswith("./"):
                assert not given, label
        assert given_count["KEIBA_STATE_KEY"] == 2, name      # 取得と保存
        assert given_count["KEIBA_SITE_PASSPHRASE"] == (1 if name in ("daily", "raceday") else 0), name


def test_state_actions_still_need_the_key_and_composites_never_read_secrets():
    # 鍵を渡す先が実際に復号・暗号化している (ここが変わったら、渡す先も見直す)
    assert "keiba state unpack" in str(_steps(GITHUB / "actions/pull-state/action.yml"))
    assert "keiba state pack" in str(_steps(GITHUB / "actions/push-state/action.yml"))
    for path in sorted((GITHUB / "actions").glob("*/action.yml")):
        assert "secrets." not in path.read_text(encoding="utf-8"), path.parent.name
