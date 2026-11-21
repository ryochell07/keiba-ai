"""エージェント (Claude) が作業中に使う簡易評価。探索期間だけを使い、1サイクルの回数に上限がある。

  agent-eval --org NAR

作業コピーのコードで学習・採点し、基準 (main) との差を表示する。結果は .agent/quick/ に残る。
最終的な判定は、Claude が終わったあとに実行役が同じ手順でやり直して決める。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import harness, settings


def _fmt(t: dict) -> str:
    lo, hi = t["d_ll_ci"]
    return (f"Log Loss {t['ll_base']:.4f} → {t['ll_cand']:.4f} (差 {t['d_ll']:+.4f}, 95%区間 [{lo:+.4f}, {hi:+.4f}]) / "
            f"Brier {t['brier_base']:.4f} → {t['brier_cand']:.4f} (差 {t['d_brier']:+.4f}) / "
            f"本命的中 {t['hit_base']:.3f} → {t['hit_cand']:.3f} / 本命の単勝回収率 {t['roi_base']:.3f} → {t['roi_cand']:.3f} (参考) / "
            f"{t['n_races']}レース・{t['n_days']}日")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agent-eval")
    ap.add_argument("--org", choices=["JRA", "NAR"], required=True)
    args = ap.parse_args(argv)
    cfg = settings.load()
    run_dir, base_dir, work = (Path(os.environ[k]) for k in ("AGENT_RUN_DIR", "AGENT_BASE_DIR", "AGENT_WORK"))
    counter = run_dir / "quick_count.json"
    used = json.loads(counter.read_text())["used"] if counter.exists() else 0
    limit = cfg["limits"]["quick_evals_per_cycle"]
    if used >= limit:
        print(f"このサイクルの簡易評価は上限の {limit} 回に達しました。今の作業コピーを最終案として終えてください。")
        return 2
    counter.write_text(json.dumps({"used": used + 1}))
    n = used + 1
    out = run_dir / "quick" / f"{n}_{args.org}.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    py = [sys.executable, "-W", "ignore", "-m", "autoresearch.harness"]
    print(f"簡易評価 {n}/{limit} ({args.org}, 探索期間 {cfg['periods']['explore'][0]}〜{cfg['periods']['explore'][1]} の前日まで) を実行します。地方は5分ほどかかります。", flush=True)
    feat = out.with_name(f"{n}_{args.org}_features.json")
    for cmd in (["features", "--code", str(work / "src"), "--org", args.org, "--out", str(feat)],
                ["score", "--code", str(work / "src"), "--org", args.org, "--stage", "explore", "--out", str(out)]):
        r = subprocess.run(py + cmd, capture_output=True, text=True)
        if r.returncode:
            print(f"{cmd[0]} が失敗しました:\n{r.stderr[-3000:]}")
            return 1
    base_rows = base_dir / f"explore_{args.org}.csv.gz"
    cmp = harness.compare(harness.read_rows(base_rows), harness.read_rows(out))
    fc = json.loads(feat.read_text())
    cols = harness.column_changes(json.loads((base_dir / f"features_{args.org}.json").read_text()), fc)
    result = {"compare": cmp, "future_leak": fc["future_leak"], "own_result_leak": fc["own_result_leak"], **cols}
    out.with_name(f"{n}_{args.org}_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"AI単独の確率: {_fmt(cmp['ai'])}")
    print(f"市場と合成した最終確率: {_fmt(cmp['final'])}")
    print(f"AI単独が基準と同一: {cmp['ai_unchanged']} (同一なら最終確率で判定されます)")
    print(f"追加した列: {cols['added'] or 'なし'}")
    print(f"値が変わった既存の列 (空でなければ不合格): {cols['changed_existing'] or 'なし'}")
    print(f"未来情報の疑い: 未来を消すと変わる列 {fc['future_leak'] or 'なし'} / 自分の結果で変わる列 {fc['own_result_leak'] or 'なし'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
