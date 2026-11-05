# keiba-ai

JRAと地方競馬のデータを蓄積し、全券種の確率と期待値を計算してスマートフォンで確認する、個人利用専用の分析システムです。

> 予測は推定であり、利益を保証しません。馬券の購入は自己責任で、余裕資金の範囲で楽しんでください。商用利用や、取得データの再配布はしないでください。

## 現在の実力 (2026-11-05時点)

地方競馬の実データ (2024年1月〜、約4.4万レース) で、学習に使っていない 2026-03〜11 の9,765レースを評価しました。

| 指標 | 結果 |
|---|---|
| AI本命の的中率 | 39.8% (確定1番人気は45.5%) |
| AI本命の単勝回収率 | 80.2% (確定1番人気は80.1%) |
| 単勝のLog Loss | AI 1.684 / 市場 1.539 |
| AIと市場の合成のLog Loss (2026-07以降) | 1.5369 / 同期間の市場 1.5380 |

**長期的にプラスになる買い方は、まだ見つかっていません。** AI単独は市場より弱く、合成しても改善はわずかです。
単勝で合成の期待値が1倍を超える馬は確定オッズ基準では出ず、他の券種で期待値プラスと判定した買い目も回収率は多くが50〜77%でした。
中央競馬は学習データが少なく、まだ評価できていません。

## できること

- JRA公式サイトから、過去結果、払戻、最終オッズ、出馬表、発売中オッズを低頻度で取得
- 地方競馬の公式ZIPを自動取得 (協会の回答に基づき個人利用に限る。詳細は [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md))
- LightGBM (Plackett-Luce) で順位を学習し、単勝オッズとも合成して単勝から三連単までの確率を計算
- 期待値ランキング (全体、券種別、レース別、AI確率、人気との乖離) を暗号化ページで表示
- 未来情報のリーク検出テストと、時系列のバックテスト

## ローカルでの使い方

```bash
uv venv --python 3.12
```

```bash
uv pip install -e '.[dev,compare]'
```

macOSでは LightGBM に libomp が必要です。

```bash
brew install libomp
```

主なコマンドは次のとおりです。

| コマンド | 内容 |
|---|---|
| `keiba jra-backfill --from 2024-01 --to 2026-10` | 過去データ取得。中断しても続きから再開 |
| `keiba jra-update` | 直近の結果を取得 |
| `keiba jra-cards` / `keiba jra-odds` | 出馬表 / 発売中オッズを取得 |
| `keiba nar-backfill --from 2024-01 --to 2026-10` | 地方の月次ZIPを取得 |
| `keiba nar-update` / `keiba nar-today` | 地方の前月・当月を更新 / 当日の出馬表とオッズを取得 |
| `keiba nar-import` | 手動で置いたZIPを取り込み |
| `keiba train` | 新モデルを学習し、旧モデルより良ければ採用 |
| `keiba models` / `keiba rollback --org JRA` | 版の一覧 / 以前のモデルへ戻す |
| `keiba reconcile` | 発走前の予測と確定結果を照合 |
| `keiba backtest --org JRA --start 2026-01-01` | ウォークフォワード評価 |
| `keiba compare --org JRA --start 2026-01-01` | 手法の比較 |
| `keiba predict` | 未確定レースを予測 |
| `keiba site` | 閲覧ページを `public/` に生成 (`KEIBA_SITE_PASSPHRASE` が必要) |
| `keiba daily` | 上記を日次でまとめて実行 |
| `keiba status` | 件数と取得元の状態 |

テストは次のコマンドで実行します (実サイトにはアクセスしません)。

```bash
.venv/bin/python -m pytest -q
```

## GitHubでの自動運転

Macを起動していなくても動く構成です。手順は [デプロイ手順](docs/DEPLOY.md) を参照してください。

動作するワークフローは `daily` (毎朝)、`raceday` (土日と月曜の30分ごと)、`backfill` (手動)、`test` (push時) です。

## ドキュメント

- [設計](docs/ARCHITECTURE.md)
- [データ取得元の調査結果](docs/DATA_SOURCES.md)
- [ロードマップ](docs/ROADMAP.md)
- [デプロイ手順](docs/DEPLOY.md)
- [日常運用マニュアル](docs/OPERATIONS.md)
- [トラブル対応マニュアル](docs/TROUBLESHOOTING.md)
