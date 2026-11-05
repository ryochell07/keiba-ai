# トラブル対応マニュアル

失敗はGitHubからメールで通知されます。Actionsの該当実行のログと、ダッシュボードの「状態」タブを確認してください。

| 症状 | 原因の候補 | 対応 |
|---|---|---|
| ページの更新が止まった | 定期実行の遅延、60日無活動での無効化 | Actionsで `daily` を手動実行。無効化されていれば Enable workflow |
| 「合言葉が違います」 | 合言葉の変更、記憶の残り | 「状態」タブか入力欄から入れ直す。忘れたら `gh secret set KEIBA_SITE_PASSPHRASE` で変更し `daily` を実行 |
| `公開リポジトリ以外では…停止` | リポジトリが非公開 | 公開に戻す。非公開で動かす場合は無料枠の超過に注意し、変数 `ALLOW_PRIVATE_RUN=true` を設定 (Pagesは別途検討が必要) |
| `HTTP 403` / `429` で停止 | 取得元が拒否。ActionsのIPが制限された可能性 | **回避しない。** 頻度を下げて様子を見る。続く場合はローカル実行に切り替え、`state pack` で同期 |
| `race_header が見つかりません` | JRAのページ構造変更 | `providers/jra_parser.py` を修正。`data/raw/` の保存ページで再解析できる |
| `カラムなし […]` | 地方CSVの仕様変更 | [説明書](https://www.keiba.go.jp/pdf/manual/data_pdf_manual.pdf) を確認し `config/nar_columns.yaml` を修正 |
| `レース数が読み込み時の…より減っている` | 復元の失敗 | 上書きを止める安全装置です。ログを確認して `daily` を再実行 |
| `状態ファイルを取得できません` | GitHubの一時障害 | 再実行。上書きはされません |
| `状態ファイルが…上限` | データが1500MB超 | 古い三連単オッズの削除などを検討。**削除は不可逆なので事前にバックアップ** |
| DBが壊れた / 誤った内容で上書きした | 実行中断など | Releaseの `keiba-state-w0〜3.enc` を `keiba-state.enc` として取り込み、`state unpack` |
| 予測が悪化した | 新モデルの問題 | `keiba rollback --org JRA` の後に `daily` |
| 暗号鍵をなくした | | 状態ファイルは復号できません。中央・地方とも `backfill` で再取得してください |
| 地方の取得条件が変わった | 協会の方針変更 | `config/sources.yaml` の `nar_csv` を `automated_fetch: false` と `permission.status: none` に戻す。手動ダウンロードしたZIPは `data/inbox/nar/` に置けば取り込める |

## ローカルでの確認

```bash
.venv/bin/keiba status
```

```bash
.venv/bin/python -m pytest -q
```

## 鍵が漏れた疑いがあるとき

1. `openssl rand -hex 32` で新しい鍵を作り、`gh secret set` で更新
2. `daily` を手動実行して新しい鍵で保存し直す
3. Releaseの古い世代 (`keiba-state-w*.enc`) は古い鍵で読めるので削除を検討 (不可逆)
