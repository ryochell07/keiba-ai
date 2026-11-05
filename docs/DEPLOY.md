# デプロイ手順

必要なのはGitHubの無料アカウントだけです。有料プランや支払い方法の登録は不要です。

## 構成の選択

| 候補 | 判断 |
|---|---|
| GitHub Actions + Pages + Release | **採用**。追加アカウント不要で、公開リポジトリの標準ランナーは無料 |
| Cloudflare Pages / Workers / D1 / R2 | 見送り。アカウント追加が必要で、学習は結局Actionsで動かすため利点が小さい。将来ログイン機能を強化したくなったら Cloudflare Access が候補 |

GitHub公式の確認結果 (2026-11-05):

- 公開リポジトリの標準ランナーは無料
- 非公開リポジトリは無料枠を超えると、支払い方法がなければ停止、あれば課金されます。ガードは公開リポジトリ以外でジョブを止めます
- ジョブは最大6時間、Pagesは1GBまで・月100GB (ソフト制限)、Releaseの1ファイルは2GiB未満

公開されるのはソースコードと暗号文だけです。データ、DB、モデルは暗号化してから置きます。
鍵や合言葉が漏れると読まれるので、十分に長いものを使ってください。

## 手順

1. 秘密情報を2つ作ります。1つ目が `KEIBA_STATE_KEY`、2つ目が `KEIBA_SITE_PASSPHRASE` です。

```bash
openssl rand -hex 32
```

```bash
openssl rand -base64 18
```

2. リポジトリを作成してpushします (公開リポジトリ)。

```bash
gh repo create keiba-ai --public --source . --push
```

3. 秘密情報を登録します。

```bash
gh secret set KEIBA_STATE_KEY
```

```bash
gh secret set KEIBA_SITE_PASSPHRASE
```

4. GitHubの Settings → Pages で Source を「GitHub Actions」にします。
5. ローカルのDBとモデルがあれば、クラウドへ移します (`.env` に鍵を設定してから)。

```bash
.venv/bin/keiba state pack data/keiba-state.enc
```

```bash
gh release create state data/keiba-state.enc --title "encrypted state" --notes "暗号化された作業データ" --latest=false
```

6. Actionsタブから `daily` を手動実行し、完了後に表示されるURLをスマホのホーム画面に追加します。

## 動作確認のチェックリスト

- [ ] `test` が成功する
- [ ] `daily` が成功し、ページが開ける
- [ ] Actionsの実行環境からJRAへ接続できる (海外IPからの取得が拒否されたら [TROUBLESHOOTING.md](TROUBLESHOOTING.md))
- [ ] 土日の `raceday` でオッズが増えている
