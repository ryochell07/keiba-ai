# 発走前予測の評価（読み取り専用）

この機能は、実際に発走前に保存された予測だけを、後から判明した着順と照合するためのものです。**バックテストや再学習とは別の検証**です。

予測時の馬番集合・実出走馬番集合・結果取得行が一致しないレースは採点しない。**出走取消、同着1着、未取得の着順**を通常の的中/不的中に混ぜず、除外件数を報告する。実データでの変更前後の対象レース数は異なる可能性がある。

## 安全な実行方法

暗号化バックアップを復元する場合は、アクセス制限された作業環境に**コピーとして**復元してください。公開GitHub Actionsのログに暗号鍵、予測台帳、オッズの生データを出さないでください。

```bash
# DB が存在する作業環境で実施する（GitHub の main や Release は変更しない）
.venv/bin/keiba reconcile --org NAR --read-only
.venv/bin/keiba reconcile --org JRA --read-only
```

`--read-only` は SQLite を `mode=ro` と `PRAGMA query_only=ON` で開き、スキーマ移行・`source_status` 更新・`backtest_result` 更新・JSONファイル生成を行いません。DBがない場合は空のDBを作らず失敗します。予測取得側も `is_final=1` の確定オッズを予測前の市場値として使用しないよう保護しています。結果は標準出力に集計値として表示されます。**その出力にも非公開の性能情報が含まれるため、公開CIではそのまま表示しないでください。**

通常の `keiba reconcile` は従来どおり集計結果とファイルを更新します。既存の毎日の運用を変更することが今回の目的ではありません。

## 暗号化Releaseを隔離復号して読み取り専用で評価する

`keiba reconcile --read-only --encrypted-state FILE --output FILE` は `KEIBA_STATE_KEY` を環境変数から読み、暗号化Releaseの `keiba.db` **だけ**をOSの権限制限付き一時ディレクトリで復号する。復元操作 `state.unpack` は呼ばず、本番の `data/keiba.db`・採用済みモデル・レポート・Releaseへは一切書き込まない。DBはSQLite `mode=ro&immutable=1` と `PRAGMA query_only=ON` で開き、一時データは処理の正常終了・例外終了時ともに削除する。ただし、プロセス強制終了・ホスト障害時の一時ファイル残存までは保証できないため、アクセス制限のある信頼済み環境で実行する。

```bash
# 信頼できる端末で、検証用ブランチをチェックアウトして依存関係をセットアップした後
# KEIBA_STATE_KEY は安全な秘密情報管理から設定し、コマンド履歴や公開CIログに表示しない
# 公開GitHub Releaseから暗号化済みファイルだけを取得（この時点では復号されない）
gh release download state --repo ryochell07/keiba-ai --pattern keiba-state.enc --dir /path/to/private-workspace
# 集計結果は非公開領域の新規ファイルへだけ保存する（標準出力には表示しない）
# private-workspace はGit管理対象のリポジトリの外側に置く
umask 077
.venv/bin/keiba reconcile --read-only --org NAR \
  --encrypted-state /path/to/private-workspace/keiba-state.enc \
  --output /path/to/private-workspace/nar-evaluation.json
.venv/bin/keiba reconcile --read-only --org JRA \
  --encrypted-state /path/to/private-workspace/keiba-state.enc \
  --output /path/to/private-workspace/jra-evaluation.json
```

暗号化状態の評価では `--output` が必須。結果ファイルは所有者だけが読めるモード `0600` で新規作成する。既存のファイルを上書きせず、**Git管理リポジトリ内への保存を拒否**する。評価データ（モデルの精度・回収率）そのものが機密になり得るため、CIログには出力しない。例の保存先パスは必ず信頼できるローカル環境に置き換える。

**運用上の重要事項：** ChatGPTや公開のIssue/PR/Actionsログに秘密鍵を貼らない。復号したDBをGitHubへアップロードしない。上記は利用者のローカル端末やアクセス制御された環境で実行する手順で、**現段階では実運用DBでの実行結果がない**。秘密鍵を利用するGitHub Actionsを新設する場合は、その実行コードを別途レビューし、本番権限付与は明示的に承認した後に行う。PR上の未承認コードに秘密鍵を渡して実行することは避ける。

## 手動GitHub Actions評価（レビュー後にのみ有効）

検証用PRに `.github/workflows/forward-evaluate-readonly.yml` を追加した。**この時点ではDraftブランチ内にあるだけで、秘密鍵付きワークフローの実行はしていない**。GitHubの手動ワークフロー登録には、独立したコード・権限レビューを終えたうえでデフォルトブランチへ明示的に採用する必要がある。今回の作業ではmainへのマージやSecretを使う実行を行わない。

**必須の実行前設定：mainブランチの保護。** 2026年11月16日の確認ではGitHubの `main.protected` が `false` だった。リポジトリの Settings → Branches（またはRulesets）で `main` を保護し、少なくとも直接pushの制限・PRレビュー・必須テストを設定すること。GitHubの `github.ref_protected == true` も実行条件に追加しているため、保護されないmainからは秘密鍵付きのジョブが**実行されない**。保護設定は本PRから自動変更しない。設定済みでも管理者による回避権限や権限変更に注意し、実行するコミットのレビューと権限管理を別途徹底する。

レビュー・採用後のワークフローは、GitHubのリポジトリ所有者が保護された `main` で `workflow_dispatch` を起動した場合だけ動く。権限は `contents: read` のみ。バックアップ取得に既存の状態復元アクションは使用せず、Releaseから暗号化された `keiba-state.enc` をダウンロードするだけ。採用モデル・DB・Releaseへの書込み、再学習、ページ更新、スケジュール実行は一切しない。本番状態書き込み用の同時実行ロック `keiba-state` も使用しない。

集計データはジョブの一時ディレクトリに `0600` で作成し、すぐにAES-GCMで別途暗号化する。レポートの暗号鍵は `KEIBA_STATE_KEY` からHKDFで用途分離して生成するため、**GitHub Actionsのartifactとして公開されるのは暗号化ファイルだけ**。メトリクスや鍵は公開ログに出さない。暗号化済みartifactの保存期間は3日。一時ファイルは通常終了・エラー時とも削除処理を試みるが、異常なホスト停止による永続消去までは保証できない。

レビュー後に手動実行できるようになったら、リポジトリのActions画面から `forward-evaluate-readonly` を開き、Run workflowで **branch: main**、`org: BOTH`（またはNAR/JRA）を選択する。完了したrunの `forward-evaluation-encrypted` artifactをローカルにダウンロードするか、GitHub CLIで次のように取得する。

```bash
gh run download <RUN_ID> --repo ryochell07/keiba-ai \
  --name forward-evaluation-encrypted --dir /path/to/private-workspace

# 元の KEIBA_STATE_KEY をローカルの秘密情報管理から設定したうえで
umask 077
.venv/bin/python -m keiba_ai.eval_artifact decrypt \
  --input /path/to/private-workspace/forward-evaluation.enc \
  --output /path/to/private-workspace/forward-evaluation.json
```

**注意：** GitHub Secretsの設定画面から秘密鍵の値を後で読み戻すことはできない。元の鍵を安全に保管していない場合、artifactは復号できない。鍵の共有やチャット貼付を依頼せず、安全な復号方法・アクセス経路を先に確保する必要がある。評価結果のJSONをそのまま公開Issue・PR・Actionsログに掲載しない。

## レポートの読み方

- `excluded_runs_without_verified_prestart_time`: 発走予定時刻が欠ける・形式が不正・予測が発走予定時刻以降など、**発走前だったことを検証できない**予測履歴の除外件数。予測時刻・発走時刻はタイムゾーン付き日時として比較し、同時点内の選択も実際の時系列で行う
- `by_time_point.*.used_all_races`: 実際に利用・表示した `p_win_used` のLogLoss/Brier/的中率。選択された予測はオッズが不完全でも評価可能なため、AI/市場/合成の**同条件比較**とは母数が異なる可能性がある
- `by_time_point.*.*.top1_roi_races` / `payout_coverage_rate`: 勝ち馬の単勝払戻データが取得済みのレース数と、そのレース数を精度評価可能数で割った比率。**着順は取れても払戻が未取得なら、回収額を0円と置かずROI算出の母数から除外**する。対象が0レースなら `top1_roi=null`。誤って高い回収率と受け取らないよう、常にカバー率と併記する
- `ev100_roi`: EV条件を満たしていても、勝ち馬の払戻が確認できないレースはベット件数・払戻を集計しない。未取得払戻と不的中は区別する。
- `evaluation_coverage.by_time_point.*.used_model_counts`: `ai` / `market` / `blend` / その他として、どの予測方式を**採用したか**の件数。採用件数と評価可能件数は区別する
- `evaluation_coverage.by_time_point.*.deployed_scored`: 採用した確率ベクトルが妥当で、着順と照合できたレース数
- `evaluation_coverage.by_time_point.*.incomplete_results`: 結果テーブルに全出走登録馬の行がそろっていないレース。結果取得中・一部欠落の可能性があるため精度とROI双方から除外
- `evaluation_coverage.by_time_point.*.runner_set_mismatch`: 予測台帳に保存した馬番集合と、確定結果の実際の出走馬番集合が異なるレース。後から取消・除外が判明した場合などを含み、評価対象外
- `evaluation_coverage.by_time_point.*.multiple_or_missing_winners`: 同着1着など勝ち馬が一意でないレース。従来の単一勝者のLogLoss/Brierでは比較不能として除外
- `n_races_with_selected_prestart_run`: 発走前の予測履歴と**結果の行が1件以上**存在したレース数（勝ち馬の有無・全頭整合性の検証前）。勝ち馬の結果だけ欠損しているレースも母数に残して除外原因に計上する。結果行が一切存在しないレースは完了・未取得を区別できないため、この母数に含めない。`n_races` は整合性検証後の券種別集計レース数
- `evaluation_coverage.by_time_point.*.outcome_valid` / `outcome_coverage_rate`: 結果の馬番集合と単一勝ち馬の条件を満たした件数・選択予測に対する割合。**結果が不完全なために評価できないレース**を把握する
- `evaluation_coverage.by_time_point.*.paired_rate_after_outcome_validation`: 正しい結果が確認できたレースを分母とした3モデルの比較可能率。オッズ取得・予測確率の不足だけを評価する補助指標。結果の整合性を満たすレースが0件なら `null`
- `evaluation_coverage.by_time_point.*.selected_runs`: 各時点で選択された発走前予測
- `ai_scored`: AIの確率と着順だけで評価可能なもの
- `fresh_complete_odds`: 全頭・同時刻の単勝オッズが元の非確定取得履歴と一致し、発走前・予測時点60分以内であるもの
- `paired_scored`: AI、市場、合成モデルすべての比較スコアが算出できたもの
- `odds_metadata_invalid`: 単勝オッズの全頭取得・同時刻フラグが不足
- `odds_timing_invalid`: オッズ取得時刻が不適切（発走後・未来・予測時点60分超など）
- `odds_snapshot_mismatch`: 元の単勝オッズ取得履歴と保存済み台帳が一致しない
- `paired_model_probabilities_invalid`: 3モデルのどれかの確率が欠損、非有限、正規化されていないなど
- `paired_model_comparison.by_time_point`: **同じレースの組**で各モデルの平均LogLoss、Brier、1着的中率の差を比較。LogLoss/Brierは「左 − 右」が負なら左が優位
- `minimum_races_for_screening: 100`: 暫定的な注意閾値。100件以上であっても有意性・将来収益は保証されません
- `n_model_versions` / `mixed_model_versions`: 同じ時点の対象レースで異なる採用モデルが混在していないか確認。混在した結果を単一モデルの精度とみなさない
- `n_race_dates`: 評価対象レースが行われた異なる開催日数
- `ci95`: 10開催日以上ある場合のみ、開催日を単位に2000回復元抽出した平均差の95%区間。**同日の複数レースを独立標本と誤認しない**ための探索的指標で、10日未満または開催日情報不足なら `null`。ロジックの再現性を保つため乱数は固定。区間が0をまたがなくても多重検定・分布変動があるので採用根拠として単独使用しない

`final_odds_reference` はあくまで参考であり、発走後に確定したオッズを購入可能な価格として扱ってはいけません。従来の `odds_basis=cutoff` は互換表示で、必ずしも発走10分前ではありません。

## オッズ取得品質の監視と本番の凍結設定

`odds_quality` の `any_rate`・`t30_rate`・`t10_rate`・`last_rate` は、**取消・除外馬を除いたすべての出走馬番について、同じ発走前スナップショットに正の単勝オッズが揃い、余分な正オッズの馬番が存在しないレース**だけを分子にする。単純な取得頭数の最大値で判定すると、部分取得と取消馬の残存で過大評価されるため、修正前の値と修正後の値をそのまま性能改善・悪化として比較しない。

`n_snapshots` も完全取得できた発走前の取得回数を数える。`missing_rate` は最後の発走前スナップショットにおける現役出走馬の未取得割合で、翌日の結果データを使う場合には取消・除外の状態も反映される。

現行の `config/sources.yaml` は `guards.freeze_models: true` で自動再学習・自動採用を停止中。また `market_only_orgs: [JRA]` のため中央はオッズが揃った場合、市場単独を予測の採用値とする。設定が存在することと、再学習が有効化されていることは別。**評価結果だけでは凍結解除やモデルの変更をしない。**

## 予測生成時の市場オッズ鮮度（Draft PR での新ルール）

過去データの照合だけでなく、予測生成時にも次のガードを適用する。単勝オッズが全頭揃っていても取得時刻が異なる、または取得時点から予測時点まで60分超経過している場合、**市場単独やAI＋市場の予測は採用せず、AI単独にフォールバックする**。各馬の取得時刻・価格は台帳にそのまま残し、`odds_status=complete` は「全頭の数値がある」という意味に限定する。`used_model` が実際の採用方式を示す。鮮度の真偽も予測入力のハッシュに含めるため、価格が変わらなくても有効期間を越えた際には改めて予測記録を作れる。

予想画面で個別オッズが60分超など利用不適格なら、価格は参考表示しても **期待値 `ev` / `ev_max` を算出しない**。表示状態は `オッズ古い・評価対象外` とし、取得のない券は `オッズ未取得` のままにする。個別券種でオッズが利用可能でも、単勝の全頭同時刻条件を満たさなければ市場合成モデルは利用しない。市場モデルの品質評価と、個別オッズに対する推定期待値の表示は異なる条件である。

最新の取得が一部馬番のみだった場合は、追加の外部アクセスをせず、保存済みの同時刻・全頭・60分以内の単勝オッズから**もっとも新しい完全なスナップショット**を探す。見つかればその一式を市場確率に使用し、見つからなければ従来どおりAI単独へ戻す。異なる時刻の馬番別データを混ぜることはしない。さらに、**正のオッズが存在する馬番集合が現在の出走馬番集合と完全一致**することも必須とし、余分な取消馬の正オッズが残る時点は不採用にする。新しい個別取得値にゼロや異常値がある場合は発売停止等の可能性があるため、過去の正オッズに戻さない。診断件数は `market_recovered_snapshot_races` として集計し、`market_ready_races` にも含める。予測台帳には採用した完全スナップショットの時刻を残す。

画面の個々の買い目に表示するオッズは別途もっとも新しい発走前の個別値を使うため、**市場確率の作成に使った全頭同時刻オッズと、表示上の個別参考価格は異なる場合がある**。両者を同一価格として扱わず、期待値表示については個別オッズの鮮度を検証する。

`latest_odds` の日時比較にはSQLiteの `julianday` によるUTCオフセット正規化を用い、辞書順で発走後の価格を拾わない。確定オッズ（`is_final=1`）を予測に使わない制限も維持する。今後本番適用する前に、取得頻度・性能・採用率への影響を検証する。

## 学習・予測特徴量における時刻の境界

`features.add_market` は、確定オッズを除外したうえで、各レースの**発走予定10分前以前**に取得された市場オッズだけを特徴量へ取り込む。さらに実運用の `predict_races` から呼ぶ場合、`as_of=predicted_at` を渡し、**予測を実行した瞬間より後の取得履歴を利用できない**。オッズ時刻はUTCオフセット込みの実時刻で比較する。タイムゾーンを持たないオッズ履歴は安全側に除外する。

`as_of` を指定しない学習・既存バックテストの動作は、従来どおりレースの発走10分前が上限。これは「学習時に予測を再現すること」と「実際に発走前に記録済みの予測を評価すること」の区別であり、**過去オッズを使った学習指標だけでは実運用の優位性は証明できない**。通常の `forward reconcile` は保存済み予測を評価する。

## 従来のバックテストの `cutoff` 価格

`backtest._cutoff_odds` の `cutoff` は、レースの発走予定10分前を基準に、それ以前の最後の非確定オッズを取得する**過去データ参考用**の基準。オッズ日時はUTCオフセット込みで正しく比較し、基準時刻より未来の取得、基準時刻から60分超古い取得、ゼロ・非有限の価格、時刻のタイムゾーン情報がない取得を除外する。修正前と修正後ではEV戦略の評価対象が変わる可能性がある。

ただし、この従来型バックテストは券種・組番ごとに直近の価格を組み合わせるため、**全馬同一時刻に購入可能だったことを保証するものではない**。信頼できる実運用の評価・採用判断には、保存済みの予測と実取得時刻・元スナップショットを照合する `forward reconcile` を優先する。発走後の確定オッズ `final_optimistic` も購入可能価格ではない。

なお、`used_model=ai` は「明示的な市場合成を採用していない」ことを意味し、学習済みAIモデル自体の特徴量に `MARKET` 系の列が含まれないことまでは保証しない。候補モデル `ai` と市場モデルを完全に独立した情報源とみなさず、モデルの使用特徴量も監査する。

## モデル採用の判断

精度改善の判断には、データ欠損による選択バイアス、評価件数、過去との期間差、キャリブレーション、確率精度、再現性、購入可能時点のオッズ、回収率の信頼区間を確認してください。この読み取り専用評価は**再学習・モデル採用・本番反映を行いません**。
