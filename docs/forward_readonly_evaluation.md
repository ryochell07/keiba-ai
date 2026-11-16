# 発走前予測の評価（読み取り専用）

この機能は、実際に発走前に保存された予測だけを、後から判明した着順と照合するためのものです。**バックテストや再学習とは別の検証**です。

## 安全な実行方法

暗号化バックアップを復元する場合は、アクセス制限された作業環境に**コピーとして**復元してください。公開GitHub Actionsのログに暗号鍵、予測台帳、オッズの生データを出さないでください。

```bash
# DB が存在する作業環境で実施する（GitHub の main や Release は変更しない）
.venv/bin/keiba reconcile --org NAR --read-only
.venv/bin/keiba reconcile --org JRA --read-only
```

`--read-only` は SQLite を `mode=ro` と `PRAGMA query_only=ON` で開き、スキーマ移行・`source_status` 更新・`backtest_result` 更新・JSONファイル生成を行いません。DBがない場合は空のDBを作らず失敗します。結果は標準出力に集計値として表示されます。**その出力にも非公開の性能情報が含まれるため、公開CIではそのまま表示しないでください。**

通常の `keiba reconcile` は従来どおり集計結果とファイルを更新します。既存の毎日の運用を変更することが今回の目的ではありません。

## レポートの読み方

- `evaluation_coverage.by_time_point.*.selected_runs`: 各時点で選択された発走前予測
- `ai_scored`: AIの確率と着順だけで評価可能なもの
- `fresh_complete_odds`: 全頭・同時刻の単勝オッズが元の取得履歴と一致し、発走前・予測時点60分以内であるもの
- `paired_scored`: AI、市場、合成モデルすべての比較スコアが算出できたもの
- `paired_model_comparison.by_time_point`: **同じレースの組**で各モデルの平均LogLoss、Brier、1着的中率の差を比較。LogLoss/Brierは「左 − 右」が負なら左が優位
- `minimum_races_for_screening: 100`: 暫定的な注意閾値。100件以上であっても有意性・将来収益は保証されません

`final_odds_reference` はあくまで参考であり、発走後に確定したオッズを購入可能な価格として扱ってはいけません。従来の `odds_basis=cutoff` は互換表示で、必ずしも発走10分前ではありません。

## モデル採用の判断

精度改善の判断には、データ欠損による選択バイアス、評価件数、過去との期間差、キャリブレーション、確率精度、再現性、購入可能時点のオッズ、回収率の信頼区間を確認してください。この読み取り専用評価は**再学習・モデル採用・本番反映を行いません**。
