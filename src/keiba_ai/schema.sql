PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS racecourse (
  course_code TEXT PRIMARY KEY,           -- J05 / N大井 など
  name TEXT NOT NULL,
  org TEXT NOT NULL CHECK (org IN ('JRA','NAR'))
);
CREATE TABLE IF NOT EXISTS meeting (      -- 開催情報
  meeting_id TEXT PRIMARY KEY,            -- org+date+course
  org TEXT NOT NULL,
  course_code TEXT NOT NULL REFERENCES racecourse(course_code),
  race_date TEXT NOT NULL,
  kai INTEGER, nichi INTEGER
);
CREATE TABLE IF NOT EXISTS race (
  race_id TEXT PRIMARY KEY,
  meeting_id TEXT NOT NULL REFERENCES meeting(meeting_id),
  org TEXT NOT NULL,
  course_code TEXT NOT NULL REFERENCES racecourse(course_code),
  race_date TEXT NOT NULL,                -- YYYY-MM-DD
  race_no INTEGER NOT NULL,
  post_time TEXT,                         -- HH:MM
  name TEXT, grade TEXT, race_kind TEXT, category TEXT, class_name TEXT,
  rule TEXT, weight_rule TEXT, conditions TEXT,
  surface TEXT, distance INTEGER, direction TEXT,
  weather TEXT, going TEXT, n_runners INTEGER,
  source TEXT NOT NULL,
  source_ref TEXT,                        -- 取得元の識別子 (JRAならCNAME)
  updated_at TEXT NOT NULL,
  UNIQUE (org, course_code, race_date, race_no)
);
CREATE INDEX IF NOT EXISTS idx_race_date ON race(race_date);
CREATE TABLE IF NOT EXISTS horse (
  horse_id TEXT PRIMARY KEY,              -- J:登録番号 / N:馬名|生年月日 (NARはID非提供)
  name TEXT NOT NULL, sex TEXT, birth_date TEXT, coat TEXT,
  sire TEXT, dam TEXT, damsire TEXT, owner TEXT, breeder TEXT
);
CREATE TABLE IF NOT EXISTS jockey  (jockey_id  TEXT PRIMARY KEY, name TEXT NOT NULL, affiliation TEXT);
CREATE TABLE IF NOT EXISTS trainer (trainer_id TEXT PRIMARY KEY, name TEXT NOT NULL, affiliation TEXT);
CREATE TABLE IF NOT EXISTS entry (         -- 出走馬 (発走前に判明する情報のみ)
  race_id TEXT NOT NULL REFERENCES race(race_id),
  umaban INTEGER NOT NULL,
  waku INTEGER,
  horse_id TEXT NOT NULL REFERENCES horse(horse_id),
  sex TEXT, age INTEGER, carried_weight REAL,
  jockey_id TEXT REFERENCES jockey(jockey_id),
  trainer_id TEXT REFERENCES trainer(trainer_id),
  body_weight REAL, body_weight_diff REAL,
  PRIMARY KEY (race_id, umaban),
  UNIQUE (race_id, horse_id)
);
CREATE INDEX IF NOT EXISTS idx_entry_horse ON entry(horse_id);
CREATE INDEX IF NOT EXISTS idx_entry_jockey ON entry(jockey_id);
CREATE INDEX IF NOT EXISTS idx_entry_trainer ON entry(trainer_id);
CREATE TABLE IF NOT EXISTS result (        -- 確定着順 (発走後に判明。特徴量の当該レース行には使わない)
  race_id TEXT NOT NULL,
  umaban INTEGER NOT NULL,
  finish_pos INTEGER,                      -- 中止/除外/取消は NULL
  status TEXT NOT NULL DEFAULT 'finished', -- finished/dnf/scratched/excluded/disq
  time_sec REAL, margin TEXT, last3f REAL, corners TEXT,
  popularity_final INTEGER,                -- 確定人気: 評価専用
  PRIMARY KEY (race_id, umaban),
  FOREIGN KEY (race_id, umaban) REFERENCES entry(race_id, umaban)
);
CREATE TABLE IF NOT EXISTS odds_snapshot ( -- 上書きせず取得時刻ごとに履歴保存
  race_id TEXT NOT NULL REFERENCES race(race_id),
  bet_type TEXT NOT NULL,
  combo TEXT NOT NULL,                     -- 組番 "3" "3-7" "3-7-11"
  captured_at TEXT NOT NULL,               -- ISO8601 JST。最終オッズの過去取得は 'final'
  odds REAL, odds_max REAL, popularity INTEGER,
  is_final INTEGER NOT NULL DEFAULT 0,
  source TEXT NOT NULL,
  PRIMARY KEY (race_id, bet_type, combo, captured_at)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_odds_time ON odds_snapshot(race_id, captured_at);
CREATE TABLE IF NOT EXISTS payout (
  race_id TEXT NOT NULL REFERENCES race(race_id),
  bet_type TEXT NOT NULL, combo TEXT NOT NULL,
  amount INTEGER NOT NULL, popularity INTEGER,
  PRIMARY KEY (race_id, bet_type, combo)
);
CREATE TABLE IF NOT EXISTS model_version (
  model_version_id TEXT PRIMARY KEY,
  org TEXT NOT NULL, algo TEXT NOT NULL, created_at TEXT NOT NULL,
  train_start TEXT, train_end TEXT, features TEXT, params TEXT, metrics TEXT, path TEXT,
  status TEXT NOT NULL DEFAULT 'active',   -- active/retired/rejected
  adopted_at TEXT, comparison TEXT
);
CREATE TABLE IF NOT EXISTS training_run (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_version_id TEXT REFERENCES model_version(model_version_id),
  started_at TEXT NOT NULL, finished_at TEXT, n_races INTEGER, n_rows INTEGER,
  status TEXT NOT NULL, detail TEXT
);
CREATE TABLE IF NOT EXISTS prediction (
  model_version_id TEXT NOT NULL REFERENCES model_version(model_version_id),
  race_id TEXT NOT NULL REFERENCES race(race_id),
  bet_type TEXT NOT NULL, combo TEXT NOT NULL,
  predicted_at TEXT NOT NULL,
  prob REAL NOT NULL, prob_std REAL,
  odds REAL, odds_max REAL, odds_captured_at TEXT,
  ev REAL, ev_max REAL, status TEXT, flags TEXT,
  PRIMARY KEY (model_version_id, race_id, bet_type, combo, predicted_at)
);
CREATE INDEX IF NOT EXISTS idx_pred_race ON prediction(race_id);
CREATE TABLE IF NOT EXISTS backtest_result (
  backtest_id TEXT NOT NULL, created_at TEXT NOT NULL,
  org TEXT NOT NULL, strategy TEXT NOT NULL, bet_type TEXT NOT NULL,
  period_start TEXT, period_end TEXT,
  n_races INTEGER, n_bets INTEGER, n_hits INTEGER,
  stake REAL, payout REAL, roi REAL, roi_ci_low REAL, roi_ci_high REAL,
  odds_basis TEXT, detail TEXT,
  PRIMARY KEY (backtest_id, org, strategy, bet_type)
);
CREATE TABLE IF NOT EXISTS fetch_log (     -- データ取得履歴
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL, url TEXT NOT NULL, ref TEXT,
  fetched_at TEXT NOT NULL, http_status INTEGER, bytes INTEGER,
  attempts INTEGER, ok INTEGER NOT NULL, error TEXT
);
CREATE INDEX IF NOT EXISTS idx_fetch_ref ON fetch_log(source, ref, ok);
CREATE TABLE IF NOT EXISTS ingested_file ( -- CSV/ZIP の差分取り込み管理
  sha256 TEXT PRIMARY KEY, source TEXT NOT NULL, filename TEXT NOT NULL,
  imported_at TEXT NOT NULL, rows INTEGER, detail TEXT
);
CREATE TABLE IF NOT EXISTS source_status (
  source TEXT PRIMARY KEY, enabled INTEGER, automated_fetch INTEGER,
  license_status TEXT, license_note TEXT, permission TEXT, last_success_at TEXT, last_error TEXT
);

CREATE TABLE IF NOT EXISTS experiment (    -- 実験記録
  experiment_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, stage TEXT NOT NULL, name TEXT NOT NULL, org TEXT NOT NULL,
  train_period TEXT, eval_period TEXT, features TEXT, algo TEXT, params TEXT, calibration TEXT,
  n_races INTEGER, log_loss REAL, brier REAL, top1_hit REAL, roi REAL, n_bets INTEGER, max_drawdown REAL,
  roi_ci_low REAL, roi_ci_high REAL, delta_log_loss REAL, delta_ci_low REAL, delta_ci_high REAL,
  decision TEXT, model_version TEXT, used_locked_test INTEGER NOT NULL DEFAULT 0, seconds REAL
);
CREATE INDEX IF NOT EXISTS idx_experiment_stage ON experiment(stage);

-- ===== Phase 1: 予測台帳。追記専用で、更新と削除はトリガーで拒否する =====
CREATE TABLE IF NOT EXISTS prediction_run (
  prediction_id TEXT PRIMARY KEY,
  race_id TEXT NOT NULL REFERENCES race(race_id),
  org TEXT NOT NULL,
  predicted_at TEXT NOT NULL,              -- 予測生成日時
  scheduled_post_at TEXT,                  -- 予定発走日時
  time_point TEXT NOT NULL,                -- day_before/morning/t30/t10/last/unknown
  minutes_to_post REAL,
  model_version_id TEXT NOT NULL REFERENCES model_version(model_version_id),
  bundle_id TEXT NOT NULL,
  git_commit TEXT,
  input_hash TEXT NOT NULL,                -- 特徴量+オッズ+モデルのハッシュ
  feature_hash TEXT NOT NULL REFERENCES prediction_input(feature_hash),
  odds_captured_at TEXT,                   -- 使ったオッズの最新の取得時刻
  odds_status TEXT NOT NULL,               -- none/partial/complete
  odds_same_time INTEGER NOT NULL DEFAULT 0,
  used_model TEXT NOT NULL,                -- ai/blend/market: 表示と期待値に使った確率
  blend_alpha REAL, blend_beta REAL,
  taus TEXT NOT NULL,                      -- AI の τ
  taus_used TEXT NOT NULL                  -- strength_used に対する τ
);
CREATE INDEX IF NOT EXISTS idx_prun_race ON prediction_run(race_id, predicted_at);
CREATE TABLE IF NOT EXISTS prediction_horse (
  prediction_id TEXT NOT NULL REFERENCES prediction_run(prediction_id),
  umaban INTEGER NOT NULL,
  horse_id TEXT NOT NULL,
  waku INTEGER,
  p_win_ai REAL NOT NULL, p_top3_ai REAL,
  p_win_market REAL,                       -- 同じ時刻の単勝オッズを正規化。全頭そろわなければ NULL
  p_win_blend REAL, p_top3_blend REAL,
  p_win_used REAL NOT NULL,
  strength_ai REAL NOT NULL,
  strength_used REAL NOT NULL,
  win_odds REAL, place_odds REAL, place_odds_max REAL,
  odds_captured_at TEXT,
  PRIMARY KEY (prediction_id, umaban)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS prediction_input (  -- モデルに渡した入力そのもの。同じ内容は1件にまとめる
  feature_hash TEXT PRIMARY KEY,
  columns_hash TEXT NOT NULL,
  n_rows INTEGER NOT NULL,
  payload BLOB NOT NULL
);
-- 表示専用の最新値。記録ではない (記録は prediction_run / prediction_horse)
CREATE TABLE IF NOT EXISTS prediction_display (
  race_id TEXT NOT NULL, bet_type TEXT NOT NULL, combo TEXT NOT NULL,
  model_version_id TEXT NOT NULL, prediction_id TEXT,
  predicted_at TEXT NOT NULL,
  prob REAL NOT NULL, prob_std REAL,
  odds REAL, odds_max REAL, odds_captured_at TEXT,
  ev REAL, ev_max REAL, status TEXT, flags TEXT,
  PRIMARY KEY (race_id, bet_type, combo)
);
-- 発走前の情報を取得するたびに、変わった値だけを記録する。published_at は提供元が公表時刻を出さない限り NULL
CREATE TABLE IF NOT EXISTS field_observation (
  observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
  race_id TEXT NOT NULL,
  umaban INTEGER NOT NULL DEFAULT 0,       -- 0 はレース全体の項目
  field TEXT NOT NULL,
  value TEXT,
  observed_at TEXT NOT NULL,               -- こちらが取得した時刻
  published_at TEXT,
  source TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fobs ON field_observation(race_id, umaban, field, observed_at);
-- モデル本体・校正・合成係数をひとまとめにした版
CREATE TABLE IF NOT EXISTS model_bundle (
  bundle_id TEXT PRIMARY KEY,
  model_version_id TEXT NOT NULL REFERENCES model_version(model_version_id),
  org TEXT NOT NULL,
  created_at TEXT NOT NULL,
  git_commit TEXT,                         -- 学習時のコミット。不明なら NULL (推測で埋めない)
  model_hash TEXT NOT NULL,
  train_start TEXT, train_end TEXT, tree_train_end TEXT,
  valid_start TEXT, valid_end TEXT, valid_period_basis TEXT,
  n_train_races INTEGER, n_valid_races INTEGER,
  feature_version TEXT, columns_hash TEXT, features TEXT,
  hyperparams TEXT, n_seeds INTEGER, n_trees TEXT,
  taus TEXT, tau_period TEXT,
  blend_alpha REAL, blend_beta REAL, blend_source TEXT, blend_fit_period TEXT, blend_ci TEXT,
  guard_recommendation TEXT, guard_reason TEXT,
  adoption_reason TEXT, note TEXT
);
CREATE INDEX IF NOT EXISTS idx_bundle_model ON model_bundle(model_version_id, created_at);
CREATE TABLE IF NOT EXISTS odds_capture_daily (  -- 集計値なので再計算で置き換える
  race_date TEXT NOT NULL, org TEXT NOT NULL,
  n_races INTEGER, any_rate REAL, t30_rate REAL, t10_rate REAL, last_rate REAL,
  median_delay_t10_min REAL, p90_delay_t10_min REAL, missing_rate REAL, n_errors INTEGER, n_snapshots INTEGER,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (race_date, org)
);
CREATE TRIGGER IF NOT EXISTS prediction_run_no_update BEFORE UPDATE ON prediction_run BEGIN SELECT RAISE(ABORT, '予測台帳は変更できません'); END;
CREATE TRIGGER IF NOT EXISTS prediction_run_no_delete BEFORE DELETE ON prediction_run BEGIN SELECT RAISE(ABORT, '予測台帳は削除できません'); END;
CREATE TRIGGER IF NOT EXISTS prediction_horse_no_update BEFORE UPDATE ON prediction_horse BEGIN SELECT RAISE(ABORT, '予測台帳は変更できません'); END;
CREATE TRIGGER IF NOT EXISTS prediction_horse_no_delete BEFORE DELETE ON prediction_horse BEGIN SELECT RAISE(ABORT, '予測台帳は削除できません'); END;
CREATE TRIGGER IF NOT EXISTS prediction_input_no_update BEFORE UPDATE ON prediction_input BEGIN SELECT RAISE(ABORT, '入力の記録は変更できません'); END;
CREATE TRIGGER IF NOT EXISTS prediction_input_no_delete BEFORE DELETE ON prediction_input BEGIN SELECT RAISE(ABORT, '入力の記録は削除できません'); END;
CREATE TRIGGER IF NOT EXISTS field_observation_no_update BEFORE UPDATE ON field_observation BEGIN SELECT RAISE(ABORT, '取得履歴は変更できません'); END;
CREATE TRIGGER IF NOT EXISTS field_observation_no_delete BEFORE DELETE ON field_observation BEGIN SELECT RAISE(ABORT, '取得履歴は削除できません'); END;
CREATE TRIGGER IF NOT EXISTS model_bundle_no_update BEFORE UPDATE ON model_bundle BEGIN SELECT RAISE(ABORT, 'モデル版の記録は変更できません'); END;
CREATE TRIGGER IF NOT EXISTS model_bundle_no_delete BEFORE DELETE ON model_bundle BEGIN SELECT RAISE(ABORT, 'モデル版の記録は削除できません'); END;
-- 以前の予測テーブルは保存したまま凍結する
CREATE TRIGGER IF NOT EXISTS prediction_legacy_no_update BEFORE UPDATE ON prediction BEGIN SELECT RAISE(ABORT, '旧予測テーブルは凍結しています'); END;
CREATE TRIGGER IF NOT EXISTS prediction_legacy_no_delete BEFORE DELETE ON prediction BEGIN SELECT RAISE(ABORT, '旧予測テーブルは凍結しています'); END;
CREATE TRIGGER IF NOT EXISTS prediction_legacy_no_insert BEFORE INSERT ON prediction BEGIN SELECT RAISE(ABORT, '旧予測テーブルは凍結しています'); END;
