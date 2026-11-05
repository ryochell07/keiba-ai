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
