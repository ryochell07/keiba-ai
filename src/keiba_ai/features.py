"""特徴量生成。学習・バックテスト・予測はすべて build_features() を通す (同一ロジック)。

未来情報の混入防止の原則:
- 着順・タイム・上がり・通過順・確定人気・払戻・確定オッズは、当該レースの特徴量に使わない。
- 過去成績は「同じ馬の前走まで」(shift) または「前日までの集計」(as-of) だけを使う。
- 市場情報は captured_at が判定時刻 (発走 cutoff 分前) 以前のスナップショットだけを使う。
"""
from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CATEGORICAL = ["course_code", "surface", "going", "weather", "direction", "sex", "weight_rule", "race_kind",
               "grade", "category", "x_course_dist", "x_cond_age"]
# 追加特徴量のグループ。採否は実験 (keiba lab groups) の結果で決める
GROUPS = {"speed": "x_speed_", "apt": "x_apt_", "opp": "x_opp_", "pace": "x_pace_", "jk": "x_jk_",
          "course": "x_course_", "day": "x_day_", "cond": "x_cond_", "pastmkt": "x_pastmkt_"}
ADOPTED_GROUPS: tuple[str, ...] = ("speed", "apt", "opp", "jk")  # 2026-11-06 の封鎖テストで採用
NUMERIC_RACE = ["distance", "dist_bucket", "n_runners", "race_no", "month", "waku", "umaban", "umaban_ratio",
                "class_ord", "age", "carried_weight", "body_weight", "body_weight_diff", "weight_ratio"]
# 発走直前まで確定しない項目。朝の予測では欠損になるので、学習時に一定割合を欠損にして慣らす。
LATE_INFO = ["body_weight", "body_weight_diff", "weight_ratio", "going", "weather"]
MARKET = ["market_win_odds", "market_log_odds", "market_pop", "market_drift"]
FORBIDDEN = re.compile(r"(^|_)(finish_pos|status|time_sec|margin|last3f|corners|popularity_final|payout|amount|final|is_win|is_top)($|_)")
WINDOWS = (3, 5, 10)
CLASS_ORD = {"新馬": 0, "未出走": 0, "未勝利": 1, "1勝": 2, "500万": 2, "2勝": 3, "1000万": 3, "3勝": 4,
             "1600万": 4, "オープン": 5, "OP": 5, "Listed": 6, "リステッド": 6, "GⅢ": 7, "GⅡ": 8, "GⅠ": 9}


def load_frame(con: sqlite3.Connection, org: str, until: str | None = None) -> pd.DataFrame:
    """entry を軸に race/result を結合する。until を渡すとその日以前のレースだけになる。"""
    sql = """
    SELECT e.race_id, e.umaban, e.waku, e.horse_id, e.sex, e.age, e.carried_weight, e.jockey_id, e.trainer_id,
           e.body_weight, e.body_weight_diff,
           r.org, r.course_code, r.race_date, r.race_no, r.post_time, r.name AS race_name, r.conditions, r.grade, r.race_kind, r.category, r.class_name,
           r.weight_rule, r.surface, r.distance, r.direction, r.weather, r.going, r.n_runners,
           h.sire,
           x.finish_pos, x.status, x.time_sec, x.margin, x.last3f, x.corners, x.popularity_final
    FROM entry e JOIN race r USING (race_id) JOIN horse h USING (horse_id)
    LEFT JOIN result x ON x.race_id=e.race_id AND x.umaban=e.umaban
    WHERE r.org=? AND r.course_code NOT LIKE 'N帯広%'
    """
    args: list = [org]
    if until:
        sql += " AND r.race_date<=?"
        args.append(until)
    df = pd.read_sql_query(sql, con, params=args)
    df = df[~df["status"].isin(["scratched", "excluded"])].copy()
    # 取消・除外を除いた実際の頭数 (発走前に判明)。
    df["n_runners"] = df.groupby("race_id")["umaban"].transform("count")
    return df.sort_values(["race_date", "post_time", "race_id", "umaban"], ignore_index=True)


def serving_frame(frame: pd.DataFrame, today: str) -> pd.DataFrame:
    """予測用の行。学習と同じく、結果のない過去の行 (取り止めのレースなど) は履歴に入れない。"""
    return frame[frame["status"].notna() | (frame["race_date"] >= today)].reset_index(drop=True)


_NAR_CLASS = re.compile(r"(?<![A-Za-z])([ABC])\s*([1-4])?(?![A-Za-z])")


def _class_ord(row_class: pd.Series, grade: pd.Series, name: pd.Series) -> pd.Series:
    """競走クラスの序列。地方はレース名の A1/B2/C3 などから読む。"""
    def one(c, g, n):
        if isinstance(g, str) and g in CLASS_ORD:
            return CLASS_ORD[g]
        if isinstance(c, str):
            for k, v in CLASS_ORD.items():
                if k in c:
                    return v
        if isinstance(n, str):
            m = _NAR_CLASS.search(unicodedata.normalize("NFKC", n))
            if m:
                return {"A": 6, "B": 4, "C": 2}[m.group(1)] + (2 - int(m.group(2) or 2)) * 0.5
        return np.nan
    return pd.Series([one(c, g, n) for c, g, n in zip(row_class, grade, name)], index=row_class.index, dtype=float)


def _first_corner(s: pd.Series, which: int) -> pd.Series:
    def one(v):
        if not isinstance(v, str) or not v:
            return np.nan
        p = v.split(",")
        try:
            return float(p[which])
        except (ValueError, IndexError):
            return np.nan
    return s.map(one)


def asof_stats(df: pd.DataFrame, keys: list[str], values: list[str], prefix: str, count: bool = True) -> pd.DataFrame:
    """キーごとに「前日まで」の累積平均を計算する。同日分は含まない。キーが欠損の行は NaN。"""
    out = pd.DataFrame(index=df.index, dtype=float)
    for v in values:
        out[f"{prefix}_{v}_rate"] = np.nan
    if count:
        out[f"{prefix}_n"] = 0.0
    sub = df.loc[df[keys].notna().all(axis=1), keys + ["race_date"] + values]
    if sub.empty:
        return out
    day = sub.groupby(keys + ["race_date"], sort=True)[values].agg(["sum", "count"])
    day.columns = [f"{a}__{b}" for a, b in day.columns]
    levels = list(range(len(keys)))
    cum = day.groupby(level=levels).cumsum().groupby(level=levels).shift(1).reset_index()
    merged = sub[keys + ["race_date"]].merge(cum, how="left", on=keys + ["race_date"])
    merged.index = sub.index
    for v in values:
        out.loc[sub.index, f"{prefix}_{v}_rate"] = merged[f"{v}__sum"] / merged[f"{v}__count"].replace(0, np.nan)
    if count:
        out.loc[sub.index, f"{prefix}_n"] = merged[f"{values[0]}__count"].fillna(0)
    return out


def add_market(df: pd.DataFrame, con: sqlite3.Connection, cutoff_min: int = 10) -> pd.DataFrame:
    """判定時刻より前に取得した単勝オッズだけを結合する。確定オッズ ('final') は決して使わない。"""
    out = df.copy()
    for c in MARKET:
        out[c] = np.nan
    if df.empty:
        return out
    q = pd.read_sql_query(
        """SELECT o.race_id, CAST(o.combo AS INTEGER) AS umaban, o.captured_at, o.odds, o.popularity
           FROM odds_snapshot o JOIN race r USING (race_id)
           WHERE o.bet_type='win' AND o.is_final=0 AND o.captured_at<>'final' AND r.org=?""",
        con, params=[df["org"].iloc[0]])
    if q.empty:
        return out
    post = df[["race_id", "race_date", "post_time"]].drop_duplicates("race_id")
    q = q.merge(post, on="race_id")
    limit = pd.to_datetime(q["race_date"] + " " + q["post_time"].fillna("00:00")) - pd.Timedelta(minutes=cutoff_min)
    q["ts"] = pd.to_datetime(q["captured_at"]).dt.tz_localize(None)
    q = q[q["ts"] <= limit].sort_values("ts")
    if q.empty:
        return out
    last = q.groupby(["race_id", "umaban"]).agg(market_win_odds=("odds", "last"), first_odds=("odds", "first"),
                                                market_pop=("popularity", "last")).reset_index()
    merged = out[["race_id", "umaban"]].merge(last, how="left", on=["race_id", "umaban"])
    out["market_win_odds"] = merged["market_win_odds"].to_numpy()
    out["market_log_odds"] = np.log(out["market_win_odds"])
    out["market_pop"] = merged["market_pop"].to_numpy()
    out["market_drift"] = (merged["market_win_odds"] / merged["first_odds"]).to_numpy()
    return out


def _smooth(rate: pd.Series, n: pd.Series, prior: float, k: float = 20.0) -> pd.Series:
    """件数の少ない組み合わせを全体平均へ縮める。"""
    return (rate.fillna(prior) * n.fillna(0) + prior * k) / (n.fillna(0) + k)


def _window_rate(d: pd.DataFrame, key: str, val: str, days: int, prior: float) -> pd.Series:
    """前日までの直近 days 日の平滑化率。当日分は含まない。履歴がなければ prior。

    結果のない日 (未確定レース) にも行を作る。行がないと結合で欠損になり、本番だけ prior に潰れる。
    """
    rows = d.loc[d[key].notna(), [key, "date"]].drop_duplicates()
    if rows.empty:
        return pd.Series(np.nan, index=d.index)
    sub = d.loc[d[key].notna() & d[val].notna(), [key, "date", val]]
    known = sub.groupby([key, "date"])[val].agg(["sum", "count"]).reset_index()
    daily = rows.merge(known, how="left", on=[key, "date"]).fillna({"sum": 0.0, "count": 0.0}).sort_values([key, "date"], ignore_index=True)
    rolled = (daily.set_index("date").groupby(key)[["sum", "count"]]
              .rolling(f"{days}D", closed="left").sum().reset_index())
    merged = d[[key, "date"]].merge(rolled, how="left", on=[key, "date"])
    merged.index = d.index
    return (merged["sum"].fillna(0) + prior * 20) / (merged["count"].fillna(0) + 20)


def _extra_features(d: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """追加特徴量。当該レースの結果は shift か前日までの集計を通してだけ使う。"""
    x: dict[str, pd.Series] = {}
    by_horse = d["horse_id"]
    roll = lambda s, w: s.groupby(by_horse, sort=False).rolling(w, min_periods=1).mean().droplevel(0)
    # --- 条件補正スピード指数: 同じ競馬場・馬場種別・距離・馬場の前日までの平均と、当日の時計の出方を除く ---
    b1 = asof_stats(d, ["course_code", "surface", "distance", "going"], ["h_speed"], "b1")
    b0 = asof_stats(d, ["course_code", "surface", "distance"], ["h_speed"], "b0", count=False)
    base = b1["b1_h_speed_rate"].where(b1["b1_n"] >= 20, b0["b0_h_speed_rate"])
    resid = d["h_speed"] - base
    day_bias = resid.groupby([d["race_date"], d["course_code"], d["surface"]], dropna=False).transform("mean")
    d["h_fig"] = resid - day_bias
    shifted = d.groupby("horse_id", sort=False)["h_fig"].shift(1)
    for w in WINDOWS:
        x[f"x_speed_m{w}"] = roll(shifted, w)
    x["x_speed_best5"] = shifted.groupby(by_horse, sort=False).rolling(5, min_periods=1).min().droplevel(0)
    x["x_speed_last"] = shifted
    x["x_speed_trend"] = x["x_speed_m3"] - x["x_speed_m10"]
    x["x_speed_behind_km_m5"] = roll((d["h_behind"] / d["distance"] * 1000).groupby(by_horse, sort=False).shift(1), 5)
    grp = x["x_speed_m5"].groupby(d["race_id"])
    x["x_speed_rel"] = (x["x_speed_m5"] - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)
    x["x_speed_rank"] = grp.rank(pct=True)
    # --- 条件別の能力 -------------------------------------------------------------
    for keys, name in ((["horse_id", "surface"], "surface"), (["horse_id", "dist_bucket"], "dist"),
                       (["horse_id", "course_code"], "course"), (["horse_id", "going"], "going")):
        a = asof_stats(d, keys, ["h_fig"], f"x_apt_{name}")
        x[f"x_apt_{name}_fig"] = a[f"x_apt_{name}_h_fig_rate"]
    long_rest = feats["rest_days"].ge(90)
    d["h_long_rest"] = long_rest.astype(float)
    a = asof_stats(d, ["horse_id", "h_long_rest"], ["h_top3"], "x_apt_rest")
    x["x_apt_rest_top3"] = _smooth(a["x_apt_rest_h_top3_rate"], a["x_apt_rest_n"], 0.3, 5)
    x["x_apt_long_rest"] = d["h_long_rest"].where(feats["rest_days"].notna())
    # --- 相手関係: 過去に戦ったレースの相手の強さ -----------------------------------
    field = x["x_speed_m5"].groupby(d["race_id"]).transform("mean")
    d["h_field"] = field
    past_field = d.groupby("horse_id", sort=False)["h_field"].shift(1)
    x["x_opp_field_m5"] = roll(past_field, 5)
    x["x_opp_now"] = field
    x["x_opp_self_vs_now"] = field - x["x_speed_m5"]
    # --- 展開: 過去走の通過順から想定する ---------------------------------------------
    style = feats["corner_first_m5"]
    bands = {"front": style.lt(0.2), "stalk": style.ge(0.2) & style.lt(0.45), "mid": style.ge(0.45) & style.lt(0.75), "close": style.ge(0.75)}
    for name, m in bands.items():
        x[f"x_pace_n_{name}"] = m.astype(float).groupby(d["race_id"]).transform("sum")
    x["x_pace_pressure"] = x["x_pace_n_front"] / d["n_runners"]
    x["x_pace_style_pressure"] = style * x["x_pace_pressure"]
    x["x_pace_style_draw"] = style * d["umaban_ratio"]
    # --- 騎手・調教師: 期間別と平滑化した組み合わせ ------------------------------------
    for days in (30, 90, 365):
        x[f"x_jk_win_{days}"] = _window_rate(d, "jockey_id", "h_win", days, 0.1)
        x[f"x_jk_top3_{days}"] = _window_rate(d, "jockey_id", "h_top3", days, 0.3)
        x[f"x_jk_tr_win_{days}"] = _window_rate(d, "trainer_id", "h_win", days, 0.1)
    for keys, name in ((["jockey_id", "horse_id"], "horse"), (["jockey_id", "trainer_id"], "trainer"),
                       (["jockey_id", "surface"], "surface"), (["jockey_id", "going"], "going"), (["trainer_id", "surface"], "tr_surface")):
        a = asof_stats(d, keys, ["h_win", "h_top3"], f"j{name}")
        x[f"x_jk_{name}_win"] = _smooth(a[f"j{name}_h_win_rate"], a[f"j{name}_n"], 0.1)
        x[f"x_jk_{name}_top3"] = _smooth(a[f"j{name}_h_top3_rate"], a[f"j{name}_n"], 0.3)
    prev_rate = x["x_jk_win_365"].groupby(by_horse, sort=False).shift(1)
    x["x_jk_upgrade"] = x["x_jk_win_365"] - prev_rate
    # --- コース傾向: 枠と脚質の有利不利 -----------------------------------------------
    d["h_draw"] = np.select([d["umaban_ratio"].le(1 / 3), d["umaban_ratio"].le(2 / 3)], [0, 1], 2)
    d["h_style"] = np.select([style.lt(0.2), style.lt(0.45), style.lt(0.75), style.ge(0.75)], [0, 1, 2, 3], np.nan)
    x["x_course_dist"] = d["course_code"].astype(str) + d["surface"].fillna("") + d["distance"].astype("Int64").astype(str)
    for keys, name in ((["course_code", "surface", "dist_bucket", "h_draw"], "draw"), (["course_code", "surface", "dist_bucket", "h_style"], "style"),
                       (["course_code", "going", "h_style"], "going_style")):
        a = asof_stats(d, keys, ["h_win"], f"c{name}")
        x[f"x_course_{name}_win"] = _smooth(a[f"c{name}_h_win_rate"], a[f"c{name}_n"], 0.1, 50)
    # --- 当日の傾向: 同じ日・同じ競馬場の、番号が前のレースだけを使う ------------------------
    winners = d.loc[pd.to_numeric(d["finish_pos"], errors="coerce").eq(1), ["race_id", "h_corner_first", "umaban_ratio"]].drop_duplicates("race_id")
    races = d[["race_id", "race_date", "course_code", "race_no"]].drop_duplicates("race_id").merge(winners, how="left", on="race_id").sort_values(["race_date", "course_code", "race_no"])
    g = races.groupby(["race_date", "course_code"], sort=False)
    races["x_day_win_corner"] = g["h_corner_first"].transform(lambda v: v.shift(1).expanding().mean())
    races["x_day_win_draw"] = g["umaban_ratio"].transform(lambda v: v.shift(1).expanding().mean())
    races["x_day_n"] = g["h_corner_first"].transform(lambda v: v.shift(1).notna().cumsum()).astype(float)
    merged = d[["race_id"]].merge(races[["race_id", "x_day_win_corner", "x_day_win_draw", "x_day_n"]], how="left", on="race_id")
    merged.index = d.index
    for c in ("x_day_win_corner", "x_day_win_draw", "x_day_n"):
        x[c] = merged[c]
    x["x_day_style_gap"] = style - x["x_day_win_corner"]
    x["x_day_draw_gap"] = d["umaban_ratio"] - x["x_day_win_draw"]
    # --- 過去走の人気: そのレースより前に市場がどう評価していたか (当該レースの人気は使わない) ------
    past_pop = pd.to_numeric(d["popularity_final"], errors="coerce") / d["n_runners"]
    shifted_pop = past_pop.groupby(by_horse, sort=False).shift(1)
    x["x_pastmkt_pop_last"] = shifted_pop
    for w in (3, 5):
        x[f"x_pastmkt_pop_m{w}"] = roll(shifted_pop, w)
    surprise = (past_pop - d["h_fin_ratio"]).groupby(by_horse, sort=False).shift(1)
    x["x_pastmkt_beat_m3"] = roll(surprise, 3)
    x["x_pastmkt_beat_m5"] = roll(surprise, 5)
    grp = x["x_pastmkt_pop_m3"].groupby(d["race_id"])
    x["x_pastmkt_rel"] = (x["x_pastmkt_pop_m3"] - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)
    x["x_pastmkt_rank"] = grp.rank(pct=True)
    d["h_pop_ratio"] = past_pop
    a = asof_stats(d, ["jockey_id"], ["h_pop_ratio"], "x_pastmkt_jk")
    x["x_pastmkt_jk_pop"] = a["x_pastmkt_jk_h_pop_ratio_rate"]
    # --- 競走条件 --------------------------------------------------------------------
    text = (d["conditions"].fillna("") if "conditions" in d else pd.Series("", index=d.index)).map(lambda v: unicodedata.normalize("NFKC", v))
    cat = d["category"].fillna("") if "category" in d else pd.Series("", index=d.index)
    x["x_cond_age"] = (text + " " + cat).str.extract(r"(\d歳以上|\d歳)", expand=False)
    x["x_cond_rank_in_class"] = feats["career_top3"] * d["class_ord"]
    return pd.DataFrame(x, index=d.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """load_frame の結果 (将来レースを含んでよい) から特徴量を作る。行順は入力と同じ。"""
    d = df.copy()
    d["date"] = pd.to_datetime(d["race_date"])
    d["month"] = d["date"].dt.month.astype(float)
    d["dist_bucket"] = pd.cut(d["distance"], [0, 1300, 1700, 2100, 2500, 9999], labels=False).astype(float)
    d["umaban_ratio"] = d["umaban"] / d["n_runners"]
    d["class_ord"] = _class_ord(d["class_name"], d["grade"], d["race_name"])
    d["weight_ratio"] = d["carried_weight"] / d["body_weight"]

    # --- 履歴計算用の成績値 (当該行の値は shift / as-of で必ず除外される) ------------------
    pos = pd.to_numeric(d["finish_pos"], errors="coerce")
    d["h_ran"] = d["status"].eq("finished").astype(float).where(d["status"].notna())
    d["h_win"] = pos.eq(1).astype(float).where(d["status"].notna())
    d["h_top2"] = pos.le(2).astype(float).where(d["status"].notna())
    d["h_top3"] = pos.le(3).astype(float).where(d["status"].notna())
    d["h_fin_ratio"] = pos / d["n_runners"]
    winner = d["time_sec"].where(pos.eq(1)).groupby(d["race_id"]).transform("min")
    d["h_behind"] = d["time_sec"] - winner
    d["h_closing_rel"] = d["last3f"] - d.groupby("race_id")["last3f"].transform("mean")
    d["h_corner_first"] = _first_corner(d["corners"], 0) / d["n_runners"]
    d["h_corner_last"] = _first_corner(d["corners"], -1) / d["n_runners"]
    d["h_pos_gain"] = d["h_corner_last"] - d["h_fin_ratio"]
    # 同条件の前日までの平均タイムとの差 (1000m あたり)
    d["h_speed"] = d["time_sec"] / d["distance"] * 1000
    base = asof_stats(d, ["course_code", "surface", "distance"], ["h_speed"], "base", count=False)
    d["h_time_index"] = d["h_speed"] - base["base_h_speed_rate"]

    feats = pd.DataFrame(index=d.index)
    g = d.groupby("horse_id", sort=False)
    # --- 馬の近走 ------------------------------------------------------------------------
    for col in ["h_fin_ratio", "h_behind", "h_time_index", "h_closing_rel", "h_corner_first", "h_corner_last",
                "h_pos_gain", "h_win", "h_top3"]:
        shifted = g[col].shift(1)
        for w in WINDOWS:
            feats[f"{col[2:]}_m{w}"] = shifted.groupby(d["horse_id"], sort=False).rolling(w, min_periods=1).mean().droplevel(0)
    feats["fin_ratio_trend"] = feats["fin_ratio_m3"] - feats["fin_ratio_m10"]
    feats["time_index_trend"] = feats["time_index_m3"] - feats["time_index_m10"]
    feats["last_finish"] = g["finish_pos"].shift(1)
    feats["last_behind"] = g["h_behind"].shift(1)
    feats["prev_closing_rel"] = g["h_closing_rel"].shift(1)
    feats["runs_count"] = g.cumcount().astype(float)
    feats["career_win"] = g["h_win"].shift(1).groupby(d["horse_id"], sort=False).expanding().mean().droplevel(0)
    feats["career_top3"] = g["h_top3"].shift(1).groupby(d["horse_id"], sort=False).expanding().mean().droplevel(0)
    feats["rest_days"] = (d["date"] - g["date"].shift(1)).dt.days.astype(float)
    feats["dist_change"] = d["distance"] - g["distance"].shift(1)
    feats["class_change"] = d["class_ord"] - g["class_ord"].shift(1)
    feats["weight_change"] = d["carried_weight"] - g["carried_weight"].shift(1)
    prev_jockey = g["jockey_id"].shift(1)
    feats["jockey_changed"] = (d["jockey_id"].ne(prev_jockey)).astype(float).where(prev_jockey.notna())
    # --- 適性 ----------------------------------------------------------------------------
    vals = ["h_win", "h_top3"]
    for keys, name in ((["horse_id", "surface"], "hs_surface"), (["horse_id", "dist_bucket"], "hs_dist"),
                       (["horse_id", "course_code"], "hs_course"), (["horse_id", "going"], "hs_going")):
        feats = feats.join(asof_stats(d, keys, vals, name))
    # --- 騎手・調教師 --------------------------------------------------------------------
    pvals = ["h_win", "h_top2", "h_top3"]
    for keys, name in ((["jockey_id"], "jk"), (["jockey_id", "course_code"], "jk_course"),
                       (["jockey_id", "dist_bucket"], "jk_dist"), (["jockey_id", "horse_id"], "jk_horse"),
                       (["trainer_id"], "tr"), (["trainer_id", "course_code"], "tr_course")):
        feats = feats.join(asof_stats(d, keys, pvals, name))
    feats = feats.join(asof_stats(d, ["sire", "surface"], ["h_win", "h_top3"], "sire_surface"))
    # --- レース内の相対値・想定ペース ------------------------------------------------------
    for col in ["fin_ratio_m5", "time_index_m5", "closing_rel_m5", "jk_h_win_rate", "behind_m5"]:
        grp = feats[col].groupby(d["race_id"])
        feats[f"rel_{col}"] = (feats[col] - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)
    front = feats["corner_first_m5"].lt(0.3).astype(float).where(feats["corner_first_m5"].notna())
    feats["pace_front_share"] = front.groupby(d["race_id"]).transform("mean")
    feats["pace_corner_mean"] = feats["corner_first_m5"].groupby(d["race_id"]).transform("mean")
    feats["style_vs_field"] = feats["corner_first_m5"] - feats["pace_corner_mean"]

    extra = {c: pd.to_numeric(d[c], errors="coerce").astype(float) for c in NUMERIC_RACE}
    extra.update({c: d[c] for c in CATEGORICAL if not c.startswith("x_")})
    extra.update({c: d[c] if c in d else pd.Series(np.nan, index=d.index) for c in MARKET})
    feats = pd.concat([feats.copy(), pd.DataFrame(extra, index=d.index)], axis=1)
    feats = pd.concat([feats, _extra_features(d, feats)], axis=1)
    assert_no_forbidden(feats.columns)
    return feats


def assert_no_forbidden(columns) -> None:
    bad = [c for c in columns if FORBIDDEN.search(c)]
    if bad:
        raise ValueError(f"禁止された列が特徴量に含まれています: {bad}")


@dataclass
class FeatureSpec:
    """カテゴリの語彙と使用列を固定し、学習時と予測時で同じ変換を保証する。"""
    columns: list[str]
    vocab: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def fit(cls, feats: pd.DataFrame, min_market_coverage: float = 0.5, groups=None) -> "FeatureSpec":
        groups = ADOPTED_GROUPS if groups is None else groups
        allowed = tuple(GROUPS[g] for g in groups)
        cols = [c for c in feats.columns if not c.startswith("x_") or c.startswith(allowed)]
        if feats["market_win_odds"].notna().mean() < min_market_coverage:
            cols = [c for c in cols if c not in MARKET]
        cols = [c for c in cols if c in CATEGORICAL or feats[c].notna().any()]
        vocab = {c: sorted(str(x) for x in feats[c].dropna().unique()) for c in CATEGORICAL if c in cols}
        return cls(cols, vocab)

    def transform(self, feats: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=feats.index)
        for c in self.columns:
            if c in self.vocab:
                mapping = {v: i for i, v in enumerate(self.vocab[c])}
                X[c] = feats[c].map(lambda v: mapping.get(str(v), np.nan) if pd.notna(v) else np.nan).astype(float)
            else:
                X[c] = feats[c].astype(float) if c in feats else np.nan
        return X

    @property
    def categorical(self) -> list[str]:
        return [c for c in self.columns if c in self.vocab]


def mask_late_info(feats: pd.DataFrame, race_ids: pd.Series, rate: float = 0.15, seed: int = 0) -> pd.DataFrame:
    """学習専用: 発走直前まで不明な情報をレース単位で欠損にする。"""
    rng = np.random.default_rng(seed)
    uniq = race_ids.unique()
    masked = set(uniq[rng.random(len(uniq)) < rate])
    idx = race_ids.isin(masked)
    out = feats.copy()
    for c in LATE_INFO:
        if c in out:
            out.loc[idx, c] = np.nan
    return out
