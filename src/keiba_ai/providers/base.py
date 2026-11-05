"""Provider 共通インターフェースと正規化レコード。JRA/NAR ともこの形式に変換して保存する。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# 共通の券種名。枠単は NAR のみ。
BET_TYPES = ("win", "place", "bracket_quinella", "quinella", "exacta", "wide", "trio", "trifecta", "bracket_exacta")
BET_LABEL = {"win": "単勝", "place": "複勝", "bracket_quinella": "枠連", "quinella": "馬連", "exacta": "馬単",
             "wide": "ワイド", "trio": "三連複", "trifecta": "三連単", "bracket_exacta": "枠単"}


@dataclass
class EntryRec:
    umaban: int
    horse_id: str
    horse_name: str
    waku: int | None = None
    sex: str | None = None
    age: int | None = None
    carried_weight: float | None = None
    jockey_id: str | None = None
    jockey_name: str | None = None
    jockey_affiliation: str | None = None
    trainer_id: str | None = None
    trainer_name: str | None = None
    trainer_affiliation: str | None = None
    body_weight: float | None = None
    body_weight_diff: float | None = None
    birth_date: str | None = None
    coat: str | None = None
    sire: str | None = None
    dam: str | None = None
    damsire: str | None = None
    owner: str | None = None
    breeder: str | None = None
    # 以下は発走後に判明
    finish_pos: int | None = None
    status: str | None = None     # None なら結果未確定
    time_sec: float | None = None
    margin: str | None = None
    last3f: float | None = None
    corners: str | None = None
    popularity_final: int | None = None


@dataclass
class OddsRec:
    bet_type: str
    combo: str
    odds: float | None
    odds_max: float | None = None
    popularity: int | None = None


@dataclass
class PayoutRec:
    bet_type: str
    combo: str
    amount: int
    popularity: int | None = None


@dataclass
class RaceRec:
    org: str
    course_code: str
    course_name: str
    race_date: str
    race_no: int
    source: str
    source_ref: str | None = None
    post_time: str | None = None
    name: str | None = None
    grade: str | None = None
    race_kind: str | None = None
    category: str | None = None
    class_name: str | None = None
    rule: str | None = None
    weight_rule: str | None = None
    conditions: str | None = None
    surface: str | None = None
    distance: int | None = None
    direction: str | None = None
    weather: str | None = None
    going: str | None = None
    n_runners: int | None = None
    kai: int | None = None
    nichi: int | None = None
    entries: list[EntryRec] = field(default_factory=list)
    payouts: list[PayoutRec] = field(default_factory=list)

    @property
    def race_id(self) -> str:
        return f"{self.org[0]}{self.race_date.replace('-', '')}{self.course_code}{self.race_no:02d}"


class Provider(ABC):
    """取得元を追加するときは、このクラスを継承して providers/__init__.py に登録する。"""
    name: str

    @abstractmethod
    def run(self, con, **kwargs) -> dict:
        """取得と保存を実行し、件数サマリを返す。"""
