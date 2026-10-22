"""학습한 분류기의 예측 결과를 태그 출처로 제공.

출처 우선순위: DB의 팀 공유 예측(가장 최근에 '완료'된 학습) → 없으면 이 PC의 predicted_tags.csv.
학습은 로컬 PC의 training.py로만 하고, 결과는 DB로 공유되어 모든 팀원이 같은 예측(AI 추천, 예측 확인, 갤러리)을 본다.
예측은 수십만 행이 될 수 있으므로 1분마다 '새 학습이 완료됐는가'만 가볍게 확인하고, 바뀌었을 때만 전체를 다시 읽는다.
"""
import pandas as pd
import streamlit as st

from core.db import DBError
from core.tagging import (PRED_CSV, PRED_THRESHOLD, file_mtime, latest_training_run, load_run_predictions, load_tags,
                          read_predictions)
from methods.base import SearchMethod, TagSource

SHARED_TTL = 60  # 새로 완료된 학습은 최대 1분 뒤에 팀 화면에 보인다


@st.cache_data(show_spinner=False)
def _load_predictions(mtime: float | None) -> pd.DataFrame:
    return read_predictions()


@st.cache_data(show_spinner=False, ttl=SHARED_TTL)
def latest_run() -> dict | None:
    """가장 최근에 완료된 학습 기록 (누가·언제·지표). DB 문제나 기록이 없으면 None."""
    try:
        return latest_training_run()
    except DBError:
        return None


@st.cache_data(show_spinner="팀 공유 예측을 불러오는 중... (새 학습 결과가 있을 때 한 번)", max_entries=2)
def _load_run(run_id: str) -> pd.DataFrame:
    return load_run_predictions(run_id)  # run_id가 같으면 내용도 같으므로 시간 제한 없이 캐시


def _current() -> tuple[str, pd.DataFrame, str, pd.Timestamp | None]:
    """(버전, 예측, 설명, 학습 시각 UTC)."""
    if (run := latest_run()) is not None:
        try:
            preds = _load_run(run["run_id"])
        except DBError:
            preds = None
        if preds is not None:
            at = pd.Timestamp(run["completed_at"]).tz_convert("UTC")
            return (f"shared:{run['run_id']}", preds,
                    f"팀 공유 · {run['trainer']} · {at.tz_convert('Asia/Seoul'):%m/%d %H:%M}", at)
    mtime = file_mtime(PRED_CSV)
    when = None if mtime is None else pd.Timestamp(mtime, unit="s", tz="UTC")
    return f"local:{mtime}", _load_predictions(mtime), "이 PC에서 학습한 결과" if mtime else "없음", when


def load_predictions() -> pd.DataFrame:
    """컬럼: image_id, tag_key, prob. 새 학습이 완료되면 자동으로 새로 읽는다."""
    return _current()[1]


def predictions_source() -> str:
    """화면 안내용: 지금 보이는 예측이 어디서 왔는지."""
    return _current()[2]


def predictions_time() -> pd.Timestamp | None:
    """지금 보이는 예측을 만든 학습 시각(UTC). 예측이 없으면 None."""
    return _current()[3]


def refresh_predictions() -> None:
    """새 학습 완료를 1분 기다리지 않고 바로 확인한다."""
    latest_run.clear()


def _active_tag_names() -> tuple[tuple[str, str], ...]:
    """사용 중인 태그 (key, 이름). DB에 문제가 있어도 갤러리는 열려야 하므로 실패하면 빈 값."""
    try:
        return tuple((t.key, t.name) for t in load_tags())
    except DBError:
        return ()


@st.cache_data(show_spinner=False)
def _predicted_map(version: str, threshold: float, tag_names: tuple[tuple[str, str], ...],
                   _preds: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    # _preds는 캐시 키에서 빠진다 (이름이 _로 시작). 내용이 바뀌면 version이 바뀐다
    names = dict(tag_names)
    p = _preds[(_preds["prob"] >= threshold) & _preds["tag_key"].isin(names)].sort_values("prob", ascending=False)
    out: dict[str, list[tuple[str, float]]] = {}
    for image_id, key, prob in zip(p["image_id"], p["tag_key"], p["prob"]):
        out.setdefault(image_id, []).append((names[key], float(prob)))
    return out


def predicted_tags_of(image_id: str, threshold: float = PRED_THRESHOLD) -> list[tuple[str, float]]:
    """(태그 표시명, 확률) 목록, 확률 높은 순."""
    version, preds, _, _ = _current()
    if preds.empty:
        return []  # 학습 전에는 태그 이름을 물어볼 필요도 없다
    return _predicted_map(version, threshold, _active_tag_names(), preds).get(image_id, [])


class PredictedTags(SearchMethod, TagSource):
    key = "predicted"
    name = "예측 태그 (학습 모델)"
    description = f"팀 라벨로 학습한 분류기가 확률 {PRED_THRESHOLD} 이상으로 예측한 태그"

    def tags_of(self, idx: int) -> list[str]:
        return [name for name, _ in predicted_tags_of(self.df["id"].iat[idx])]
