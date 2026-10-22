"""검색 방식 등록부. 새 방식은 METHOD_CLASSES에 추가하면 해당 페이지에 자동으로 나타난다."""
import streamlit as st

from core.dataset import load_metadata, metadata_key
from methods.base import Hit, ImageSearch, PairFinder, SearchMethod, TagSource
from methods.phash import PhashMethod
from methods.predicted_tags import PredictedTags
from methods.safebooru_tags import SafebooruTags

METHOD_CLASSES: list[type[SearchMethod]] = [
    PhashMethod,
    SafebooruTags,
    PredictedTags,
    # ClipMethod, WdTagger, PoseMethod ...
]


@st.cache_resource(show_spinner="검색 인덱스 준비 중... (처음 한 번만 오래 걸려요)")
def _build_methods(class_ids: tuple[int, ...], meta: tuple) -> list[SearchMethod]:  # 인자 이름에 _를 붙이면 캐시 키에서 빠지므로 주의
    df = load_metadata()
    return [cls(df) for cls in METHOD_CLASSES]


def get_methods() -> list[SearchMethod]:
    # 앱 실행 중 코드를 고치면 Streamlit이 모듈을 다시 불러와 클래스 객체가 새로 생긴다.
    # 캐시가 예전 클래스의 인스턴스를 돌려주면 isinstance가 실패하므로, 클래스 id를 캐시 키에 넣는다.
    class_ids = tuple(id(c) for c in (*METHOD_CLASSES, ImageSearch, PairFinder, TagSource))
    # 이미지 목록이 바뀌면(새 이미지 등록) 행 번호가 달라지므로 검색 인덱스도 새로 만든다
    return _build_methods(class_ids, metadata_key(load_metadata()))


def methods_of(capability: type) -> list:
    return [m for m in get_methods() if isinstance(m, capability)]


__all__ = ["Hit", "ImageSearch", "PairFinder", "SearchMethod", "TagSource", "get_methods", "methods_of"]
