"""드로잉 레퍼런스 기술 검증 웹앱.  실행: streamlit run app.py"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
PROJECT_MODULES = ("core", "methods", "views", "components", "training")


@st.cache_resource
def _code_state() -> dict:
    return {}  # 이 서버 프로세스가 마지막으로 불러온 코드 버전 (재실행·세션 사이에 유지)


def _reload_changed_code() -> None:
    """코드가 바뀌었으면 메모리에 남은 예전 모듈을 버려, 이번 실행에서 새 파일을 불러오게 한다.

    Streamlit Cloud는 push한 파일을 받아도 서버를 재시작하지 않는 경우가 있다. 그러면 app.py는 새것인데
    이미 불러온 views/common.py 등은 예전 것이라 ImportError가 난다 (관리자가 Reboot해야만 풀림).
    """
    files = [*ROOT.glob("*.py"), *(f for p in PROJECT_MODULES[:-1] for f in (ROOT / p).glob("*.py")),
             *(ROOT / "web").rglob("*.js"), *(ROOT / "components").glob("*.js")]
    version = max((f.stat().st_mtime_ns for f in files), default=0)
    state = _code_state()
    if state.get("version") == version:
        return
    for name in [n for n in sys.modules if n.split(".")[0] in PROJECT_MODULES]:
        sys.modules.pop(name, None)
    state["version"] = version


_reload_changed_code()
from views import auth  # noqa: E402  (예전 모듈을 버린 뒤에 불러와야 새 코드가 쓰인다)

st.set_page_config(page_title="SPECTRUM · 드로잉 레퍼런스 검증", page_icon="🎨", layout="wide",
                   initial_sidebar_state="auto" if auth.role() else "collapsed")

from core.db import is_configured, set_write_guard  # noqa: E402
set_write_guard(auth.write_guard)  # 방문자 세션은 DB 쓰기를 막는다 (매 실행마다: 모듈을 다시 불러오면 초기화되므로)

pages = {
    "팀": [
        st.Page("views/team.py", title="팀 현황", icon="👥"),
    ],
    "검증": [
        st.Page("views/gallery.py", title="갤러리", icon="🖼️", default=True),
        st.Page("views/similar.py", title="유사 검색", icon="🔍"),
        st.Page("views/duplicates.py", title="중복 확인", icon="👯"),
    ],
    "자동 태깅": [
        st.Page("views/labeling.py", title="라벨링", icon="🏷️"),
        st.Page("views/train.py", title="학습 현황", icon="🧠"),
        st.Page("views/predictions.py", title="예측 확인", icon="🤖"),
        st.Page("views/tags_admin.py", title="태그 관리", icon="🗂️"),
    ],
    "포즈": [
        st.Page("views/pose_label.py", title="포즈 라벨링", icon="🦴"),
        st.Page("views/pose_search.py", title="포즈 검색 (2D)", icon="🔎"),
        st.Page("views/pose_search_3d.py", title="포즈 검색 (3D)", icon="🧍"),
        st.Page("views/pose_accuracy.py", title="포즈 정확도", icon="📏"),
    ],
}

from core.dataset import META_CSV  # noqa: E402
if not is_configured() and not META_CSV.exists():  # 이미지 목록을 읽을 곳이 없으면 모든 페이지가 오류로 멈춘다
    st.error("**접속 정보 파일(`.env`)이 없습니다.** 처음 설치했다면 아래 순서대로 해 주세요.\n\n"
             "1. 프로젝트 폴더의 `.env.example`을 복사해 이름을 `.env`로 바꿉니다 (`.env.txt`가 되지 않게 주의)\n"
             "2. 팀장에게 받은 `SUPABASE_URL`, `SUPABASE_KEY` 값을 넣습니다\n"
             "3. 터미널에서 `Ctrl+C`로 앱을 끄고 `python -m streamlit run app.py`로 다시 켭니다\n\n"
             "자세한 안내는 `README.md`에 있습니다.")
    st.stop()

if not auth.role():
    st.navigation(pages, position="hidden")  # 주소의 페이지 경로(예: /pose_label?image=...)는 입장 뒤에도 유지된다
    auth.render_gate()
    st.stop()

nav = st.navigation(pages)
# SNB(사이드 메뉴) 글자를 Streamlit 기본 14px보다 1px 크게. <style>만 있는 st.html은 화면 자리를 차지하지 않는다
st.html("""<style>
[data-testid="stSidebarNavLink"] p, [data-testid="stNavSectionHeader"] { font-size: 15px; }
[data-testid="stSidebarNavLink"] [data-testid="stIconEmoji"] { font-size: 15.4px; }
</style>""")
with st.sidebar:
    auth.sidebar()
    st.caption("태그·라벨 저장소: **Supabase**" if is_configured() else ":red[Supabase 접속 정보 없음 — .env 확인]")

nav.run()
