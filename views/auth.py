"""입장 화면(팀원 이메일 확인 / 방문자)과 사이드바 로그인 표시. app.py가 쓴다.

- 팀원: 이메일이 DB 명단(members)에 있으면 입장. '내 이름'은 명단의 이름으로 고정된다(이름이 갈라지지 않게).
- 방문자: 모든 화면을 볼 수 있지만 저장·수정은 막힌다. 화면 버튼은 이름이 없어서 꺼지고,
  DB 쓰기 직전에도 write_guard가 한 번 더 막는다 (core.db.check_write).
상태는 세션(st.session_state)에만 있어 브라우저를 새로고침하면 다시 입장해야 한다.
"""
import time

import streamlit as st

from core.db import DBError
from core.members import is_email, member_name, normalize_email

ROLE, EMAIL, NAME, FAILS = "auth_role", "auth_email", "auth_name", "auth_fails"
MEMBER, VISITOR = "member", "visitor"
MAX_FAILS, LOCK_SECONDS = 5, 60  # 1분 안에 5번 틀리면 1분 동안 막는다 (이메일 마구 넣어 보기 방지)
VISITOR_BLOCKED = "방문자는 볼 수만 있어 저장되지 않았습니다. 팀원은 로그아웃 후 이메일로 입장해 주세요."

_STYLE = """
<style>
/* Streamlit 기본 글꼴 크기보다 우선하도록 div + !important */
div.spectrum-title { text-align: center; font-size: clamp(2.8rem, 11vw, 4.5rem) !important; font-weight: 800 !important;
                     letter-spacing: -0.025em; line-height: 1.1; margin: 0; padding: 0; }
div.spectrum-sub { text-align: center; opacity: 0.65; margin: 0.6rem 0 2rem; }
</style>
"""


def md_escape(text: str) -> str:
    """이름·이메일을 마크다운 그대로 보이게 (예: '__test__'가 굵은 글씨 기호로 사라지지 않게)."""
    return "".join("\\" + ch if ch in "\\`*_{}[]()#+-.!<>|~$" else ch for ch in text)


def role() -> str | None:
    return st.session_state.get(ROLE)


def is_member() -> bool:
    return role() == MEMBER


def write_guard() -> None:
    """core.db.set_write_guard에 넣는 함수. 팀원으로 입장한 세션만 DB에 쓸 수 있다.

    화면 세션이 없는 스레드(팀원이 '학습 시작'을 눌러 뒤에서 도는 학습, core.train_job)는 막지 않는다.
    그런 스레드는 팀원 확인을 거친 뒤에만 만들어진다.
    """
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None:
        return
    if not is_member():
        raise DBError(VISITOR_BLOCKED)


def _recent_fails() -> list[float]:
    now = time.time()
    fails = [t for t in st.session_state.get(FAILS, []) if now - t < LOCK_SECONDS]
    st.session_state[FAILS] = fails
    return fails


def _enter_as_visitor() -> None:
    st.session_state.update({ROLE: VISITOR, EMAIL: "", NAME: "", "evaluator": ""})


def _leave() -> None:
    for k in (ROLE, EMAIL, NAME, "evaluator"):
        st.session_state.pop(k, None)


def _try_member(email: str) -> None:
    if len(_recent_fails()) >= MAX_FAILS:
        st.error("여러 번 틀려서 잠시 막혔습니다. 1분 뒤에 다시 시도해 주세요.")
        return
    if not is_email(email):
        st.error("이메일 형식이 아닙니다. 예: someone@example.com")
        return
    try:
        name = member_name(email)
    except DBError as e:
        st.error(f"팀원 확인에 실패했습니다. {e}")
        return
    if name is None:
        st.session_state[FAILS] = _recent_fails() + [time.time()]
        st.error("등록된 팀원 이메일이 아닙니다. 팀원이 아니라면 아래 **입장하기**로 둘러볼 수 있어요.")
        return
    st.session_state.update({ROLE: MEMBER, EMAIL: normalize_email(email), NAME: name, "evaluator": name,
                             FAILS: []})
    st.rerun()


def render_gate() -> None:
    """입장 화면. 가운데 SPECTRUM, 팀원 이메일 입력, 아래에 방문자 입장."""
    st.markdown(_STYLE, unsafe_allow_html=True)
    st.markdown("<div style='height: 14vh'></div>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown("<div class='spectrum-title'>SPECTRUM</div>", unsafe_allow_html=True)
        st.markdown("<div class='spectrum-sub'>드로잉 레퍼런스 서비스 · 기술 검증 도구</div>", unsafe_allow_html=True)
        with st.form("auth_form", border=False):
            email = st.text_input("팀원 이메일", placeholder="you@example.com", key="auth_email_input")
            submitted = st.form_submit_button("팀원으로 입장", type="primary", width="stretch")
        if submitted:
            _try_member(email)
        st.divider()
        # 가로 컨테이너: 좁은 화면(휴대폰)에서도 문구 오른쪽에 버튼이 붙어 있다 (columns는 세로로 쌓임)
        with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center"):
            st.markdown("방문자이신가요?", width="content")
            st.button("입장하기", key="auth_visitor", on_click=_enter_as_visitor)
        st.caption("방문자는 모든 화면을 볼 수 있지만 저장·수정은 할 수 없습니다.", text_alignment="right")


def sidebar() -> None:
    """사이드바 맨 위: 누구로 입장했는지 + 로그아웃."""
    if is_member():
        st.session_state["evaluator"] = st.session_state[NAME]  # 저장할 때 쓰는 이름은 명단의 이름으로 고정
        st.markdown(f"**{md_escape(st.session_state[NAME])}** 님")
        st.caption(md_escape(st.session_state[EMAIL]))
        st.button("로그아웃", key="auth_logout", on_click=_leave)
    else:
        st.session_state["evaluator"] = ""
        st.markdown("👀 **방문자** · 보기 전용")
        st.caption("저장·수정은 팀원만 할 수 있습니다.")
        st.button("나가기", key="auth_logout", on_click=_leave)
