"""학습 현황: (이 PC에서 켠 앱이면) 학습 시작, 마지막 학습 결과(DB 공유), 태그별 라벨 현황, 실행 방법.

학습은 이미지가 많아 배포 서버(ON_CLOUD)에서는 하지 않고, 팀원 각자 PC에서 켠 로컬 웹앱에서 한다.
학습은 core.train_job이 화면과 분리해 뒤에서 돌리므로, 다른 페이지로 가거나 새로고침해도 계속된다.
결과는 DB(training_runs, predictions)로 공유된다.
"""
import time

import pandas as pd
import streamlit as st

import training
from core import team
from core.dataset import ON_CLOUD
from core.db import DBError
from core.tagging import active_training_runs, consolidate
from core.train_job import DONE, FAILED, RUNNING, TrainJob
from methods.predicted_tags import latest_run, refresh_predictions
from views.auth import md_escape
from views.common import cannot_save_notice, current_user, is_visitor, load_labels_or_stop, load_tags_or_stop

WATCHING = "train_watching"  # 이 화면이 지켜보는 실행 번호 (끝나면 화면 전체를 새로 그려 '마지막 학습'을 갱신)
OTHERS_SEEN = "train_others_seen"  # 다른 팀원의 학습을 보고 있었는지 (끝나면 화면 전체를 새로 그린다)


@st.cache_resource
def get_job() -> TrainJob:
    return TrainJob()  # 서버 프로세스에 하나 → 한 번에 하나만 학습


def _kst(ts: float) -> str:
    return pd.Timestamp(ts, unit="s", tz="UTC").tz_convert("Asia/Seoul").strftime("%H:%M")


def _others_training() -> list[dict]:
    """다른 PC(또는 이 PC의 터미널)에서 지금 학습 중인 기록. 이 앱에서 돌리는 학습은 위 진행률로 보여주므로 뺀다."""
    try:
        runs = active_training_runs()
    except DBError:
        return []
    if not ON_CLOUD:
        s = get_job().snapshot()
        if s.status == RUNNING:  # 이 앱의 학습: 같은 사람이 이 앱에서 시작한 시각 이후에 만든 기록
            runs = [r for r in runs if not (r["trainer"] == s.trainer
                                            and pd.Timestamp(r["started_at"]).timestamp() >= s.started_at - 5)]
    return runs


st.title("🧠 학습 현황")
st.caption(f"임베딩 모델: `{training.EMBED_ID}` · 학습 조건: 태그마다 양성 {training.MIN_POS}건 이상, 음성 {training.MIN_NEG}건 이상 "
           "· 학습은 각자 PC에서 켠 웹앱에서 하고(팀원 누구나), 결과는 DB로 팀 전체에 공유됩니다.")

tags = load_tags_or_stop()
labels = load_labels_or_stop()
status = training.label_status(tags, consolidate(labels))


@st.fragment(run_every=10 if _others_training() else None)
def others_panel() -> None:
    runs = _others_training()
    if runs:
        st.session_state[OTHERS_SEEN] = True
        for r in runs:
            started = pd.Timestamp(r["started_at"]).tz_convert("Asia/Seoul")
            st.info(f"🔄 **{md_escape(r['trainer'])}** 님이 {started:%H:%M}부터 학습 중입니다 · {md_escape(r.get('stage') or '')}  \n"
                    "끝나면 아래 **마지막 학습**에 결과가 나옵니다. 지금 또 학습을 시작하면 나중에 끝난 쪽 결과만 남아요.")
    elif st.session_state.pop(OTHERS_SEEN, None):  # 보고 있던 다른 팀원의 학습이 방금 끝남
        refresh_predictions()
        st.rerun(scope="app")


others_panel()

# ---------- 학습 실행 (이 PC에서 켠 앱에서만) ----------
if not ON_CLOUD:
    job = get_job()
    st.subheader("학습 실행 (이 PC)")
    snap = job.snapshot()
    me = current_user()
    trainable = int(status["trainable"].sum())
    if is_visitor():
        cannot_save_notice("학습 실행")

    def start() -> None:
        if job.start(current_user()):
            st.session_state[WATCHING] = job.snapshot().run_no

    st.button("🚀 학습 시작", type="primary", on_click=start,
              disabled=snap.status == RUNNING or not me or not trainable,
              help="뒤에서 학습합니다. 다른 페이지에서 라벨링을 계속해도 됩니다.")
    if not trainable:
        st.caption("학습할 수 있는 태그가 없습니다. 아래 **태그별 라벨 현황**에서 부족한 라벨 수를 확인하세요.")
    if snap.status == RUNNING:
        st.session_state[WATCHING] = snap.run_no  # 다른 페이지에 다녀와도 끝나는 순간을 잡는다

    @st.fragment(run_every=2 if snap.status == RUNNING else None)
    def job_panel() -> None:
        s = job.snapshot()
        if s.status == RUNNING:
            st.progress(min(max(s.frac or 0.0, 0.0), 1.0), text=f"**{s.stage}** · {s.message}")
            st.caption(f"{md_escape(s.trainer)} 님이 {_kst(s.started_at)}에 시작 · 경과 {training._duration(time.time() - s.started_at)} · "
                       "다른 페이지로 가거나 새로고침해도 계속됩니다. 이 PC의 앱(검은 창)을 끄면 멈추고, 다시 시작하면 이어서 합니다.")
        elif st.session_state.get(WATCHING) == s.run_no:  # 지켜보던 학습이 방금 끝남 → 위아래 내용을 새로 그린다
            st.session_state.pop(WATCHING, None)
            refresh_predictions()
            st.rerun(scope="app")
        elif s.status in (DONE, FAILED):
            r = s.result
            took = training._duration(s.finished_at - s.started_at)
            if s.status == FAILED:
                st.error(f"학습이 끝나지 않았습니다 ({_kst(s.finished_at)}). {s.error}  \n다시 누르면 이미 한 부분은 건너뛰고 이어서 합니다.")
            elif not r.trained:
                st.warning(f"학습할 수 있는 태그가 없어 학습하지 않았습니다 ({_kst(s.finished_at)}). 라벨을 더 모은 뒤 다시 실행하세요.")
            else:
                st.success(f"학습 완료 ({_kst(s.finished_at)}) · {len(r.report)}개 태그 · 이미지 {r.n_images:,}장 · "
                           f"새로 추출 {r.n_new_embeddings:,}장 · {took}")
                if r.share_error:
                    st.warning(f"팀 공유에 실패했습니다(이 PC에는 저장됨). 다시 학습하면 공유됩니다. {r.share_error}")
                elif r.shared_rows is not None:
                    st.info(f"예측 {r.shared_rows:,}행을 팀에 공유했습니다. 팀원 화면에는 1분 안에 반영됩니다.")
            if r is not None and r.skipped:
                with st.expander(f"열 수 없어 건너뛴 이미지 {len(r.skipped)}장"):
                    st.code("\n".join(f"{i}: {why}" for i, why in r.skipped[:200]), language=None)
        if s.log:
            with st.expander("실행 기록"):
                st.code("\n".join(s.log[-60:]), language=None)

    job_panel()

# ---------- 마지막 학습 ----------
run = latest_run()
st.subheader("마지막 학습")
st.button("🔄 새로 확인", on_click=refresh_predictions, help="방금 학습을 마쳤다면 눌러서 바로 확인합니다 (평소에는 1분마다 자동 확인).")
if run is None:
    st.info("아직 팀에 공유된 학습이 없습니다. 아래 **학습하는 방법**대로 실행하면 여기에 결과가 나옵니다.")
else:
    done_at = pd.Timestamp(run["completed_at"]).tz_convert("Asia/Seoul")
    new_labels = team.labels_since(labels, done_at.tz_convert("UTC"))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("학습 완료", f"{done_at:%m/%d %H:%M}")
    m2.metric("학습한 사람", md_escape(run["trainer"]))
    m3.metric("학습 이미지", f"{run['n_images']:,}장")
    m4.metric("이후 새 라벨", f"{new_labels:,}건", help="마지막 학습 뒤에 저장·수정된 라벨. 많이 쌓였으면 다시 학습할 때입니다.")

    report = pd.DataFrame(run["report"] or [])
    if not report.empty:
        st.markdown(f"**태그별 평가 지표** (평가용 20%) · F1 {training.F1_WARN} 미만은 빨간색")
        table = pd.DataFrame({
            "태그": report["name"], "양성": report["n_pos"], "음성": report["n_neg"], "평가 건수": report["n_test"],
            "Precision": report["precision"], "Recall": report["recall"], "F1": report["f1"],
        })
        low = report["f1"] < training.F1_WARN
        st.dataframe(
            table.style.format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1": "{:.2f}"})
            .apply(lambda r: ["background-color: rgba(255, 75, 75, 0.25)" if low.iat[r.name] else ""] * len(r), axis=1),
            hide_index=True, width="stretch",
        )
    st.caption(f"예측 {run['n_rows']:,}행 공유 · 모델 `{run['embed_model']}`")

# ---------- 라벨 현황 ----------
st.subheader("태그별 라벨 현황")
st.dataframe(
    pd.DataFrame({
        "태그": status["name"], "분류": status["group"], "양성": status["n_pos"], "음성": status["n_neg"],
        "학습 가능": status["trainable"].map({True: "✅ 가능", False: "❌ 불가"}), "사유": status["reason"],
    }),
    hide_index=True, width="stretch",
)

# ---------- 학습하는 방법 ----------
st.subheader("학습하는 방법")
if ON_CLOUD:
    st.info("이 주소(배포 서버)에서는 서버 메모리가 작아 학습할 수 없습니다. **자기 PC에서 웹앱을 켜서** 학습하세요.")
st.markdown(
    "1. 자기 PC의 프로젝트 폴더에서 **`SPECTRUM 실행.bat`을 더블클릭** → 브라우저에 웹앱이 열립니다.\n"
    "2. 팀원 이메일로 입장 → **자동 태깅 → 학습 현황** → **🚀 학습 시작**.\n"
    "3. 학습은 뒤에서 진행됩니다. 기다리는 동안 다른 페이지에서 라벨링을 계속해도 됩니다.\n\n"
    "- 팀원 누구나 학습할 수 있고, 누가 하든 팀 전체 라벨과 DB의 이미지 목록으로 학습하므로 결과가 같습니다. "
    "다른 팀원이 학습 중이면 이 페이지 위쪽에 표시되니 끝난 뒤 시작하세요.\n"
    "- 측정 기준(14코어 PC) 약 **25~30장/초**(1만 장 ≈ 6~7분), 최대 메모리 약 1.7GB. "
    "그 PC에서 처음 학습할 때만 오래 걸리고(이미지 폴더가 없으면 내려받는 시간 추가), 다음부터는 새 이미지만 처리합니다.\n"
    "- 중간에 앱(검은 창)을 꺼도 **다시 학습 시작을 누르면 이어서** 합니다(이미 추출한 이미지는 건너뜀, 최대 30초 분량만 다시).\n"
    "- 열 수 없는 이미지는 건너뛰고 결과에 목록을 보여줍니다.\n"
    "- 예측은 **다 올라간 뒤에** 팀원 화면(AI 추천·예측 확인·갤러리)에 1분 안에 반영됩니다. 올리다 실패하면 팀 화면은 직전 결과 그대로입니다.\n"
    "- 새 이미지를 추가했다면 먼저 관리자 PC에서 `python migrate_to_supabase.py`로 DB에 등록하세요(등록 안 된 이미지의 예측은 공유되지 않음).")
with st.expander("개발자용: 터미널에서 학습"):
    st.code("python training.py --email 내이메일@example.com   # 팀에 공유\npython training.py --no-share                      # 이 PC에만 저장",
            language="bash")
