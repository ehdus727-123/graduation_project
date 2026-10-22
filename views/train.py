"""학습 현황: 태그별 라벨 현황, 마지막 학습 결과(DB 공유), 로컬 PC 실행 방법.

학습은 이미지가 많아 웹앱에서 하지 않고 로컬 PC의 training.py로만 한다. 결과는 DB(training_runs, predictions)로 공유된다.
"""
import pandas as pd
import streamlit as st

import training
from core import team
from core.tagging import consolidate
from methods.predicted_tags import latest_run, refresh_predictions
from views.common import load_labels_or_stop, load_tags_or_stop

st.title("🧠 학습 현황")
st.caption(f"임베딩 모델: `{training.EMBED_ID}` · 학습 조건: 태그마다 양성 {training.MIN_POS}건 이상, 음성 {training.MIN_NEG}건 이상 "
           "· 학습은 로컬 PC에서 `training.py`로 하고, 결과는 DB로 팀 전체에 공유됩니다.")

tags = load_tags_or_stop()
labels = load_labels_or_stop()
status = training.label_status(tags, consolidate(labels))
run = latest_run()

# ---------- 마지막 학습 ----------
st.subheader("마지막 학습")
st.button("🔄 새로 확인", on_click=refresh_predictions, help="방금 학습을 마쳤다면 눌러서 바로 확인합니다 (평소에는 1분마다 자동 확인).")
if run is None:
    st.info("아직 팀에 공유된 학습이 없습니다. 아래 **로컬 PC에서 학습하기** 순서대로 실행하면 여기에 결과가 나옵니다.")
else:
    done_at = pd.Timestamp(run["completed_at"]).tz_convert("Asia/Seoul")
    new_labels = team.labels_since(labels, done_at.tz_convert("UTC"))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("학습 완료", f"{done_at:%m/%d %H:%M}")
    m2.metric("학습한 사람", run["trainer"])
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

# ---------- 실행 방법 ----------
st.subheader("로컬 PC에서 학습하기")
st.markdown(
    "학습은 **이미지 파일(`drawing_ref_test/`)이 있는 PC 한 곳**에서 합니다. "
    "프로젝트 폴더에서 터미널(PowerShell)을 열고 순서대로 실행하세요.")
st.markdown("**1. (새 이미지를 추가했다면) DB에 이미지 등록** — 관리자 키가 있는 PC에서")
st.code("python migrate_to_supabase.py", language="bash")
st.markdown("**2. 학습 실행** — 이메일은 웹앱 입장에 쓰는 팀원 이메일")
st.code("python training.py --email 내이메일@example.com", language="bash")
st.markdown(
    "- 진행률·속도·남은 시간이 표시됩니다. 측정 기준(14코어 PC) 약 **25~30장/초**(1만 장 ≈ 6~7분), 최대 메모리 약 1.7GB.\n"
    "- 중간에 멈춰도 **같은 명령을 다시 실행하면 이어서** 합니다. `Ctrl+C`로 멈추면 그때까지 전부 남고, "
    "창이 갑자기 닫히면 마지막 30초 분량만 다시 합니다.\n"
    "- 열 수 없는 이미지는 건너뛰고 마지막에 목록을 보여줍니다.\n"
    "- 끝나면 예측이 DB에 올라가고, **다 올라간 뒤에** 팀원 화면(AI 추천·예측 확인·갤러리)에 1분 안에 반영됩니다. "
    "올리다 실패하면 팀 화면은 직전 학습 결과 그대로입니다.\n"
    "- DB에 등록되지 않은 이미지가 있으면 시작할 때 경고합니다(그 이미지의 예측은 공유되지 않음).")
with st.expander("이 PC에만 저장하고 공유하지 않으려면"):
    st.code("python training.py --no-share", language="bash")
