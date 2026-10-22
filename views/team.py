import pandas as pd
import streamlit as st

import training
from core import poses as store
from core import team
from core.dataset import load_metadata
from core.db import DBError
from core.image_labeling import team_finished_images
from core.tagging import consolidate
from methods.predicted_tags import predictions_source, predictions_time
from views.common import current_user, load_labels_or_stop, load_tags_or_stop

st.title("👥 팀 현황")
st.caption("누가 얼마나 했는지, 지금 무엇이 부족한지 봅니다. 모든 팀원이 같은 DB 데이터를 봅니다.")
df = load_metadata()
ids = set(df["id"])
tags = load_tags_or_stop()
active = [t.key for t in tags]
labels = load_labels_or_stop()
try:
    poses = store.load_poses()
except DBError as e:
    poses = None
    st.caption(f"포즈 데이터를 불러오지 못해 포즈 항목은 비워 둡니다. {e}")
try:
    evals = store.load_pose_evals()
except DBError:
    evals = None

status = training.label_status(tags, consolidate(labels[labels["image_id"].isin(ids)]))
members = team.member_table(labels, poses, evals, active)
pose_prog = store.progress(poses) if poses is not None else None

# ---------- 요약 ----------
m1, m2, m3, m4 = st.columns(4)
m1.metric("참여 팀원", len(members))
m2.metric("라벨링 끝낸 이미지", f"{len(team_finished_images(labels, active) & ids)} / {len(df)}",
          help="한 명 이상이 사용 중인 모든 태그를 라벨링한 이미지")
m3.metric("학습 가능한 태그", f"{int(status['trainable'].sum())} / {len(tags)}",
          help=f"양성 {training.MIN_POS}건, 음성 {training.MIN_NEG}건 이상 (팀 다수결 기준)")
m4.metric("포즈 완료 이미지", f"{pose_prog['done_images']} / {len(df)}" if pose_prog else "-")

# ---------- 지금 필요한 일 ----------
st.subheader("📌 지금 필요한 일")
todo = 0

when = predictions_time()
new_labels = team.labels_since(labels, when)
if when is None and status["trainable"].any():
    todo += 1
    st.info(f"학습 가능한 태그가 {int(status['trainable'].sum())}개 있지만 아직 학습하지 않았습니다. "
            "이미지가 있는 PC 한 곳에서 `training.py`를 실행하면 팀 전체에 AI 추천이 생깁니다.")
    st.page_link("views/train.py", label="학습 현황 · 실행 방법 보기", icon="🧠")
elif new_labels:
    todo += 1
    st.info(f"마지막 학습({predictions_source()}) 이후 새 라벨 **{new_labels}건**이 쌓였습니다. "
            "로컬 PC에서 다시 학습하면 AI 추천이 더 정확해집니다.")
    st.page_link("views/train.py", label="학습 현황 · 실행 방법 보기", icon="🧠")

lacking = status[~status["trainable"]]
if not lacking.empty:
    todo += 1
    st.markdown(f"**라벨이 부족한 태그 {len(lacking)}개** — 학습하려면 아래만큼 더 필요합니다.")
    hints = {t.key: t.booru_hint or "" for t in tags}
    st.dataframe(pd.DataFrame({
        "태그": lacking["name"], "분류": lacking["group"], "해당": lacking["n_pos"], "아님": lacking["n_neg"],
        "부족": lacking["reason"].str.replace("라벨 부족: ", "", regex=False),
        "Safebooru 힌트": lacking["tag_key"].map(hints),
    }), hide_index=True, width="stretch")
    st.caption("양성(해당)이 부족하면: **라벨링 → 태그별 (하나씩)**에서 그 태그를 고르고 "
               "'Safebooru 힌트 태그 우선' 순서로 보면 해당 예시를 빨리 모을 수 있습니다.")
    st.page_link("views/labeling.py", label="라벨링으로 이동", icon="🏷️")

if pose_prog:
    left = len(df) - pose_prog["done_images"] - pose_prog["skipped_images"]
    if left > 0:
        todo += 1
        st.markdown(f"**포즈 라벨링이 남은 이미지 {left}장** — 누군가 완료한 이미지는 자동으로 건너뛰니 동시에 해도 겹치지 않습니다.")
        st.page_link("views/pose_label.py", label="포즈 라벨링으로 이동", icon="🦴")

if not todo:
    st.success("지금 급한 일은 없습니다. 🎉")

# ---------- 팀원별 ----------
st.subheader("🙋 팀원별 활동")
if members.empty:
    st.info("아직 기록이 없습니다. 사이드바에 이름을 입력하고 라벨링을 시작해 보세요.")
else:
    me = current_user()
    last = pd.to_datetime(members["last_active"], utc=True).dt.tz_convert("Asia/Seoul").dt.strftime("%m/%d %H:%M")
    st.dataframe(pd.DataFrame({
        "이름": members["name"] + members["name"].map(lambda n: "  ← 나" if n == me else ""),
        "라벨링 끝낸 이미지": members["label_images"], "라벨 수": members["labels"], "최근 7일 라벨": members["labels_7d"],
        "포즈 완료": members["pose_done"], "포즈 임시저장": members["pose_draft"], "포즈 검색 평가": members["evals"],
        "최근 활동": last.fillna("-"),
    }), hide_index=True, width="stretch")

    if pairs := team.suspicious_pairs(members["name"]):
        st.warning("같은 사람일 수 있는 이름이 있습니다: " + ", ".join(f"**{a}** / **{b}**" for a, b in pairs)
                   + ". 같은 사람이라면 앞으로 한 이름만 써 주세요 (이름이 다르면 다수결에서 두 명으로 계산됩니다).")
