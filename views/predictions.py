import streamlit as st

from core.dataset import load_metadata
from core.tagging import NEGATIVE, POSITIVE
from methods.predicted_tags import load_predictions, predictions_source
from views.common import (cannot_save_notice, current_user, image_card, load_labels_or_stop, load_tags_or_stop, paginate,
                          save_label_safely, tag_label)

PER_PAGE = 12
COLS = 4
ABOVE, BELOW = "임계값 이상 (해당으로 예측)", "임계값 미만 (해당 아님으로 예측)"

st.title("🤖 예측 확인")
df = load_metadata()
preds = load_predictions()
tags = [t for t in load_tags_or_stop() if t.key in set(preds["tag_key"])]
if not tags:
    st.info("아직 예측 결과가 없습니다. 로컬 PC에서 `training.py`로 학습하면 여기에 나옵니다. 방법은 **학습 현황** 페이지에 있어요.")
    st.page_link("views/train.py", label="학습 현황으로 이동", icon="🧠")
    st.stop()

labeler = current_user()
if not labeler:
    cannot_save_notice("맞음/틀림 저장")

tag = st.selectbox("태그", tags, format_func=tag_label, key="pred_tag")
c1, c2, c3 = st.columns([2, 2, 1])
threshold = c1.slider("확률 임계값", 0.0, 1.0, 0.5, 0.05, key="pred_threshold")
view = c2.radio("보기", [ABOVE, BELOW], key="pred_view")
hide_done = c3.toggle("내가 라벨링한 것 숨기기", value=False, key="pred_hide")
st.caption(f"예측 출처: {predictions_source()} · 맞음/틀림은 DB에 내 라벨로 저장되어 다음 학습에 반영됩니다.")

labels = load_labels_or_stop()
mine_rows = labels[(labels["tag_key"] == tag.key) & (labels["labeler"] == labeler)]
mine = dict(zip(mine_rows["image_id"], mine_rows["value"]))

p = preds[preds["tag_key"] == tag.key]
p = p[p["image_id"].isin(set(df["id"]))]
predicted_positive = view == ABOVE
p = p[(p["prob"] >= threshold) if predicted_positive else (p["prob"] < threshold)]
if hide_done:
    p = p[~p["image_id"].isin(mine)]
p = p.sort_values("prob", ascending=False)

n_above = int((preds.loc[preds["tag_key"] == tag.key, "prob"] >= threshold).sum())
st.metric(f"'{tag.name}'로 예측된 이미지", f"{n_above} / {len(df)}장")
if p.empty:
    st.info("조건에 맞는 이미지가 없습니다.")
    st.stop()

row = {i: n for n, i in enumerate(df["id"])}
start, end = paginate(len(p), PER_PAGE, key="pred_page")
cols = st.columns(COLS)
predicted_value = POSITIVE if predicted_positive else NEGATIVE
for n, (image_id, prob) in enumerate(zip(p["image_id"].iloc[start:end], p["prob"].iloc[start:end])):
    current = mine.get(image_id)
    with cols[n % COLS]:
        image_card(df, row[image_id], caption=f"확률 **{prob:.2f}**")
        b1, b2 = st.columns(2)
        # 맞음 = 예측대로 라벨, 틀림 = 예측의 반대로 라벨
        for col, text, value in ((b1, "⭕ 맞음", predicted_value), (b2, "❌ 틀림", 1 - predicted_value)):
            col.button(text, key=f"pr_{tag.key}_{image_id}_{text}", width="stretch",
                       type="primary" if current == value else "secondary", disabled=not labeler,
                       on_click=save_label_safely, args=(image_id, tag.key, value, labeler))
        if current is not None:
            st.caption(f"내 라벨: {'해당' if current == POSITIVE else '해당 아님'}")
