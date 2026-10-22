"""사람이 확정한 포즈를 COCO keypoints 형식으로 내보낸다 (포즈 모델 파인튜닝용, 학습은 Colab에서 별도 진행).

  python export_coco.py                       # → drawing_ref_test/export/coco_poses.json
  python export_coco.py --out my.json

- 대상: status='done' 이고 source가 manual 또는 model_corrected 인 포즈. 한 이미지에 여러 작성자가 있으면 가장 최근 것.
- 관절 13개(우리 정의). COCO 표시값 v: 2=보임, 1=가려짐(위치 추정), 0=없음(x=y=0).
- 좌표는 로컬 images/ 파일의 픽셀 크기 기준. file_name은 images/ 기준 상대 경로.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from core.cli import quiet_streamlit, utf8_console

quiet_streamlit()

from PIL import Image  # noqa: E402

from core import db  # noqa: E402
from core import pose as P  # noqa: E402
from core import poses as store  # noqa: E402
from core.dataset import DATA_DIR, local_image, read_metadata  # noqa: E402

VISIBILITY = {P.VISIBLE: 2, P.OCCLUDED: 1, P.ABSENT: 0}


def build(poses, meta) -> dict:
    df = poses[(poses["status"] == P.DONE) & poses["source"].isin(P.SEARCHABLE_SOURCES)]
    df = df.sort_values("updated_at").drop_duplicates("image_id", keep="last")
    paths = dict(zip(meta["id"], meta["img_path"]))
    skeleton = [[a + 1, b + 1] for _, a, b, _, _ in P.BONES if a < P.N_JOINTS and b < P.N_JOINTS]  # COCO는 1부터
    images, annotations = [], []
    for n, r in enumerate(df.itertuples(), start=1):
        if r.image_id not in paths:
            continue
        with Image.open(local_image(paths[r.image_id])) as img:  # 이 PC에 파일이 없으면 원본 주소에서 내려받는다
            w, h = img.size
        kps = P.validate_keypoints(r.keypoints)
        flat, xs, ys = [], [], []
        for k in kps:
            v = VISIBILITY[k["state"]]
            x, y = (k["x"] * w, k["y"] * h) if v else (0.0, 0.0)
            flat += [round(x, 2), round(y, 2), v]
            if v:
                xs.append(x)
                ys.append(y)
        if not xs:  # 보이는 관절이 하나도 없으면 학습에 쓸 수 없다
            continue
        pad = 0.1 * max(max(xs) - min(xs), max(ys) - min(ys))
        x0, y0 = max(min(xs) - pad, 0), max(min(ys) - pad, 0)
        x1, y1 = min(max(xs) + pad, w), min(max(ys) + pad, h)
        images.append({"id": n, "file_name": Path(paths[r.image_id]).name, "width": w, "height": h,
                       "safebooru_id": r.image_id})
        annotations.append({"id": n, "image_id": n, "category_id": 1, "keypoints": flat,
                            "num_keypoints": sum(1 for k in kps if k["state"] != P.ABSENT),
                            "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)],
                            "area": round((x1 - x0) * (y1 - y0), 2), "iscrowd": 0,
                            "attributes": {"facing": r.facing, "annotator": r.annotator, "source": r.source}})
    return {
        "info": {"description": "드로잉 레퍼런스 포즈 라벨 (팀 내부 검증용, 이미지 재배포 금지)",
                 "date_created": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        "images": images, "annotations": annotations,
        "categories": [{"id": 1, "name": "person", "supercategory": "person",
                        "keypoints": P.JOINT_KEYS, "skeleton": skeleton}],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA_DIR / "export" / "coco_poses.json"))
    args = ap.parse_args()
    try:
        coco = build(store.load_poses(), read_metadata())
    except db.DBError as e:
        print(f"실패: {e}", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    print(f"저장: {out} · 이미지 {len(coco['images'])}장")
    return 0


if __name__ == "__main__":
    utf8_console()
    sys.exit(main())
