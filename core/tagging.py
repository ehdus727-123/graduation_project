"""팀 태그(Supabase tags), 라벨(Supabase labels), 예측 결과(predicted_tags.csv) 읽기/쓰기.

태그와 라벨은 모두 Supabase가 기준이다. DB 함수는 실패하면 db.DBError(한국어 메시지)를 던진다.
"""
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from core import db
from core.dataset import DATA_DIR

LABELS_CSV = DATA_DIR / "labels.csv"   # 예전 로컬 라벨. DB 이전(migrate_to_supabase.py)에만 쓴다
TAGS_YAML = DATA_DIR / "tags.yaml"     # 예전 태그 정의. DB 이전에만 쓴다
MODELS_DIR = DATA_DIR / "models"
REPORT_CSV = DATA_DIR / "train_report.csv"
PRED_CSV = DATA_DIR / "predicted_tags.csv"
EMB_NPY = DATA_DIR / "embeddings.npy"
EMB_IDS_CSV = DATA_DIR / "embeddings_ids.csv"

PRED_THRESHOLD = 0.5  # 갤러리에서 예측 태그로 표시할 최소 확률
LABEL_COLUMNS = ["image_id", "tag_key", "value", "labeler", "timestamp"]
POSITIVE, NEGATIVE = 1, 0
KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
TAGS_CACHE_SECONDS = 30  # 다른 팀원이 바꾼 태그는 최대 30초 뒤에 보인다 (내가 바꾼 건 즉시)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except db.DBError:
        raise
    except Exception as e:
        raise db.DBError(db.friendly_error(e)) from e


# ---------- 태그 ----------

@dataclass(frozen=True)
class TagDef:
    key: str
    name: str
    group: str = ""
    definition: str = ""
    booru_hint: str | None = None
    active: bool = True


class TagError(ValueError):
    """입력값이 잘못된 경우 (사용자에게 그대로 보여줄 한국어 메시지)."""


_tags_cache: tuple[float, list[TagDef]] | None = None


def invalidate_tags_cache() -> None:
    global _tags_cache
    _tags_cache = None


def load_tags(include_inactive: bool = False) -> list[TagDef]:
    """기본은 '사용 중'인 태그만. 태그 관리 페이지에서는 include_inactive=True."""
    global _tags_cache
    if _tags_cache is None or time.monotonic() - _tags_cache[0] > TAGS_CACHE_SECONDS:
        rows = _db_call(db.fetch_all, "tags", "key, name, tag_group, definition, booru_hint, active, sort_order")
        rows.sort(key=lambda r: (r["sort_order"], r["key"]))
        tags = [TagDef(key=r["key"], name=r["name"], group=r["tag_group"] or "", definition=r["definition"] or "",
                       booru_hint=r["booru_hint"] or None, active=bool(r["active"])) for r in rows]
        _tags_cache = (time.monotonic(), tags)
    tags = _tags_cache[1]
    return tags if include_inactive else [t for t in tags if t.active]


def _clean_hint(booru_hint: str | None) -> str | None:
    hint = (booru_hint or "").strip().lower().replace(" ", "_")
    return hint or None


def add_tag(key: str, name: str, group: str, definition: str, booru_hint: str | None = None) -> None:
    key, name, group, definition = key.strip(), name.strip(), group.strip(), definition.strip()
    if not KEY_PATTERN.match(key):
        raise TagError("key는 영문 소문자로 시작하고 영문 소문자·숫자·밑줄(_)만 쓸 수 있습니다 (최대 40자). 예: low_angle")
    if not name:
        raise TagError("표시 이름을 입력해 주세요.")
    if not group:
        raise TagError("분류 축을 선택하거나 입력해 주세요.")
    existing = load_tags(include_inactive=True)
    if any(t.key == key for t in existing):
        raise TagError(f"key '{key}'는 이미 있습니다. (사용 안 함 태그도 포함)")
    db.check_write()
    row = {"key": key, "name": name, "tag_group": group, "definition": definition,
           "booru_hint": _clean_hint(booru_hint), "active": True, "sort_order": len(existing)}
    _db_call(lambda: db.get_client().table("tags").insert(row, returning="minimal").execute())
    invalidate_tags_cache()


def update_tag(key: str, name: str, definition: str) -> None:
    name = name.strip()
    if not name:
        raise TagError("표시 이름은 비워 둘 수 없습니다.")
    db.check_write()
    _db_call(lambda: db.get_client().table("tags")
             .update({"name": name, "definition": definition.strip(), "updated_at": _now()}).eq("key", key).execute())
    invalidate_tags_cache()


def set_tag_active(key: str, active: bool) -> None:
    db.check_write()
    _db_call(lambda: db.get_client().table("tags").update({"active": active, "updated_at": _now()}).eq("key", key).execute())
    invalidate_tags_cache()


def delete_tag(key: str) -> None:
    """라벨이 하나도 없는 태그만 삭제할 수 있다 (DB 정책과 외래키로도 막혀 있음)."""
    db.check_write()
    n = _db_call(lambda: db.get_client().table("labels").select("id", count="exact").eq("tag_key", key).limit(1).execute()).count
    if n:
        raise TagError(f"라벨이 {n}건 붙어 있어 삭제할 수 없습니다. '사용 안 함'으로 바꿔 주세요.")
    deleted = _db_call(lambda: db.get_client().table("tags").delete().eq("key", key).execute()).data
    if not deleted:
        raise TagError("삭제되지 않았습니다. 그 사이 라벨이 추가되었거나 권한이 없습니다.")
    invalidate_tags_cache()


# ---------- 라벨 ----------

def load_labels() -> pd.DataFrame:
    """컬럼: image_id, tag_key, value, labeler, timestamp."""
    rows = _db_call(db.fetch_all, "labels", "image_id, tag_key, value, labeler, updated_at")
    df = pd.DataFrame(rows, columns=["image_id", "tag_key", "value", "labeler", "updated_at"])
    return df.rename(columns={"updated_at": "timestamp"}).astype({"image_id": str, "value": int})


def save_label(image_id: str, tag_key: str, value: int, labeler: str) -> None:
    """같은 (image_id, tag_key, labeler)는 덮어쓴다."""
    row = {"image_id": str(image_id), "tag_key": tag_key, "value": int(value), "labeler": labeler, "updated_at": _now()}
    _db_call(db.upsert, "labels", [row], on_conflict="image_id,tag_key,labeler")


def save_labels(image_id: str, values: dict[str, int], labeler: str) -> None:
    """이미지 한 장의 여러 태그 라벨을 한 번에 저장. 같은 (image_id, tag_key, labeler)는 덮어쓴다.

    여러 행을 한 번의 요청으로 보내므로 DB에서 하나의 INSERT ... ON CONFLICT 문으로 실행된다 → 전부 저장되거나 전부 실패.
    """
    if not values:
        return
    if not labeler.strip():
        raise TagError("이름이 비어 있습니다.")
    db.check_write()
    now = _now()
    rows = [{"image_id": str(image_id), "tag_key": key, "value": int(v), "labeler": labeler, "updated_at": now}
            for key, v in values.items()]
    _db_call(lambda: db.get_client().table("labels")
             .upsert(rows, on_conflict="image_id,tag_key,labeler", returning="minimal").execute())


def load_labels_csv(path=LABELS_CSV) -> pd.DataFrame:
    """예전 로컬 CSV 라벨 (DB 이전용)."""
    if not path.exists():
        return pd.DataFrame(columns=LABEL_COLUMNS).astype({"value": int})
    return pd.read_csv(path, dtype={"image_id": str, "tag_key": str, "labeler": str}, encoding="utf-8-sig")


def consolidate(labels: pd.DataFrame) -> pd.DataFrame:
    """여러 라벨러의 라벨을 (image_id, tag_key)당 하나로. 다수결, 동률이면 제외."""
    if labels.empty:
        return pd.DataFrame(columns=["image_id", "tag_key", "value"])
    mean = labels.groupby(["image_id", "tag_key"])["value"].mean().reset_index()
    mean = mean[mean["value"] != 0.5]
    mean["value"] = (mean["value"] > 0.5).astype(int)
    return mean


# ---------- 예측 ----------

def read_predictions() -> pd.DataFrame:
    """이 PC에서 마지막으로 학습한 예측 (predicted_tags.csv)."""
    if not PRED_CSV.exists():
        return pd.DataFrame(columns=["image_id", "tag_key", "prob"])
    return pd.read_csv(PRED_CSV, dtype={"image_id": str, "tag_key": str}, encoding="utf-8-sig")


# ---------- 팀 공유 예측 (학습 묶음 단위) ----------
# 학습 한 번 = training_runs 한 행. 예측은 그 run_id로 여러 번에 나눠 올리고, 다 올라가면 finish_training_run()으로
# '완료' 표시한다. 웹앱은 가장 최근에 완료된 학습의 예측만 읽으므로 올리는 도중·실패한 학습은 팀 화면에 보이지 않는다.

UPLOAD_CHUNK = 5000   # 한 요청에 보내는 예측 행 수 (약 0.8MB)
UPLOAD_RETRIES = 3    # 요청 하나가 실패하면 다시 보내는 횟수 (같은 행은 덮어쓰므로 다시 보내도 안전)
RUN_COLUMNS = "run_id, started_at, completed_at, trainer, embed_model, n_images, n_rows, report"


def known_image_ids() -> set[str]:
    """DB images에 등록된 이미지. 등록되지 않은 이미지의 예측은 외래키 때문에 공유할 수 없다."""
    return {str(r["id"]) for r in _db_call(db.fetch_all, "images", "id")}


ACTIVE_SECONDS = 120  # 이 시간 안에 진행 상황을 남긴 미완료 학습 = 지금 학습 중 (학습 PC는 HEARTBEAT_EVERY마다 남김)
HEARTBEAT_EVERY = 20


def start_training_run(trainer: str, embed_model: str) -> str:
    """학습을 시작할 때 기록을 '진행 중'으로 만들고 run_id를 돌려준다. 다른 팀원 화면에 '학습 중'으로 보인다."""
    import uuid
    if not trainer.strip():
        raise TagError("학습한 사람 이름이 비어 있습니다.")
    db.check_write()
    run_id = str(uuid.uuid4())
    row = {"run_id": run_id, "trainer": trainer.strip(), "embed_model": embed_model,
           "heartbeat_at": _now(), "stage": "준비"}
    _db_call(lambda: db.get_client().table("training_runs").insert(row, returning="minimal").execute())
    return run_id


def touch_training_run(run_id: str, stage: str) -> None:
    """진행 상황을 남긴다 (다른 팀원 화면의 '학습 중' 표시용)."""
    db.check_write()
    _db_call(lambda: db.get_client().rpc("touch_training_run", {"p_run": run_id, "p_stage": stage[:300]}).execute())


def active_training_runs() -> list[dict]:
    """지금 학습 중인 기록 (완료 전이고 ACTIVE_SECONDS 안에 진행 상황을 남긴 것). 최근 순."""
    since = (datetime.now(timezone.utc) - timedelta(seconds=ACTIVE_SECONDS)).isoformat()
    return _db_call(lambda: db.get_client().table("training_runs").select("run_id, trainer, started_at, heartbeat_at, stage")
                    .is_("completed_at", "null").gte("heartbeat_at", since)
                    .order("heartbeat_at", desc=True).execute().data)


def upload_predictions(run_id: str, preds: pd.DataFrame, model: str, known: set[str] | None = None,
                       progress=None) -> int:
    """예측을 UPLOAD_CHUNK행씩 나눠 올린다. 올린 행 수를 돌려준다. progress(올린 수, 전체 수)는 선택.

    known(DB에 있는 이미지 id)에 없는 이미지는 뺀다. 요청 하나가 실패하면 UPLOAD_RETRIES번까지 다시 보낸다.
    """
    db.check_write()
    p = preds if known is None else preds[preds["image_id"].astype(str).isin(known)]
    total = len(p)
    probs = p["prob"].astype(float).clip(0.0, 1.0).round(4)
    for start in range(0, total, UPLOAD_CHUNK):
        part = slice(start, start + UPLOAD_CHUNK)  # 행 dict는 조각마다 만든다 (수십만 행을 한꺼번에 만들지 않게)
        rows = [{"run_id": run_id, "image_id": str(i), "tag_key": k, "prob": float(v), "model": model}
                for i, k, v in zip(p["image_id"].iloc[part], p["tag_key"].iloc[part], probs.iloc[part])]
        for attempt in range(1, UPLOAD_RETRIES + 1):
            try:
                _db_call(lambda: db.get_client().table("predictions")
                         .upsert(rows, on_conflict="run_id,image_id,tag_key", returning="minimal").execute())
                break
            except db.DBError:
                if attempt == UPLOAD_RETRIES:
                    raise
                time.sleep(2 * attempt)
        if progress:
            progress(min(start + UPLOAD_CHUNK, total), total)
    return total


def finish_training_run(run_id: str, n_images: int, report: list[dict]) -> None:
    """다 올라간 학습을 '완료'로 표시하고 이미지 수·태그별 지표를 남긴다. 이때부터 팀 화면에 보인다.

    다른 학습 중 이미 완료된 것(옛 결과)과 30분 넘게 멈춘 것의 예측은 DB 함수가 지운다 (같은 시간에 올리는 중인 학습은 남김).
    """
    db.check_write()
    _db_call(lambda: db.get_client().rpc("finish_training_run",
                                         {"p_run": run_id, "p_n_images": int(n_images), "p_report": report}).execute())


def latest_training_run() -> dict | None:
    """가장 최근에 완료된 학습 기록 (없으면 None)."""
    rows = _db_call(lambda: db.get_client().table("training_runs").select(RUN_COLUMNS)
                    .not_.is_("completed_at", "null").order("completed_at", desc=True).limit(1).execute().data)
    return rows[0] if rows else None


def load_run_predictions(run_id: str) -> pd.DataFrame:
    """한 학습의 예측 전체. 컬럼: image_id, tag_key, prob."""
    def fetch():
        out, start = [], 0
        while True:  # 페이지를 넘길 때 순서가 바뀌지 않도록 기본 키 순으로 정렬
            page = (db.get_client().table("predictions").select("image_id, tag_key, prob").eq("run_id", run_id)
                    .order("image_id").order("tag_key").range(start, start + db.PAGE_SIZE - 1).execute().data)
            out += page
            if len(page) < db.PAGE_SIZE:
                return out
            start += db.PAGE_SIZE
    df = pd.DataFrame(_db_call(fetch), columns=["image_id", "tag_key", "prob"])
    return df.astype({"image_id": str, "tag_key": str, "prob": float})


def file_mtime(path) -> float | None:
    """캐시 키로 쓰기 위한 수정 시각. 파일이 바뀌면 캐시가 자동으로 무효화된다."""
    return path.stat().st_mtime if path.exists() else None
