"""데이터셋(metadata.csv + images/) 로딩과 썸네일 관리."""
import hashlib
import os
import urllib.request
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "drawing_ref_test"
# 배포 서버(Streamlit Community Cloud)인지. 그 서버는 코드를 항상 /mount/src/ 아래에서 실행한다.
# 서버는 메모리가 작아 학습·AI 일괄 추정처럼 무거운 작업을 하면 앱 전체가 멈출 수 있으므로 화면에서 막는다.
# 시험용으로 DRAWING_REF_CLOUD=1(서버처럼) / 0(PC처럼)을 주면 그 값을 따른다.
ON_CLOUD = os.getenv("DRAWING_REF_CLOUD", "1" if str(ROOT).startswith("/mount/src/") else "0") == "1"
CLOUD_HEAVY_NOTICE = ("배포 서버에서는 메모리가 부족해 앱 전체가 멈출 수 있어 이 작업을 막아 두었습니다. "
                      "각자 PC에서 앱을 실행해 진행해 주세요. 결과는 DB에 저장되어 서버 화면에도 그대로 보입니다.")
IMG_DIR = DATA_DIR / "images"
META_CSV = DATA_DIR / "metadata.csv"
CACHE_DIR = DATA_DIR / "cache"
THUMB_DIR = CACHE_DIR / "thumbs"
REMOTE_DIR = CACHE_DIR / "remote"  # 서버: URL 이미지를 한 번 내려받아 두는 곳
THUMB_SIZE = 384
DOWNLOAD_TIMEOUT = 30


def is_url(img_path: str) -> bool:
    return img_path.startswith(("http://", "https://"))


def local_image(img_path: str) -> str:
    """파일로 열 수 있는 이미지 경로. 서버처럼 img_path가 URL이면 내려받아 캐시하고 그 경로를 돌려준다.

    PIL·OpenCV·CLIP 등 파일 경로가 필요한 곳은 img_path를 바로 쓰지 말고 이 함수를 거친다.
    """
    if not is_url(img_path):
        return img_path
    suffix = Path(img_path.split("?", 1)[0]).suffix.lower()
    dst = REMOTE_DIR / (hashlib.md5(img_path.encode()).hexdigest() + (suffix if len(suffix) <= 5 else ""))
    if not dst.exists():
        REMOTE_DIR.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(img_path, headers={"User-Agent": "drawing-ref-validation/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as r:
                data = r.read()
        except OSError as e:
            raise FileNotFoundError(f"이미지를 내려받지 못했습니다 ({img_path}): {e}") from e
        tmp = dst.with_name(dst.name + ".part")  # 받다가 끊겨도 깨진 파일이 캐시로 남지 않게
        tmp.write_bytes(data)
        tmp.replace(dst)
    return str(dst)


def _local_path(image_id: str, csv_path) -> Path:
    # csv의 path는 만든 사람 PC의 절대경로라서, 파일명만 따서 images/ 기준으로 다시 잡는다
    if isinstance(csv_path, str) and csv_path:
        return IMG_DIR / csv_path.replace("\\", "/").rsplit("/", 1)[-1]
    return IMG_DIR / f"{image_id}.jpg"


def read_metadata() -> pd.DataFrame:
    """팀 공통 이미지 목록. 행 번호(0..n-1)가 모든 검색 모듈에서 공통 인덱스로 쓰인다. 웹앱에서는 load_metadata()를 쓴다.

    DB(images)가 기준이다 → 어느 PC에서든(예전 drawing_ref_test를 복사해 둔 PC도) 같은 이미지 목록을 본다.
    이 PC에 이미지 파일(images/{id}.jpg)이 있으면 그 파일을, 없으면 원본 주소(URL)를 img_path로 쓴다.
    DB 접속 정보가 없을 때만(오프라인 개발) 로컬 metadata.csv를 쓴다.
    """
    from core import db  # db.py가 dataset.ROOT를 import하므로 순환 import를 피하려고 함수 안에서 import
    if not db.is_configured():
        return read_local_metadata()
    df = _read_metadata_from_db()
    local = [IMG_DIR / f"{i}.jpg" for i in df["id"]]
    df["img_path"] = [str(p) if p.exists() else url for p, url in zip(local, df["url"])]
    return df


def read_local_metadata() -> pd.DataFrame:
    """이 PC의 metadata.csv + images/ (이미지 파일이 있는 행만). DB 등록(migrate_to_supabase.py)과 오프라인 개발용."""
    df = pd.read_csv(META_CSV, dtype={"id": str})
    df = df.drop_duplicates(subset="id")
    df["tags"] = df["tags"].fillna("")
    df["img_path"] = [str(_local_path(i, p)) for i, p in zip(df["id"], df.get("path", [None] * len(df)))]
    df = df[df["img_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
    df["tag_list"] = df["tags"].str.split()
    return df


def _read_metadata_from_db() -> pd.DataFrame:
    from core import db  # db.py가 dataset.ROOT를 import하므로 순환 import를 피하려고 함수 안에서 import

    rows = db.call(db.fetch_all, "images", "id,url,source_page,tags,width,height,phash")
    df = pd.DataFrame(rows, columns=["id", "url", "source_page", "tags", "width", "height", "phash"])
    df["id"] = df["id"].astype(str)
    # DB는 순서를 보장하지 않으므로 id로 정렬한다 (행 번호가 공통 인덱스라 실행마다·PC마다 같아야 한다)
    df = df.drop_duplicates(subset="id")
    df = df.iloc[sorted(range(len(df)), key=lambda n: (len(df["id"].iat[n]), df["id"].iat[n]))].reset_index(drop=True)
    df["tag_list"] = df["tags"].map(lambda t: list(t) if isinstance(t, list) else str(t or "").split())
    df["tags"] = df["tag_list"].map(" ".join)
    df["img_path"] = df["url"]
    return df


METADATA_TTL = 300  # 다른 팀원이 DB에 등록한 새 이미지는 최대 5분 뒤에 보인다 (앱을 다시 켜지 않아도)
load_metadata = st.cache_data(show_spinner="이미지 목록 불러오는 중...", ttl=METADATA_TTL)(read_metadata)


def metadata_key(df: pd.DataFrame) -> tuple:
    """이미지 목록이 바뀌었는지 비교하는 값 (검색 인덱스 등 행 번호에 묶인 캐시의 키)."""
    return len(df), hash(tuple(df["id"]))


def thumbnail(img_path: str) -> str:
    """갤러리용 축소 이미지 경로. 없으면 만들어서 cache/thumbs/에 저장한다."""
    if is_url(img_path):  # 서버: URL은 브라우저가 직접 불러오므로 그대로 쓴다
        return img_path
    src = Path(img_path)
    dst = THUMB_DIR / src.name
    if not dst.exists():
        THUMB_DIR.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as img:
            img = img.convert("RGB")
            img.thumbnail((THUMB_SIZE, THUMB_SIZE))
            img.save(dst, quality=85)
    return str(dst)