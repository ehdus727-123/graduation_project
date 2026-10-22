"""새 이미지 등록(migrate_to_supabase.py --images-only) 단위 테스트 (실제 DB 없이)."""
import pandas as pd
import pytest
from PIL import Image

import migrate_to_supabase as M


@pytest.fixture
def local_images(tmp_path, monkeypatch):
    paths = []
    for i, color in enumerate(["red", "green", "blue"]):
        p = tmp_path / f"{i}.jpg"
        Image.new("RGB", (40, 60), color).save(p)
        paths.append(str(p))
    df = pd.DataFrame({"id": ["1", "2", "3"], "url": [f"https://x/{i}.jpg" for i in range(3)],
                       "source_page": "https://x", "tag_list": [["a"], ["b"], ["c"]],
                       "width": [40, 40, None], "height": [60, 60, None], "img_path": paths})
    monkeypatch.setattr(M, "read_local_metadata", lambda: df)
    monkeypatch.setattr(M, "CACHE_DIR", tmp_path / "cache")  # 이 PC의 phash 캐시를 읽지 않게
    sent = []
    monkeypatch.setattr(M.db, "fetch_all", lambda table, cols: [{"id": "1"}])  # 1번은 이미 DB에 있음
    monkeypatch.setattr(M.db, "upsert", lambda table, rows, on_conflict, **k: sent.append((table, rows, k)))
    return sent


def test_adds_only_new_images_without_touching_existing(local_images):
    total, new = M.migrate_images(progress=lambda *_: None)
    assert (total, new) == (3, 2)
    table, rows, kwargs = local_images[0]
    assert table == "images" and [r["id"] for r in rows] == ["2", "3"]  # 이미 있는 1번은 보내지 않는다
    assert kwargs.get("ignore_duplicates") is True  # 그 사이 누가 먼저 등록해도 덮어쓰지 않는다 (팀원 키는 추가만 가능)
    assert all(isinstance(r["phash"], str) and len(r["phash"]) == 16 for r in rows)  # 해시를 빈 값으로 보내지 않는다
    assert rows[1]["width"] is None  # 크기를 모르면 비워 둔다


def test_nothing_to_add(local_images, monkeypatch):
    monkeypatch.setattr(M.db, "fetch_all", lambda table, cols: [{"id": i} for i in ("1", "2", "3")])
    assert M.migrate_images(progress=lambda *_: None) == (3, 0)
    assert local_images == []  # 보낼 것이 없으면 요청도 하지 않는다
