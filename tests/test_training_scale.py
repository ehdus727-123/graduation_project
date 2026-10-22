"""대량 학습 대비 단위 테스트 (실제 DB·CLIP 모델 없이).

- 임베딩 캐시: 중간에 멈춰도 이어서 하기, 저장 도중 끊긴 캐시 복구, 열 수 없는 이미지 건너뛰기
- 팀 공유: 나눠 올리기·재시도, 다 올라간 뒤에만 '완료', 실패하면 완료하지 않음
"""
import zlib

import numpy as np
import pandas as pd
import pytest

import training
from core import db
from core import tagging


def fake_vec(path: str) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(path.encode()))
    return rng.random(512, dtype=np.float32)


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    """결과 파일은 모두 임시 폴더에 (이 PC의 캐시·결과를 덮어쓰지 않게). 임베딩은 경로로 정해지는 가짜 벡터."""
    for name in ("EMB_NPY", "EMB_IDS_CSV"):
        monkeypatch.setattr(training, name, tmp_path / getattr(training, name).name)
    monkeypatch.setattr(training, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(training, "REPORT_CSV", tmp_path / "train_report.csv")
    monkeypatch.setattr(training, "PRED_CSV", tmp_path / "predicted_tags.csv")
    monkeypatch.setattr(training, "SAVE_EVERY", 0)  # 배치마다 저장 (멈춘 지점 확인용)
    calls = []

    def embed(_embedder, paths):
        calls.append(list(paths))
        if any("bad" in p for p in paths):
            raise OSError("cannot identify image file")
        return np.stack([fake_vec(p) for p in paths])

    monkeypatch.setattr(training, "embed_images", embed)
    return calls


def frame(n: int, bad: set[int] = frozenset()) -> pd.DataFrame:
    ids = [f"img{i:04d}" for i in range(n)]
    return pd.DataFrame({"id": ids, "img_path": [f"/x/{'bad' if i in bad else 'ok'}_{i}.jpg" for i in range(n)]})


def quiet(*_):
    pass


def test_resume_after_interrupt(tmp_cache, monkeypatch):
    df = frame(100)
    real = training.embed_images

    def stop_on_third(embedder, paths):
        if len(tmp_cache) == 2:  # 세 번째 배치에서 Ctrl+C
            raise KeyboardInterrupt
        return real(embedder, paths)

    monkeypatch.setattr(training, "embed_images", stop_on_third)
    with pytest.raises(KeyboardInterrupt):
        training.ensure_embeddings(df, lambda: object(), quiet)
    ids, vecs = training.load_embedding_cache()
    assert len(ids) == 2 * training.BATCH_SIZE  # 멈추기 전 두 배치는 저장됨

    monkeypatch.setattr(training, "embed_images", real)
    tmp_cache.clear()
    ok, X, n_new, skipped = training.ensure_embeddings(df, lambda: object(), quiet)
    assert n_new == 100 - 2 * training.BATCH_SIZE  # 나머지만 추출
    assert sum(len(c) for c in tmp_cache) == n_new
    assert list(ok["id"]) == list(df["id"]) and not skipped
    assert np.allclose(X, np.stack([fake_vec(p) for p in df["img_path"]]))  # df 순서와 정확히 맞는다


def test_cache_cut_between_files_keeps_common_prefix(tmp_cache):
    df = frame(70)
    training.ensure_embeddings(df, lambda: object(), quiet)
    ids = pd.read_csv(training.EMB_IDS_CSV, dtype=str, encoding="utf-8-sig")
    ids.iloc[:40].to_csv(training.EMB_IDS_CSV, index=False, encoding="utf-8-sig")  # id 목록을 쓰기 전에 끊긴 상황
    got_ids, vecs = training.load_embedding_cache()
    assert len(got_ids) == len(vecs) == 40
    tmp_cache.clear()
    ok, X, n_new, _ = training.ensure_embeddings(df, lambda: object(), quiet)
    assert n_new == 30 and len(ok) == 70


def test_broken_images_are_skipped(tmp_cache):
    df = frame(80, bad={3, 50})
    ok, X, n_new, skipped = training.ensure_embeddings(df, lambda: object(), quiet)
    assert {i for i, _ in skipped} == {"img0003", "img0050"}
    assert len(ok) == 78 == len(X) and n_new == 78
    assert "img0003" not in set(ok["id"])


def test_stops_when_nothing_can_be_read(tmp_cache):
    df = frame(200, bad=set(range(200)))
    with pytest.raises(training.TrainingError, match="한 장도 읽지 못했습니다"):
        training.ensure_embeddings(df, lambda: object(), quiet)


def test_progress_shows_speed_and_eta(tmp_cache):
    messages = []
    training.ensure_embeddings(frame(40), lambda: object(), lambda stage, msg, frac: messages.append(msg))
    assert any("장/초" in m and "남은 시간" in m for m in messages)


# ---------- 팀 공유 ----------

class FakeTable:
    def __init__(self, store, fail_times):
        self.store, self.fail_times = store, fail_times

    def upsert(self, rows, on_conflict, **_):
        self.rows = rows
        return self

    def execute(self):
        if self.fail_times[0] > 0:
            self.fail_times[0] -= 1
            raise RuntimeError("일시적 네트워크 오류")
        self.store.append(len(self.rows))
        return type("Res", (), {"data": None})()


@pytest.fixture
def fake_predictions(monkeypatch):
    store, fail = [], [0]
    client = type("C", (), {"table": lambda self, name: FakeTable(store, fail)})()
    monkeypatch.setattr(db, "get_client", lambda: client)
    monkeypatch.setattr(tagging.time, "sleep", lambda s: None)
    return store, fail


def test_upload_in_chunks_with_retry(fake_predictions, monkeypatch):
    store, fail = fake_predictions
    monkeypatch.setattr(tagging, "UPLOAD_CHUNK", 1000)
    preds = pd.DataFrame({"image_id": [str(i % 2600) for i in range(5200)], "tag_key": ["a"] * 2600 + ["b"] * 2600,
                          "prob": np.linspace(0, 1, 5200)})
    known = {str(i) for i in range(2500)}  # 100장은 DB에 없음 → 빠진다
    fail[0] = 2  # 처음 두 번 실패 → 재시도로 통과
    seen = []
    n = tagging.upload_predictions("run-1", preds, "m", known=known, progress=lambda d, t: seen.append((d, t)))
    assert n == 5000 and sum(store) == 5000 and max(store) <= 1000
    assert seen[-1] == (5000, 5000)


def test_upload_gives_up_after_retries(fake_predictions):
    store, fail = fake_predictions
    fail[0] = 99
    preds = pd.DataFrame({"image_id": ["1"], "tag_key": ["a"], "prob": [0.5]})
    with pytest.raises(db.DBError):
        tagging.upload_predictions("run-1", preds, "m")


@pytest.fixture
def fake_training_data(tmp_cache, monkeypatch):
    df = frame(60)
    tags = [tagging.TagDef("a", "태그A", "앵글")]
    labels = pd.DataFrame({"image_id": df["id"][:50], "tag_key": "a", "value": [1] * 25 + [0] * 25,
                           "labeler": "김하영", "timestamp": "2026-10-22T00:00:00+00:00"})
    monkeypatch.setattr(training, "read_metadata", lambda: df)
    monkeypatch.setattr(training, "load_tags", lambda: tags)
    monkeypatch.setattr(training, "load_labels", lambda: labels)
    monkeypatch.setattr(training, "known_image_ids", lambda: set(df["id"][:55]))
    calls = []
    monkeypatch.setattr(training, "start_training_run",
                        lambda trainer, model, n, report: calls.append(("start", trainer, n, report)) or "run-9")
    monkeypatch.setattr(training, "finish_training_run", lambda run_id: calls.append(("finish", run_id)))
    return calls


def test_publish_marks_complete_only_after_upload(fake_training_data, monkeypatch):
    calls = fake_training_data
    monkeypatch.setattr(training, "upload_predictions",
                        lambda run_id, preds, model, known=None, progress=None:
                        calls.append(("upload", run_id, len(preds[preds["image_id"].isin(known)]))) or 55)
    result = training.train_all(get_embedder=lambda: object(), progress=quiet, publish=True, trainer="김하영")
    assert [c[0] for c in calls] == ["start", "upload", "finish"]
    start = calls[0]
    assert start[1] == "김하영" and start[2] == 60
    assert start[3][0]["tag_key"] == "a" and isinstance(start[3][0]["f1"], float)  # DB(jsonb)로 보낼 수 있는 숫자
    assert calls[1] == ("upload", "run-9", 55) and calls[2] == ("finish", "run-9")
    assert result.shared_rows == 55 and not result.share_error
    assert training.PRED_CSV.exists()


def test_failed_upload_is_not_marked_complete(fake_training_data, monkeypatch):
    calls = fake_training_data

    def broken(*a, **k):
        raise db.DBError("DB 오류: 요청 시간이 초과되었습니다")

    monkeypatch.setattr(training, "upload_predictions", broken)
    result = training.train_all(get_embedder=lambda: object(), progress=quiet, publish=True, trainer="김하영")
    assert [c[0] for c in calls] == ["start"]  # finish를 부르지 않으므로 팀 화면은 직전 학습 그대로
    assert "시간이 초과" in result.share_error and result.trained
    assert training.PRED_CSV.exists()  # 이 PC의 결과는 남는다


def test_publish_requires_trainer(fake_training_data):
    with pytest.raises(training.TrainingError, match="이름"):
        training.train_all(get_embedder=lambda: object(), progress=quiet, publish=True, trainer=" ")
