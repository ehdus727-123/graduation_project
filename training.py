"""CLIP 임베딩 + 태그별 로지스틱 회귀 학습. 로컬 PC 전용 (웹앱 '학습 현황'은 결과만 보여준다).

실행:  python training.py --email 팀명단의_내_이메일      (결과를 DB에 올려 팀과 공유)
       python training.py --no-share                      (이 PC에만 저장, 이메일 불필요)
순서: 임베딩 추출(캐시에 없는 것만) → 태그별 학습(80/20 층화 분할) → 평가 → 전체 이미지 예측 → 팀 공유

이미지 수천~수만 장을 전제로 한다.
- 임베딩은 SAVE_EVERY초마다·Ctrl+C로 멈출 때·끝날 때 캐시에 저장 → 같은 명령으로 다시 실행하면 이어서 한다.
- 열 수 없는 이미지는 건너뛰고 마지막에 목록을 보여준다.
- 진행률에 속도와 남은 시간을 표시한다. 메모리는 최대 약 1.7GB(3,000장 실측, CLIP 모델 약 1.1GB) + 1만 장당 임베딩 약 20MB.
- 예측은 나눠 올리고 다 올라간 뒤에만 '완료' 표시 → 팀 화면에는 완료된 학습만 보인다 (core.tagging 참고).
"""
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from core.cli import quiet_streamlit, utf8_console

quiet_streamlit()

from core.dataset import local_image, read_metadata  # noqa: E402
from core.tagging import (EMB_IDS_CSV, EMB_NPY, HEARTBEAT_EVERY, MODELS_DIR, NEGATIVE, POSITIVE,  # noqa: E402
                          PRED_CSV, REPORT_CSV, TagDef, consolidate, finish_training_run, load_labels, load_tags,
                          start_training_run, touch_training_run, upload_predictions)

# 임베딩 모델 설정. WD Tagger 등으로 바꿀 때는 이 상수와 load_embedder / embed_images만 고치면 된다.
# EMBED_ID가 바뀌면 기존 임베딩 캐시는 자동으로 버리고 새로 추출한다.
EMBED_MODEL = "ViT-B-32"
EMBED_PRETRAINED = "laion2b_s34b_b79k"
EMBED_ID = f"open_clip/{EMBED_MODEL}/{EMBED_PRETRAINED}"

BATCH_SIZE = 32
MIN_POS = 20
MIN_NEG = 20
TEST_SIZE = 0.2
SEED = 42
F1_WARN = 0.7
SAVE_EVERY = 30         # 임베딩 캐시 저장 간격(초). 창이 갑자기 닫히면 마지막 저장 뒤 작업만 잃는다 (배치마다 쓰면 장수가 많을 때 느림)
PROGRESS_EVERY = 3      # 진행률 표시 간격(초)
MAX_FAILED_BATCHES = 3  # 연속으로 이만큼의 배치에서 한 장도 못 읽으면 이미지가 아니라 환경 문제로 보고 멈춘다

Progress = Callable[[str, str, float | None], None]  # (단계, 메시지, 진행률 0~1 또는 None)


class TrainingError(Exception):
    def __init__(self, stage: str, target: str | None, cause: Exception):
        self.stage, self.target, self.cause = stage, target, cause
        where = f" ({target})" if target else ""
        super().__init__(f"'{stage}' 단계{where}에서 실패했습니다: {type(cause).__name__}: {cause}")


@dataclass
class TrainResult:
    status: pd.DataFrame                    # 태그별 라벨 현황
    report: pd.DataFrame | None = None      # 이번에 학습한 태그의 지표
    n_new_embeddings: int = 0
    seconds: float = 0.0
    log: list[str] = field(default_factory=list)
    shared_rows: int | None = None          # 팀에 공유한 예측 행 수 (공유하지 않았으면 None)
    share_error: str = ""                   # 공유 실패 사유 (로컬 결과 파일은 이미 저장된 상태)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # 열 수 없어 건너뛴 이미지 (id, 사유)
    n_images: int = 0                       # 임베딩이 있어 학습·예측에 쓴 이미지 수

    @property
    def trained(self) -> bool:
        return self.report is not None and not self.report.empty


def _print_progress(stage: str, message: str, frac: float | None) -> None:
    pct = f" {frac:.0%}" if frac is not None else ""
    print(f"[{stage}]{pct} {message}", flush=True)


# ---------- 임베딩 ----------

def load_embedder():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(EMBED_MODEL, pretrained=EMBED_PRETRAINED, device="cpu")
    model.eval()
    return model, preprocess


def embed_images(embedder, paths: list[str]) -> np.ndarray:
    import torch
    model, preprocess = embedder
    batch = []
    for p in paths:
        with Image.open(local_image(p)) as img:
            batch.append(preprocess(img.convert("RGB")))
    with torch.no_grad():
        feats = model.encode_image(torch.stack(batch))
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.numpy().astype(np.float32)


def load_embedding_cache() -> tuple[list[str], np.ndarray | None]:
    if not (EMB_NPY.exists() and EMB_IDS_CSV.exists()):
        return [], None
    try:
        ids = pd.read_csv(EMB_IDS_CSV, dtype=str, encoding="utf-8-sig")
        vecs = np.load(EMB_NPY)
    except Exception:
        return [], None  # 깨진 캐시는 버리고 새로 추출한다
    if ids.empty or (ids["model"] != EMBED_ID).any():
        return [], None  # 다른 모델로 만든 캐시는 쓰지 않는다
    # 두 파일을 따로 저장하므로 그 사이에 멈추면 개수가 다를 수 있다. 캐시는 뒤에 덧붙이기만 하므로 앞쪽 공통 부분은 맞다
    n = min(len(ids), len(vecs))
    return list(ids["image_id"][:n]), vecs[:n]


def _replace_atomically(path, write) -> None:
    """임시 파일에 다 쓴 뒤 바꿔치기 → 저장 도중에 멈춰도 예전 파일이 깨지지 않는다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    write(tmp)
    os.replace(tmp, path)


def _save_embedding_cache(ids: list[str], vecs: np.ndarray) -> None:
    def write_npy(tmp):
        with open(tmp, "wb") as f:  # 파일 객체로 저장 (경로로 주면 numpy가 이름 끝에 .npy를 덧붙인다)
            np.save(f, vecs)
    _replace_atomically(EMB_NPY, write_npy)  # 벡터 먼저, id 목록 나중 (읽을 때 앞쪽 공통 부분만 쓰므로 안전)
    _replace_atomically(EMB_IDS_CSV, lambda tmp: pd.DataFrame({"image_id": ids, "model": EMBED_ID})
                        .to_csv(tmp, index=False, encoding="utf-8-sig"))


def _embed_batch(embedder, chunk: list[tuple[str, str]]) -> tuple[list[str], np.ndarray | None, list[tuple[str, str]]]:
    """(성공한 id, 벡터, 실패 [(id, 사유)]). 배치가 실패하면 한 장씩 다시 해서 열 수 없는 이미지만 뺀다."""
    try:
        return [i for i, _ in chunk], embed_images(embedder, [p for _, p in chunk]), []
    except Exception:
        pass
    ok_ids, vecs, bad = [], [], []
    for i, p in chunk:
        try:
            vecs.append(embed_images(embedder, [p]))
            ok_ids.append(i)
        except Exception as e:
            bad.append((i, f"{type(e).__name__}: {e}"))
    return ok_ids, (np.vstack(vecs) if vecs else None), bad


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}시간 {seconds % 3600 // 60}분"
    return f"{seconds // 60}분 {seconds % 60}초" if seconds >= 60 else f"{seconds}초"


def ensure_embeddings(df: pd.DataFrame, get_embedder: Callable,
                      progress: Progress) -> tuple[pd.DataFrame, np.ndarray, int, list[tuple[str, str]]]:
    """(임베딩이 있는 이미지만 남긴 df, 그 순서의 임베딩 행렬, 새로 추출한 장수, 건너뛴 이미지 [(id, 사유)]).

    SAVE_EVERY초마다, 그리고 멈출 때(Ctrl+C·오류 포함)와 끝날 때 캐시에 저장한다 → 다시 실행하면 이어서 한다.
    """
    ids, base = load_embedding_cache()
    have = set(ids)
    missing = [(i, p) for i, p in zip(df["id"], df["img_path"]) if i not in have]
    new_ids: list[str] = []
    new_vecs: list[np.ndarray] = []
    skipped: list[tuple[str, str]] = []

    def save() -> None:
        nonlocal base, ids, new_ids, new_vecs
        if not new_vecs:
            return
        base = np.concatenate(([base] if base is not None else []) + new_vecs)
        ids = ids + new_ids
        new_ids, new_vecs = [], []
        _save_embedding_cache(ids, base)

    if missing:
        progress("임베딩 추출", f"모델 불러오는 중 ({EMBED_MODEL}) · 새로 추출할 이미지 {len(missing):,}장"
                 + (f" (캐시에 {len(have):,}장 있음 — 이어서 합니다)" if have else ""), 0.0)
        embedder = get_embedder()
        t0 = last_save = last_report = time.perf_counter()
        failed_in_a_row = 0
        try:
            for start in range(0, len(missing), BATCH_SIZE):
                chunk = missing[start:start + BATCH_SIZE]
                ok_ids, vecs, bad = _embed_batch(embedder, chunk)
                skipped += bad
                failed_in_a_row = 0 if ok_ids else failed_in_a_row + 1
                if failed_in_a_row >= MAX_FAILED_BATCHES:
                    raise TrainingError("임베딩 추출", f"이미지 {chunk[0][0]} ~ {chunk[-1][0]}",
                                        RuntimeError(f"연속 {failed_in_a_row}개 배치에서 이미지를 한 장도 읽지 못했습니다. "
                                                     f"마지막 오류: {bad[-1][1]}"))
                if ok_ids:
                    new_ids += ok_ids
                    new_vecs.append(vecs)
                now = time.perf_counter()
                if now - last_save >= SAVE_EVERY:
                    save()
                    last_save = now
                done = start + len(chunk)
                if now - last_report >= PROGRESS_EVERY or done == len(missing):
                    rate = done / max(now - t0, 1e-6)
                    left = (len(missing) - done) / rate
                    progress("임베딩 추출", f"{done:,}/{len(missing):,}장 · {rate:.1f}장/초 · 남은 시간 약 {_duration(left)}"
                             + (f" · 건너뜀 {len(skipped)}장" if skipped else ""), done / len(missing))
                    last_report = now
        finally:
            save()  # 멈춰도(Ctrl+C·오류) 그때까지 추출한 것은 남긴다
    else:
        progress("임베딩 추출", f"캐시 사용 ({len(df):,}장 모두 있음)", 1.0)
    row = {i: n for n, i in enumerate(ids)}
    ok = df[df["id"].isin(row)].reset_index(drop=True)
    X = base[[row[i] for i in ok["id"]]] if len(ok) else np.zeros((0, 512), dtype=np.float32)
    return ok, X, len(missing) - len(skipped), skipped


# ---------- 라벨 현황 ----------

def label_status(tags: list[TagDef], consolidated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in tags:
        v = consolidated.loc[consolidated["tag_key"] == t.key, "value"]
        pos, neg = int((v == POSITIVE).sum()), int((v == NEGATIVE).sum())
        lacks = []
        if pos < MIN_POS:
            lacks.append(f"양성 {MIN_POS - pos}건")
        if neg < MIN_NEG:
            lacks.append(f"음성 {MIN_NEG - neg}건")
        rows.append({"tag_key": t.key, "name": t.name, "group": t.group, "n_pos": pos, "n_neg": neg,
                     "trainable": not lacks, "reason": "" if not lacks else "라벨 부족: " + ", ".join(lacks) + " 더 필요"})
    return pd.DataFrame(rows, columns=["tag_key", "name", "group", "n_pos", "n_neg", "trainable", "reason"])


# ---------- 학습 ----------

def _train_one(X: np.ndarray, y: np.ndarray) -> tuple[LogisticRegression, dict]:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
    clf.fit(X_tr, y_tr)
    p, r, f1, _ = precision_recall_fscore_support(y_te, clf.predict(X_te), average="binary", zero_division=0)
    return clf, {"n_train": len(y_tr), "n_test": len(y_te), "precision": p, "recall": r, "f1": f1}


def train_all(get_embedder: Callable = load_embedder, progress: Progress = _print_progress,
              publish: bool = False, trainer: str = "") -> TrainResult:
    """publish=True면 예측을 DB에 올려 팀 전체가 같은 결과를 본다 (CLI 기본). trainer = 학습 기록에 남는 이름.

    기본값이 False인 이유: 테스트처럼 함수를 직접 부르는 곳이 팀 공유 예측을 덮어쓰지 않게.
    """
    t0 = time.perf_counter()
    log: list[str] = []
    run_id: str | None = None  # 팀 공유 학습 기록 (시작할 때 만들어 진행 상황을 남긴다)
    last_beat = 0.0

    def report_progress(stage, message, frac=None):
        nonlocal last_beat
        log.append(f"[{stage}] {message}")
        progress(stage, message, frac)
        if run_id and time.monotonic() - last_beat >= HEARTBEAT_EVERY:  # 다른 팀원 화면의 '학습 중' 표시
            last_beat = time.monotonic()
            try:
                touch_training_run(run_id, f"{stage} · {message}")
            except Exception:
                pass  # 진행 상황 공유가 안 돼도 학습은 계속한다

    if publish and not trainer.strip():
        raise TrainingError("준비", None, ValueError("팀에 공유하려면 학습한 사람 이름이 필요합니다."))
    try:
        df = read_metadata()
        tags = load_tags()
        all_labels = consolidate(load_labels())
    except Exception as e:
        raise TrainingError("준비", None, e) from e
    labels = all_labels[all_labels["image_id"].isin(set(df["id"]))]  # 다른 데이터셋에서 온 라벨은 무시
    report_progress("준비", f"이미지 {len(df):,}장 · 사용 중 태그 {len(tags)}개")

    status = label_status(tags, labels)
    if not status["trainable"].any():
        for _, s in status.iterrows():
            report_progress("준비", f"{s['name']} 건너뜀 — {s['reason']}")
        report_progress("준비", "학습 가능한 태그가 없습니다 (라벨 부족)")
        return TrainResult(status=status, seconds=time.perf_counter() - t0, log=log)

    share_error = ""
    if publish:
        try:
            run_id = start_training_run(trainer, EMBED_ID)
            touch_training_run(run_id, "준비")
            last_beat = time.monotonic()
        except Exception as e:  # 기록을 못 만들면 공유만 못 한다 (학습은 이 PC에서 계속)
            share_error = f"학습 기록을 만들지 못했습니다: {e}"
            report_progress("팀 공유", f"⚠️ {share_error} — 학습은 계속하고 결과는 이 PC에만 저장합니다")

    df, X, n_new, skipped = ensure_embeddings(df, get_embedder, report_progress)
    if skipped:
        report_progress("임베딩 추출", f"열 수 없어 건너뛴 이미지 {len(skipped)}장 (학습·예측에서 빠짐)")
    # 건너뛴 이미지의 라벨은 쓸 수 없으므로 현황을 다시 계산한다
    labels = labels[labels["image_id"].isin(set(df["id"]))]
    status = label_status(tags, labels)
    targets = [t for t in tags if status.set_index("tag_key").at[t.key, "trainable"]]
    for _, s in status[~status["trainable"]].iterrows():
        report_progress("준비", f"{s['name']} 건너뜀 — {s['reason']}")
    if not targets:
        report_progress("준비", "학습 가능한 태그가 없습니다 (라벨 부족)")
        return TrainResult(status=status, seconds=time.perf_counter() - t0, log=log, skipped=skipped, n_images=len(df))
    row = {i: n for n, i in enumerate(df["id"])}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    trained_at = datetime.now().isoformat(timespec="seconds")
    reports, models = [], {}
    for n, t in enumerate(targets, start=1):
        report_progress("태그별 학습", f"{t.name} ({n}/{len(targets)})", (n - 1) / len(targets))
        try:
            data = labels[labels["tag_key"] == t.key]
            clf, metrics = _train_one(X[[row[i] for i in data["image_id"]]], data["value"].to_numpy())
            joblib.dump({"model": clf, "tag_key": t.key, "embed_id": EMBED_ID, "trained_at": trained_at},
                        MODELS_DIR / f"{t.key}.joblib")
        except Exception as e:
            raise TrainingError("태그별 학습", f"태그 {t.name}", e) from e
        models[t.key] = clf
        s = status.set_index("tag_key").loc[t.key]
        reports.append({"tag_key": t.key, "name": t.name, "group": t.group, "n_pos": s["n_pos"], "n_neg": s["n_neg"],
                        **metrics, "trained_at": trained_at, "embed_model": EMBED_ID})
    report = pd.DataFrame(reports)
    report.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")
    report_progress("평가", "지표 저장 완료 (train_report.csv)", 1.0)

    preds = []
    for n, (key, clf) in enumerate(models.items(), start=1):
        try:
            prob = clf.predict_proba(X)[:, list(clf.classes_).index(POSITIVE)]
        except Exception as e:
            raise TrainingError("전체 예측", f"태그 {key}", e) from e
        preds.append(pd.DataFrame({"image_id": df["id"], "tag_key": key, "prob": prob.round(4)}))
        report_progress("전체 예측", f"{n}/{len(models)} 태그", n / len(models))
    all_preds = pd.concat(preds)
    all_preds.to_csv(PRED_CSV, index=False, encoding="utf-8-sig")

    shared_rows = None
    if publish and run_id:
        report_progress("팀 공유", f"예측 {len(all_preds):,}행을 DB에 올리는 중 (다 올라간 뒤에만 팀 화면에 보입니다)")
        try:
            shared_rows = upload_predictions(
                run_id, all_preds, EMBED_ID,
                progress=lambda done, total: report_progress("팀 공유", f"{done:,}/{total:,}행", done / total if total else 1.0))
            finish_training_run(run_id, len(df), _report_rows(report))
            run_id = None  # 끝났으므로 진행 상황을 더 남기지 않는다
            report_progress("팀 공유", f"{shared_rows:,}행 공유 완료 — 팀원 화면에 1분 안에 반영됩니다", 1.0)
        except Exception as e:  # 공유 실패는 학습 실패가 아니다 (로컬 결과는 이미 저장됨, 팀 화면은 직전 학습 그대로)
            run_id = None
            share_error = str(e)
            report_progress("팀 공유", f"실패 (이 PC의 결과 파일은 저장됨, 팀 화면은 직전 학습 결과 그대로): {e}")

    seconds = time.perf_counter() - t0
    report_progress("완료", f"{len(models)}개 태그 학습 · 이미지 {len(df):,}장 · 새 임베딩 {n_new:,}장 · {_duration(seconds)}")
    return TrainResult(status=status, report=report, n_new_embeddings=n_new, seconds=seconds, log=log,
                       shared_rows=shared_rows, share_error=share_error, skipped=skipped, n_images=len(df))


def _report_rows(report: pd.DataFrame) -> list[dict]:
    """학습 기록(DB jsonb)에 넣을 태그별 지표. numpy 숫자는 JSON으로 보낼 수 없어 파이썬 숫자로 바꾼다."""
    cols = ["tag_key", "name", "group", "n_pos", "n_neg", "n_train", "n_test", "precision", "recall", "f1"]
    out = []
    for r in report[cols].to_dict("records"):
        out.append({k: (round(float(v), 4) if k in ("precision", "recall", "f1")
                        else int(v) if k.startswith("n_") else v) for k, v in r.items()})
    return out


def _peak_memory_mb() -> float | None:
    """이 프로세스의 최대 메모리 사용량(MB). 알 수 없으면 None."""
    try:
        import psutil
        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None)  # Windows
        if peak:
            return peak / 2**20
        import resource  # macOS·Linux
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / 2**20 if sys.platform == "darwin" else kb / 2**10
    except Exception:
        return None


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="태그별 분류기 학습 (로컬 PC 전용). 중간에 멈춰도 같은 명령으로 이어서 합니다.")
    ap.add_argument("--email", default="", help="팀 명단에 등록된 내 이메일. 학습 기록에 명단의 이름이 남는다")
    ap.add_argument("--no-share", action="store_true", help="예측을 DB에 올리지 않고 이 PC에만 저장 (이메일 불필요)")
    args = ap.parse_args()

    trainer = ""
    if not args.no_share:
        if not args.email.strip():
            ap.error("팀에 공유하려면 --email이 필요합니다. 예: python training.py --email you@example.com "
                     "(공유하지 않으려면 --no-share)")
        from core.db import DBError
        from core.members import member_name
        try:
            trainer = member_name(args.email) or ""
        except DBError as e:
            print(f"팀원 확인에 실패했습니다: {e}", file=sys.stderr)
            return 1
        if not trainer:
            print("팀 명단에 없는 이메일입니다. 웹앱 입장에 쓰는 이메일을 넣어 주세요.", file=sys.stderr)
            return 1
        print(f"학습한 사람: {trainer}", flush=True)
    try:
        result = train_all(publish=not args.no_share, trainer=trainer)
    except TrainingError as e:
        print(f"\n실패: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n멈췄습니다. 그때까지 추출한 임베딩은 저장했습니다. 같은 명령으로 다시 실행하면 이어서 합니다.", file=sys.stderr)
        return 130
    if result.skipped:
        print(f"\n열 수 없어 건너뛴 이미지 {len(result.skipped)}장 (처음 10장):")
        for image_id, reason in result.skipped[:10]:
            print(f"  {image_id}: {reason}")
    if (peak := _peak_memory_mb()) is not None:
        print(f"\n최대 메모리 사용량: 약 {peak:,.0f}MB")
    print("\n라벨 현황")
    print(result.status[["name", "n_pos", "n_neg", "reason"]].to_string(index=False))
    if result.trained:
        print("\n평가 결과 (평가용 20%)")
        r = result.report[["name", "n_test", "precision", "recall", "f1"]].copy()
        r["주의"] = np.where(r["f1"] < F1_WARN, f"F1 < {F1_WARN}", "")
        print(r.to_string(index=False, float_format="%.2f"))
        if result.share_error:
            print(f"\n팀 공유 실패 (이 PC의 결과 파일은 저장됨): {result.share_error}", file=sys.stderr)
            return 3
        return 0
    return 2  # 학습 가능한 태그가 없어 학습하지 않음


if __name__ == "__main__":
    utf8_console()
    sys.exit(main())
