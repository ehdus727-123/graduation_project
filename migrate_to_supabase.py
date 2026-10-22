"""로컬 데이터(metadata.csv, tags.yaml, labels.csv, eval_*.csv)를 Supabase로 옮긴다.

실행: python migrate_to_supabase.py               (전체, 관리자 secret 키 필요: 이미지 표 수정 권한)
      python migrate_to_supabase.py --evals-only  (유사 검색 평가만, 팀원 키로도 가능)
여러 번 실행해도 안전하다.
  - 이미지: 같은 id는 덮어씀 (이미지 파일 자체는 올리지 않고 메타데이터만)
  - 태그: DB에 없는 key만 추가 (웹 '태그 관리'에서 고친 내용은 건드리지 않음)
  - 라벨: 같은 (이미지, 태그, 라벨러)는 덮어씀
  - 유사 검색 평가(eval_*.csv): DB에 없는 평가만 추가 (웹에서 새로 누른 평가는 덮어쓰지 않음)
"""
import sys

import pandas as pd
import yaml

from core.cli import quiet_streamlit, utf8_console

quiet_streamlit()

from core import db  # noqa: E402
from core.dataset import CACHE_DIR, DATA_DIR, read_local_metadata  # noqa: E402
from core.evaluation import TABLE as EVAL_TABLE, csv_to_rows, load_evals_csv  # noqa: E402
from core.tagging import LABELS_CSV, TAGS_YAML, invalidate_tags_cache, load_labels_csv  # noqa: E402


def migrate_images() -> int:
    df = read_local_metadata()  # 이 PC의 metadata.csv (DB 목록이 아니라 올릴 원본)
    phash_csv = CACHE_DIR / "phash.csv"
    phash = dict(pd.read_csv(phash_csv, dtype=str).values) if phash_csv.exists() else {}

    def num(v):
        return None if pd.isna(v) else int(v)

    rows = [{"id": r.id, "url": r.url, "source_page": r.source_page, "tags": r.tag_list,
             "width": num(r.width), "height": num(r.height), "phash": phash.get(r.id)}
            for r in df.itertuples()]
    db.upsert("images", rows, on_conflict="id")
    return len(rows)


def migrate_tags_yaml(path=TAGS_YAML) -> int:
    """tags.yaml의 태그 중 DB에 없는 것만 추가. 추가 시도한 수를 돌려준다."""
    if not path.exists():
        return 0
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    rows = [{"key": str(t["key"]), "name": str(t["name"]), "tag_group": str(t.get("group") or ""),
             "definition": str(t.get("definition") or ""), "booru_hint": t.get("booru_hint") or None,
             "sort_order": n} for n, t in enumerate(raw) if isinstance(t, dict) and t.get("key") and t.get("name")]
    db.upsert("tags", rows, on_conflict="key", ignore_duplicates=True)
    invalidate_tags_cache()
    return len(rows)


def migrate_labels(path=LABELS_CSV) -> tuple[int, int]:
    """(올린 수, 건너뛴 수). DB에 없는 이미지·태그의 라벨은 건너뛴다."""
    labels = load_labels_csv(path)
    if labels.empty:
        return 0, 0
    image_ids = {r["id"] for r in db.fetch_all("images", "id")}
    tag_keys = {r["key"] for r in db.fetch_all("tags", "key")}
    labels = labels.drop_duplicates(subset=["image_id", "tag_key", "labeler"], keep="last")
    ok = labels["image_id"].isin(image_ids) & labels["tag_key"].isin(tag_keys) & labels["labeler"].notna()
    rows = [{"image_id": r.image_id, "tag_key": r.tag_key, "value": int(r.value), "labeler": r.labeler,
             **({"updated_at": r.timestamp} if isinstance(r.timestamp, str) and r.timestamp else {})}
            for r in labels[ok].itertuples()]
    db.upsert("labels", rows, on_conflict="image_id,tag_key,labeler")
    return len(rows), int((~ok).sum())


def migrate_evals() -> list[tuple[str, int, int]]:
    """eval_{방식}.csv마다 (방식, 올린 수, 건너뛴 수). 이미 DB에 있는 평가는 그대로 둔다."""
    files = sorted(DATA_DIR.glob("eval_*.csv"))
    if not files:
        return []
    image_ids = {r["id"] for r in db.fetch_all("images", "id")}
    out = []
    for path in files:
        method_key = path.stem.removeprefix("eval_")
        rows, skipped = csv_to_rows(load_evals_csv(path), method_key, image_ids)
        if rows:
            db.upsert(EVAL_TABLE, rows, on_conflict="method,evaluator,query_id,result_id", ignore_duplicates=True)
        out.append((method_key, len(rows), skipped))
    return out


def print_evals(results: list[tuple[str, int, int]]) -> None:
    if not results:
        print("    eval_*.csv 없음 — 건너뜀")
    for key, n, skipped in results:
        print(f"    {key}: {n}건 확인 (DB에 이미 있던 평가는 그대로 둠)"
              + (f", {skipped}건 건너뜀 (DB에 없는 이미지·이름 없음)" if skipped else ""))


def main() -> int:
    if "--evals-only" in sys.argv[1:]:
        try:
            print("유사 검색 평가 (eval_*.csv) ...", flush=True)
            print_evals(migrate_evals())
        except Exception as e:
            print(f"\n실패: {db.friendly_error(e)}", file=sys.stderr)
            return 1
        print("\n완료. 이제 유사 검색 평가는 Supabase가 기준입니다 (eval_*.csv는 더 이상 읽지 않음).")
        return 0
    try:
        print("1/4 이미지 메타데이터 (metadata.csv) ...", flush=True)
        print(f"    {migrate_images()}건 업로드")
        print("2/4 팀 태그 (tags.yaml) ...", flush=True)
        if TAGS_YAML.exists():
            migrate_tags_yaml()
            print(f"    DB 태그 {len(db.fetch_all('tags', 'key'))}개 (이미 있던 key는 그대로 둠)")
        else:
            print("    tags.yaml 없음 — 건너뜀 (태그는 웹앱 '태그 관리'에서 추가)")
        print("3/4 라벨 (labels.csv) ...", flush=True)
        if LABELS_CSV.exists():
            n, skipped = migrate_labels()
            print(f"    {n}건 업로드" + (f", {skipped}건 건너뜀 (DB에 없는 이미지/태그)" if skipped else ""))
        else:
            print("    labels.csv 없음 — 건너뜀")
        print("4/4 유사 검색 평가 (eval_*.csv) ...", flush=True)
        print_evals(migrate_evals())
    except Exception as e:
        print(f"\n실패: {db.friendly_error(e)}", file=sys.stderr)
        return 1
    print("\n완료. 이제 태그·라벨·유사 검색 평가는 Supabase가 기준입니다 "
          "(tags.yaml, labels.csv, eval_*.csv는 더 이상 읽지 않음).")
    return 0


if __name__ == "__main__":
    utf8_console()
    sys.exit(main())
