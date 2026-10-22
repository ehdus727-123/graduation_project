# CLAUDE.md

드로잉 특화 레퍼런스 서비스(졸업 프로젝트)의 **기술 검증용 내부 웹앱**. 팀원 4~5명(비개발자 포함)이 각자 PC에서 로컬로 실행한다.
Windows, Python 3.14, GPU 없음(모든 기능이 CPU에서 동작해야 함).

## 실행

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt          # 팀원용
pip install -r requirements-dev.txt      # 개발자용 (테스트 도구 추가)
python -m playwright install chromium    # 개발자용, 브라우저 E2E 테스트
streamlit run app.py
```

- 비개발자용 실행: `SPECTRUM 실행.bat` 더블클릭(= `python -m streamlit run app.py`). bat은 **UTF-8 + CRLF**(`.gitattributes`의 `*.bat eol=crlf`)이고 첫 줄 근처에서 `chcp 65001`
- 태그 학습은 **팀원 누구나 자기 PC에서 켠 웹앱**의 '학습 현황 → 학습 시작'으로 한다(배포 서버 `ON_CLOUD`에서는 버튼 없음). 학습은 `core/train_job.py`가 화면과 분리된 스레드로 돌린다(서버 프로세스에 하나, `st.cache_resource`) → 다른 페이지로 가거나 새로고침해도 계속, 한 번에 하나만. 학습 기록(`training_runs`)은 시작할 때 만들고 `touch_training_run`으로 20초마다 진행 상황을 남겨, 다른 팀원 화면에 '○○ 님이 학습 중'이 보인다(`active_training_runs`, 2분 안에 소식이 있는 미완료 기록)
  - 개발자용 CLI: `python training.py --email 팀원이메일`(DB 공유) / `--no-share`(이 PC에만). 종료 코드 0 완료·1 실패·2 학습할 태그 없음·3 팀 공유 실패·130 Ctrl+C
  - 뒤에서 도는 학습 스레드에는 화면 세션이 없으므로 방문자 차단(`auth.write_guard`)은 세션이 없는 스레드를 통과시킨다. 그런 스레드는 팀원 확인 뒤에만 만든다
- 단독 스크립트: `python migrate_to_supabase.py`(로컬 데이터 → DB, secret 키 필요. 팀원은 `--evals-only`로 예전 `eval_*.csv` 평가만 올림), `python check_supabase.py`(DB 연결 테스트)
- 포즈: `python predict_poses.py`(DWPose 미리 계산, 아직 없는 이미지만, 500장 약 25분), `python predict_poses.py --import x.json`(별도 환경 결과 가져오기), `python export_coco.py`(사람이 확정한 포즈 → COCO)
- MediaPipe는 팀 환경에 넣지 않는다(numpy·OpenCV 충돌). 비교용은 `tools/mediapipe_predict.py` 상단 안내대로 별도 가상환경에서 실행
- 테스트: `python -m pytest tests`
  - `tests/test_pose.py`: 매칭 단위 테스트 (DB·브라우저 불필요)
  - `tests/test_team.py`: 팀 현황 집계·비슷한 이름 감지 단위 테스트
  - `tests/test_auth.py`: 팀원 이메일 확인·방문자 쓰기 차단 (가짜 DB, 실제 DB 불필요)
  - `tests/test_train_job.py`: 뒤에서 도는 학습 작업 — 한 번에 하나, 진행·성공·실패 기록, 방문자 차단이 학습 스레드를 막지 않음
  - `tests/test_training_scale.py`: 대량 학습 대비 — 임베딩 이어서 하기·캐시 복구·깨진 이미지 건너뛰기, 예측 나눠 올리기·실패 시 완료 안 함 (가짜 DB·임베딩)
  - `tests/test_search_evals.py`: 유사 검색 평가 저장·읽기·예전 CSV 변환 (가짜 DB 클라이언트, 실제 DB 불필요)
  - `st.page_link`는 `st.navigation` 안에서만 동작한다. 페이지를 `AppTest.from_file`로 단독 실행해 볼 때는 `st.page_link`를 대체해 둔다
  - `tests/e2e/test_pin_editor.py`: 핀 편집기를 Chromium에서 직접 검증 (Streamlit·DB 불필요)
  - `tests/e2e/test_pose_app.py`, `test_pose_step2_app.py`, `test_pose_3d_app.py`: 실제 서버 + 브라우저 + Supabase 전체 흐름. 테이블이 없으면 자동으로 건너뜀
  - `tests/e2e/test_mannequin.py`: 마네킹을 Streamlit 없이 검증. 외부 네트워크 요청이 하나라도 있으면 실패(오프라인 보장)
  - Streamlit 화면 E2E에서 결과를 읽을 때는 고정 대기 대신 "바뀐 상태"가 화면에 나타나거나 사라질 때까지 기다린다 (재실행 전 DOM을 읽는 실수 방지)
  - 기존 페이지 확인은 `AppTest`로 **화면만 그려 볼 것**. 평가 버튼을 누르는 스모크 테스트는 실제 DB(`search_evals`)에 기록하므로 돌리지 않는다(누르려면 `core.evaluation`의 저장 함수를 가짜로 바꾼다)

## 구조

| 경로 | 역할 |
| --- | --- |
| `app.py` | `st.navigation` 진입점. 입장 전에는 입장 화면(`views/auth.py`)만 보여준다. 매 실행마다 `db.set_write_guard(auth.write_guard)`. `.env`와 로컬 데이터가 모두 없으면 설치 안내만 보여준다 |
| `views/auth.py` | 입장 화면(SPECTRUM, 팀원 이메일 / "방문자이신가요? 입장하기"), 사이드바 로그인 표시·로그아웃. 팀원이면 `st.session_state["evaluator"]` = 명단의 이름(고정), 방문자면 빈 값 |
| `core/members.py` | 팀원 확인: DB 함수 `member_name(이메일)` 호출. 앱은 명단 전체를 읽지 못한다 |
| `core/team.py` | 팀 활동 집계(팀원별 작업량, 비슷한 이름 감지, 마지막 학습 이후 라벨 수) — 순수 pandas. `views/team.py`(팀 현황)가 쓴다 |
| `views/` | 페이지 스크립트. 공통 UI는 `views/common.py` |
| `methods/` | 검색 방식 모듈. `SearchMethod` + 기능 믹스인(`ImageSearch`/`PairFinder`/`TagSource`), `METHOD_CLASSES`에 등록하면 해당 페이지에 자동 노출 |
| `core/dataset.py` | 이미지 목록. `read_metadata`는 **항상 DB(`images`) 기준**(id로 정렬)이고 이 PC에 `images/{id}.jpg`가 있으면 그 파일, 없으면 URL을 `img_path`로 쓴다. DB 접속 정보가 없을 때만 로컬 `metadata.csv`. `read_local_metadata`는 이 PC의 `metadata.csv`(DB 등록용). `load_metadata`는 5분 캐시. 썸네일, `local_image`, 배포 서버 여부 `ON_CLOUD` |
| `core/evaluation.py` | 이미지 유사 검색 평가(비슷함/다름) DB 입출력(`search_evals`), 예전 `eval_*.csv` 변환 |
| `core/db.py` | Supabase 클라이언트, `fetch_all`/`upsert`, `friendly_error`(한국어 오류) |
| `core/tagging.py` | 팀 태그(`tags`)·라벨(`labels`) DB 입출력, 예측 CSV, 팀 공유 예측(학습 묶음: `start_training_run` → `upload_predictions`(5,000행씩, 재시도) → `finish_training_run`). `save_labels()`는 이미지 한 장의 여러 태그를 한 번의 요청으로 저장(전부 저장 또는 전부 실패) |
| `methods/predicted_tags.py` | 예측 읽기 공통 입구 `load_predictions()`: DB 공유 예측(가장 최근에 **완료된** 학습. 1분마다 새 학습 여부만 확인하고 바뀌면 전체를 다시 읽음) → 없으면 로컬 `predicted_tags.csv` |
| `core/train_job.py` | 로컬 웹앱 '학습 시작'을 뒤에서 돌리는 작업 관리자(스레드, 한 번에 하나, 진행 상태·결과 보관). Streamlit 의존 없음 |
| `training.py` | 학습 본체(웹앱 버튼·CLI 공용). 임베딩 캐시는 30초마다·멈출 때·끝날 때 저장(임시 파일 후 교체, 읽을 때 두 파일의 앞쪽 공통 부분만 사용), 열 수 없는 이미지 건너뜀, 진행률에 속도·남은 시간 |
| `core/image_labeling.py` | 이미지별 라벨링 규칙: 체크=1·해제=0·보류=저장 안 함, 끝낸 이미지 = 사용 중 태그 전부에 내 라벨이 있음 (순수 함수) |
| `views/labeling.py` | 라벨링 방식 전환. `labeling_by_image.py`(기본, 체크박스 한 번에) / `labeling_by_tag.py`(기존 화면을 함수로 감싼 것 — 동작 변경 금지) |
| `core/pose.py` | 포즈 관절 정의·정규화·매칭 (순수 numpy, Streamlit/DB 의존 없음) |
| `core/poses.py` | `poses`·`pose_evals` DB 입출력 |
| `core/pose_models.py` | 사전학습 포즈 모델 → 13관절 변환(COCO·MediaPipe 매핑, 신뢰도 → 상태, 방향 추정) |
| `core/pose_predict.py` | 모델 추정 실행·저장 (웹 '바로 추정'·'미리 계산'과 CLI 공용) |
| `core/pose_accuracy.py` | 모델 vs 사람 정확도(관절별 오차, PCK@0.2, 검출률) — 순수 pandas |
| `web/` | **Streamlit에 의존하지 않는** 프론트엔드 모듈(순수 ES 모듈). MVP 화면으로 옮길 수 있게 유지 |
| `components/` | `web/` 모듈을 Streamlit에 붙이는 얇은 어댑터(`st.components.v2`) |
| `web/mannequin/` | 3D 마네킹(three.js를 생성자로 받음 → 번들러·로컬·CDN 어디서나). `demo.html`은 Streamlit 없이 쓰는 예시 |
| `static/vendor/three/` | three.js 0.186.1 원본(수정 금지). `.streamlit/config.toml`의 `enableStaticServing`으로 `/app/static/`에서 제공 → 오프라인 동작. **설정은 서버를 켤 때만 읽히므로, 바꾸면 앱을 다시 시작해야 한다** |
| `supabase/schema.sql` | DB 스키마 전체. 여러 번 실행해도 안전해야 함(`if not exists`, 정책 존재 확인) |
| `third_party/` | 프로젝트에 포함한 외부 코드와 라이선스 원문 |
| `drawing_ref_test/` | 로컬 데이터(이미지, 캐시, 모델). **Git 제외** |

## 규칙

- 서비스 Primary color는 `#7022EF`(SPECTRUM). `.streamlit/config.toml`의 `[theme.light]`·`[theme.dark]`에 둔다(`[theme]`에 바로 쓰면 다크 모드를 따르지 않음). 직접 색을 넣는 강조·CTA도 이 색을 쓰고, 오류·경고·나쁨을 뜻하는 빨강(상태 색)은 그대로 둔다. 설정 파일은 서버를 켤 때만 읽힌다.
- **화면 문구는 모두 한국어.** 오류도 팀원이 이해할 수 있는 한국어로(`db.friendly_error` 참고).
- CSV는 `utf-8-sig`로 쓰고 읽는다(엑셀 한글 깨짐 방지).
- 페이지 간 공유 상태는 `st.session_state`, 위젯 키 이름은 페이지 접두사를 붙인다(`label_`, `pose_` 등).
- `st.cache_resource`/`st.cache_data` 함수의 인자 이름을 `_`로 시작하면 캐시 키에서 빠진다. 모듈 리로드 대비는 `methods/__init__.py`의 클래스 id 키 참고. 행 번호에 묶인 캐시(검색 인덱스 등)는 `metadata_key(df)`를 키에 넣어 이미지 목록이 바뀌면 다시 만든다.
- 무거운 계산 결과(Phash, 임베딩)는 `drawing_ref_test/cache/` 등에 파일로 캐시한다.
- `st.components.v2` 컴포넌트 키에는 `__`를 쓸 수 없다. 작성자 이름처럼 임의 문자가 들어가는 키는 해시로 바꾼다(`components/pose_editor.py`).
- 위젯 키에 대상 ID와 '저장 상태 요약값'을 넣어, 대상이 바뀌거나 저장 직후 다시 열 때 위젯이 새로 초기화되게 한다(라벨링 체크박스, 포즈 편집기 공통).
- 체크할 때마다 즉시 바뀌어야 하는 안내가 있는 입력 영역은 `st.form` 대신 `@st.fragment`로 감싼다(그 영역만 다시 그림, DB 재조회 없음).
- 화면에 그리지 않은 위젯의 값은 Streamlit이 지운다. 화면 전환 후에도 유지해야 하는 값은 `views/labeling.py`처럼 매 실행 시 다시 넣어 둔다.
- v2 컴포넌트는 같은 키로 다시 그려질 때 JS 함수가 다시 호출된다. 편집 중 상태는 `data.version`이 같으면 유지하고, 바꾸려면 version을 바꾼다.
- Streamlit Cloud는 push한 파일을 받아도 서버를 재시작하지 않을 수 있다(예전 모듈이 메모리에 남아 ImportError). `app.py`의 `_reload_changed_code()`가 코드 파일 수정 시각이 바뀌면 프로젝트 모듈(`PROJECT_MODULES`)을 버리고 다시 불러온다. 새 최상위 패키지를 만들면 `PROJECT_MODULES`에 추가한다.
- 서버(로컬 `metadata.csv` 없음)에서는 `img_path`가 이미지 URL이다. 파일로 열어야 하는 곳(PIL·OpenCV·CLIP)은 반드시 `core.dataset.local_image(img_path)`를 거친다(내려받아 `cache/remote/`에 캐시). `st.image`는 URL을 그대로 받으므로 필요 없다.
- 배포 서버(`core.dataset.ON_CLOUD`, 코드가 `/mount/src/`에서 실행됨)는 메모리가 작다. 학습·AI 포즈 추정처럼 무거운 작업은 서버에서 버튼을 막고 `CLOUD_HEAVY_NOTICE`로 PC 실행을 안내한다. 새로 무거운 기능을 만들면 같은 방식으로 막는다. 시험할 때는 환경 변수 `DRAWING_REF_CLOUD=1`/`0`.
- 방문자는 보기 전용이다. 화면은 저장 버튼을 `current_user()`가 비었을 때 꺼서 막고(`views.common.cannot_save_notice`로 안내), DB 쪽은 `db.upsert`가 `db.check_write()`로 한 번 더 막는다. `db.upsert`를 거치지 않고 직접 쓰는 코드(`client.table().insert/update/delete`)는 반드시 먼저 `db.check_write()`를 부른다. 이름과 무관한 쓰기 버튼(태그 관리, AI 미리 계산 등)은 `is_visitor()`로 끈다.
- 서버의 파일(`drawing_ref_test/` 아래)은 재시작 때 지워진다. 팀이 만든 데이터는 파일이 아니라 DB에 저장한다(캐시만 파일로).
- 좌표는 이미지 크기에 대한 0~1 비율로 저장하고, 각도 계산 전에는 반드시 가로세로 비율을 보정한다.
- 포즈의 좌우는 **캐릭터 기준**(캐릭터의 왼팔 = `l_`). 화면에 안내 문구를 항상 표시한다.

## 데이터 안전 (중요)

- `SUPABASE.env`/`.env`의 키 값은 **절대 출력하지 않는다.** 키 종류 확인이 필요하면 접두사(`sb_secret_`/`sb_publishable_`)만 본다.
- 태그를 새로 만들어야 하는 테스트는 key `zz_test_*` + 이름 `[테스트] ...`로 만들고, fixture에서 **시작 전·종료 시(실패해도)** 정리한다.
  임시 태그의 라벨을 먼저 지워야 태그가 지워진다(`labels.tag_key`는 외래키 restrict). 예: `tests/e2e/test_labeling_image_app.py`
- `training.train_all()`은 기본으로 예측을 DB에 올리지 않는다(`publish=False`). CLI만 `publish=True`. 테스트에서 실제 DB로 켜면 팀 공유 예측이 바뀐다.
  예측은 학습 묶음(`training_runs.run_id`) 단위로 올리고, **다 올라간 뒤에만** `finish_training_run()`으로 완료 표시한다(완료할 때 다른 학습 중 이미 완료된 것과 30분 넘게 멈춘 것의 예측은 DB 함수가 지움 — 같은 시간에 올리는 중인 학습은 남김, 학습 기록·지표는 남음).
  실제 DB로 올리기를 시험할 때는 trainer `__test__`로 묶음을 만들되 **완료하지 않고**, 끝나면 그 `training_runs` 행을 지운다(예측은 cascade로 함께 지워짐).
- 학습을 실행하는 테스트는 `training`의 결과 경로(MODELS_DIR, REPORT_CSV, PRED_CSV, 임베딩 캐시)를 임시 폴더로 바꿔서 이 PC의 결과 파일을 덮어쓰지 않는다.
- `poses`의 `source='model'` 행(annotator = 모델 이름: `dwpose`, `mediapipe-heavy`)은 모델 결과다. 사람 데이터와 섞어 지우지 말 것. 사람 라벨은 `manual`/`model_corrected`.
- 테스트는 실제 팀 데이터를 건드리지 않는다. DB에 쓰는 테스트는 작성자/라벨러 이름 `__test__`를 쓰고 끝나면 그 행만 지운다.
- 브라우저 E2E는 `members`에 임시 팀원(`__test__@test.invalid` → `__test__`)을 넣어 입장 화면을 통과하고 끝나면 그 행만 지운다(`tests/e2e/test_pose_app.py`의 `login`). `page.goto`로 주소를 새로 열면 세션이 새로 생기므로 다시 `login`한다.
- `members`의 이메일은 팀원 개인정보다. 출력·로그·커밋에 넣지 않는다(저장소가 공개).
- DB의 `labels`·`poses`·`pose_evals`·`search_evals`(예전 `eval_*.csv`)는 팀이 직접 만든 데이터다. 덮어쓰기·삭제 전에 반드시 확인한다.
- secret 키(`service_role`)는 RLS를 무시한다. 팀원에게는 publishable 키를 나눠준다.

## DB (Supabase)

- 테이블: `images`, `tags`, `labels`, `predictions`, `poses`, `pose_evals`, `search_evals`, `training_runs`(학습 기록·지표), `members`(팀원 명단 — 팀원 키로는 읽기 불가, 함수 `member_name`으로만 확인)
- 스키마 변경 시 `supabase/schema.sql`에 추가하고, 사용자가 SQL Editor에서 실행해야 한다(API 키로는 DDL 불가).
- 새 테이블에는 GRANT(이 프로젝트는 자동 부여 안 됨)와 RLS 정책을 함께 추가한다.

## 라이선스

- 외부 코드를 포함하면 `third_party/<이름>/LICENSE`에 원문을 넣고 `THIRD_PARTY_NOTICES.md`에 기록한다.
- x6ud/pose-search(MIT)는 **아이디어만 참고**했다(코드 미포함). 3D 마네킹은 three.js로 직접 구현한다.
- 수집 이미지(Safebooru)는 팀 내부 검증용이며 저장소·외부에 재배포하지 않는다.

## 작업 방식

- 큰 기능은 명세서(`SPEC_*.md`) 기준으로 단계별 구현, 단계마다 커밋하고 멈춰서 보고한다.
- 기존 페이지 동작을 바꾸지 않는다. 바꿔야 하면 보고에 명시한다.
