-- 드로잉 레퍼런스 검증 도구 DB 스키마
-- Supabase 대시보드 > SQL Editor > New query 에 전체를 붙여넣고 Run.
-- 여러 번 실행해도 안전하다 (이미 있는 테이블·정책은 건너뜀).

-- ---------- 이미지 (metadata.csv) ----------
create table if not exists public.images (
  id          text primary key,                 -- Safebooru post id
  url         text,
  source_page text,
  tags        text[] not null default '{}',     -- Safebooru 원본 태그
  width       int,
  height      int,
  phash       text,
  created_at  timestamptz not null default now()
);
create index if not exists images_tags_gin on public.images using gin (tags);

-- ---------- 팀 태그 정의 (웹앱 '태그 관리' 페이지에서 편집) ----------
create table if not exists public.tags (
  key         text primary key check (key ~ '^[a-z][a-z0-9_]*$'),  -- 영문 식별자. 만든 뒤에는 바꾸지 않는다
  name        text not null check (length(trim(name)) > 0),        -- 화면 표시명
  tag_group   text not null default '',         -- 분류 축 (앵글, 포즈, 구도 ...)
  definition  text not null default '',         -- 라벨링 기준
  booru_hint  text,                             -- 대응되는 Safebooru 태그 (선택)
  active      boolean not null default true,    -- false = 사용 안 함 (라벨이 있어 삭제할 수 없는 태그)
  sort_order  int  not null default 0,
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

-- ---------- 라벨 ----------
create table if not exists public.labels (
  id          bigint generated always as identity primary key,
  image_id    text not null references public.images(id) on delete cascade,
  tag_key     text not null references public.tags(key) on update cascade on delete restrict,  -- 라벨 있는 태그는 삭제 불가
  value       smallint not null check (value in (0, 1)),   -- 1=해당, 0=해당 아님
  labeler     text not null check (length(trim(labeler)) > 0),
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now(),
  unique (image_id, tag_key, labeler)           -- 같은 사람이 다시 저장하면 덮어쓰기
);
create index if not exists labels_tag_key_idx on public.labels (tag_key);

-- ---------- 예측 결과 ----------
create table if not exists public.predictions (
  image_id     text not null references public.images(id) on delete cascade,
  tag_key      text not null references public.tags(key) on update cascade on delete cascade,
  prob         real not null check (prob between 0 and 1),
  model        text not null,
  predicted_at timestamptz not null default now(),
  primary key (image_id, tag_key)
);

-- ---------- 테이블 사용 권한 (GRANT) ----------
-- 이 프로젝트는 SQL로 만든 테이블에 API 역할 권한을 자동으로 주지 않으므로 직접 준다.
-- (없으면 secret 키로도 "permission denied for table ..." 오류가 난다)
--   service_role  = secret 키 (관리용: 이전 스크립트, 정리 작업)
--   anon          = publishable 키 (팀원용)
grant usage on schema public to anon, authenticated, service_role;
grant select, insert, update, delete on public.images, public.tags, public.labels, public.predictions to service_role;
grant select                         on public.images                    to anon, authenticated;
grant select, insert, update, delete on public.tags                      to anon, authenticated;  -- 삭제는 아래 RLS가 '라벨 없는 태그'로 제한
grant select, insert, update         on public.labels, public.predictions to anon, authenticated;
grant usage, select on all sequences in schema public to anon, authenticated, service_role;

-- ---------- 행 단위 접근 규칙 (RLS) ----------
-- secret 키는 RLS를 무시하므로 아래 정책은 publishable(anon) 키로 접속하는 팀원에게 적용된다.
-- 팀원은 읽기 + 라벨/태그/예측 추가·수정만 가능하고, 이미지 수정과 삭제는 막는다.
-- 예외: 라벨이 하나도 없는 태그는 삭제할 수 있다 (잘못 만든 태그 정리용).
alter table public.images      enable row level security;
alter table public.tags        enable row level security;
alter table public.labels      enable row level security;
alter table public.predictions enable row level security;

do $$
declare
  t text;
begin
  -- 모든 테이블: 읽기 허용
  foreach t in array array['images', 'tags', 'labels', 'predictions'] loop
    if not exists (select 1 from pg_policies where schemaname = 'public' and tablename = t and policyname = t || '_select') then
      execute format('create policy %I on public.%I for select to anon, authenticated using (true)', t || '_select', t);
    end if;
  end loop;
  -- 태그·라벨·예측: 추가·수정 허용
  foreach t in array array['tags', 'labels', 'predictions'] loop
    if not exists (select 1 from pg_policies where schemaname = 'public' and tablename = t and policyname = t || '_insert') then
      execute format('create policy %I on public.%I for insert to anon, authenticated with check (true)', t || '_insert', t);
    end if;
    if not exists (select 1 from pg_policies where schemaname = 'public' and tablename = t and policyname = t || '_update') then
      execute format('create policy %I on public.%I for update to anon, authenticated using (true) with check (true)', t || '_update', t);
    end if;
  end loop;
  if not exists (select 1 from pg_policies where schemaname = 'public' and tablename = 'tags' and policyname = 'tags_delete_unused') then
    create policy tags_delete_unused on public.tags for delete to anon, authenticated
      using (not exists (select 1 from public.labels l where l.tag_key = tags.key));
  end if;
end $$;

-- =====================================================================
-- 포즈 핀 라벨링 + 포즈 검색 (SPEC_pose_annotation_search 단계 1)
-- 이 블록만 따로 실행해도 된다. 여러 번 실행해도 안전하다.
-- =====================================================================

-- ---------- 포즈 (일러스트 위 13개 관절 핀) ----------
create table if not exists public.poses (
  image_id     text not null references public.images(id) on delete cascade,
  person_index smallint not null default 0 check (person_index >= 0),       -- 한 이미지 내 인물 번호
  annotator    text not null check (length(trim(annotator)) > 0),           -- 작성자 이름 (모델 결과면 모델 이름)
  source       text not null default 'manual' check (source in ('manual', 'model', 'model_corrected')),
  facing       text check (facing in ('front', 'side', 'back')),             -- 몸이 향한 방향 (건너뛴 경우 비어 있음)
  keypoints    jsonb not null check (jsonb_typeof(keypoints) = 'array' and jsonb_array_length(keypoints) = 13),
                                                                             -- 13개 {x, y, state}, 좌표는 이미지 크기 대비 0~1
  status       text not null default 'draft' check (status in ('draft', 'done', 'skipped')),
  created_at   timestamptz not null default now(),
  updated_at   timestamptz not null default now(),
  primary key (image_id, person_index, annotator)
);
create index if not exists poses_status_idx on public.poses (status, source);

-- ---------- 포즈 검색 평가 (비슷함/다름) ----------
create table if not exists public.pose_evals (
  id               bigint generated always as identity primary key,
  evaluator        text not null check (length(trim(evaluator)) > 0),
  query_hash       text not null,                  -- 같은 검색(질의 포즈 + 옵션)을 묶는 키
  query            jsonb not null,                 -- 질의 포즈 13개 {x, y, state} + aspect
  options          jsonb not null default '{}',    -- 부위, 좌우 반전, 방향 필터, 허용 범위
  result_image_id  text not null references public.images(id) on delete cascade,
  result_annotator text,
  rank             int,
  score            real,
  flip             boolean not null default false,
  verdict          text not null check (verdict in ('similar', 'different')),
  created_at       timestamptz not null default now(),
  updated_at       timestamptz not null default now(),
  unique (evaluator, query_hash, result_image_id)  -- 같은 사람이 다시 누르면 덮어쓰기
);

-- ---------- 권한 ----------
grant select, insert, update, delete on public.poses, public.pose_evals to service_role;
grant select, insert, update         on public.poses, public.pose_evals to anon, authenticated;
grant usage, select on all sequences in schema public to anon, authenticated, service_role;

alter table public.poses      enable row level security;
alter table public.pose_evals enable row level security;

do $$
declare
  t text;
begin
  foreach t in array array['poses', 'pose_evals'] loop
    if not exists (select 1 from pg_policies where schemaname = 'public' and tablename = t and policyname = t || '_select') then
      execute format('create policy %I on public.%I for select to anon, authenticated using (true)', t || '_select', t);
    end if;
    if not exists (select 1 from pg_policies where schemaname = 'public' and tablename = t and policyname = t || '_insert') then
      execute format('create policy %I on public.%I for insert to anon, authenticated with check (true)', t || '_insert', t);
    end if;
    if not exists (select 1 from pg_policies where schemaname = 'public' and tablename = t and policyname = t || '_update') then
      execute format('create policy %I on public.%I for update to anon, authenticated using (true) with check (true)', t || '_update', t);
    end if;
  end loop;
end $$;

-- =====================================================================
-- 이미지 유사 검색 평가 (비슷함/다름). 예전에는 각자 PC의 eval_{방식}.csv에 저장했다.
-- 배포 서버는 재시작 때 파일이 지워지므로 DB로 옮긴다. 이 블록만 따로 실행해도 되고, 여러 번 실행해도 안전하다.
-- 예전 CSV는 python migrate_to_supabase.py 로 올린다 (DB에 이미 있는 평가는 덮어쓰지 않음).
-- =====================================================================
create table if not exists public.search_evals (
  id         bigint generated always as identity primary key,
  method     text not null,                        -- 검색 방식 key (예: phash)
  evaluator  text not null check (length(trim(evaluator)) > 0),
  query_id   text not null references public.images(id) on delete cascade,
  result_id  text not null references public.images(id) on delete cascade,
  rank       int,
  score      real,                                 -- 방식별 점수 (Phash는 해밍 거리)
  verdict    text not null check (verdict in ('similar', 'different')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (method, evaluator, query_id, result_id)  -- 같은 사람이 다시 누르면 덮어쓰기
);

grant select, insert, update, delete on public.search_evals to service_role;
grant select, insert, update         on public.search_evals to anon, authenticated;
grant usage, select on all sequences in schema public to anon, authenticated, service_role;

alter table public.search_evals enable row level security;

do $$
begin
  if not exists (select 1 from pg_policies where schemaname = 'public' and tablename = 'search_evals' and policyname = 'search_evals_select') then
    create policy search_evals_select on public.search_evals for select to anon, authenticated using (true);
  end if;
  if not exists (select 1 from pg_policies where schemaname = 'public' and tablename = 'search_evals' and policyname = 'search_evals_insert') then
    create policy search_evals_insert on public.search_evals for insert to anon, authenticated with check (true);
  end if;
  if not exists (select 1 from pg_policies where schemaname = 'public' and tablename = 'search_evals' and policyname = 'search_evals_update') then
    create policy search_evals_update on public.search_evals for update to anon, authenticated using (true) with check (true);
  end if;
end $$;

-- =====================================================================
-- 팀원 명단 (입장 화면의 이메일 확인용). 이 블록만 따로 실행해도 되고, 여러 번 실행해도 안전하다.
-- 앱(팀원 키)은 명단을 직접 읽을 수 없고, member_name(이메일) 함수로 "이 이메일이 누구인지"만 물어본다.
-- 이름은 기존 라벨·포즈 기록의 작성자 이름과 똑같이 넣는다 (다르면 다른 사람으로 집계됨).
--
-- 팀원 추가·수정 예시 (SQL Editor에서 실행, 이메일은 소문자로):
--   insert into public.members (email, name) values ('someone@example.com', '김하영')
--   on conflict (email) do update set name = excluded.name, active = true;
-- 입장 막기: update public.members set active = false where email = 'someone@example.com';
-- =====================================================================
create table if not exists public.members (
  email      text primary key check (email = lower(trim(email)) and position('@' in email) > 1),
  name       text not null check (length(trim(name)) > 0),
  active     boolean not null default true,
  created_at timestamptz not null default now()
);

grant select, insert, update, delete on public.members to service_role;
revoke all on public.members from anon, authenticated;
alter table public.members enable row level security;  -- 정책을 만들지 않는다 → 팀원 키로는 명단을 읽을 수 없음

create or replace function public.member_name(p_email text) returns text
language sql stable security definer set search_path = public as $$
  select name from public.members where email = lower(trim(p_email)) and active limit 1
$$;
revoke all on function public.member_name(text) from public;
grant execute on function public.member_name(text) to anon, authenticated, service_role;

-- =====================================================================
-- 학습 기록 + 예측을 학습 묶음(run) 단위로 (2026-10, 태그 학습을 로컬 PC의 training.py로만 하면서 추가)
-- 이 블록만 따로 실행해도 되고, 여러 번 실행해도 안전하다.
--
-- - training.py가 학습 한 번마다 training_runs에 기록(누가, 언제, 이미지 수, 태그별 지표)을 넣고,
--   (학습을 시작할 때 기록을 만들고) 예측을 그 run_id로 여러 번에 나눠 올린 뒤 finish_training_run()으로 '완료' 표시한다.
-- - 웹앱은 가장 최근에 '완료'된 학습의 예측만 쓴다 → 올리는 도중이거나 실패한 학습은 팀 화면에 보이지 않는다.
-- - 완료할 때 다른 학습(이미 완료됐거나 30분 넘게 멈춘 것)의 예측은 지운다(학습 기록·지표는 남김). 팀원 키에는 삭제 권한이 없어 함수로 한다.
-- - 학습 중인 PC는 touch_training_run()으로 진행 상황을 남겨, 다른 팀원 화면에 '○○ 님이 학습 중'이 보인다.
-- - 2026-10-22에 처음 실행한 뒤 바뀐 부분(진행 상황 칸, 함수 인자)이 있으므로 이 블록 전체를 다시 실행하면 된다.
-- =====================================================================
create table if not exists public.training_runs (
  run_id       uuid primary key,
  started_at   timestamptz not null default now(),
  completed_at timestamptz,                          -- 비어 있으면 올리는 중이거나 중간에 실패한 학습
  trainer      text not null check (length(trim(trainer)) > 0),
  embed_model  text not null,
  n_images     int not null default 0,               -- 학습에 쓴(임베딩이 있는) 이미지 수
  n_rows       int not null default 0,               -- 올린 예측 행 수 (완료할 때 DB가 센다)
  report       jsonb not null default '[]'::jsonb,   -- 태그별 지표 [{tag_key, name, n_pos, n_neg, precision, recall, f1, ...}]
  heartbeat_at timestamptz,                          -- 학습 중인 PC가 주기적으로 갱신 → 다른 팀원 화면에 '학습 중' 표시
  stage        text                                  -- 지금 하는 일 (예: '임베딩 추출 · 45% 남은 시간 약 3분')
);
-- 2026-10-22 처음 만든 뒤 추가한 칸 (이미 테이블이 있으면 여기서 더한다)
alter table public.training_runs add column if not exists heartbeat_at timestamptz;
alter table public.training_runs add column if not exists stage text;
create index if not exists training_runs_completed_idx on public.training_runs (completed_at desc);

-- 예측에 학습 묶음 번호를 붙인다. 기본 키를 (run_id, image_id, tag_key)로 바꾼다.
-- 예전 방식 행(run_id 없음)은 새 방식에서 읽지 않으므로 지운다 (전환 시점 2026-10-22에 0행).
alter table public.predictions add column if not exists run_id uuid references public.training_runs(run_id) on delete cascade;
do $$
begin
  if not exists (select 1 from pg_index i join pg_attribute a on a.attrelid = i.indrelid and a.attnum = any(i.indkey)
                 where i.indrelid = 'public.predictions'::regclass and i.indisprimary and a.attname = 'run_id') then
    delete from public.predictions where run_id is null;
    alter table public.predictions drop constraint if exists predictions_pkey;
    alter table public.predictions alter column run_id set not null;
    alter table public.predictions add primary key (run_id, image_id, tag_key);
  end if;
end $$;

-- 학습 기록은 학습을 시작할 때 만들고(지표 없이), 끝날 때 이 함수로 이미지 수·지표를 넣으며 '완료' 표시한다.
drop function if exists public.finish_training_run(uuid);  -- 2026-10-22 첫 버전 (인자 하나)
create or replace function public.finish_training_run(p_run uuid, p_n_images int, p_report jsonb) returns void
language plpgsql security definer set search_path = public as $$
begin
  update public.training_runs
     set completed_at = now(), heartbeat_at = now(), stage = '완료',
         n_images = p_n_images, report = coalesce(p_report, '[]'::jsonb),
         n_rows = (select count(*) from public.predictions where run_id = p_run)
   where run_id = p_run and completed_at is null;
  if not found then
    raise exception '학습 기록(%)이 없거나 이미 완료되었습니다', p_run;
  end if;
  -- 다른 학습의 예측은 지운다: 이미 완료된 학습(이제 옛 결과)과 30분 넘게 소식이 없는 중단된 학습.
  -- 같은 시간에 다른 PC에서 아직 올리는 중인 학습의 예측은 남긴다. 학습 기록과 지표는 모두 남긴다
  delete from public.predictions p using public.training_runs r
   where p.run_id = r.run_id and r.run_id <> p_run
     and (r.completed_at is not null or coalesce(r.heartbeat_at, r.started_at) < now() - interval '30 minutes');
end $$;

-- 학습 중인 PC가 주기적으로 부른다 → 다른 팀원 화면에 '○○ 님이 학습 중 · 지금 하는 일'을 보여준다
create or replace function public.touch_training_run(p_run uuid, p_stage text) returns void
language sql security definer set search_path = public as $$
  update public.training_runs set heartbeat_at = now(), stage = left(p_stage, 300)
   where run_id = p_run and completed_at is null
$$;

grant select, insert, update, delete on public.training_runs to service_role;
grant select, insert                 on public.training_runs to anon, authenticated;
alter table public.training_runs enable row level security;
do $$
begin
  if not exists (select 1 from pg_policies where schemaname = 'public' and tablename = 'training_runs' and policyname = 'training_runs_select') then
    create policy training_runs_select on public.training_runs for select to anon, authenticated using (true);
  end if;
  if not exists (select 1 from pg_policies where schemaname = 'public' and tablename = 'training_runs' and policyname = 'training_runs_insert') then
    create policy training_runs_insert on public.training_runs for insert to anon, authenticated with check (completed_at is null);
  end if;
end $$;
revoke all on function public.finish_training_run(uuid, int, jsonb) from public;
grant execute on function public.finish_training_run(uuid, int, jsonb) to anon, authenticated, service_role;
revoke all on function public.touch_training_run(uuid, text) from public;
grant execute on function public.touch_training_run(uuid, text) to anon, authenticated, service_role;

-- =====================================================================
-- 팀원 키(publishable)로도 새 이미지를 '추가'할 수 있게 (2026-10-22)
-- 이미지 수집 담당자가 관리자 키 없이 자기 PC에서 python migrate_to_supabase.py --images-only 로 등록한다.
-- 이미 있는 이미지는 건드리지 않는다(ON CONFLICT DO NOTHING). 수정·삭제는 계속 관리자 키만.
-- 이 블록만 따로 실행해도 되고, 여러 번 실행해도 안전하다.
-- =====================================================================
grant insert on public.images to anon, authenticated;
do $$
begin
  if not exists (select 1 from pg_policies where schemaname = 'public' and tablename = 'images' and policyname = 'images_insert') then
    create policy images_insert on public.images for insert to anon, authenticated with check (true);
  end if;
end $$;
