"""로컬 웹앱 '학습 시작'의 뒤에서 도는 작업(core.train_job) 단위 테스트. 실제 학습·DB 없이 가짜 학습 함수로."""
import threading
import time

import pytest

import training
from core import db
from core.train_job import DONE, FAILED, IDLE, RUNNING, TrainJob


def wait_until(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_runs_in_background_one_at_a_time():
    gate, calls = threading.Event(), []

    def fake_train(get_embedder, progress, publish, trainer):
        calls.append((publish, trainer))
        progress("임베딩 추출", "10/100장 · 남은 시간 약 9초", 0.1)
        gate.wait(5)  # 화면이 다른 일을 하는 동안 학습이 계속되는 상황
        return training.TrainResult(status=None, n_images=100)

    job = TrainJob()
    assert job.snapshot().status == IDLE
    assert job.start("김하영", run=fake_train)
    assert wait_until(lambda: job.snapshot().frac == 0.1)
    s = job.snapshot()
    assert s.status == RUNNING and s.stage == "임베딩 추출" and "남은 시간" in s.message and s.trainer == "김하영"
    assert not job.start("김도연", run=fake_train)  # 실행 중에는 두 번째를 시작하지 않는다
    gate.set()
    assert wait_until(lambda: job.snapshot().status == DONE)
    assert calls == [(True, "김하영")]  # 팀 공유(publish=True)로, 누른 사람 이름으로
    assert job.snapshot().result.n_images == 100 and job.snapshot().run_no == 1


def test_failure_is_recorded_and_can_restart():
    def broken(**_):
        raise training.TrainingError("임베딩 추출", "이미지 1 ~ 32", RuntimeError("디스크가 가득 찼습니다"))

    job = TrainJob()
    job.start("김하영", run=broken)
    assert wait_until(lambda: job.snapshot().status == FAILED)
    assert "디스크가 가득" in job.snapshot().error
    job.start("김하영", run=lambda **_: training.TrainResult(status=None))  # 실패 뒤에는 다시 시작할 수 있다
    assert wait_until(lambda: job.snapshot().status == DONE) and job.snapshot().run_no == 2


def test_unexpected_error_does_not_kill_silently():
    job = TrainJob()
    job.start("김하영", run=lambda **_: 1 / 0)
    assert wait_until(lambda: job.snapshot().status == FAILED)
    assert "ZeroDivisionError" in job.snapshot().error


def test_requires_trainer():
    with pytest.raises(ValueError):
        TrainJob().start("  ")


def test_background_thread_is_not_blocked_by_visitor_guard():
    """방문자 차단은 화면 세션 기준. 팀원이 시작한 뒤에서 도는 학습 스레드는 세션이 없으므로 막지 않는다."""
    from views.auth import write_guard
    db.set_write_guard(write_guard)
    try:
        result = []
        t = threading.Thread(target=lambda: result.append(db.check_write() or "ok"))
        t.start()
        t.join()
        assert result == ["ok"]
    finally:
        db.set_write_guard(None)
