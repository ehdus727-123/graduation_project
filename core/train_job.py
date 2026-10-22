"""로컬 웹앱의 '학습 시작'을 화면과 분리해 뒤에서(스레드) 돌리는 작업 관리자. Streamlit에 의존하지 않는다.

Streamlit은 버튼을 누르거나 페이지를 옮기거나 새로고침하면 실행 중인 화면 코드를 멈춘다. 학습을 화면 코드 안에서
돌리면 그동안 앱을 쓸 수 없으므로, 서버 프로세스에 하나뿐인 TrainJob이 스레드로 학습하고 화면은 상태만 읽는다.
- 한 번에 하나만 실행된다 (start가 False를 돌려줌).
- 서버(터미널 창)를 끄면 학습도 멈춘다. 임베딩 캐시 덕분에 다시 시작하면 이어서 한다 (training.ensure_embeddings).
"""
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

IDLE, RUNNING, DONE, FAILED = "idle", "running", "done", "failed"
MAX_LOG = 300


@dataclass
class JobState:
    status: str = IDLE
    trainer: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    stage: str = ""
    message: str = ""
    frac: float | None = None
    log: list[str] = field(default_factory=list)
    result: object = None   # training.TrainResult
    error: str = ""
    run_no: int = 0          # 실행할 때마다 1씩 증가 (화면이 '끝난 것을 봤는지' 구분할 때 쓴다)


class TrainJob:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = JobState()
        self._embedder = None  # CLIP 모델은 한 번 불러오면 다음 학습에도 다시 쓴다

    def snapshot(self) -> JobState:
        with self._lock:
            s = self._state
            return JobState(**{**s.__dict__, "log": list(s.log)})

    def start(self, trainer: str, run: Callable | None = None) -> bool:
        """학습을 뒤에서 시작한다. 이미 실행 중이면 False. run은 테스트용 (기본: training.train_all)."""
        if not trainer.strip():
            raise ValueError("학습한 사람 이름이 필요합니다.")
        with self._lock:
            if self._state.status == RUNNING:
                return False
            self._state = JobState(status=RUNNING, trainer=trainer.strip(), started_at=time.time(),
                                   stage="준비", message="시작하는 중", run_no=self._state.run_no + 1)
        threading.Thread(target=self._run, args=(trainer.strip(), run), daemon=True, name="spectrum-training").start()
        return True

    def _progress(self, stage: str, message: str, frac: float | None) -> None:
        with self._lock:
            s = self._state
            s.stage, s.message, s.frac = stage, message, frac
            s.log.append(f"[{stage}] {message}")
            del s.log[:-MAX_LOG]

    def _get_embedder(self):
        if self._embedder is None:
            import training
            self._embedder = training.load_embedder()
        return self._embedder

    def _run(self, trainer: str, run: Callable | None) -> None:
        import training
        run = run or training.train_all
        try:
            result = run(get_embedder=self._get_embedder, progress=self._progress, publish=True, trainer=trainer)
            status, error = DONE, ""
        except training.TrainingError as e:
            result, status, error = None, FAILED, str(e)
        except BaseException as e:  # 예상 못 한 오류도 화면에 한국어로 남긴다 (스레드가 조용히 죽지 않게)
            result, status, error = None, FAILED, f"알 수 없는 오류로 실패했습니다: {type(e).__name__}: {e}"
        with self._lock:
            s = self._state
            s.status, s.result, s.error, s.finished_at = status, result, error, time.time()
