"""입장(팀원 이메일 확인)·방문자 쓰기 차단 단위 테스트 (실제 DB 없이)."""
import pytest

from core import db
from core import members as M
from core import tagging


@pytest.fixture
def fake_rpc(monkeypatch):
    """member_name 함수를 흉내 낸다. 호출 인자를 기록한다."""
    calls = []
    roster = {"kim@example.com": "김하영"}

    class Rpc:
        def __init__(self, email):
            self.email = email

        def execute(self):
            return type("Res", (), {"data": roster.get(self.email)})()

    class Client:
        def rpc(self, fn, params):
            calls.append((fn, params))
            return Rpc(params["p_email"])

    monkeypatch.setattr(db, "get_client", lambda: Client())
    return calls


def test_member_lookup_normalizes_email(fake_rpc):
    assert M.member_name("  Kim@Example.COM ") == "김하영"
    assert fake_rpc == [("member_name", {"p_email": "kim@example.com"})]
    assert M.member_name("other@example.com") is None


def test_invalid_email_does_not_query(fake_rpc):
    for bad in ("", "kim", "kim@", "@example.com", "kim @example.com"):
        assert M.member_name(bad) is None
    assert fake_rpc == []


@pytest.fixture
def visitor_guard():
    def guard():
        raise db.DBError("방문자는 볼 수만 있어 저장되지 않았습니다.")
    db.set_write_guard(guard)
    yield
    db.set_write_guard(None)


def test_guard_blocks_every_write_path(visitor_guard, monkeypatch):
    def no_network():
        raise AssertionError("쓰기가 막혔다면 DB에 접속하지 않아야 한다")
    monkeypatch.setattr(db, "get_client", no_network)
    writes = [
        lambda: db.upsert("labels", [{"x": 1}], on_conflict="x"),
        lambda: tagging.save_labels("1", {"t": 1}, "김하영"),
        lambda: tagging.save_label("1", "t", 1, "김하영"),
        lambda: tagging.set_tag_active("t", False),
        lambda: tagging.update_tag("t", "이름", "정의"),
        lambda: tagging.delete_tag("t"),
    ]
    for write in writes:
        with pytest.raises(db.DBError, match="방문자"):
            write()


def test_no_guard_outside_web_app():
    db.set_write_guard(None)
    db.check_write()  # 스크립트·테스트에서는 막지 않는다


def test_sidebar_shows_names_as_typed():
    from views.auth import md_escape as _md
    assert _md("__test__") == r"\_\_test\_\_"   # 마크다운 굵은 글씨 기호로 사라지지 않게
    assert _md(r"a\b.c") == r"a\\b\.c"
    assert _md("김하영") == "김하영"
