from unittest.mock import patch
import pytest
from orchestra.chat import store


def _patch_root(tmp_path):
    return patch("orchestra.config.orchestra_root", return_value=tmp_path)


def test_new_session_id_shape(tmp_path):
    with _patch_root(tmp_path):
        sid = store.new_session_id("cdx-deep@windows")
    parts = sid.split("-")
    assert len(parts) == 3
    assert parts[0].isdigit() and len(parts[0]) == 8      # YYYYMMDD
    assert parts[1] == "cdxdeepwindows"                    # slug, non-alnum stripped
    assert len(parts[2]) == 4                              # short hex


def test_write_and_load_round_trip(tmp_path):
    with _patch_root(tmp_path):
        sid = "20260713-x-aaaa"
        store.write_meta(sid, "oll-fast", "ollama/qwen2.5:7b")
        store.append_message(sid, "user", "hi")
        store.append_message(sid, "assistant", "hello")
        meta, msgs = store.load_session(sid)
    assert meta["alias"] == "oll-fast"
    assert meta["model"] == "ollama/qwen2.5:7b"
    assert [m["content"] for m in msgs] == ["hi", "hello"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]


def test_load_missing_raises_with_available_ids(tmp_path):
    with _patch_root(tmp_path):
        store.write_meta("20260713-a-bbbb", "oll-fast", "m")
        with pytest.raises(FileNotFoundError) as exc:
            store.load_session("nope")
    assert "20260713-a-bbbb" in str(exc.value)


def test_list_sessions_returns_snippet(tmp_path):
    with _patch_root(tmp_path):
        store.write_meta("20260713-a-cccc", "oll-fast", "m")
        store.append_message("20260713-a-cccc", "user", "first question here")
        listing = store.list_sessions()
    assert listing[0]["session_id"] == "20260713-a-cccc"
    assert listing[0]["alias"] == "oll-fast"
    assert "first question" in listing[0]["first_user"]
