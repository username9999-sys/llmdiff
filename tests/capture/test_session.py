"""Tests for llmdiff.capture.session."""

from __future__ import annotations

import pytest

from llmdiff.capture.errors import SessionError
from llmdiff.capture.record import TrafficRecorder
from llmdiff.capture.session import CaptureSession, list_sessions
from llmdiff.core.corpus import Corpus


def test_session_records_with_session_id(tmp_path):
    with CaptureSession(out_dir=tmp_path, session_id="s1") as session:
        session.record("hi", "hello", model="m1")
        session.record("yo", "hey")
    assert len(session.recorder) == 2
    for case in session.recorder.cases:
        assert case.metadata["session_id"] == "s1"
        assert "session:s1" in case.metadata["tags"]


def test_session_writes_rotated_file_on_end(tmp_path):
    with CaptureSession(out_dir=tmp_path, session_id="abc") as session:
        session.record("p", "o")
    expected = tmp_path / "traffic.abc.jsonl"
    assert expected.exists()
    corpus = Corpus.from_jsonl(expected)
    assert len(corpus) == 1
    assert corpus[0].metadata["session_id"] == "abc"


def test_session_summary(tmp_path):
    session = CaptureSession(out_dir=tmp_path, session_id="s9")
    session.record("p1", "o1", model="m1")
    session.record("p2", "o2", model="m2")
    summary = session.end()
    assert summary.session_id == "s9"
    assert summary.recorded == 2
    assert summary.models == ("m1", "m2")
    assert summary.output_path is not None
    d = summary.to_dict()
    assert d["recorded"] == 2 and "started_at" in d and "ended_at" in d


def test_double_end_raises(tmp_path):
    session = CaptureSession(out_dir=tmp_path)
    session.end()
    with pytest.raises(SessionError):
        session.end()


def test_record_after_end_raises(tmp_path):
    session = CaptureSession(out_dir=tmp_path)
    session.end()
    with pytest.raises(SessionError):
        session.record("p", "o")


def test_auto_generated_session_id(tmp_path):
    s = CaptureSession(out_dir=tmp_path)
    assert s.session_id.startswith("sess-")
    assert s.is_active


def test_wraps_existing_recorder(tmp_path):
    rec = TrafficRecorder()
    session = CaptureSession(out_dir=tmp_path, recorder=rec, session_id="w1")
    session.record("p", "o")
    assert len(rec) == 1
    assert rec.cases[0].metadata["session_id"] == "w1"


def test_invalid_recorder_rejected(tmp_path):
    with pytest.raises(SessionError):
        CaptureSession(out_dir=tmp_path, recorder=object())  # type: ignore[arg-type]


def test_custom_file_pattern(tmp_path):
    with CaptureSession(
        out_dir=tmp_path, session_id="xyz", file_pattern="{date}-{session_id}.jsonl"
    ) as session:
        session.record("p", "o")
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    assert "xyz" in files[0].name


def test_list_sessions(tmp_path):
    with CaptureSession(out_dir=tmp_path, session_id="sA") as s:
        s.record("p", "o", model="m1")
    with CaptureSession(out_dir=tmp_path, session_id="sB") as s:
        s.record("p", "o", model="m2")
        s.record("p2", "o2", model="m2")
    sessions = list_sessions(tmp_path)
    assert len(sessions) == 2
    by_id = {s["session_id"]: s for s in sessions}
    assert by_id["sA"]["recorded"] == 1
    assert by_id["sB"]["recorded"] == 2
    assert by_id["sB"]["models"] == ["m2"]


def test_list_sessions_missing_dir(tmp_path):
    with pytest.raises(SessionError):
        list_sessions(tmp_path / "nope")


def test_summary_while_active(tmp_path):
    session = CaptureSession(out_dir=tmp_path, session_id="live")
    session.record("p", "o", model="m")
    s = session.summary()
    assert s["active"] is True and s["recorded"] == 1
