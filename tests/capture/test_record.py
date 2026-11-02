"""Tests for llmdiff.capture.record.TrafficRecorder."""

from __future__ import annotations

import json
import threading

import pytest

from llmdiff.capture.errors import RecordError
from llmdiff.capture.record import TrafficRecorder
from llmdiff.core.corpus import Corpus


def test_record_basic_fields():
    rec = TrafficRecorder()
    case = rec.record("hello", "world", model="gpt-4o", tags=["t1"])
    assert case is not None
    assert case.prompt == "hello"
    assert case.candidate_output == "world"
    assert case.baseline_output == ""
    assert case.metadata["model"] == "gpt-4o"
    assert case.metadata["tags"] == ["t1"]
    assert "timestamp" in case.metadata
    assert len(rec) == 1


def test_sequential_ids_unique_and_prefixed():
    rec = TrafficRecorder(id_prefix="cap")
    ids = [rec.record("p", "o").id for _ in range(5)]
    assert ids == [f"cap-{i:06d}" for i in range(1, 6)]
    assert len(set(ids)) == 5


def test_uuid_id_strategy():
    rec = TrafficRecorder(id_strategy="uuid")
    a = rec.record("p", "o")
    b = rec.record("p", "o")
    assert a.id != b.id
    assert a.id.startswith("rec-")


def test_invalid_id_strategy():
    with pytest.raises(RecordError):
        TrafficRecorder(id_strategy="bogus")


def test_duplicate_explicit_id_rejected():
    rec = TrafficRecorder()
    rec.record("p", "o", case_id="x")
    with pytest.raises(RecordError):
        rec.record("p", "o", case_id="x")


def test_redaction_hooks_applied():
    rec = TrafficRecorder(redaction_hooks=[lambda s: s.replace("secret", "[X]")])
    case = rec.record("my secret prompt", "output secret")
    assert case.prompt == "my [X] prompt"
    assert case.candidate_output == "output [X]"


def test_non_callable_hook_rejected():
    with pytest.raises(RecordError):
        TrafficRecorder(redaction_hooks=[42])


def test_sampler_drops_records():
    class DropAll:
        def should_keep(self, record):
            return False

    rec = TrafficRecorder(sampler=DropAll())
    assert rec.record("p", "o") is None
    assert len(rec) == 0
    assert rec.dropped_count == 1


def test_thread_safety_many_threads():
    rec = TrafficRecorder()

    def worker(n):
        for i in range(50):
            rec.record(f"p{n}-{i}", "o")

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(rec) == 400
    ids = [c.id for c in rec.cases]
    assert len(set(ids)) == 400


def test_save_load_roundtrip(tmp_path):
    rec = TrafficRecorder()
    rec.record("hi", "hello", model="m1", tags=["a"])
    rec.record("yo", "hey")
    path = tmp_path / "traffic.jsonl"
    rec.save(path)
    text = path.read_text(encoding="utf-8")
    lines = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    assert len(lines) == 2
    assert lines[0]["prompt"] == "hi"

    loaded = TrafficRecorder.load(path)
    assert len(loaded) == 2
    assert loaded.cases[0].prompt == "hi"
    assert loaded.cases[0].metadata["model"] == "m1"
    # loaded corpus is a valid Corpus
    corpus = loaded.corpus()
    assert isinstance(corpus, Corpus)
    assert len(corpus) == 2


def test_load_missing_file_raises_record_error(tmp_path):
    with pytest.raises(RecordError):
        TrafficRecorder.load(tmp_path / "nope.jsonl")


def test_load_invalid_jsonl_raises(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(RecordError):
        TrafficRecorder.load(bad)


def test_from_jsonl_text():
    rec = TrafficRecorder()
    rec.record("a", "b")
    text = rec.to_jsonl_text()
    rec2 = TrafficRecorder.from_jsonl_text(text)
    assert len(rec2) == 1
    assert rec2.cases[0].prompt == "a"


def test_stats():
    rec = TrafficRecorder()
    rec.record("p1", "o1", model="m1", tags=["x"])
    rec.record("p2", "o2", model="m2")
    stats = rec.stats()
    assert stats["recorded"] == 2
    assert stats["models"] == ["m1", "m2"]
    assert stats["tags"] == ["x"]
    assert stats["first_id"] == rec.cases[0].id


def test_clear():
    rec = TrafficRecorder()
    rec.record("p", "o")
    rec.clear()
    assert len(rec) == 0
    # counter resets: next id is 000001 again
    assert rec.record("p", "o").id.endswith("000001")


def test_non_string_prompt_rejected():
    rec = TrafficRecorder()
    with pytest.raises(RecordError):
        rec.record(123, "o")  # type: ignore[arg-type]


def test_default_metadata_and_tags_merged():
    rec = TrafficRecorder(tags=["base"], metadata={"source": "test"})
    case = rec.record("p", "o", tags=["extra"], extra_field=1)
    assert case.metadata["source"] == "test"
    assert case.metadata["extra_field"] == 1
    assert set(case.metadata["tags"]) == {"base", "extra"}
