# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = Path("scripts/strip_event_timestamps.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("strip_event_timestamps", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load_module()


def _run(tmp_path, records, *flags):
    """Write records as JSONL, run the utility to a file, return output dicts."""
    src = tmp_path / "events.jsonl"
    lines = [r if isinstance(r, str) else json.dumps(r) for r in records]
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = tmp_path / "out.jsonl"
    mod.main([str(src), "--output", str(out), *flags])
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]


# A failed sample: the load generator deliberately emits ERROR before COMPLETE
# even though COMPLETE's timestamp is earlier. File order is the intended order.
_ERR = {"event_type": "error", "sample_uuid": "s", "timestamp_ns": 200}
_DONE = {"event_type": "complete", "sample_uuid": "s", "timestamp_ns": 100}


def test_file_order_preserved_and_numbered_sequentially(tmp_path):
    recs = [
        {"event_type": "c", "timestamp_ns": 30},
        {"event_type": "a", "timestamp_ns": 10},
        {"event_type": "b", "timestamp_ns": 20},
    ]
    out = _run(tmp_path, recs)

    assert [r["event_type"] for r in out] == ["c", "a", "b"]  # file order kept
    assert [r["seq"] for r in out] == [0, 1, 2]
    assert [r["timestamp_ns"] for r in out] == [30, 10, 20]  # timestamp retained


def test_error_before_complete_preserved(tmp_path):
    out = _run(tmp_path, [_ERR, _DONE])

    assert [r["event_type"] for r in out] == ["error", "complete"]
    assert [r["seq"] for r in out] == [0, 1]


def test_drop_timestamp_removes_original(tmp_path):
    (out,) = _run(
        tmp_path, [{"event_type": "x", "timestamp_ns": 5}], "--drop-timestamp"
    )

    assert "timestamp_ns" not in out
    assert out["seq"] == 0


def test_custom_seq_name(tmp_path):
    (out,) = _run(
        tmp_path, [{"event_type": "x", "timestamp_ns": 5}], "--seq-name", "order"
    )

    assert out["order"] == 0
    assert "seq" not in out


def test_seq_placed_adjacent_to_timestamp(tmp_path):
    # The ordinal is inserted in place of the timestamp, not appended at the end.
    recs = [{"event_type": "a", "timestamp_ns": 5, "tail": 1}]

    (kept,) = _run(tmp_path, recs)
    assert list(kept.keys()) == ["event_type", "timestamp_ns", "seq", "tail"]

    (dropped,) = _run(tmp_path, recs, "--drop-timestamp")
    assert list(dropped.keys()) == ["event_type", "seq", "tail"]


def test_always_drops_approx_datetime_str_and_keeps_other_fields(tmp_path):
    legacy = {
        "approx_datetime_str": "2026-02-11T08:06:17.4",
        "event_type": "test_started",
        "sample_uuid": "",
        "timestamp_ns": 675243120177080,
        "value": "hello",
    }
    (out,) = _run(tmp_path, [legacy])

    assert "approx_datetime_str" not in out
    assert out["event_type"] == "test_started"
    assert out["value"] == "hello"  # untouched passthrough
    assert out["seq"] == 0


def test_blank_lines_skipped(tmp_path):
    recs = [
        {"event_type": "a", "timestamp_ns": 1},
        "",
        {"event_type": "b", "timestamp_ns": 2},
    ]
    out = _run(tmp_path, recs)

    assert [r["event_type"] for r in out] == ["a", "b"]
    assert [r["seq"] for r in out] == [0, 1]


def test_bad_lines_are_skipped_counted_and_summarized(tmp_path, capsys):
    src = tmp_path / "events.jsonl"
    src.write_text(
        "\n".join(
            [
                json.dumps({"event_type": "a", "timestamp_ns": 1}),
                '{"event_type": "b",',  # malformed JSON
                "[1, 2]",  # valid JSON, not an object
                json.dumps({"event_type": "c"}),  # missing timestamp_ns
                json.dumps({"event_type": "d", "timestamp_ns": 2}),
            ]
        )
        + "\n"
    )
    out = tmp_path / "out.jsonl"
    mod.main([str(src), "--output", str(out)])

    written = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["event_type"] for r in written] == ["a", "d"]  # only usable records
    assert [r["seq"] for r in written] == [0, 1]  # contiguous over kept records

    err = capsys.readouterr().err
    assert "skipped 3" in err
    assert "malformed JSON" in err
    assert "not a JSON object" in err
    assert "timestamp_ns" in err


def test_invalid_utf8_line_is_skipped_and_counted(tmp_path, capsys):
    # A line with invalid UTF-8 must be tallied like any other bad line, not crash.
    src = tmp_path / "events.jsonl"
    src.write_bytes(
        json.dumps({"event_type": "a", "timestamp_ns": 1}).encode()
        + b"\n\xff\xfe not utf8\n"
        + json.dumps({"event_type": "b", "timestamp_ns": 2}).encode()
        + b"\n"
    )
    out = tmp_path / "out.jsonl"
    mod.main([str(src), "--output", str(out)])

    written = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["event_type"] for r in written] == ["a", "b"]
    assert [r["seq"] for r in written] == [0, 1]
    assert "invalid UTF-8" in capsys.readouterr().err


def test_bad_output_path_is_clean_error(tmp_path):
    src = tmp_path / "events.jsonl"
    src.write_text(json.dumps({"event_type": "a", "timestamp_ns": 1}) + "\n")

    with pytest.raises(SystemExit):
        mod.main([str(src), "--output", str(tmp_path / "no_such_dir" / "out.jsonl")])


def test_clean_run_writes_no_summary(tmp_path, capsys):
    _run(tmp_path, [{"event_type": "a", "timestamp_ns": 1}])

    assert capsys.readouterr().err == ""


def test_seq_name_collision_is_a_hard_error(tmp_path):
    # A record already containing the chosen column name must abort, not corrupt.
    with pytest.raises(SystemExit):
        _run(tmp_path, [{"event_type": "x", "timestamp_ns": 1, "seq": 999}])

    with pytest.raises(SystemExit):
        _run(
            tmp_path,
            [{"event_type": "x", "timestamp_ns": 1}],
            "--seq-name",
            "event_type",
        )


def test_refuses_to_overwrite_input(tmp_path):
    # The streaming path truncates --output before reading the input; aliasing
    # them must be rejected with the input left intact.
    src = tmp_path / "events.jsonl"
    src.write_text(json.dumps({"event_type": "a", "timestamp_ns": 1}) + "\n")
    before = src.read_text()

    with pytest.raises(SystemExit):
        mod.main([str(src), "--output", str(src)])

    assert src.read_text() == before  # not truncated


def test_missing_input_is_clean_error(tmp_path):
    with pytest.raises(SystemExit):
        mod.main([str(tmp_path / "does_not_exist.jsonl")])


def test_bad_input_does_not_truncate_existing_output(tmp_path):
    # A bad input path must fail before the (distinct) --output is truncated.
    out = tmp_path / "precious.jsonl"
    out.write_text("KEEP ME\n")
    with pytest.raises(SystemExit):
        mod.main([str(tmp_path / "nope.jsonl"), "--output", str(out)])

    assert out.read_text() == "KEEP ME\n"  # untouched


def test_writes_to_stdout_when_no_output(tmp_path, capsys):
    src = tmp_path / "events.jsonl"
    src.write_text(json.dumps({"event_type": "x", "timestamp_ns": 5}) + "\n")
    mod.main([str(src)])

    line = capsys.readouterr().out.strip()
    assert json.loads(line) == {"event_type": "x", "timestamp_ns": 5, "seq": 0}
