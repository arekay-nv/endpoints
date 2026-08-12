#!/usr/bin/env python3
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

"""Replace raw timestamps in an events.jsonl with an ordinal sequence.

Records are emitted in file order — the intended emission order, e.g. the load
generator deliberately writes ERROR before COMPLETE for a failed sample even
though COMPLETE's timestamp is fractionally earlier — and each gets a 0-based
``seq``. The tool streams line-by-line, so memory stays constant regardless of
file size.

Schema-agnostic: each line is handled as a plain JSON object, so both the current
``EventRecord`` schema and older ``events.jsonl`` variants (``approx_datetime_str``,
un-namespaced ``event_type``) pass through with all other fields preserved.
``approx_datetime_str`` is always dropped; ``--drop-timestamp`` also drops
``timestamp_ns``; ``--seq-name`` renames the ordinal column.

Lines that can't be used — malformed JSON, a non-object value, or a record with
no ``timestamp_ns`` — are skipped and tallied, with a summary written to stderr
at the end (the run still exits 0). A ``--seq-name`` that collides with a key
already present in a record is a hard error.

usage:
  strip_event_timestamps.py INPUT.jsonl [--output OUT.jsonl]
      [--drop-timestamp] [--seq-name seq]

  --drop-timestamp   remove the original timestamp_ns too (default: keep it)
  --seq-name NAME    name of the sequence column (default: seq); must not
                     collide with an existing field
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import BinaryIO, TextIO

# Wall-clock timestamp field that is always stripped (an ordinal replaces it).
_ALWAYS_DROP = ("approx_datetime_str",)
_TIMESTAMP_KEY = "timestamp_ns"
_PROG = "strip_event_timestamps"


def _classify(line: bytes) -> tuple[dict | None, str]:
    """Parse a JSONL line into a usable record, or return why it can't be used."""
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError:
        return None, "invalid UTF-8"
    try:
        rec = json.loads(text)
    except json.JSONDecodeError:
        return None, "malformed JSON"
    if not isinstance(rec, dict):
        return None, "not a JSON object"
    if _TIMESTAMP_KEY not in rec:
        return None, f"missing {_TIMESTAMP_KEY!r}"
    return rec, ""


def _rewrite(rec: dict, seq: int, seq_name: str, drop_timestamp: bool) -> dict:
    """Rebuild the record preserving key order, placing the ordinal at timestamp_ns."""
    out: dict = {}
    for key, value in rec.items():
        if key in _ALWAYS_DROP:
            continue
        if key == _TIMESTAMP_KEY:
            if not drop_timestamp:
                out[key] = value
            out[seq_name] = seq  # place the ordinal where the timestamp was
            continue
        out[key] = value
    return out


def _same_file(a: str, b: str) -> bool:
    """True if a and b point at the same file (handles symlinks/hardlinks)."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        # b may not exist yet; fall back to comparing resolved paths.
        return os.path.realpath(a) == os.path.realpath(b)


def _process(
    fh: BinaryIO, out_fh: TextIO, path: str, seq_name: str, drop_timestamp: bool
) -> tuple[int, Counter[str]]:
    """Stream records to out_fh; return (records_written, skip_counts_by_reason).

    The input is read as bytes so a line with invalid UTF-8 is tallied like any
    other unusable line rather than aborting the whole run.
    """
    errors: Counter[str] = Counter()
    written = 0
    for lineno, line in enumerate(fh, start=1):
        line = line.strip()
        if not line:
            continue
        rec, reason = _classify(line)
        if rec is None:
            errors[reason] += 1
            continue
        if seq_name in rec:
            raise SystemExit(
                f"{path}:{lineno}: record already has a {seq_name!r} key; "
                "choose a different --seq-name"
            )
        record = _rewrite(rec, written, seq_name, drop_timestamp)
        out_fh.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False))
        out_fh.write("\n")
        written += 1
    return written, errors


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog=_PROG, description=__doc__.splitlines()[0])
    parser.add_argument("input", help="path to events.jsonl")
    parser.add_argument(
        "-o", "--output", default=None, help="output path (default: stdout)"
    )
    parser.add_argument(
        "--drop-timestamp",
        action="store_true",
        help="also drop the original timestamp_ns",
    )
    parser.add_argument(
        "--seq-name",
        default="seq",
        help="name of the sequence column (default: seq); "
        "must not collide with an existing field",
    )
    args = parser.parse_args(argv)

    # Open the input first: a bad path fails cleanly here, before any output is
    # opened/truncated. Bytes mode so invalid UTF-8 is a tallied line, not a crash.
    try:
        fh = open(args.input, "rb")
    except OSError as exc:
        raise SystemExit(f"{args.input}: cannot open input: {exc}") from exc

    try:
        # The input is opened (read-only) but not yet consumed; refuse an --output
        # that aliases it before the write handle truncates the source.
        if args.output is not None and _same_file(args.input, args.output):
            raise SystemExit(
                f"{args.output}: refusing to write output over the input file"
            )
        try:
            out_fh = (
                open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
            )
        except OSError as exc:
            raise SystemExit(f"{args.output}: cannot open output: {exc}") from exc
        try:
            written, errors = _process(
                fh, out_fh, args.input, args.seq_name, args.drop_timestamp
            )
        finally:
            if out_fh is not sys.stdout:
                out_fh.close()
    finally:
        fh.close()

    if errors:
        total = sum(errors.values())
        detail = ", ".join(f"{n} {reason}" for reason, n in sorted(errors.items()))
        print(
            f"{_PROG}: wrote {written} record(s); skipped {total} line(s): {detail}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # Downstream (e.g. `head`) closed the pipe. Redirect stdout to devnull so
        # the interpreter's shutdown flush doesn't raise a second BrokenPipeError.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(1)
