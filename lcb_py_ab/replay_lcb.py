#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay a captured LCB codes_dict against a running lcb-service.

Spike helper for the lcb-service Python 3.11-vs-3.14 A/B. The live benchmark run
(with ``LCB_DUMP_CODES_DIR`` set) writes one ``codes_NNN.json`` per evaluate
call. This script replays that exact payload against whichever lcb-service is
listening on ``--port``, so the ONLY variable between two replays is the
container's Python interpreter. Uses ``_lcb_ws_evaluate`` -- the exact shared
client both production scorers use -- so grading is identical to a real run.

Run from the endpoints checkout, e.g.:
    env -u VIRTUAL_ENV PYTHONPATH=src python lcb_py_ab/replay_lcb.py \
        --codes run_py311/lcb_codes/codes_000.json --port 13835 \
        --label py311 --out results_py311.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

from inference_endpoint.evaluation.scoring import _lcb_ws_evaluate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codes", required=True, help="captured codes_NNN.json payload")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=13835)
    ap.add_argument("--timeout", type=int, default=None, help="override timeout_sec")
    ap.add_argument("--label", default="", help="tag stored in output (e.g. py311)")
    ap.add_argument("--out", required=True, help="results json output path")
    args = ap.parse_args()

    # Never re-capture while replaying.
    os.environ.pop("LCB_DUMP_CODES_DIR", None)

    payload = json.loads(Path(args.codes).read_text())
    codes_dict = payload["codes_dict"]
    timeout = (
        args.timeout
        if args.timeout is not None
        else int(payload.get("timeout_sec", 60))
    )
    url = f"ws://{args.host}:{args.port}/evaluate"

    res = _lcb_ws_evaluate(url, codes_dict, timeout)
    if res is None:
        print(f"ERROR: lcb-service unreachable/failed at {url}", file=sys.stderr)
        sys.exit(2)

    total = int(res.get("total_samples", 0))
    per_problem = res.get("results", {})
    passed = sum(sum(v) for v in per_problem.values())
    pass_at_1 = passed / total if total else 0.0
    out = {
        "label": args.label,
        "url": url,
        "codes_file": str(args.codes),
        "total_samples": total,
        "total_passed": passed,
        "pass_at_1": pass_at_1,
        "results": per_problem,  # {question_id: [pass_bool, ...]}
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(
        f"[{args.label or url}] pass@1={pass_at_1:.4f} "
        f"passed={passed}/{total} -> {args.out}"
    )


if __name__ == "__main__":
    main()
