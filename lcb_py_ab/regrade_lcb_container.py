#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Re-grade a completed LiveCodeBench run's generations against a running
lcb-service container, with NO re-inference.

Reconstructs codes_dict from events.jsonl (manual EventRecord parse, mirroring
rescore_two_ways.py so it runs from the same checkout venv that produced the
run), then grades via the container WebSocket and dumps per-problem pass/fail +
pass@1. Run once per container build (py311 on one port, py314 on another) over
the SAME events to isolate the container's Python version, then diff with
compare_lcb.py.

    <venv>/bin/python regrade_lcb_container.py \
        --report-dir <RUN>/results --dataset <LCB_release_v6.parquet> \
        --extract full --port 13835 --label py311 --out results_py311.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import msgspec

try:
    import websocket
except ImportError:
    websocket = None

from inference_endpoint.core.record import EventRecord, EventType, SampleEventType
from inference_endpoint.core.types import TextModelOutput
from inference_endpoint.dataset_manager.dataset import Dataset, DatasetFormat
from inference_endpoint.evaluation.extractor import PythonCodeExtractor
from inference_endpoint.evaluation.scoring import LiveCodeBenchScorer


def _join(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (tuple, list)):
        return "".join(str(p) for p in value)
    return str(value)


def ws_evaluate(url: str, codes_dict: dict, timeout_sec: int) -> dict:
    if websocket is None:
        sys.exit("websocket-client not installed in this venv")
    ws = websocket.create_connection(
        url, timeout=7200, ping_interval=30, ping_timeout=10
    )
    try:
        ws.send(
            msgspec.json.encode(
                {"codes_dict": codes_dict, "timeout_sec": timeout_sec}
            ).decode("utf-8")
        )
        while True:
            msg = ws.recv()
            if not msg:
                sys.exit("empty frame from lcb-service")
            data = msgspec.json.decode(msg)
            status = data.get("status")
            if status == "completed":
                return data.get("result")
            if status == "error":
                sys.exit(f"lcb-service error: {data.get('error')}")
    finally:
        try:
            ws.close()
        except Exception:  # noqa: BLE001 - ignore close errors
            pass


def resolve_name(report_dir: Path, requested: str) -> str:
    keys = list(json.load((report_dir / "sample_idx_map.json").open()).keys())
    if requested and requested != "auto" and requested in keys:
        return requested
    lcb = [k for k in keys if "livecodebench" in k.lower() or "code_bench" in k.lower()]
    if len(lcb) == 1:
        return lcb[0]
    if len(keys) == 1:
        return keys[0]
    sys.exit(f"pass --dataset-name; keys={keys}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report-dir", required=True, help="run dir holding events.jsonl")
    ap.add_argument("--dataset", required=True, help="LCB release_v6 parquet")
    ap.add_argument("--dataset-name", default="auto")
    ap.add_argument(
        "--extract",
        choices=("full", "output"),
        default="full",
        help="'full' = str(TextModelOutput) (reasoning+output); "
        "'output' = output only (fallback to reasoning if empty)",
    )
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=13835)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--lcb-version", default="release_v6")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    report_dir = Path(a.report_dir)
    name = resolve_name(report_dir, a.dataset_name)
    ds = Dataset.load_from_file(a.dataset, format=DatasetFormat.PARQUET)
    ds.load()
    assert ds.dataframe is not None

    scorer = LiveCodeBenchScorer(
        dataset_name=name,
        dataset=ds,
        report_dir=report_dir,
        extractor=PythonCodeExtractor,
        lcb_version=a.lcb_version,
        lcb_websocket_port=None,
        show_lcb_runner_output=False,
        timeout=a.timeout,
    )
    smap = scorer.sample_index_map
    qcol = scorer.question_id_column

    dec = msgspec.json.Decoder(type=EventRecord, dec_hook=EventType.decode_hook)
    codes_dict: dict[str, list[str]] = defaultdict(list)
    n_matched = 0
    with (report_dir / "events.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = dec.decode(line)
            if r.event_type != SampleEventType.COMPLETE:
                continue
            si = smap.get(r.sample_uuid)
            if si is None:
                continue
            qid = str(ds.dataframe.iloc[si][qcol])
            d = r.data
            if isinstance(d, TextModelOutput):
                if a.extract == "full":
                    text = str(d)
                else:
                    text = _join(d.output)
                    if not text.strip() and d.reasoning is not None:
                        text = _join(d.reasoning)
            else:
                text = str(d) if d is not None else ""
            code = PythonCodeExtractor.extract(text, default="# FAILED TO EXTRACT CODE")
            codes_dict[qid].append(code)
            n_matched += 1

    codes_dict = dict(codes_dict)
    print(
        f"[{a.label}] matched {n_matched} samples over {len(codes_dict)} problems"
        f" (extract={a.extract}); grading via {a.host}:{a.port}",
        flush=True,
    )

    url = f"ws://{a.host}:{a.port}/evaluate"
    result = ws_evaluate(url, codes_dict, a.timeout)
    total = int(result.get("total_samples", 0))
    per_problem = result.get("results", {})
    passed = sum(sum(v) for v in per_problem.values())
    pass_at_1 = passed / total if total else 0.0
    out = {
        "label": a.label,
        "url": url,
        "report_dir": str(report_dir),
        "dataset_name": name,
        "extract": a.extract,
        "total_samples": total,
        "total_passed": passed,
        "pass_at_1": pass_at_1,
        "results": per_problem,
    }
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"[{a.label}] pass@1={pass_at_1:.4f} passed={passed}/{total} -> {a.out}")


if __name__ == "__main__":
    main()
