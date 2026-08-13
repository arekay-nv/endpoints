#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diff two LCB replay results (e.g. py314 baseline vs py311 variant).

Aligns per-problem, per-sample pass/fail across the two replays. Because both
replays graded the SAME captured codes_dict, index alignment within a problem is
valid and every difference is attributable to the container's Python version.

    python lcb_py_ab/compare_lcb.py --a results_py314.json --b results_py311.json
"""

import argparse
import json
from pathlib import Path


def _load(p: str) -> dict:
    return json.loads(Path(p).read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="baseline results json (e.g. py314)")
    ap.add_argument("--b", required=True, help="variant results json (e.g. py311)")
    ap.add_argument("--max-list", type=int, default=200, help="max flips to print")
    args = ap.parse_args()

    A, B = _load(args.a), _load(args.b)
    la = A.get("label") or "A"
    lb = B.get("label") or "B"

    print(f"{'':24}{la:>12}{lb:>12}")
    print(f"{'pass@1':24}{A['pass_at_1']:12.4f}{B['pass_at_1']:12.4f}")
    print(
        f"{'passed/total':24}"
        f"{str(A['total_passed']) + '/' + str(A['total_samples']):>12}"
        f"{str(B['total_passed']) + '/' + str(B['total_samples']):>12}"
    )
    print(f"{'delta pass@1 (b - a)':24}{B['pass_at_1'] - A['pass_at_1']:>+12.4f}")

    ra, rb = A["results"], B["results"]
    qids = sorted(set(ra) | set(rb))
    flips: list[tuple[str, int, bool, bool]] = []
    n_common = a_pass_b_fail = a_fail_b_pass = 0
    for q in qids:
        va, vb = ra.get(q, []), rb.get(q, [])
        for i in range(min(len(va), len(vb))):
            n_common += 1
            pa, pb = bool(va[i]), bool(vb[i])
            if pa != pb:
                flips.append((q, i, pa, pb))
                if pa and not pb:
                    a_pass_b_fail += 1
                else:
                    a_fail_b_pass += 1

    print(f"\ncompared samples: {n_common}")
    print(
        f"flips: {len(flips)}  "
        f"(pass-on-{la}/fail-on-{lb}: {a_pass_b_fail}   "
        f"fail-on-{la}/pass-on-{lb}: {a_fail_b_pass})"
    )
    for q, i, pa, pb in flips[: args.max_list]:
        print(
            f"  {q}[{i}]: {la}={'PASS' if pa else 'FAIL'} -> "
            f"{lb}={'PASS' if pb else 'FAIL'}"
        )
    if len(flips) > args.max_list:
        print(f"  ... {len(flips) - args.max_list} more")


if __name__ == "__main__":
    main()
