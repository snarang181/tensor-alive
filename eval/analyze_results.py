#!/usr/bin/env python3
"""Analyze evaluation results and produce summary tables.

Usage:
    python analyze_results.py [results.json]
"""

import json
import os
import sys
from collections import defaultdict

import config


def load_results(path=None):
    if path is None:
        path = os.path.join(config.RESULTS_DIR, "results.json")
    with open(path) as f:
        return json.load(f)


def summary_table(results):
    """Print summary table grouped by pass."""
    by_pass = defaultdict(list)
    for r in results:
        by_pass[r["pass"]].append(r)

    header = f"{'Pass':<30} {'Total':>6} {'SrcOK':>6} {'OptOK':>6} {'TgtOK':>6} {'Equiv':>6} {'NEQ':>5} {'Unk':>5} {'P50ms':>8} {'P95ms':>8}"
    print(header)
    print("=" * len(header))

    totals = {"total": 0, "src": 0, "opt": 0, "tgt": 0, "eq": 0, "neq": 0, "unk": 0}

    for pass_name in sorted(by_pass.keys()):
        records = by_pass[pass_name]
        total = len(records)
        src_ok = sum(1 for r in records if r.get("src_parseable"))
        opt_ok = sum(1 for r in records if r.get("mlir_opt_ok"))
        tgt_ok = sum(1 for r in records if r.get("tgt_parseable"))
        equiv = sum(1 for r in records if r["status"] == "equivalent")
        neq = sum(1 for r in records if r["status"] == "not_equivalent")
        unk = sum(1 for r in records if r["status"] in ("unknown", "timeout"))

        times = sorted(r["time_ms"] for r in records if r["time_ms"] > 0)
        p50 = times[len(times) // 2] if times else 0
        p95 = times[int(len(times) * 0.95)] if times else 0

        print(
            f"{pass_name:<30} {total:>6} {src_ok:>6} {opt_ok:>6} {tgt_ok:>6} {equiv:>6} {neq:>5} {unk:>5} {p50:>8.1f} {p95:>8.1f}"
        )

        totals["total"] += total
        totals["src"] += src_ok
        totals["opt"] += opt_ok
        totals["tgt"] += tgt_ok
        totals["eq"] += equiv
        totals["neq"] += neq
        totals["unk"] += unk

    print("=" * len(header))
    print(
        f"{'TOTAL':<30} {totals['total']:>6} {totals['src']:>6} {totals['opt']:>6} {totals['tgt']:>6} {totals['eq']:>6} {totals['neq']:>5} {totals['unk']:>5}"
    )


def latex_table(results):
    """Generate LaTeX table."""
    by_pass = defaultdict(list)
    for r in results:
        by_pass[r["pass"]].append(r)

    print("\n% LaTeX table")
    print("\\begin{tabular}{lrrrrrrr}")
    print("\\toprule")
    print("Pass & Total & Src OK & Tgt OK & Equiv & NEQ & Unk & P50 (ms) \\\\")
    print("\\midrule")

    for pass_name in sorted(by_pass.keys()):
        records = by_pass[pass_name]
        total = len(records)
        src_ok = sum(1 for r in records if r.get("src_parseable"))
        tgt_ok = sum(1 for r in records if r.get("tgt_parseable"))
        equiv = sum(1 for r in records if r["status"] == "equivalent")
        neq = sum(1 for r in records if r["status"] == "not_equivalent")
        unk = sum(1 for r in records if r["status"] in ("unknown", "timeout"))
        times = sorted(r["time_ms"] for r in records if r["time_ms"] > 0)
        p50 = times[len(times) // 2] if times else 0

        name_escaped = pass_name.replace("_", "\\_")
        print(
            f"{name_escaped} & {total} & {src_ok} & {tgt_ok} & {equiv} & {neq} & {unk} & {p50:.1f} \\\\"
        )

    print("\\bottomrule")
    print("\\end{tabular}")


def not_equivalent_report(results):
    """Report all not-equivalent findings."""
    neq = [r for r in results if r["status"] == "not_equivalent"]
    if not neq:
        print("\nNo not-equivalent findings.")
        return

    print(f"\n=== NOT EQUIVALENT FINDINGS ({len(neq)}) ===")
    print("These require manual investigation:\n")
    for r in neq:
        print(f"  Function: {r['function']}")
        print(f"  Pass:     {r['pass']}")
        print(f"  Output:   {r.get('output', 'N/A')[:200]}")
        if r.get("source_file"):
            print(f"  Source:   {r['source_file']}")
        print()


def coverage_analysis(results):
    """Analyze what fraction of functions are in scope."""
    statuses = defaultdict(int)
    for r in results:
        statuses[r["status"]] += 1

    total = len(results)
    print(f"\n=== Coverage Analysis ({total} total pairs) ===")
    for status in sorted(statuses.keys()):
        count = statuses[status]
        pct = 100 * count / total if total else 0
        print(f"  {status:<20} {count:>6} ({pct:>5.1f}%)")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    results = load_results(path)
    print(f"Loaded {len(results)} results\n")
    summary_table(results)
    latex_table(results)
    not_equivalent_report(results)
    coverage_analysis(results)


if __name__ == "__main__":
    main()
