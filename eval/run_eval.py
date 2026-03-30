#!/usr/bin/env python3
"""Main evaluation orchestrator for tensor-alive.

Runs tensor-alive against real MLIR compiler passes and collects results.

Usage:
    python run_eval.py --synthetic     # Run on synthetic benchmarks only
    python run_eval.py --corpus        # Run on MLIR test corpus only
    python run_eval.py                 # Run both
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import config
from extract_functions import extract_corpus, filter_supported
from preprocess import preprocess


def run_mlir_opt(source_mlir: str, pass_flag: str) -> Tuple[Optional[str], str]:
    """Run mlir-opt on source MLIR text. Returns (output_text, error_msg)."""
    try:
        result = subprocess.run(
            [config.MLIR_OPT, pass_flag],
            input=source_mlir,
            capture_output=True,
            text=True,
            timeout=config.TIMEOUT_MLIR_OPT,
        )
        if result.returncode == 0:
            return result.stdout, ""
        return None, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return None, "mlir-opt timeout"
    except Exception as e:
        return None, str(e)


def run_tensor_alive(src_path: str, tgt_path: str, timeout_ms: int = 60000) -> Dict:
    """Run tensor-alive on two files. Returns result dict."""
    try:
        result = subprocess.run(
            [config.TENSOR_ALIVE, src_path, tgt_path, "--timeout", str(timeout_ms)],
            capture_output=True,
            text=True,
            timeout=config.TIMEOUT_TENSOR_ALIVE + 5,
        )
        output = result.stdout.strip() + result.stderr.strip()

        status_map = {
            0: "equivalent",
            1: "not_equivalent",
            2: "unknown",
            3: "shape_mismatch",
            4: "parse_error",
        }
        status = status_map.get(result.returncode, f"exit_{result.returncode}")

        # Extract timing
        time_ms = 0.0
        import re

        m = re.search(r"solved in ([\d.]+)ms", output)
        if m:
            time_ms = float(m.group(1))

        return {"status": status, "time_ms": time_ms, "output": output}
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "time_ms": config.TIMEOUT_TENSOR_ALIVE * 1000,
            "output": "timeout",
        }
    except Exception as e:
        return {"status": "error", "time_ms": 0, "output": str(e)}


def check_parseable(mlir_text: str) -> bool:
    """Check if tensor-alive can parse this MLIR text."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
        f.write(mlir_text)
        f.flush()
        try:
            result = subprocess.run(
                [config.TENSOR_ALIVE, "--parse-only", f.name],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except:
            return False
        finally:
            os.unlink(f.name)


def eval_single(
    func_name: str, source_mlir: str, pass_name: str, pass_flag: str
) -> Dict:
    """Evaluate a single (function, pass) pair."""
    record = {
        "function": func_name,
        "pass": pass_name,
        "src_parseable": True,
        "tgt_parseable": True,
        "mlir_opt_ok": False,
        "status": "skipped",
        "time_ms": 0.0,
        "output": "",
    }

    # Run mlir-opt
    transformed, err = run_mlir_opt(source_mlir, pass_flag)
    if transformed is None:
        record["status"] = "mlir_opt_error"
        record["output"] = err
        return record
    record["mlir_opt_ok"] = True

    # Run tensor-alive directly — it reports parse errors via exit code
    with (
        tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as sf,
        tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tf,
    ):
        sf.write(source_mlir)
        sf.flush()
        tf.write(transformed)
        tf.flush()
        try:
            result = run_tensor_alive(
                sf.name, tf.name, timeout_ms=config.TIMEOUT_TENSOR_ALIVE * 1000
            )
            record.update(result)
            if record["status"] == "parse_error":
                out = record.get("output", "")
                if "Source:" in out:
                    record["src_parseable"] = False
                    record["status"] = "src_parse_error"
                else:
                    record["tgt_parseable"] = False
                    record["status"] = "tgt_parse_error"
        finally:
            os.unlink(sf.name)
            os.unlink(tf.name)

    return record


def run_corpus_eval(
    functions: List[Tuple[str, str, str]], passes: List[Tuple[str, str]]
) -> List[Dict]:
    """Run evaluation on corpus × passes."""
    results = []
    total = len(functions) * len(passes)
    done = 0

    for func_name, source_mlir, source_file in functions:
        for pass_name, pass_flag in passes:
            record = eval_single(func_name, source_mlir, pass_name, pass_flag)
            record["source_file"] = source_file
            results.append(record)
            done += 1
            status = record["status"]
            sym = {
                "equivalent": "+",
                "not_equivalent": "!",
                "timeout": "T",
                "parse_error": "P",
                "tgt_parse_error": "p",
                "src_parse_error": "s",
                "mlir_opt_error": "m",
                "unknown": "?",
                "shape_mismatch": "S",
            }.get(status, ".")
            print(f"\r  [{done}/{total}] {sym}", end="", flush=True)

    print()
    return results


def run_synthetic_eval() -> List[Dict]:
    """Run evaluation on synthetic benchmarks."""
    synth_dir = config.SYNTH_DIR
    if not os.path.isdir(synth_dir):
        print(f"No synthetic benchmarks at {synth_dir}. Run synth_benchmarks.py first.")
        return []

    results = []
    src_files = sorted(f for f in os.listdir(synth_dir) if f.endswith(".src.mlir"))

    for src_name in src_files:
        base = src_name.replace(".src.mlir", "")
        src_path = os.path.join(synth_dir, src_name)
        tgt_path = os.path.join(synth_dir, base + ".tgt.mlir")
        if not os.path.exists(tgt_path):
            continue

        # Read the metadata (pass name) from filename convention: category_params_pass
        parts = base.rsplit("_", 1)
        pass_name = parts[-1] if len(parts) > 1 else "unknown"

        result = run_tensor_alive(src_path, tgt_path)
        result["function"] = base
        result["pass"] = pass_name
        result["src_parseable"] = True
        result["tgt_parseable"] = True
        result["mlir_opt_ok"] = True
        results.append(result)

    return results


def print_summary(results: List[Dict]):
    """Print a summary table of results."""
    from collections import Counter

    # Group by pass
    by_pass = {}
    for r in results:
        p = r["pass"]
        if p not in by_pass:
            by_pass[p] = []
        by_pass[p].append(r)

    print(
        f"\n{'Pass':<30} {'Total':>6} {'SrcOK':>6} {'TgtOK':>6} {'Equiv':>6} {'NEQ':>5} {'Unk':>5} {'Time(ms)':>10}"
    )
    print("-" * 90)

    for pass_name in sorted(by_pass.keys()):
        records = by_pass[pass_name]
        total = len(records)
        src_ok = sum(1 for r in records if r.get("src_parseable"))
        tgt_ok = sum(1 for r in records if r.get("tgt_parseable"))
        equiv = sum(1 for r in records if r["status"] == "equivalent")
        neq = sum(1 for r in records if r["status"] == "not_equivalent")
        unk = sum(1 for r in records if r["status"] in ("unknown", "timeout"))
        times = [r["time_ms"] for r in records if r["time_ms"] > 0]
        avg_time = sum(times) / len(times) if times else 0

        print(
            f"{pass_name:<30} {total:>6} {src_ok:>6} {tgt_ok:>6} {equiv:>6} {neq:>5} {unk:>5} {avg_time:>10.1f}"
        )

    # Overall
    total = len(results)
    equiv = sum(1 for r in results if r["status"] == "equivalent")
    neq = sum(1 for r in results if r["status"] == "not_equivalent")
    print(f"\nOverall: {total} pairs, {equiv} equivalent, {neq} not-equivalent")

    # Report any not-equivalent findings
    neq_records = [r for r in results if r["status"] == "not_equivalent"]
    if neq_records:
        print(f"\n=== NOT EQUIVALENT FINDINGS ({len(neq_records)}) ===")
        for r in neq_records:
            print(f"  {r['function']} [{r['pass']}]: {r.get('output', '')[:100]}")


def main():
    parser = argparse.ArgumentParser(description="tensor-alive evaluation pipeline")
    parser.add_argument(
        "--synthetic", action="store_true", help="Run synthetic benchmarks only"
    )
    parser.add_argument("--corpus", action="store_true", help="Run MLIR corpus only")
    args = parser.parse_args()

    run_both = not args.synthetic and not args.corpus
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    all_results = []

    if args.corpus or run_both:
        print("=== MLIR Corpus Evaluation ===")
        print("Extracting functions...")
        all_funcs = extract_corpus(config.MLIR_TEST_DIRS)
        supported = filter_supported(all_funcs)
        print(f"  {len(all_funcs)} total chunks, {len(supported)} in supported subset")
        print("Running evaluation...")
        corpus_results = run_corpus_eval(supported, config.PASSES)
        all_results.extend(corpus_results)

    if args.synthetic or run_both:
        print("\n=== Synthetic Benchmark Evaluation ===")
        synth_results = run_synthetic_eval()
        all_results.extend(synth_results)

    # Save results
    results_path = os.path.join(config.RESULTS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
