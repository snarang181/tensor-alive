#!/usr/bin/env python3
"""Main fuzzing harness for finding MLIR bugs with tensor-alive.

Combines three fuzzing strategies:
1. Random generation — diverse programs from gen_random.py
2. Mutation — mutate MLIR test corpus seeds via mutate.py
3. Differential — test pass ordering equivalence via diff_test.py

Usage:
    python fuzz.py                         # Run all strategies (default 100 programs)
    python fuzz.py --mode random -n 200    # Random generation only
    python fuzz.py --mode mutate -n 100    # Mutational fuzzing only
    python fuzz.py --mode diff -n 50       # Differential testing only
    python fuzz.py --mode all -n 300       # All strategies
    python fuzz.py --seed 123              # Set RNG seed
    python fuzz.py --timeout 60000         # Set Z3 timeout in ms
    python fuzz.py --save-all              # Save all generated programs (not just bugs)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

import config
from gen_random import generate_batch, ALL_GENERATORS
from mutate import apply_n_mutations, PRESERVING_MUTATIONS, BREAKING_MUTATIONS
from diff_test import diff_test_program, PIPELINES
from extract_functions import extract_corpus, filter_supported
from preprocess import is_supported


# ---------------------------------------------------------------------------
# Core fuzzing operations
# ---------------------------------------------------------------------------


def run_mlir_opt(source: str, pass_flag: str) -> Tuple[Optional[str], str]:
    """Run mlir-opt. Returns (output, error)."""
    try:
        result = subprocess.run(
            [config.MLIR_OPT, pass_flag],
            input=source,
            capture_output=True,
            text=True,
            timeout=config.TIMEOUT_MLIR_OPT,
        )
        if result.returncode == 0:
            return result.stdout, ""
        return None, result.stderr[:200]
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except Exception as e:
        return None, str(e)


def run_tensor_alive(src_text: str, tgt_text: str, timeout_ms: int = 30000) -> Dict:
    """Run tensor-alive on two MLIR texts. Returns result dict."""
    with (
        tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as sf,
        tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tf,
    ):
        sf.write(src_text)
        sf.flush()
        tf.write(tgt_text)
        tf.flush()
        try:
            result = subprocess.run(
                [config.TENSOR_ALIVE, sf.name, tf.name, "--timeout", str(timeout_ms)],
                capture_output=True,
                text=True,
                timeout=timeout_ms // 1000 + 10,
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

            time_ms = 0.0
            m = re.search(r"solved in ([\d.]+)ms", output)
            if m:
                time_ms = float(m.group(1))

            return {"status": status, "time_ms": time_ms, "output": output}
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "time_ms": timeout_ms, "output": "timeout"}
        except Exception as e:
            return {"status": "error", "time_ms": 0, "output": str(e)}
        finally:
            os.unlink(sf.name)
            os.unlink(tf.name)


# ---------------------------------------------------------------------------
# Bug reporting
# ---------------------------------------------------------------------------


class BugReporter:
    """Collects and saves fuzzing findings."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.bugs = []
        self.stats = {
            "total_programs": 0,
            "total_checks": 0,
            "equivalent": 0,
            "not_equivalent": 0,
            "unknown": 0,
            "timeout": 0,
            "parse_error": 0,
            "mlir_opt_error": 0,
            "shape_mismatch": 0,
        }
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "bugs"), exist_ok=True)

    def record_check(self, status: str):
        self.stats["total_checks"] += 1
        if status in self.stats:
            self.stats[status] += 1

    def report_bug(self, bug: Dict):
        """Record and save a potential bug."""
        bug_id = len(self.bugs)
        bug["id"] = bug_id
        bug["timestamp"] = datetime.now().isoformat()
        self.bugs.append(bug)

        # Save bug details
        bug_dir = os.path.join(self.output_dir, "bugs", f"bug_{bug_id:04d}")
        os.makedirs(bug_dir, exist_ok=True)

        if "source_mlir" in bug:
            with open(os.path.join(bug_dir, "source.mlir"), "w") as f:
                f.write(bug["source_mlir"])
        if "target_mlir" in bug:
            with open(os.path.join(bug_dir, "target.mlir"), "w") as f:
                f.write(bug["target_mlir"])

        info = {k: v for k, v in bug.items() if k not in ("source_mlir", "target_mlir")}
        with open(os.path.join(bug_dir, "info.json"), "w") as f:
            json.dump(info, f, indent=2)

        kind = bug.get("kind", "unknown")
        name = bug.get("program_name", "?")
        print(f"  BUG #{bug_id} [{kind}] {name}: {bug.get('description', '')}")

    def save_summary(self):
        summary = {
            "stats": self.stats,
            "bugs_found": len(self.bugs),
            "bug_summaries": [
                {
                    "id": b["id"],
                    "kind": b.get("kind"),
                    "program_name": b.get("program_name"),
                    "description": b.get("description"),
                    "pass": b.get("pass_name"),
                }
                for b in self.bugs
            ],
        }
        with open(os.path.join(self.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    def print_summary(self):
        s = self.stats
        print(f"\n{'=' * 60}")
        print(f"FUZZING SUMMARY")
        print(f"{'=' * 60}")
        print(f"Programs generated:  {s['total_programs']}")
        print(f"Equivalence checks:  {s['total_checks']}")
        print(f"  Equivalent:        {s['equivalent']}")
        print(f"  NOT equivalent:    {s['not_equivalent']}")
        print(f"  Unknown/timeout:   {s['unknown'] + s['timeout']}")
        print(f"  Parse errors:      {s['parse_error']}")
        print(f"  Shape mismatches:  {s['shape_mismatch']}")
        print(f"  mlir-opt errors:   {s['mlir_opt_error']}")
        print(f"{'=' * 60}")
        print(f"POTENTIAL BUGS FOUND: {len(self.bugs)}")
        if self.bugs:
            for b in self.bugs:
                print(
                    f"  #{b['id']}: [{b.get('kind')}] {b.get('program_name')} — {b.get('description', '')[:80]}"
                )
        print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Fuzzing strategy: Random generation
# ---------------------------------------------------------------------------


def fuzz_random(
    n: int,
    seed: int,
    timeout_ms: int,
    reporter: BugReporter,
    save_all: bool,
    output_dir: str,
):
    """Generate random programs, run through passes, check equivalence."""
    print(f"\n--- Random Generation ({n} programs) ---")
    programs = generate_batch(n, seed)
    reporter.stats["total_programs"] += len(programs)

    for i, prog in enumerate(programs):
        if save_all:
            prog_dir = os.path.join(output_dir, "all_programs", prog.name)
            os.makedirs(prog_dir, exist_ok=True)
            with open(os.path.join(prog_dir, "source.mlir"), "w") as f:
                f.write(prog.mlir)

        for pass_flag in prog.suggested_passes:
            target, err = run_mlir_opt(prog.mlir, pass_flag)
            if target is None:
                reporter.record_check("mlir_opt_error")
                continue

            result = run_tensor_alive(prog.mlir, target, timeout_ms)
            reporter.record_check(result["status"])

            if result["status"] == "not_equivalent":
                reporter.report_bug(
                    {
                        "kind": "pass_unsound",
                        "program_name": prog.name,
                        "pass_name": pass_flag,
                        "description": f"Pass {pass_flag} produced non-equivalent output",
                        "source_mlir": prog.mlir,
                        "target_mlir": target,
                        "counterexample": result["output"],
                        "time_ms": result["time_ms"],
                    }
                )

            if save_all and target:
                prog_dir = os.path.join(output_dir, "all_programs", prog.name)
                os.makedirs(prog_dir, exist_ok=True)
                pass_name = pass_flag.lstrip("-").replace("-", "_")
                with open(os.path.join(prog_dir, f"target_{pass_name}.mlir"), "w") as f:
                    f.write(target)

        # Progress
        sym = "." if i % 10 != 9 else str((i + 1) // 10 % 10)
        print(sym, end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# Fuzzing strategy: Mutation-based
# ---------------------------------------------------------------------------


def fuzz_mutate(
    n: int, seed: int, timeout_ms: int, reporter: BugReporter, output_dir: str
):
    """Mutate MLIR corpus seeds and check equivalence."""
    print(f"\n--- Mutational Fuzzing ({n} mutations) ---")

    # Collect seeds from MLIR test corpus
    print("  Extracting corpus seeds...")
    all_funcs = extract_corpus(config.MLIR_TEST_DIRS)
    supported = filter_supported(all_funcs)
    print(f"  {len(supported)} supported functions from corpus")

    if not supported:
        print(
            "  No supported functions found. Falling back to random generation seeds."
        )
        programs = generate_batch(min(n, 20), seed)
        supported = [(p.name, p.mlir, "generated") for p in programs]

    import random

    rng = random.Random(seed)
    mutations_done = 0

    while mutations_done < n and supported:
        # Pick a random seed program
        func_name, source_mlir, source_file = rng.choice(supported)

        # Apply mutations
        results = apply_n_mutations(
            source_mlir, min(3, n - mutations_done), seed=seed + mutations_done
        )

        for mutated_mlir, mutation_name, is_breaking in results:
            reporter.stats["total_programs"] += 1

            # For preserving mutations: run pass on both original and mutated,
            # check that pass(original) ≡ pass(mutated) is consistent
            for pass_name, pass_flag in config.PASSES[:3]:  # Test top 3 passes
                target_orig, err1 = run_mlir_opt(source_mlir, pass_flag)
                target_mut, err2 = run_mlir_opt(mutated_mlir, pass_flag)

                if target_orig is None or target_mut is None:
                    reporter.record_check("mlir_opt_error")
                    continue

                if not is_breaking:
                    # Shape-mutated program: check pass(mutated_src) ≡ mutated_src
                    result = run_tensor_alive(mutated_mlir, target_mut, timeout_ms)
                    reporter.record_check(result["status"])

                    if result["status"] == "not_equivalent":
                        reporter.report_bug(
                            {
                                "kind": "pass_unsound_mutated",
                                "program_name": f"{func_name}_mut_{mutation_name}",
                                "pass_name": pass_flag,
                                "mutation": mutation_name,
                                "description": f"Pass {pass_flag} non-equivalent on mutated program ({mutation_name})",
                                "source_mlir": mutated_mlir,
                                "target_mlir": target_mut,
                                "counterexample": result["output"],
                                "original_source": source_file,
                            }
                        )
                else:
                    # Breaking mutation: check that original ≢ mutated
                    # If pass makes them "equivalent", that's suspicious
                    result = run_tensor_alive(source_mlir, mutated_mlir, timeout_ms)
                    reporter.record_check(result["status"])

                    if result["status"] == "equivalent":
                        # The breaking mutation didn't change semantics?
                        # This might be expected (e.g., the mutation was in dead code)
                        # or it might indicate a tensor-alive false positive.
                        # Log but don't flag as bug.
                        pass

            mutations_done += 1
            if mutations_done % 10 == 0:
                print(f"  [{mutations_done}/{n}]", end="", flush=True)

    print()


# ---------------------------------------------------------------------------
# Fuzzing strategy: Differential pass ordering
# ---------------------------------------------------------------------------


def fuzz_differential(
    n: int, seed: int, timeout_ms: int, reporter: BugReporter, output_dir: str
):
    """Test pass ordering equivalence on random + corpus programs."""
    print(f"\n--- Differential Testing ({n} programs) ---")

    # Mix of random and corpus programs
    programs = []

    # Random programs
    random_progs = generate_batch(n // 2, seed)
    for p in random_progs:
        programs.append((p.name, p.mlir))

    # Corpus programs
    all_funcs = extract_corpus(config.MLIR_TEST_DIRS)
    supported = filter_supported(all_funcs)
    import random as rng_mod

    rng = rng_mod.Random(seed)
    if supported:
        sample_size = min(n - n // 2, len(supported))
        corpus_sample = rng.sample(supported, sample_size)
        for func_name, mlir, _ in corpus_sample:
            programs.append((func_name, mlir))

    reporter.stats["total_programs"] += len(programs)

    for i, (name, mlir) in enumerate(programs):
        findings = diff_test_program(mlir, name)

        for f in findings:
            reporter.stats["total_checks"] += 1
            if f["status"] == "NOT_EQUIVALENT":
                reporter.record_check("not_equivalent")
                reporter.report_bug(
                    {
                        "kind": "pass_ordering",
                        "program_name": name,
                        "description": f"Pipeline {f['pipeline_a']} vs {f['pipeline_b']} differ",
                        "pipeline_a": f["pipeline_a"],
                        "pipeline_b": f["pipeline_b"],
                        "source_mlir": f.get("result_a", ""),
                        "target_mlir": f.get("result_b", ""),
                        "counterexample": f.get("output", ""),
                    }
                )
            else:
                reporter.record_check(f["status"])

        if (i + 1) % 5 == 0:
            print(f"  [{i + 1}/{len(programs)}]", end="", flush=True)

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fuzz MLIR passes using tensor-alive translation validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fuzz.py                          # All strategies, 100 programs
  python fuzz.py --mode random -n 500     # 500 random programs
  python fuzz.py --mode diff -n 50        # Differential testing on 50 programs
  python fuzz.py --seed 0 -n 1000         # Reproducible large run
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["random", "mutate", "diff", "all"],
        default="all",
        help="Fuzzing strategy (default: all)",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=100,
        help="Number of programs to generate per strategy (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30000,
        help="Z3 solver timeout in ms (default: 30000)",
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Save all generated programs, not just bugs",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output directory (default: fuzz/results/<timestamp>)",
    )
    args = parser.parse_args()

    # Output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(__file__), "results", timestamp)

    print(f"Tensor-Alive MLIR Fuzzer")
    print(f"  Mode: {args.mode}")
    print(f"  Programs per strategy: {args.n}")
    print(f"  Seed: {args.seed}")
    print(f"  Z3 timeout: {args.timeout}ms")
    print(f"  Output: {output_dir}")

    # Verify binaries exist
    if not os.path.isfile(config.TENSOR_ALIVE):
        print(f"\nERROR: tensor-alive not found at {config.TENSOR_ALIVE}")
        print("  Build with: cd build && cmake .. && make")
        sys.exit(1)
    if not os.path.isfile(config.MLIR_OPT):
        print(f"\nERROR: mlir-opt not found at {config.MLIR_OPT}")
        sys.exit(1)

    reporter = BugReporter(output_dir)
    start = time.time()

    if args.mode in ("random", "all"):
        fuzz_random(
            args.n, args.seed, args.timeout, reporter, args.save_all, output_dir
        )

    if args.mode in ("mutate", "all"):
        fuzz_mutate(args.n, args.seed + 10000, args.timeout, reporter, output_dir)

    if args.mode in ("diff", "all"):
        fuzz_differential(args.n, args.seed + 20000, args.timeout, reporter, output_dir)

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.1f}s")

    reporter.save_summary()
    reporter.print_summary()
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
