#!/usr/bin/env python3
"""Differential testing across MLIR pass orderings.

Runs the same MLIR input through different pass orderings and checks that
all outputs are semantically equivalent using tensor-alive.

Bug patterns this catches:
- Pass A then B != Pass B then A (ordering sensitivity)
- Pass A applied twice != Pass A applied once (non-idempotence)
- Pipeline P1 != Pipeline P2 for semantically equivalent pipelines
"""

import itertools
import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

# Add parent dir for config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
import config


# ---------------------------------------------------------------------------
# Pass ordering configurations
# ---------------------------------------------------------------------------

# Individual passes to combine
INDIVIDUAL_PASSES = [
    ("canonicalize", "--canonicalize"),
    ("fuse-elementwise", "--linalg-fuse-elementwise-ops"),
    ("fold-unit-dims", "--linalg-fold-unit-extent-dims"),
    ("generalize", "--linalg-generalize-named-ops"),
    ("fold-into-elem", "--linalg-fold-into-elementwise"),
    ("elem-to-linalg", "--convert-elementwise-to-linalg"),
]

# Pre-defined pipelines to test (each is a list of pass flags)
PIPELINES = {
    # Ordering variants
    "canon_then_fuse": ["--canonicalize", "--linalg-fuse-elementwise-ops"],
    "fuse_then_canon": ["--linalg-fuse-elementwise-ops", "--canonicalize"],
    "canon_then_fold_unit": ["--canonicalize", "--linalg-fold-unit-extent-dims"],
    "fold_unit_then_canon": ["--linalg-fold-unit-extent-dims", "--canonicalize"],
    "gen_then_canon": ["--linalg-generalize-named-ops", "--canonicalize"],
    "gen_then_fuse": ["--linalg-generalize-named-ops", "--linalg-fuse-elementwise-ops"],
    # Idempotence tests
    "canon_once": ["--canonicalize"],
    "canon_twice": ["--canonicalize", "--canonicalize"],
    "canon_thrice": ["--canonicalize", "--canonicalize", "--canonicalize"],
    "fuse_once": ["--linalg-fuse-elementwise-ops"],
    "fuse_twice": ["--linalg-fuse-elementwise-ops", "--linalg-fuse-elementwise-ops"],
    # Three-pass combos
    "gen_canon_fuse": [
        "--linalg-generalize-named-ops",
        "--canonicalize",
        "--linalg-fuse-elementwise-ops",
    ],
    "gen_fuse_canon": [
        "--linalg-generalize-named-ops",
        "--linalg-fuse-elementwise-ops",
        "--canonicalize",
    ],
    # Longer pipelines
    "full_pipeline_a": [
        "--linalg-generalize-named-ops",
        "--linalg-fuse-elementwise-ops",
        "--linalg-fold-unit-extent-dims",
        "--canonicalize",
    ],
    "full_pipeline_b": [
        "--canonicalize",
        "--linalg-generalize-named-ops",
        "--linalg-fold-unit-extent-dims",
        "--linalg-fuse-elementwise-ops",
        "--canonicalize",
    ],
}


def run_pipeline(mlir_text: str, passes: List[str]) -> Tuple[Optional[str], str]:
    """Run a sequence of mlir-opt passes. Returns (result_text, error)."""
    current = mlir_text
    for pass_flag in passes:
        try:
            result = subprocess.run(
                [config.MLIR_OPT, pass_flag],
                input=current,
                capture_output=True,
                text=True,
                timeout=config.TIMEOUT_MLIR_OPT,
            )
            if result.returncode != 0:
                return None, f"mlir-opt {pass_flag} failed: {result.stderr[:200]}"
            current = result.stdout
        except subprocess.TimeoutExpired:
            return None, f"mlir-opt {pass_flag} timeout"
        except Exception as e:
            return None, str(e)
    return current, ""


def check_equivalence(mlir_a: str, mlir_b: str, timeout_ms: int = 30000) -> Dict:
    """Check if two MLIR programs are equivalent using tensor-alive."""
    with (
        tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as fa,
        tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as fb,
    ):
        fa.write(mlir_a)
        fa.flush()
        fb.write(mlir_b)
        fb.flush()
        try:
            import re

            result = subprocess.run(
                [config.TENSOR_ALIVE, fa.name, fb.name, "--timeout", str(timeout_ms)],
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
            os.unlink(fa.name)
            os.unlink(fb.name)


def diff_test_program(
    mlir_text: str,
    program_name: str,
    pipeline_pairs: Optional[List[Tuple[str, str]]] = None,
) -> List[Dict]:
    """Run differential testing on a single program across pipeline pairs.

    If pipeline_pairs is None, tests all pairwise combinations of PIPELINES.
    Returns list of findings (only non-equivalent or errors).
    """
    if pipeline_pairs is None:
        # Default: test ordering pairs + idempotence
        pipeline_pairs = [
            ("canon_then_fuse", "fuse_then_canon"),
            ("canon_then_fold_unit", "fold_unit_then_canon"),
            ("canon_once", "canon_twice"),
            ("canon_twice", "canon_thrice"),
            ("fuse_once", "fuse_twice"),
            ("gen_canon_fuse", "gen_fuse_canon"),
            ("full_pipeline_a", "full_pipeline_b"),
        ]

    findings = []

    # Run all pipelines and cache results
    pipeline_results = {}
    needed = set()
    for a, b in pipeline_pairs:
        needed.add(a)
        needed.add(b)

    for name in needed:
        if name not in PIPELINES:
            continue
        result, err = run_pipeline(mlir_text, PIPELINES[name])
        pipeline_results[name] = (result, err)

    # Compare pairs
    for name_a, name_b in pipeline_pairs:
        if name_a not in pipeline_results or name_b not in pipeline_results:
            continue

        result_a, err_a = pipeline_results[name_a]
        result_b, err_b = pipeline_results[name_b]

        if result_a is None or result_b is None:
            # One pipeline failed — skip (not a tensor-alive finding)
            continue

        # Quick textual check — if identical, skip expensive SMT check
        if result_a.strip() == result_b.strip():
            continue

        # Check semantic equivalence
        equiv = check_equivalence(result_a, result_b)

        if equiv["status"] == "not_equivalent":
            findings.append(
                {
                    "program": program_name,
                    "pipeline_a": name_a,
                    "pipeline_b": name_b,
                    "status": "NOT_EQUIVALENT",
                    "output": equiv["output"],
                    "time_ms": equiv["time_ms"],
                    "pipeline_a_passes": PIPELINES[name_a],
                    "pipeline_b_passes": PIPELINES[name_b],
                    "result_a": result_a[:500],
                    "result_b": result_b[:500],
                }
            )
        elif equiv["status"] not in ("equivalent", "parse_error", "shape_mismatch"):
            findings.append(
                {
                    "program": program_name,
                    "pipeline_a": name_a,
                    "pipeline_b": name_b,
                    "status": equiv["status"],
                    "output": equiv["output"],
                    "time_ms": equiv["time_ms"],
                }
            )

    return findings


def generate_all_pairs(
    pass_names: List[str], max_len: int = 2
) -> List[Tuple[str, str]]:
    """Generate all pairs of pass orderings up to max_len passes."""
    pairs = []
    for length in range(1, max_len + 1):
        perms = list(itertools.permutations(pass_names, length))
        for i, p1 in enumerate(perms):
            for p2 in perms[i + 1 :]:
                if set(p1) == set(p2) and p1 != p2:  # Same passes, different order
                    pairs.append(("_".join(p1), "_".join(p2)))
    return pairs


if __name__ == "__main__":
    example = """func.func @test(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %init: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %add = arith.addf %a, %b : f32
    linalg.yield %add : f32
  } -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
"""
    print("Running differential testing on example program...")
    findings = diff_test_program(example, "example_add")
    if findings:
        print(f"\n=== FINDINGS ({len(findings)}) ===")
        for f in findings:
            print(
                f"  {f['program']}: {f['pipeline_a']} vs {f['pipeline_b']} -> {f['status']}"
            )
            if "output" in f:
                print(f"    {f['output'][:200]}")
    else:
        print("No discrepancies found.")
