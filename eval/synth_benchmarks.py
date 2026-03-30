#!/usr/bin/env python3
"""Generate synthetic MLIR benchmark programs for tensor-alive evaluation.

Each benchmark is a source .mlir file. The target is produced by running
mlir-opt with the specified pass.
"""

import os
import subprocess
import sys
from itertools import product
from typing import List, Tuple

import config


def write_pair(name: str, source: str, pass_flag: str):
    """Write source and generate target via mlir-opt."""
    os.makedirs(config.SYNTH_DIR, exist_ok=True)
    src_path = os.path.join(config.SYNTH_DIR, f"{name}.src.mlir")
    tgt_path = os.path.join(config.SYNTH_DIR, f"{name}.tgt.mlir")

    with open(src_path, "w") as f:
        f.write(source)

    try:
        result = subprocess.run(
            [config.MLIR_OPT, pass_flag],
            input=source,
            capture_output=True,
            text=True,
            timeout=config.TIMEOUT_MLIR_OPT,
        )
        if result.returncode == 0 and result.stdout.strip():
            with open(tgt_path, "w") as f:
                f.write(result.stdout)
            return True
        else:
            print(f"  Warning: mlir-opt failed for {name}: {result.stderr[:100]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  Warning: mlir-opt timeout for {name}")
        return False


def gen_elementwise_chains() -> List[Tuple[str, str, str]]:
    """Generate elementwise operation chains."""
    benchmarks = []
    shapes = [(4, 4), (8, 8), (16, 16), (2, 3, 4)]
    ops = [
        ("add", "arith.addf"),
        ("mul", "arith.mulf"),
        ("sub", "arith.subf"),
    ]
    depths = [1, 2, 3, 4]

    for shape in shapes:
        shape_str = "x".join(str(d) for d in shape)
        ttype = f"tensor<{shape_str}xf32>"

        for depth in depths:
            for op_name, op_mlir in ops:
                name = f"elem_{op_name}_d{depth}_{shape_str}"
                lines = [
                    f"func.func @test(%arg0: {ttype}, %arg1: {ttype}) -> {ttype} {{"
                ]
                prev = "%arg0"
                for i in range(depth):
                    rhs = "%arg1" if i % 2 == 0 else "%arg0"
                    lines.append(f"  %{i} = {op_mlir} {prev}, {rhs} : {ttype}")
                    prev = f"%{i}"
                lines.append(f"  return {prev} : {ttype}")
                lines.append("}")
                source = "\n".join(lines) + "\n"
                benchmarks.append((name, source, "--convert-elementwise-to-linalg"))

    return benchmarks


def gen_matmul() -> List[Tuple[str, str, str]]:
    """Generate matmul benchmarks."""
    benchmarks = []
    sizes = [(2, 3, 4), (4, 4, 4), (4, 8, 4), (8, 8, 8), (16, 8, 16)]

    for M, K, N in sizes:
        name = f"matmul_{M}x{K}x{N}"
        source = f"""func.func @test(%A: tensor<{M}x{K}xf32>, %B: tensor<{K}x{N}xf32>, %C: tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32> {{
  %0 = linalg.matmul ins(%A, %B : tensor<{M}x{K}xf32>, tensor<{K}x{N}xf32>) outs(%C : tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
  return %0 : tensor<{M}x{N}xf32>
}}
"""
        benchmarks.append((name, source, "--linalg-generalize-named-ops"))

    return benchmarks


def gen_batch_matmul() -> List[Tuple[str, str, str]]:
    """Generate batch matmul benchmarks."""
    benchmarks = []
    sizes = [(2, 4, 4, 4), (4, 8, 4, 8)]

    for B, M, K, N in sizes:
        name = f"batch_matmul_{B}x{M}x{K}x{N}"
        source = f"""func.func @test(%A: tensor<{B}x{M}x{K}xf32>, %B_: tensor<{B}x{K}x{N}xf32>, %C: tensor<{B}x{M}x{N}xf32>) -> tensor<{B}x{M}x{N}xf32> {{
  %0 = linalg.batch_matmul ins(%A, %B_ : tensor<{B}x{M}x{K}xf32>, tensor<{B}x{K}x{N}xf32>) outs(%C : tensor<{B}x{M}x{N}xf32>) -> tensor<{B}x{M}x{N}xf32>
  return %0 : tensor<{B}x{M}x{N}xf32>
}}
"""
        benchmarks.append((name, source, "--linalg-generalize-named-ops"))

    return benchmarks


def gen_reduce_sum() -> List[Tuple[str, str, str]]:
    """Generate reduce-sum benchmarks."""
    benchmarks = []
    shapes = [(4, 4), (8, 8), (4, 16), (16, 4)]

    for M, N in shapes:
        name = f"reduce_sum_{M}x{N}"
        source = f"""func.func @test(%arg0: tensor<{M}x{N}xf32>, %init: tensor<{M}xf32>) -> tensor<{M}xf32> {{
  %0 = linalg.generic {{
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  }} ins(%arg0 : tensor<{M}x{N}xf32>) outs(%init : tensor<{M}xf32>) {{
  ^bb0(%a: f32, %acc: f32):
    %add = arith.addf %acc, %a : f32
    linalg.yield %add : f32
  }} -> tensor<{M}xf32>
  return %0 : tensor<{M}xf32>
}}
"""
        benchmarks.append((name, source, "--canonicalize"))

    return benchmarks


def gen_reduce_max() -> List[Tuple[str, str, str]]:
    """Generate reduce-max benchmarks."""
    benchmarks = []
    shapes = [(4, 4), (8, 8), (4, 16)]

    for M, N in shapes:
        name = f"reduce_max_{M}x{N}"
        source = f"""func.func @test(%arg0: tensor<{M}x{N}xf32>, %init: tensor<{M}xf32>) -> tensor<{M}xf32> {{
  %0 = linalg.generic {{
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  }} ins(%arg0 : tensor<{M}x{N}xf32>) outs(%init : tensor<{M}xf32>) {{
  ^bb0(%a: f32, %acc: f32):
    %max = arith.maximumf %acc, %a : f32
    linalg.yield %max : f32
  }} -> tensor<{M}xf32>
  return %0 : tensor<{M}xf32>
}}
"""
        benchmarks.append((name, source, "--canonicalize"))

    return benchmarks


def gen_fused_elementwise() -> List[Tuple[str, str, str]]:
    """Generate two-op elementwise chains for fusion testing."""
    benchmarks = []
    shapes = [(4, 4), (8, 8), (16, 16)]
    op_pairs = [
        ("add_mul", "arith.addf", "arith.mulf"),
        ("mul_add", "arith.mulf", "arith.addf"),
        ("sub_mul", "arith.subf", "arith.mulf"),
    ]

    for shape in shapes:
        M, N = shape
        ttype = f"tensor<{M}x{N}xf32>"
        mapdef = f"affine_map<(d0, d1) -> (d0, d1)>"

        for pair_name, op1, op2 in op_pairs:
            name = f"fuse_{pair_name}_{M}x{N}"
            source = f"""func.func @test(%arg0: {ttype}, %arg1: {ttype}, %init: {ttype}) -> {ttype} {{
  %0 = linalg.generic {{
    indexing_maps = [{mapdef}, {mapdef}, {mapdef}],
    iterator_types = ["parallel", "parallel"]
  }} ins(%arg0, %arg1 : {ttype}, {ttype}) outs(%init : {ttype}) {{
  ^bb0(%a: f32, %b: f32, %c: f32):
    %r = {op1} %a, %b : f32
    linalg.yield %r : f32
  }} -> {ttype}
  %1 = linalg.generic {{
    indexing_maps = [{mapdef}, {mapdef}, {mapdef}],
    iterator_types = ["parallel", "parallel"]
  }} ins(%0, %arg0 : {ttype}, {ttype}) outs(%init : {ttype}) {{
  ^bb0(%a: f32, %b: f32, %c: f32):
    %r = {op2} %a, %b : f32
    linalg.yield %r : f32
  }} -> {ttype}
  return %1 : {ttype}
}}
"""
            benchmarks.append((name, source, "--linalg-fuse-elementwise-ops"))

    return benchmarks


def gen_transpose() -> List[Tuple[str, str, str]]:
    """Generate transpose benchmarks."""
    benchmarks = []
    shapes = [(2, 3), (4, 8), (8, 16)]

    for M, N in shapes:
        # Double transpose = identity (canonicalize should fold)
        name = f"transpose_double_{M}x{N}"
        source = f"""func.func @test(%arg0: tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32> {{
  %0 = tensor.transpose %arg0 [1, 0] : tensor<{M}x{N}xf32> -> tensor<{N}x{M}xf32>
  %1 = tensor.transpose %0 [1, 0] : tensor<{N}x{M}xf32> -> tensor<{M}x{N}xf32>
  return %1 : tensor<{M}x{N}xf32>
}}
"""
        benchmarks.append((name, source, "--canonicalize"))

    return benchmarks


def gen_reshape() -> List[Tuple[str, str, str]]:
    """Generate reshape benchmarks."""
    benchmarks = []
    configs = [
        ((2, 3), (6,)),
        ((2, 3, 4), (6, 4)),
        ((4, 4), (16,)),
    ]

    for orig_shape, collapsed_shape in configs:
        orig_str = "x".join(str(d) for d in orig_shape)
        coll_str = "x".join(str(d) for d in collapsed_shape)

        # Build reassociation
        if len(orig_shape) == 2 and len(collapsed_shape) == 1:
            reassoc = "[[0, 1]]"
        elif len(orig_shape) == 3 and len(collapsed_shape) == 2:
            reassoc = "[[0, 1], [2]]"
        else:
            continue

        name = f"reshape_fold_{orig_str}_to_{coll_str}"
        source = f"""func.func @test(%arg0: tensor<{orig_str}xf32>) -> tensor<{orig_str}xf32> {{
  %0 = tensor.collapse_shape %arg0 {reassoc} : tensor<{orig_str}xf32> into tensor<{coll_str}xf32>
  %1 = tensor.expand_shape %0 {reassoc} : tensor<{coll_str}xf32> into tensor<{orig_str}xf32>
  return %1 : tensor<{orig_str}xf32>
}}
"""
        benchmarks.append((name, source, "--canonicalize"))

    return benchmarks


def main():
    generators = [
        ("Elementwise chains", gen_elementwise_chains),
        ("Matmul", gen_matmul),
        ("Batch matmul", gen_batch_matmul),
        ("Reduce sum", gen_reduce_sum),
        ("Reduce max", gen_reduce_max),
        ("Fused elementwise", gen_fused_elementwise),
        ("Transpose", gen_transpose),
        ("Reshape", gen_reshape),
    ]

    total = 0
    success = 0

    for category, gen_fn in generators:
        benchmarks = gen_fn()
        cat_success = 0
        for name, source, pass_flag in benchmarks:
            if write_pair(name, source, pass_flag):
                cat_success += 1
            total += 1
        success += cat_success
        print(f"{category}: {cat_success}/{len(benchmarks)} generated")

    print(f"\nTotal: {success}/{total} benchmarks generated in {config.SYNTH_DIR}")


if __name__ == "__main__":
    main()
