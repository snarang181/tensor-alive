#!/usr/bin/env python3
"""Mutational fuzzer for MLIR programs.

Takes MLIR programs (from the MLIR test corpus or generated programs) and
applies semantics-preserving and semantics-breaking mutations to stress-test
tensor-alive and mlir-opt.

Mutation strategies:
1. Shape mutation — change tensor dimensions while keeping rank
2. Op substitution — swap arith ops (addf <-> mulf, etc.)
3. Affine map permutation — shuffle indexing map expressions
4. Rank change — add/remove dimensions (with matching map changes)
5. Combiner swap — change reduction combiner (sum <-> max)
"""

import re
import random
from typing import List, Optional, Tuple

# Regex patterns for MLIR constructs
TENSOR_TYPE_RE = re.compile(r"tensor<([\dx]+)x(f\d+|i\d+|bf16)>")
SHAPE_RE = re.compile(r"(\d+)x")
ARITH_OP_RE = re.compile(r"(arith\.\w+)")
DIM_RE = re.compile(r"tensor<((?:\d+x)+)")


ARITH_BINARY_OPS = ["arith.addf", "arith.subf", "arith.mulf"]
ARITH_SWAPS = {
    "arith.addf": ["arith.subf", "arith.mulf"],
    "arith.subf": ["arith.addf", "arith.mulf"],
    "arith.mulf": ["arith.addf", "arith.subf"],
    "arith.maximumf": ["arith.minimumf", "arith.addf"],
    "arith.minimumf": ["arith.maximumf", "arith.addf"],
}


def mutate_shapes(mlir: str, rng: random.Random) -> Optional[str]:
    """Change tensor dimensions while keeping the program structurally valid.

    Strategy: pick one dimension size that appears in the program and replace
    ALL occurrences with a different size. This preserves shape consistency.

    Skips programs with tensor.empty (undefined outs values), expand_shape /
    collapse_shape (dimension products must match), and linalg.pack/unpack
    (tile sizes are shape-coupled).
    """
    # Skip programs where shape changes would break internal invariants
    skip_patterns = [
        "tensor.empty",  # Undefined outs values
        "expand_shape",  # Dim product constraints
        "collapse_shape",  # Dim product constraints
        "linalg.pack",  # Tile size coupling
        "linalg.unpack",  # Tile size coupling
        "output_shape",  # Explicit shape literals
    ]
    for pat in skip_patterns:
        if pat in mlir:
            return None

    # Find all dimension sizes used
    dims_used = set()
    for m in TENSOR_TYPE_RE.finditer(mlir):
        shape_str = m.group(1)  # e.g. "4x8"
        for d in shape_str.split("x"):
            if d.isdigit():
                dims_used.add(int(d))

    if not dims_used:
        return None

    # Pick a dim to mutate (avoid 1 since that's special for unit dims)
    candidates = [d for d in dims_used if d > 1]
    if not candidates:
        return None

    old_dim = rng.choice(candidates)
    # Pick a new dim that's different
    possible = [d for d in [2, 3, 4, 5, 6, 7, 8] if d != old_dim]
    new_dim = rng.choice(possible)

    # Replace all occurrences of this dim in tensor type contexts
    # Be careful: replace "4x" with "5x" and "x4x" with "x5x" and "x4>" with "x5>"
    result = mlir
    # Replace in tensor types: NxMx... patterns
    result = re.sub(
        rf"(?<=[<x]){old_dim}(?=[x>])",
        str(new_dim),
        result,
    )

    return result


def mutate_swap_op(mlir: str, rng: random.Random) -> Optional[str]:
    """Swap an arithmetic operation with a different one.

    This is a semantics-BREAKING mutation — if mlir-opt treats the result
    as equivalent, that's a bug.
    """
    ops_found = ARITH_OP_RE.findall(mlir)
    swappable = [op for op in ops_found if op in ARITH_SWAPS]
    if not swappable:
        return None

    op_to_swap = rng.choice(swappable)
    new_op = rng.choice(ARITH_SWAPS[op_to_swap])

    # Replace first occurrence only
    return mlir.replace(op_to_swap, new_op, 1)


def mutate_add_negation(mlir: str, rng: random.Random) -> Optional[str]:
    """Wrap a value in arith.negf (double negation should be identity)."""
    # Find a linalg.yield line
    m = re.search(r"(\s+)linalg\.yield (%\w+) : f32", mlir)
    if not m:
        return None

    indent = m.group(1)
    val = m.group(2)
    neg_line = f"{indent}%_neg = arith.negf {val} : f32\n"
    neg2_line = f"{indent}%_neg2 = arith.negf %_neg : f32\n"
    new_yield = f"{indent}linalg.yield %_neg2 : f32"

    return mlir[: m.start()] + neg_line + neg2_line + new_yield + mlir[m.end() :]


def mutate_iterator_types(mlir: str, rng: random.Random) -> Optional[str]:
    """Flip a parallel dim to reduction or vice versa.

    This is semantics-BREAKING — if the pass still considers it equivalent, bug.
    """
    # Only do this rarely and on programs with multiple parallel dims
    m = re.search(r"iterator_types\s*=\s*\[([^\]]+)\]", mlir)
    if not m:
        return None

    types_str = m.group(1)
    types = [t.strip().strip('"') for t in types_str.split(",")]

    par_indices = [i for i, t in enumerate(types) if t == "parallel"]
    if len(par_indices) < 2:
        return None  # Need at least 2 parallel to safely flip one

    # This would change semantics — intentional for finding bugs
    idx = rng.choice(par_indices)
    types[idx] = "reduction"
    new_types = ", ".join(f'"{t}"' for t in types)

    return mlir[: m.start(1)] + new_types + mlir[m.end(1) :]


def mutate_transpose_perm(mlir: str, rng: random.Random) -> Optional[str]:
    """Mutate a transpose permutation."""
    m = re.search(r"tensor\.transpose\s+%\w+\s+\[([^\]]+)\]", mlir)
    if not m:
        return None

    perm = [int(x.strip()) for x in m.group(1).split(",")]
    if len(perm) < 2:
        return None

    # Swap two elements in the permutation
    i, j = rng.sample(range(len(perm)), 2)
    perm[i], perm[j] = perm[j], perm[i]
    new_perm = ", ".join(str(p) for p in perm)

    return mlir[: m.start(1)] + new_perm + mlir[m.end(1) :]


# ---------------------------------------------------------------------------
# Mutation strategies
# ---------------------------------------------------------------------------

# Semantics-preserving mutations (good for testing pass correctness)
PRESERVING_MUTATIONS = [
    ("shape_change", mutate_shapes),
    ("double_negation", mutate_add_negation),
]

# Semantics-breaking mutations (should NOT be equivalent after pass)
BREAKING_MUTATIONS = [
    ("swap_op", mutate_swap_op),
    ("flip_iterator", mutate_iterator_types),
    ("swap_transpose_perm", mutate_transpose_perm),
]

ALL_MUTATIONS = PRESERVING_MUTATIONS + BREAKING_MUTATIONS


def apply_random_mutation(
    mlir: str,
    rng: random.Random,
    breaking: bool = False,
) -> Optional[Tuple[str, str, bool]]:
    """Apply a random mutation. Returns (mutated_mlir, mutation_name, is_breaking) or None."""
    mutations = BREAKING_MUTATIONS if breaking else PRESERVING_MUTATIONS
    rng.shuffle(mutations)

    for name, mutator in mutations:
        result = mutator(mlir, rng)
        if result is not None and result != mlir:
            is_breaking = name in dict(BREAKING_MUTATIONS)
            return result, name, is_breaking

    return None


def apply_n_mutations(
    mlir: str,
    n: int,
    seed: int = 0,
    mix_breaking: bool = True,
) -> List[Tuple[str, str, bool]]:
    """Apply n different mutations to the same base program.

    Returns list of (mutated_mlir, mutation_description, is_breaking).
    """
    results = []
    for i in range(n):
        rng = random.Random(seed + i)
        breaking = mix_breaking and rng.random() < 0.3
        result = apply_random_mutation(mlir, rng, breaking=breaking)
        if result is not None:
            results.append(result)
    return results


if __name__ == "__main__":
    import sys

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
    print("=== Original ===")
    print(example)

    mutations = apply_n_mutations(example, 5, seed=42)
    for mutated, desc, breaking in mutations:
        tag = "BREAKING" if breaking else "preserving"
        print(f"\n=== Mutation: {desc} ({tag}) ===")
        print(mutated)
