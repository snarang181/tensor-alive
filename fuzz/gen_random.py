#!/usr/bin/env python3
"""Random MLIR program generator for fuzzing tensor-alive.

Generates valid MLIR programs in tensor-alive's supported subset:
- linalg.generic with random affine maps and body ops
- Elementwise chains (arith.addf/mulf/subf/etc)
- Reshape chains (collapse_shape + expand_shape)
- Transpose compositions
- Broadcast patterns
- Matmul / batch_matmul
- Reduction patterns (sum, max, min)

All programs use static shapes and f32 element type.
"""

import random
from dataclasses import dataclass, field
from itertools import product
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def tensor_type(shape: Tuple[int, ...], dtype: str = "f32") -> str:
    return f"tensor<{'x'.join(str(d) for d in shape)}x{dtype}>"


def affine_map(dims: int, exprs: List[str]) -> str:
    dim_names = [f"d{i}" for i in range(dims)]
    return f"affine_map<({', '.join(dim_names)}) -> ({', '.join(exprs)})>"


# ---------------------------------------------------------------------------
# Shape utilities
# ---------------------------------------------------------------------------

SMALL_DIMS = [1, 2, 3, 4, 5, 6, 7, 8]
MEDIUM_DIMS = [1, 2, 3, 4, 8, 12, 16]


def rand_shape(rank: int, dims=SMALL_DIMS) -> Tuple[int, ...]:
    return tuple(random.choice(dims) for _ in range(rank))


def rand_rank(lo=1, hi=4) -> int:
    return random.randint(lo, hi)


def _prod(iterable) -> int:
    result = 1
    for x in iterable:
        result *= x
    return result


def factorizations(n: int) -> List[Tuple[int, ...]]:
    """Return all ordered factorizations of n into 2 factors > 0."""
    facts = []
    for i in range(1, n + 1):
        if n % i == 0:
            facts.append((i, n // i))
    return facts


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

ELEMENTWISE_OPS = [
    ("arith.addf", 2),
    ("arith.subf", 2),
    ("arith.mulf", 2),
    ("arith.negf", 1),
]

REDUCTION_COMBINERS = [
    ("arith.addf", "sum"),
    ("arith.mulf", "prod"),
    ("arith.maximumf", "max"),
    ("arith.minimumf", "min"),
]


@dataclass
class GeneratedProgram:
    name: str
    mlir: str
    suggested_passes: List[str] = field(default_factory=list)


def gen_elementwise_chain(seed: int) -> GeneratedProgram:
    """Generate a chain of elementwise linalg.generic ops."""
    rng = random.Random(seed)
    rank = rng.randint(1, 3)
    shape = tuple(rng.choice(SMALL_DIMS) for _ in range(rank))
    ttype = tensor_type(shape)
    depth = rng.randint(2, 5)
    num_inputs = rng.randint(2, 3)

    dim_names = [f"d{i}" for i in range(rank)]
    identity = affine_map(rank, dim_names)

    args = [f"%arg{i}: {ttype}" for i in range(num_inputs)]
    args.append(f"%init: {ttype}")
    sig = ", ".join(args)

    lines = [f"func.func @test({sig}) -> {ttype} {{"]

    prev = "%arg0"
    for step in range(depth):
        op_name, arity = rng.choice(ELEMENTWISE_OPS)
        if arity == 2:
            rhs = f"%arg{rng.randint(0, num_inputs - 1)}"
            ins_list = f"{prev}, {rhs} : {ttype}, {ttype}"
            maps = f"[{identity}, {identity}, {identity}]"
            body_args = "%a: f32, %b: f32, %c: f32"
            body_op = f"    %r{step} = {op_name} %a, %b : f32"
        else:
            ins_list = f"{prev} : {ttype}"
            maps = f"[{identity}, {identity}]"
            body_args = "%a: f32, %c: f32"
            body_op = f"    %r{step} = {op_name} %a : f32"

        result = f"%v{step}"
        lines.append(f"  {result} = linalg.generic {{")
        lines.append(f"    indexing_maps = {maps},")
        par_types = ", ".join(['"parallel"'] * rank)
        lines.append(f"    iterator_types = [{par_types}]")
        lines.append(f"  }} ins({ins_list}) outs(%init : {ttype}) {{")
        lines.append(f"  ^bb0({body_args}):")
        lines.append(body_op)
        lines.append(f"    linalg.yield %r{step} : f32")
        lines.append(f"  }} -> {ttype}")
        prev = result

    lines.append(f"  return {prev} : {ttype}")
    lines.append("}")

    return GeneratedProgram(
        name=f"elem_chain_s{seed}",
        mlir="\n".join(lines) + "\n",
        suggested_passes=["--canonicalize", "--linalg-fuse-elementwise-ops"],
    )


def gen_linalg_generic_mixed(seed: int) -> GeneratedProgram:
    """Generate a linalg.generic with mixed parallel/reduction dims and random body."""
    rng = random.Random(seed)
    n_par = rng.randint(1, 3)
    n_red = rng.randint(0, 2)
    total = n_par + n_red

    par_dims = tuple(rng.choice(SMALL_DIMS) for _ in range(n_par))
    red_dims = tuple(rng.choice([2, 3, 4]) for _ in range(n_red))

    dim_names = [f"d{i}" for i in range(total)]
    par_names = dim_names[:n_par]
    red_names = dim_names[n_par:]

    # Input shape includes all dims, output shape only parallel dims
    in_shape = par_dims + red_dims
    out_shape = par_dims

    in_type = tensor_type(in_shape)
    out_type = tensor_type(out_shape)
    in_map = affine_map(total, dim_names)
    out_map = (
        affine_map(total, par_names)
        if par_names
        else affine_map(total, ["0"] if total > 0 else [])
    )

    # Choose combiner
    combiner_op, comb_name = rng.choice(REDUCTION_COMBINERS)

    iter_types = ['"parallel"'] * n_par + ['"reduction"'] * n_red

    # Optional: add a second input for binary pre-processing before reduction
    use_two_inputs = rng.random() < 0.4 and n_red > 0
    if use_two_inputs:
        pre_op = rng.choice(["arith.addf", "arith.mulf"])
        lines = [
            f"func.func @test(%arg0: {in_type}, %arg1: {in_type}, %init: {out_type}) -> {out_type} {{"
        ]
        ins_str = f"%arg0, %arg1 : {in_type}, {in_type}"
        maps = f"[{in_map}, {in_map}, {out_map}]"
        body_args = "%a: f32, %b: f32, %acc: f32"
        body_lines = [
            f"    %pre = {pre_op} %a, %b : f32",
            f"    %r = {combiner_op} %acc, %pre : f32",
        ]
    else:
        lines = [
            f"func.func @test(%arg0: {in_type}, %init: {out_type}) -> {out_type} {{"
        ]
        ins_str = f"%arg0 : {in_type}"
        maps = f"[{in_map}, {out_map}]"
        body_args = "%a: f32, %acc: f32"
        if n_red > 0:
            body_lines = [f"    %r = {combiner_op} %acc, %a : f32"]
        else:
            # Pure elementwise with single input
            unary_op = rng.choice(["arith.negf"])
            body_lines = [f"    %r = {unary_op} %a : f32"]

    lines.append(f"  %0 = linalg.generic {{")
    lines.append(f"    indexing_maps = {maps},")
    lines.append(f"    iterator_types = [{', '.join(iter_types)}]")
    lines.append(f"  }} ins({ins_str}) outs(%init : {out_type}) {{")
    lines.append(f"  ^bb0({body_args}):")
    lines.extend(body_lines)
    lines.append(f"    linalg.yield %r : f32")
    lines.append(f"  }} -> {out_type}")
    lines.append(f"  return %0 : {out_type}")
    lines.append("}")

    passes = ["--canonicalize"]
    if n_red == 0:
        passes.append("--linalg-fuse-elementwise-ops")

    return GeneratedProgram(
        name=f"generic_mixed_s{seed}",
        mlir="\n".join(lines) + "\n",
        suggested_passes=passes,
    )


def gen_reshape_chain(seed: int) -> GeneratedProgram:
    """Generate collapse_shape -> expand_shape chains that should fold to identity."""
    rng = random.Random(seed)
    rank = rng.randint(2, 4)
    shape = tuple(rng.choice([2, 3, 4, 6]) for _ in range(rank))
    ttype = tensor_type(shape)

    # Pick a random contiguous grouping for collapse
    # e.g. for rank=4: [[0,1],[2,3]] or [[0],[1,2,3]] or [[0,1,2],[3]]
    splits = sorted(rng.sample(range(1, rank), rng.randint(1, rank - 1)))
    groups = []
    prev = 0
    for s in splits:
        groups.append(list(range(prev, s)))
        prev = s
    groups.append(list(range(prev, rank)))

    collapsed_shape = tuple(
        1 if len(g) == 0 else _prod(shape[i] for i in g) for g in groups
    )

    reassoc = (
        "[" + ", ".join("[" + ", ".join(str(i) for i in g) + "]" for g in groups) + "]"
    )
    collapsed_type = tensor_type(collapsed_shape)

    lines = [
        f"func.func @test(%arg0: {ttype}) -> {ttype} {{",
        f"  %0 = tensor.collapse_shape %arg0 {reassoc} : {ttype} into {collapsed_type}",
        f"  %1 = tensor.expand_shape %0 {reassoc} : {collapsed_type} into {ttype}",
        f"  return %1 : {ttype}",
        "}",
    ]

    return GeneratedProgram(
        name=f"reshape_chain_s{seed}",
        mlir="\n".join(lines) + "\n",
        suggested_passes=["--canonicalize"],
    )


def gen_transpose_chain(seed: int) -> GeneratedProgram:
    """Generate chains of transposes that may simplify."""
    rng = random.Random(seed)
    rank = rng.randint(2, 4)
    shape = tuple(rng.choice(SMALL_DIMS) for _ in range(rank))
    depth = rng.randint(2, 4)

    lines = [f"func.func @test(%arg0: {tensor_type(shape)}) -> {tensor_type(shape)} {{"]
    current_shape = list(shape)
    current_perm = list(range(rank))  # tracks cumulative permutation
    prev = "%arg0"

    for step in range(depth):
        perm = list(range(rank))
        rng.shuffle(perm)
        new_shape = tuple(current_shape[perm[i]] for i in range(rank))
        perm_str = ", ".join(str(p) for p in perm)
        result = f"%t{step}"
        lines.append(
            f"  {result} = tensor.transpose {prev} [{perm_str}] : {tensor_type(tuple(current_shape))} -> {tensor_type(new_shape)}"
        )
        current_shape = list(new_shape)
        prev = result

    # Add final transpose to get back to original shape if needed
    # Compute inverse of current cumulative permutation
    cumul = list(range(rank))
    cur_shape_track = list(shape)
    # Replay all transposes to find the final permutation
    # Simpler: just add one more transpose back to original shape
    if tuple(current_shape) != shape:
        # Need to invert: find perm such that current_shape[perm[i]] = shape[i]
        # This is a bit tricky; just add the inverse transpose
        inv = [0] * rank
        # Map from current positions back to original
        # Actually just check if the shapes even allow it
        # Easier: ensure the chain returns to original shape by construction
        # Undo by applying inverse of cumulative permutation
        pass  # We'll return whatever shape we end up with

    final_type = tensor_type(tuple(current_shape))

    # Rewrite signature to match output
    lines[0] = f"func.func @test(%arg0: {tensor_type(shape)}) -> {final_type} {{"
    lines.append(f"  return {prev} : {final_type}")
    lines.append("}")

    return GeneratedProgram(
        name=f"transpose_chain_s{seed}",
        mlir="\n".join(lines) + "\n",
        suggested_passes=["--canonicalize"],
    )


def gen_double_transpose(seed: int) -> GeneratedProgram:
    """Generate A -> transpose -> transpose -> A (should fold to identity)."""
    rng = random.Random(seed)
    rank = rng.randint(2, 4)
    shape = tuple(rng.choice(SMALL_DIMS) for _ in range(rank))

    perm = list(range(rank))
    rng.shuffle(perm)
    inv_perm = [0] * rank
    for i, p in enumerate(perm):
        inv_perm[p] = i

    mid_shape = tuple(shape[perm[i]] for i in range(rank))

    ttype = tensor_type(shape)
    mid_type = tensor_type(mid_shape)
    perm_str = ", ".join(str(p) for p in perm)
    inv_str = ", ".join(str(p) for p in inv_perm)

    lines = [
        f"func.func @test(%arg0: {ttype}) -> {ttype} {{",
        f"  %0 = tensor.transpose %arg0 [{perm_str}] : {ttype} -> {mid_type}",
        f"  %1 = tensor.transpose %0 [{inv_str}] : {mid_type} -> {ttype}",
        f"  return %1 : {ttype}",
        "}",
    ]

    return GeneratedProgram(
        name=f"double_transpose_s{seed}",
        mlir="\n".join(lines) + "\n",
        suggested_passes=["--canonicalize"],
    )


def gen_broadcast_elementwise(seed: int) -> GeneratedProgram:
    """Generate broadcast + elementwise patterns."""
    rng = random.Random(seed)
    out_rank = rng.randint(2, 3)
    out_shape = tuple(rng.choice(SMALL_DIMS) for _ in range(out_rank))

    # Create a broadcast-compatible input (some dims = 1 or missing)
    in_shape = list(out_shape)
    broadcast_dims = []
    for i in range(out_rank):
        if rng.random() < 0.4:
            in_shape[i] = 1
            broadcast_dims.append(i)

    in_shape = tuple(in_shape)
    out_type = tensor_type(out_shape)
    in_type = tensor_type(in_shape)

    dim_names = [f"d{i}" for i in range(out_rank)]
    out_map = affine_map(out_rank, dim_names)
    in_exprs = ["0" if i in broadcast_dims else f"d{i}" for i in range(out_rank)]
    in_map = affine_map(out_rank, in_exprs)

    op_name = rng.choice(["arith.addf", "arith.mulf", "arith.subf"])
    iter_types = ", ".join(['"parallel"'] * out_rank)

    lines = [
        f"func.func @test(%arg0: {in_type}, %arg1: {out_type}, %init: {out_type}) -> {out_type} {{",
        f"  %0 = linalg.generic {{",
        f"    indexing_maps = [{in_map}, {out_map}, {out_map}],",
        f"    iterator_types = [{iter_types}]",
        f"  }} ins(%arg0, %arg1 : {in_type}, {out_type}) outs(%init : {out_type}) {{",
        f"  ^bb0(%a: f32, %b: f32, %c: f32):",
        f"    %r = {op_name} %a, %b : f32",
        f"    linalg.yield %r : f32",
        f"  }} -> {out_type}",
        f"  return %0 : {out_type}",
        "}",
    ]

    return GeneratedProgram(
        name=f"broadcast_elem_s{seed}",
        mlir="\n".join(lines) + "\n",
        suggested_passes=["--canonicalize", "--linalg-fuse-elementwise-ops"],
    )


def gen_matmul_variants(seed: int) -> GeneratedProgram:
    """Generate matmul with varying sizes for generalization testing."""
    rng = random.Random(seed)
    use_batch = rng.random() < 0.3
    M = rng.choice([2, 3, 4, 8])
    K = rng.choice([2, 3, 4, 8])
    N = rng.choice([2, 3, 4, 8])

    if use_batch:
        B = rng.choice([2, 3, 4])
        a_type = tensor_type((B, M, K))
        b_type = tensor_type((B, K, N))
        c_type = tensor_type((B, M, N))
        op = "linalg.batch_matmul"
        name = f"batch_matmul_{B}x{M}x{K}x{N}_s{seed}"
    else:
        a_type = tensor_type((M, K))
        b_type = tensor_type((K, N))
        c_type = tensor_type((M, N))
        op = "linalg.matmul"
        name = f"matmul_{M}x{K}x{N}_s{seed}"

    lines = [
        f"func.func @test(%A: {a_type}, %B: {b_type}, %C: {c_type}) -> {c_type} {{",
        f"  %0 = {op} ins(%A, %B : {a_type}, {b_type}) outs(%C : {c_type}) -> {c_type}",
        f"  return %0 : {c_type}",
        "}",
    ]

    return GeneratedProgram(
        name=name,
        mlir="\n".join(lines) + "\n",
        suggested_passes=["--linalg-generalize-named-ops", "--canonicalize"],
    )


def gen_reduction_variants(seed: int) -> GeneratedProgram:
    """Generate reductions over different axes with different combiners."""
    rng = random.Random(seed)
    rank = rng.randint(2, 3)
    shape = tuple(rng.choice(SMALL_DIMS) for _ in range(rank))

    # Choose which dims to reduce
    n_red = rng.randint(1, rank - 1)
    red_axes = sorted(rng.sample(range(rank), n_red))
    par_axes = [i for i in range(rank) if i not in red_axes]

    out_shape = tuple(shape[i] for i in par_axes)
    in_type = tensor_type(shape)
    out_type = tensor_type(out_shape)

    dim_names = [f"d{i}" for i in range(rank)]
    in_map = affine_map(rank, dim_names)
    out_exprs = [f"d{i}" for i in par_axes]
    out_map = affine_map(rank, out_exprs)

    iter_types = []
    for i in range(rank):
        if i in red_axes:
            iter_types.append('"reduction"')
        else:
            iter_types.append('"parallel"')

    combiner_op, comb_name = rng.choice(REDUCTION_COMBINERS)

    lines = [
        f"func.func @test(%arg0: {in_type}, %init: {out_type}) -> {out_type} {{",
        f"  %0 = linalg.generic {{",
        f"    indexing_maps = [{in_map}, {out_map}],",
        f"    iterator_types = [{', '.join(iter_types)}]",
        f"  }} ins(%arg0 : {in_type}) outs(%init : {out_type}) {{",
        f"  ^bb0(%a: f32, %acc: f32):",
        f"    %r = {combiner_op} %acc, %a : f32",
        f"    linalg.yield %r : f32",
        f"  }} -> {out_type}",
        f"  return %0 : {out_type}",
        "}",
    ]

    return GeneratedProgram(
        name=f"reduce_{comb_name}_s{seed}",
        mlir="\n".join(lines) + "\n",
        suggested_passes=["--canonicalize"],
    )


def gen_elementwise_then_reduce(seed: int) -> GeneratedProgram:
    """Generate elementwise op followed by reduction — tests fusion into reduction."""
    rng = random.Random(seed)
    rank = rng.randint(2, 3)
    shape = tuple(rng.choice(SMALL_DIMS) for _ in range(rank))
    ttype = tensor_type(shape)

    # Reduce last dim
    out_shape = shape[:-1]
    out_type = tensor_type(out_shape)

    dim_names = [f"d{i}" for i in range(rank)]
    identity = affine_map(rank, dim_names)
    out_map = affine_map(rank, dim_names[:-1])

    elem_op = rng.choice(["arith.addf", "arith.mulf", "arith.subf"])
    red_op, red_name = rng.choice(REDUCTION_COMBINERS)

    iter_types_par = ", ".join(['"parallel"'] * rank)
    iter_types_red = ", ".join(['"parallel"'] * (rank - 1) + ['"reduction"'])

    lines = [
        f"func.func @test(%arg0: {ttype}, %arg1: {ttype}, %init_full: {ttype}, %init_red: {out_type}) -> {out_type} {{",
        f"  %0 = linalg.generic {{",
        f"    indexing_maps = [{identity}, {identity}, {identity}],",
        f"    iterator_types = [{iter_types_par}]",
        f"  }} ins(%arg0, %arg1 : {ttype}, {ttype}) outs(%init_full : {ttype}) {{",
        f"  ^bb0(%a: f32, %b: f32, %c: f32):",
        f"    %r = {elem_op} %a, %b : f32",
        f"    linalg.yield %r : f32",
        f"  }} -> {ttype}",
        f"  %1 = linalg.generic {{",
        f"    indexing_maps = [{identity}, {out_map}],",
        f"    iterator_types = [{iter_types_red}]",
        f"  }} ins(%0 : {ttype}) outs(%init_red : {out_type}) {{",
        f"  ^bb0(%a: f32, %acc: f32):",
        f"    %r = {red_op} %acc, %a : f32",
        f"    linalg.yield %r : f32",
        f"  }} -> {out_type}",
        f"  return %1 : {out_type}",
        "}",
    ]

    return GeneratedProgram(
        name=f"elem_then_reduce_{red_name}_s{seed}",
        mlir="\n".join(lines) + "\n",
        suggested_passes=[
            "--canonicalize",
            "--linalg-fuse-elementwise-ops",
            "--linalg-fold-into-elementwise",
        ],
    )


def gen_unit_dim_generic(seed: int) -> GeneratedProgram:
    """Generate linalg.generic with unit extent dims for fold-unit-extent testing."""
    rng = random.Random(seed)
    rank = rng.randint(2, 4)
    shape = list(rng.choice(SMALL_DIMS) for _ in range(rank))

    # Insert 1-2 unit dims
    n_unit = rng.randint(1, min(2, rank))
    unit_positions = rng.sample(range(rank), n_unit)
    for p in unit_positions:
        shape[p] = 1

    shape = tuple(shape)
    ttype = tensor_type(shape)
    dim_names = [f"d{i}" for i in range(rank)]
    identity = affine_map(rank, dim_names)
    iter_types = ", ".join(['"parallel"'] * rank)

    op = rng.choice(["arith.addf", "arith.mulf"])

    lines = [
        f"func.func @test(%arg0: {ttype}, %arg1: {ttype}, %init: {ttype}) -> {ttype} {{",
        f"  %0 = linalg.generic {{",
        f"    indexing_maps = [{identity}, {identity}, {identity}],",
        f"    iterator_types = [{iter_types}]",
        f"  }} ins(%arg0, %arg1 : {ttype}, {ttype}) outs(%init : {ttype}) {{",
        f"  ^bb0(%a: f32, %b: f32, %c: f32):",
        f"    %r = {op} %a, %b : f32",
        f"    linalg.yield %r : f32",
        f"  }} -> {ttype}",
        f"  return %0 : {ttype}",
        "}",
    ]

    return GeneratedProgram(
        name=f"unit_dim_s{seed}",
        mlir="\n".join(lines) + "\n",
        suggested_passes=["--linalg-fold-unit-extent-dims", "--canonicalize"],
    )


def gen_permuted_affine_maps(seed: int) -> GeneratedProgram:
    """Generate linalg.generic with permuted indexing maps (non-identity access patterns).

    This targets bugs in affine map composition during fusion and canonicalization.
    """
    rng = random.Random(seed)
    rank = rng.randint(2, 3)
    shape = tuple(rng.choice(SMALL_DIMS) for _ in range(rank))
    ttype = tensor_type(shape)

    dim_names = [f"d{i}" for i in range(rank)]
    identity = affine_map(rank, dim_names)

    # Permuted map for one input
    perm = list(range(rank))
    rng.shuffle(perm)
    perm_shape = tuple(shape[p] for p in perm)
    perm_exprs = [f"d{perm[i]}" for i in range(rank)]
    perm_map = affine_map(rank, perm_exprs)
    perm_type = tensor_type(perm_shape)

    iter_types = ", ".join(['"parallel"'] * rank)
    op = rng.choice(["arith.addf", "arith.mulf", "arith.subf"])

    lines = [
        f"func.func @test(%arg0: {perm_type}, %arg1: {ttype}, %init: {ttype}) -> {ttype} {{",
        f"  %0 = linalg.generic {{",
        f"    indexing_maps = [{perm_map}, {identity}, {identity}],",
        f"    iterator_types = [{iter_types}]",
        f"  }} ins(%arg0, %arg1 : {perm_type}, {ttype}) outs(%init : {ttype}) {{",
        f"  ^bb0(%a: f32, %b: f32, %c: f32):",
        f"    %r = {op} %a, %b : f32",
        f"    linalg.yield %r : f32",
        f"  }} -> {ttype}",
        f"  return %0 : {ttype}",
        "}",
    ]

    return GeneratedProgram(
        name=f"perm_maps_s{seed}",
        mlir="\n".join(lines) + "\n",
        suggested_passes=["--canonicalize", "--linalg-fuse-elementwise-ops"],
    )


# ---------------------------------------------------------------------------
# Master generator
# ---------------------------------------------------------------------------

ALL_GENERATORS = [
    gen_elementwise_chain,
    gen_linalg_generic_mixed,
    gen_reshape_chain,
    gen_transpose_chain,
    gen_double_transpose,
    gen_broadcast_elementwise,
    gen_matmul_variants,
    gen_reduction_variants,
    gen_elementwise_then_reduce,
    gen_unit_dim_generic,
    gen_permuted_affine_maps,
]


def generate_batch(n: int, seed: int = 0) -> List[GeneratedProgram]:
    """Generate n random programs, cycling through generator types."""
    programs = []
    for i in range(n):
        gen = ALL_GENERATORS[i % len(ALL_GENERATORS)]
        try:
            prog = gen(seed + i)
            programs.append(prog)
        except Exception as e:
            pass  # Skip malformed generations
    return programs


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    programs = generate_batch(n, seed)
    print(f"Generated {len(programs)} programs:")
    for p in programs:
        print(f"  {p.name} — passes: {p.suggested_passes}")
        if "--verbose" in sys.argv:
            print(p.mlir)
