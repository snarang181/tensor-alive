"""Preprocess MLIR text for tensor-alive consumption.

Resolves affine_map aliases (#map -> inline), strips unsupported patterns.
"""

import re


def resolve_map_aliases(mlir_text: str) -> str:
    """Replace #map references with inline affine_map<...> definitions."""
    aliases = {}
    body_lines = []
    for line in mlir_text.splitlines():
        m = re.match(r"^(#\w+)\s*=\s*(affine_map<[^>]+>)", line)
        if m:
            aliases[m.group(1)] = m.group(2)
        else:
            body_lines.append(line)

    result = "\n".join(body_lines)
    # Sort by length descending to avoid partial replacement (#map1 before #map)
    for name in sorted(aliases.keys(), key=len, reverse=True):
        result = result.replace(name, aliases[name])
    return result


def strip_comments(mlir_text: str) -> str:
    """Remove // RUN: and // CHECK lines."""
    lines = []
    for line in mlir_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("// RUN:") or stripped.startswith("// CHECK"):
            continue
        lines.append(line)
    return "\n".join(lines)


def preprocess(mlir_text: str) -> str:
    """Full preprocessing pipeline."""
    text = strip_comments(mlir_text)
    # Note: we don't resolve aliases here since the C++ parser now handles them.
    # This function is available as a fallback.
    return text


UNSUPPORTED_PATTERNS = [
    "memref",
    "tensor<?",
    "tensor<*",
    "scf.",
    "vector.",
    # tensor ops not supported by parser
    "tensor.pad",
    "tensor.extract_slice",
    "tensor.insert_slice",
    "tensor.generate",
    "tensor.yield",
    "tensor.from_elements",
    "tensor.scatter",
    "tensor.gather",
    "tensor.concat",
    "tensor.extract",
    "tensor.rank",
    "tensor.cast",
    "tensor.splat",
    "tensor.dim",
    "tensor.bitcast",
    "bufferization.",
    "sparse_tensor.",
    "complex<",
    "tensor<?",
    "?x",
    "x?x",
    "x?>",
    "affine.",
    # linalg named ops not supported
    "linalg.index",
    "linalg.winograd",
    "linalg.pooling",
    "linalg.conv",
    "linalg.depthwise",
    "linalg.reduce",
    "linalg.map",
    "linalg.copy",
    "linalg.dot",
    "linalg.log",
    "linalg.exp",
    "linalg.abs",
    "linalg.ceil",
    "linalg.floor",
    "linalg.negf",
    "linalg.div",
    "linalg.min",
    "linalg.max",
    "linalg.select",
    "linalg.softmax",
    "linalg.elemwise",
    "linalg.matvec",
    "linalg.vecmat",
    "linalg.add",
    "linalg.sub",
    "linalg.mul",
    # arith ops not supported by parser
    "arith.index_cast",
    "arith.extf",
    "arith.truncf",
    "arith.sitofp",
    "arith.fptosi",
    "arith.uitofp",
    "arith.fptoui",
    "arith.extsi",
    "arith.extui",
    "arith.trunci",
    "arith.bitcast",
    "arith.cmpf",
    "arith.cmpi",
    "arith.select",
    "arith.remf",
    "arith.xori",
    "arith.shli",
    "arith.shrsi",
    "arith.shrui",
    # other dialects
    "transform.",
    "cf.",
    "gpu.",
    "ub.poison",
    "test.",
]


def is_supported(mlir_text: str) -> bool:
    """Check if an MLIR function is in tensor-alive's supported subset."""
    for pattern in UNSUPPORTED_PATTERNS:
        if pattern in mlir_text:
            return False
    # Must have at least one func.func
    if "func.func" not in mlir_text and "func @" not in mlir_text:
        return False
    # Must return a tensor value (skip void functions)
    if "-> tensor<" not in mlir_text and "-> (" not in mlir_text:
        return False
    # Skip functions with no return value
    if "return" not in mlir_text and "func.return" not in mlir_text:
        return False
    return True
