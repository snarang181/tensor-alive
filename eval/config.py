"""Configuration for the tensor-alive evaluation pipeline."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MLIR_OPT = "/Users/samarthnarang/Downloads/llvm-project/build/bin/mlir-opt"
TENSOR_ALIVE = os.path.join(BASE_DIR, "build", "tensor-alive")

MLIR_TEST_DIRS = [
    "/Users/samarthnarang/Downloads/llvm-project/mlir/test/Dialect/Linalg",
    "/Users/samarthnarang/Downloads/llvm-project/mlir/test/Dialect/Tensor",
]

PASSES = [
    ("canonicalize", "--canonicalize"),
    ("generalize-named-ops", "--linalg-generalize-named-ops"),
    ("fuse-elementwise", "--linalg-fuse-elementwise-ops"),
    ("fold-unit-dims", "--linalg-fold-unit-extent-dims"),
    ("fold-tensor-subset", "--fold-tensor-subset-ops"),
    ("fold-into-elementwise", "--linalg-fold-into-elementwise"),
    ("elementwise-to-linalg", "--convert-elementwise-to-linalg"),
]

TIMEOUT_MLIR_OPT = 5  # seconds
TIMEOUT_TENSOR_ALIVE = 10  # seconds
MAX_PARALLEL = 4

RESULTS_DIR = os.path.join(BASE_DIR, "eval", "results")
SYNTH_DIR = os.path.join(BASE_DIR, "eval", "synthetic")
