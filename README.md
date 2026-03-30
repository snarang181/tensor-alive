# tensor-alive

Translation validation for tensor transformations in MLIR.

Given a source and target MLIR program (before/after a compiler pass), `tensor-alive` encodes their semantics into Z3 and checks equivalence — proving correctness or producing a concrete counterexample.

## Building

Requires C++17 and [Z3](https://github.com/Z3Prover/z3).

```bash
# Install Z3 (macOS)
brew install z3

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Usage

```
./build/tensor-alive <source.mlir> <target.mlir> [options]
./build/tensor-alive --parse-only <file.mlir>
```

**Options:**
| Flag | Description |
|------|-------------|
| `--timeout <ms>` | Z3 solver timeout (default: 30000) |
| `--verbose` | Print intermediate encoding info |
| `--dump-smt` | Dump SMT-LIB2 formula |
| `--parse-only` | Check if a file parses successfully |

**Example:**
```bash
./build/tensor-alive tests/mlir_inputs/broadcast_add_src.mlir \
                      tests/mlir_inputs/broadcast_add_tgt.mlir
```

## Supported MLIR Operations

- **Elementwise arithmetic** (`arith.addf`, `arith.mulf`, etc.)
- **Broadcasting** (implicit dimension expansion)
- **Reshape / collapse / expand** (`tensor.collapse_shape`, `tensor.expand_shape`)
- **Transpose** (`linalg.transpose`)
- **Pack / unpack** (`tensor.pack`, `tensor.unpack`, with padding)
- **Reductions** (`linalg.generic` with reduction iterators)
- Subsets of the `arith`, `tensor`, and `linalg` dialects

## How It Works

1. **Parse** — A custom MLIR parser (no LLVM dependency) builds an internal IR
2. **Encode** — Tensors are modeled as functions from indices to Z3 Reals; operations become logical constraints over these functions
3. **Check** — The negation of output equivalence is passed to Z3. UNSAT = equivalent; SAT = counterexample

## Evaluation

The `eval/` directory contains tooling to run validation across MLIR compiler passes:

```bash
# Run evaluation against MLIR test corpus
python eval/run_eval.py --mlir-opt $(which mlir-opt) --corpus <path>

# Generate synthetic benchmarks
python eval/synth_benchmarks.py

# Analyze results
python eval/analyze_results.py eval/results/
```

## Fuzzing

The `fuzz/` directory provides a fuzzing harness for finding bugs:

```bash
# Random MLIR generation + validation
python fuzz/fuzz.py --mode random

# Mutation-based fuzzing over existing corpus
python fuzz/fuzz.py --mode mutate

# Differential testing (pass ordering)
python fuzz/fuzz.py --mode diff
```

## Project Structure

```
src/
  parser/       # MLIR lexer and parser
  ir/           # Internal representation (operations, programs, types)
  smt/          # Z3 encoding (tensor, operation, program encoders)
  checker/      # Equivalence checker and result types
  main.cpp
eval/           # Evaluation framework (Python)
fuzz/           # Fuzzing harness (Python)
tests/          # Hand-written MLIR test cases
```
