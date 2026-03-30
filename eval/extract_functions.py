"""Extract individual functions from MLIR test files.

Handles the MLIR test format:
- Split on '// -----' separators
- Strip RUN/CHECK directives
- Keep #map aliases with their chunk
- Filter to supported subset
"""

import os
import re
from typing import List, Tuple
from preprocess import strip_comments, is_supported


def split_mlir_file(filepath: str) -> List[str]:
    """Split an MLIR test file into individual chunks on '// -----'."""
    with open(filepath) as f:
        content = f.read()
    chunks = re.split(r"\n\s*//\s*-----\s*\n", content)
    return [c.strip() for c in chunks if c.strip()]


def extract_func_name(chunk: str) -> str:
    """Extract function name from a chunk."""
    m = re.search(r"func(?:\.func)?\s+@(\w+)", chunk)
    return m.group(1) if m else "unknown"


def extract_functions_from_file(filepath: str) -> List[Tuple[str, str, str]]:
    """Extract (func_name, clean_mlir, source_file) tuples from a test file."""
    results = []
    chunks = split_mlir_file(filepath)
    basename = os.path.basename(filepath)

    for i, chunk in enumerate(chunks):
        clean = strip_comments(chunk)
        if not clean.strip():
            continue
        func_name = extract_func_name(clean)
        label = f"{basename}::{func_name}_{i}"
        results.append((label, clean, filepath))

    return results


def extract_corpus(test_dirs: List[str]) -> List[Tuple[str, str, str]]:
    """Extract all functions from MLIR test directories."""
    all_functions = []
    for test_dir in test_dirs:
        if not os.path.isdir(test_dir):
            print(f"Warning: {test_dir} not found, skipping")
            continue
        for fname in sorted(os.listdir(test_dir)):
            if not fname.endswith(".mlir"):
                continue
            filepath = os.path.join(test_dir, fname)
            try:
                funcs = extract_functions_from_file(filepath)
                all_functions.extend(funcs)
            except Exception as e:
                print(f"Warning: failed to process {filepath}: {e}")
    return all_functions


def filter_supported(
    functions: List[Tuple[str, str, str]],
) -> List[Tuple[str, str, str]]:
    """Filter to only supported functions."""
    return [(name, mlir, src) for name, mlir, src in functions if is_supported(mlir)]


if __name__ == "__main__":
    import config

    print("Extracting MLIR test corpus...")
    all_funcs = extract_corpus(config.MLIR_TEST_DIRS)
    print(f"  Total chunks: {len(all_funcs)}")
    supported = filter_supported(all_funcs)
    print(f"  Supported (post-filter): {len(supported)}")
    for name, _, _ in supported[:20]:
        print(f"    {name}")
    if len(supported) > 20:
        print(f"    ... and {len(supported) - 20} more")
