# Coding Style Guide

## Formatting

### C++ (src/, tests/)
- **Formatter:** `clang-format` with LLVM style (see `.clang-format`).
- Run: `clang-format -i -style=file <files>`

### Python (eval/, fuzz/)
- **Formatter:** `ruff format`.
- Run: `ruff format eval/ fuzz/`

## C++ Conventions

### Naming
| Entity            | Style            | Example                    |
|-------------------|------------------|----------------------------|
| Classes / Structs | `PascalCase`     | `EquivalenceChecker`       |
| Enum types        | `PascalCase`     | `OpKind`                   |
| Enum values       | `PascalCase`     | `OpKind::LinalgMatmul`     |
| Functions         | `camelCase`      | `mkTensorFunc`, `addValue` |
| Local variables   | `camelCase`      | `srcFile`, `timeoutMs`     |
| Member variables  | `camelCase_`     | `ctx_`, `varCounter_`      |
| Constants         | `UPPER_SNAKE`    | (macros / constexpr)       |
| Namespaces        | `snake_case`     | `tensor_alive`             |
| Files             | `PascalCase`     | `EquivalenceChecker.cpp`   |

### Headers
- Use `#pragma once` (no include guards).
- Order includes: project headers first, then standard library, then third-party (`<z3++.h>`). Separate each group with a blank line.

### Namespaces
- All code lives in `namespace tensor_alive { ... }`.
- Close with `} // namespace tensor_alive`.
- Do not indent namespace body.
- `using namespace tensor_alive;` is allowed only in `.cpp` files, never in headers.

### Classes & Structs
- Prefer `struct` for plain data aggregates (no invariants). Use `class` when there are private members or non-trivial invariants.
- Public section first, then private.
- Use `explicit` on single-argument constructors.

### Error Handling
- Throw `std::runtime_error` for unrecoverable errors (parse failures, missing SSA values).
- Use `std::optional` for values that may be absent, not sentinel values (except `-1` for IDs where already established).
- No exceptions in hot SMT-encoding paths; prefer returning error codes or result structs (`CheckResult`).

### Modern C++
- Target C++17.
- Prefer `std::string_view` for non-owning string parameters where appropriate.
- Use `auto` when the type is obvious from the right-hand side; spell it out otherwise.
- Prefer range-based for loops.
- Use structured bindings where they improve clarity.
- Use `std::variant` and `std::optional` over union types or raw pointers.

### Comments
- Use `//` for single-line comments.
- Comment _why_, not _what_.
- No Doxygen unless the function's contract is non-obvious from its name and signature.

## Python Conventions

### General
- Target Python 3.8+.
- Follow PEP 8 (enforced by `ruff format`).
- Use type hints for function signatures (`-> Optional[str]`, `List[Dict]`).
- Module-level docstrings describing purpose and usage.
- Use `snake_case` for functions, variables, and modules.
- Use `PascalCase` for classes.
- Use `UPPER_SNAKE` for module-level constants.

### Imports
- Standard library first, then third-party, then project-local. Separate with blank lines.
- Avoid wildcard imports (`from x import *`).
- Keep imports at the top of the file, not inside functions (unless avoiding circular imports or heavy deps).

### Subprocess Calls
- Always set `timeout` on `subprocess.run`.
- Always use `capture_output=True, text=True`.
- Check `returncode` explicitly rather than using `check=True` (we map exit codes to semantic statuses).

## General Principles

- Keep files focused: one class or closely related set of types per header.
- Prefer composition over inheritance.
- No global mutable state.
- Avoid premature abstraction — three concrete uses before extracting a helper.
