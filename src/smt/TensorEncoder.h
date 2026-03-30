#pragma once
#include "../ir/Program.h"
#include "Z3Context.h"
#include <functional>
#include <unordered_map>
#include <vector>

namespace tensor_alive {

// Represents a tensor's Z3 encoding.
// Two modes:
// 1. UF-based: a Z3 uninterpreted function (for inputs, tensor.empty)
// 2. Expr-based: a lambda that computes the element at given indices
//    by composing operand expressions. No quantifiers needed.
struct TensorRepr {
  z3::func_decl func;         // UF for this tensor (used for inputs)
  std::vector<int64_t> shape; // tensor shape
  bool isScalar;

  // Optional: expression-based element computation.
  // If set, apply() uses this instead of the UF.
  using ElemFn = std::function<z3::expr(const std::vector<z3::expr> &)>;
  ElemFn elemFn;

  // Apply to index variables to get the element expression
  z3::expr apply(const std::vector<z3::expr> &indices) const {
    if (elemFn) {
      return elemFn(indices);
    }
    if (isScalar) {
      z3::expr_vector args(func.ctx());
      return func(args);
    }
    z3::expr_vector args(func.ctx());
    for (auto &idx : indices)
      args.push_back(idx);
    return func(args);
  }

  // For scalars, get the value directly
  z3::expr scalarExpr() const {
    if (elemFn) {
      return elemFn({});
    }
    z3::expr_vector args(func.ctx());
    return func(args);
  }
};

class TensorEncoder {
public:
  TensorEncoder(Z3Ctx &z3ctx, const std::string &prefix);

  // Create encoding for a program input (uninterpreted function)
  TensorRepr encodeInput(const Value &val);

  // Register an encoding for an operation result
  void registerResult(int valueId, TensorRepr repr);

  // Get the encoding for a value
  const TensorRepr &getRepr(int valueId) const;

  bool hasRepr(int valueId) const;

private:
  Z3Ctx &z3ctx_;
  std::string prefix_;
  std::unordered_map<int, TensorRepr> encodings_;
};

} // namespace tensor_alive
