#pragma once
#include "../ir/Program.h"
#include "Z3Context.h"
#include <unordered_map>
#include <vector>

namespace tensor_alive {

// Represents a tensor's Z3 encoding.
// For scalars (rank 0): expr holds the single Real value.
// For tensors (rank > 0): func is a UF from Int^rank -> Real.
struct TensorRepr {
  z3::func_decl func;         // UF for this tensor
  std::vector<int64_t> shape; // tensor shape
  bool isScalar;

  // Apply the UF to index variables to get the element expression
  z3::expr apply(const std::vector<z3::expr> &indices) const {
    if (isScalar) {
      z3::expr_vector args(func.ctx());
      return func(args);
    }
    z3::expr_vector args(func.ctx());
    for (auto &idx : indices)
      args.push_back(idx);
    return func(args);
  }

  // For scalars, get the value directly (0-arg function application)
  z3::expr scalarExpr() const {
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
