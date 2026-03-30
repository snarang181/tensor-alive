#pragma once
#include "../ir/TensorType.h"
#include <string>
#include <vector>
#include <z3++.h>

namespace tensor_alive {

class Z3Ctx {
public:
  Z3Ctx();

  z3::context &ctx() { return ctx_; }

  // Sort helpers
  z3::sort intSort();
  z3::sort realSort();
  z3::sort boolSort();

  // Create an uninterpreted function for a tensor: Int^rank -> Real
  z3::func_decl mkTensorFunc(const std::string &name, int rank);

  // Create a fresh integer variable (for index quantification)
  z3::expr mkIntVar(const std::string &name);

  // Create a fresh real variable
  z3::expr mkRealVar(const std::string &name);

  // Bounds constraint: 0 <= idx < dim
  z3::expr mkBoundsConstraint(const z3::expr &idx, int64_t dim);

  // Full bounds for a set of indices against a shape
  z3::expr mkShapeBounds(const std::vector<z3::expr> &indices,
                         const std::vector<int64_t> &shape);

  // Create index variables for a given rank
  std::vector<z3::expr> mkIndexVars(const std::string &prefix, int rank);

  // Real constant
  z3::expr mkReal(double v);

  // Integer constant
  z3::expr mkInt(int64_t v);

private:
  z3::context ctx_;
  int varCounter_ = 0;
};

} // namespace tensor_alive
