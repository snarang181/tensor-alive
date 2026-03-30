#include "Z3Context.h"
#include <sstream>

namespace tensor_alive {

Z3Ctx::Z3Ctx() {}

z3::sort Z3Ctx::intSort() { return ctx_.int_sort(); }
z3::sort Z3Ctx::realSort() { return ctx_.real_sort(); }
z3::sort Z3Ctx::boolSort() { return ctx_.bool_sort(); }

z3::func_decl Z3Ctx::mkTensorFunc(const std::string &name, int rank) {
  z3::sort_vector domain(ctx_);
  for (int i = 0; i < rank; i++) {
    domain.push_back(intSort());
  }
  if (rank == 0) {
    // Scalar: zero-argument function (constant)
    return z3::function(name.c_str(), domain, realSort());
  }
  return z3::function(name.c_str(), domain, realSort());
}

z3::expr Z3Ctx::mkIntVar(const std::string &name) {
  return ctx_.int_const(name.c_str());
}

z3::expr Z3Ctx::mkRealVar(const std::string &name) {
  return ctx_.real_const(name.c_str());
}

z3::expr Z3Ctx::mkBoundsConstraint(const z3::expr &idx, int64_t dim) {
  return idx >= 0 && idx < ctx_.int_val(static_cast<int>(dim));
}

z3::expr Z3Ctx::mkShapeBounds(const std::vector<z3::expr> &indices,
                              const std::vector<int64_t> &shape) {
  z3::expr result = ctx_.bool_val(true);
  for (size_t i = 0; i < indices.size(); i++) {
    result = result && mkBoundsConstraint(indices[i], shape[i]);
  }
  return result;
}

std::vector<z3::expr> Z3Ctx::mkIndexVars(const std::string &prefix, int rank) {
  std::vector<z3::expr> vars;
  for (int i = 0; i < rank; i++) {
    vars.push_back(mkIntVar(prefix + "_i" + std::to_string(i)));
  }
  return vars;
}

z3::expr Z3Ctx::mkReal(double v) {
  // Use rational representation for exactness
  std::ostringstream oss;
  oss << std::fixed << v;
  return ctx_.real_val(oss.str().c_str());
}

z3::expr Z3Ctx::mkInt(int64_t v) {
  return ctx_.int_val(static_cast<int64_t>(v));
}

} // namespace tensor_alive
