#pragma once
#include "../ir/Program.h"
#include "TensorEncoder.h"
#include "Z3Context.h"
#include <vector>

namespace tensor_alive {

// Encodes individual operations into Z3 constraints.
// For each operation, defines the result tensor's UF in terms of operand UFs.
//
// The encoding follows the formal semantics:
//   For elementwise binary op (e.g., addf):
//     result(i0,...,in) = lhs(i0,...,in) OP rhs(i0,...,in)
//   For constant:
//     result() = constant_value   (scalar)
//   For negf:
//     result(i0,...,in) = -operand(i0,...,in)
class OpEncoder {
public:
  OpEncoder(Z3Ctx &z3ctx, TensorEncoder &tensors, const Program &prog,
            const std::string &prefix);

  // Encode one operation. Returns constraints that define the result.
  // Also registers the result TensorRepr in the TensorEncoder.
  z3::expr encode(const Operation &op);

private:
  Z3Ctx &z3ctx_;
  TensorEncoder &tensors_;
  const Program &prog_;
  std::string prefix_;
  int freshCounter_ = 0;
  int emptyCounter_ = 0; // shared counter for tensor.empty (reset per program)

  std::string freshName(const std::string &base);

  // Specific op encoders
  z3::expr encodeConstant(const Operation &op);
  z3::expr encodeBinaryF(const Operation &op, const std::string &opStr);
  z3::expr encodeNegF(const Operation &op);
  z3::expr encodeBinaryI(const Operation &op, const std::string &opStr);
  z3::expr encodeTranspose(const Operation &op);
  z3::expr encodeBroadcast(const Operation &op);
  z3::expr encodeCollapseShape(const Operation &op);
  z3::expr encodeExpandShape(const Operation &op);
  z3::expr encodeLinalgGeneric(const Operation &op);
  z3::expr encodeTensorEmpty(const Operation &op);
  z3::expr encodeLinalgFill(const Operation &op);
  z3::expr encodeLinalgPack(const Operation &op);
  z3::expr encodeLinalgUnpack(const Operation &op);
  z3::expr encodeFmaF(const Operation &op);
  z3::expr encodeReshape(const Operation &op);
  z3::expr encodeMaxF(const Operation &op);
  z3::expr encodeMinF(const Operation &op);

  // Evaluate the body of a linalg.generic region given block arg values
  z3::expr evalBody(const Operation::LinalgRegion &region,
                    std::vector<z3::expr> blockVals);

  // Map an affine map's result expressions to Z3 index expressions,
  // substituting constants where resultDims[i] == -1.
  std::vector<z3::expr> mapAffineIndices(const Operation::AffineMap &map,
                                         const std::vector<z3::expr> &iterVars);

  // Helper: compute linear index from multi-dimensional indices and shape
  z3::expr linearIndex(const std::vector<z3::expr> &indices,
                       const std::vector<int64_t> &shape);
};

} // namespace tensor_alive
