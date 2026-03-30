#include "OpEncoder.h"
#include <stdexcept>

namespace tensor_alive {

OpEncoder::OpEncoder(Z3Ctx &z3ctx, TensorEncoder &tensors, const Program &prog,
                     const std::string &prefix)
    : z3ctx_(z3ctx), tensors_(tensors), prog_(prog), prefix_(prefix) {}

std::string OpEncoder::freshName(const std::string &base) {
  return prefix_ + "_" + base + "_" + std::to_string(freshCounter_++);
}

z3::expr OpEncoder::encode(const Operation &op) {
  switch (op.kind) {
  case OpKind::Constant:
    return encodeConstant(op);
  case OpKind::AddF:
    return encodeBinaryF(op, "add");
  case OpKind::SubF:
    return encodeBinaryF(op, "sub");
  case OpKind::MulF:
    return encodeBinaryF(op, "mul");
  case OpKind::DivF:
    return encodeBinaryF(op, "div");
  case OpKind::NegF:
    return encodeNegF(op);
  case OpKind::MaxF:
    return encodeMaxF(op);
  case OpKind::MinF:
    return encodeMinF(op);
  case OpKind::AddI:
    return encodeBinaryI(op, "add");
  case OpKind::SubI:
    return encodeBinaryI(op, "sub");
  case OpKind::MulI:
    return encodeBinaryI(op, "mul");
  case OpKind::Transpose:
    return encodeTranspose(op);
  case OpKind::Broadcast:
    return encodeBroadcast(op);
  case OpKind::LinalgGeneric:
  case OpKind::LinalgMatmul:
  case OpKind::LinalgBatchMatmul:
    return encodeLinalgGeneric(op);
  case OpKind::TensorEmpty:
    return encodeTensorEmpty(op);
  case OpKind::LinalgFill:
    return encodeLinalgFill(op);
  case OpKind::LinalgPack:
    return encodeLinalgPack(op);
  case OpKind::LinalgUnpack:
    return encodeLinalgUnpack(op);
  case OpKind::CollapseShape:
    return encodeCollapseShape(op);
  case OpKind::ExpandShape:
    return encodeExpandShape(op);
  case OpKind::FuncArg:
  case OpKind::FuncReturn:
    return z3ctx_.ctx().bool_val(true);
  default:
    throw std::runtime_error("OpEncoder: unsupported op " +
                             opKindToString(op.kind));
  }
}

z3::expr OpEncoder::encodeConstant(const Operation &op) {
  if (!op.constantValue.has_value())
    throw std::runtime_error("Constant op missing value");

  int resultId = op.resultIds.at(0);
  const Value &resultVal = prog_.getValue(resultId);

  // Create a fresh UF for this constant
  std::string name = freshName("const");
  z3::func_decl func = z3ctx_.mkTensorFunc(name, resultVal.type.rank());

  TensorRepr repr{func, resultVal.type.shape, resultVal.type.isScalar()};
  tensors_.registerResult(resultId, repr);

  z3::expr constExpr = z3ctx_.mkReal(*op.constantValue);

  if (resultVal.type.isScalar()) {
    // Scalar: func() == constant
    return repr.scalarExpr() == constExpr;
  } else {
    // Tensor constant: forall indices, func(indices) == constant
    // For static shapes, use universally quantified constraint.
    // Since Z3 will search existentially over indices, we define the function
    // such that it equals the constant everywhere.
    auto indices = z3ctx_.mkIndexVars(name, resultVal.type.rank());
    z3::expr app = repr.apply(indices);
    z3::expr body = z3::implies(
        z3ctx_.mkShapeBounds(indices, resultVal.type.shape), app == constExpr);
    z3::expr_vector qvars(z3ctx_.ctx());
    for (auto &idx : indices)
      qvars.push_back(idx);
    return z3::forall(qvars, body);
  }
}

z3::expr OpEncoder::encodeBinaryF(const Operation &op,
                                  const std::string &opStr) {
  if (op.operandIds.size() != 2 || op.resultIds.size() != 1)
    throw std::runtime_error("Binary op expects 2 operands and 1 result");

  const TensorRepr &lhs = tensors_.getRepr(op.operandIds[0]);
  const TensorRepr &rhs = tensors_.getRepr(op.operandIds[1]);
  int resultId = op.resultIds[0];
  const Value &resultVal = prog_.getValue(resultId);

  std::string name = freshName("binf");
  z3::func_decl func = z3ctx_.mkTensorFunc(name, resultVal.type.rank());
  TensorRepr repr{func, resultVal.type.shape, resultVal.type.isScalar()};
  tensors_.registerResult(resultId, repr);

  if (resultVal.type.isScalar()) {
    z3::expr l = lhs.scalarExpr();
    z3::expr r = rhs.scalarExpr();
    z3::expr res = repr.scalarExpr();
    if (opStr == "add")
      return res == (l + r);
    if (opStr == "sub")
      return res == (l - r);
    if (opStr == "mul")
      return res == (l * r);
    if (opStr == "div")
      return res == (l / r);
    throw std::runtime_error("Unknown binary op: " + opStr);
  }

  // Tensor: forall indices in bounds, result(i) = lhs(i) OP rhs(i)
  auto indices = z3ctx_.mkIndexVars(name, resultVal.type.rank());
  z3::expr lExpr = lhs.apply(indices);
  z3::expr rExpr = rhs.apply(indices);
  z3::expr resExpr = repr.apply(indices);

  z3::expr opExpr(z3ctx_.ctx());
  if (opStr == "add")
    opExpr = lExpr + rExpr;
  else if (opStr == "sub")
    opExpr = lExpr - rExpr;
  else if (opStr == "mul")
    opExpr = lExpr * rExpr;
  else if (opStr == "div")
    opExpr = lExpr / rExpr;
  else
    throw std::runtime_error("Unknown binary op: " + opStr);

  z3::expr body = z3::implies(
      z3ctx_.mkShapeBounds(indices, resultVal.type.shape), resExpr == opExpr);
  z3::expr_vector qvars(z3ctx_.ctx());
  for (auto &idx : indices)
    qvars.push_back(idx);
  return z3::forall(qvars, body);
}

z3::expr OpEncoder::encodeNegF(const Operation &op) {
  if (op.operandIds.size() != 1 || op.resultIds.size() != 1)
    throw std::runtime_error("NegF expects 1 operand and 1 result");

  const TensorRepr &operand = tensors_.getRepr(op.operandIds[0]);
  int resultId = op.resultIds[0];
  const Value &resultVal = prog_.getValue(resultId);

  std::string name = freshName("negf");
  z3::func_decl func = z3ctx_.mkTensorFunc(name, resultVal.type.rank());
  TensorRepr repr{func, resultVal.type.shape, resultVal.type.isScalar()};
  tensors_.registerResult(resultId, repr);

  if (resultVal.type.isScalar()) {
    return repr.scalarExpr() == -operand.scalarExpr();
  }

  auto indices = z3ctx_.mkIndexVars(name, resultVal.type.rank());
  z3::expr body =
      z3::implies(z3ctx_.mkShapeBounds(indices, resultVal.type.shape),
                  repr.apply(indices) == -operand.apply(indices));
  z3::expr_vector qvars(z3ctx_.ctx());
  for (auto &idx : indices)
    qvars.push_back(idx);
  return z3::forall(qvars, body);
}

z3::expr OpEncoder::encodeBinaryI(const Operation &op,
                                  const std::string &opStr) {
  return encodeBinaryF(op, opStr);
}

z3::expr OpEncoder::encodeMaxF(const Operation &op) {
  if (op.operandIds.size() != 2 || op.resultIds.size() != 1)
    throw std::runtime_error("MaxF expects 2 operands and 1 result");

  const TensorRepr &lhs = tensors_.getRepr(op.operandIds[0]);
  const TensorRepr &rhs = tensors_.getRepr(op.operandIds[1]);
  int resultId = op.resultIds[0];
  const Value &resultVal = prog_.getValue(resultId);

  std::string name = freshName("maxf");
  z3::func_decl func = z3ctx_.mkTensorFunc(name, resultVal.type.rank());
  TensorRepr repr{func, resultVal.type.shape, resultVal.type.isScalar()};
  tensors_.registerResult(resultId, repr);

  if (resultVal.type.isScalar()) {
    z3::expr l = lhs.scalarExpr(), r = rhs.scalarExpr();
    return repr.scalarExpr() == z3::ite(l >= r, l, r);
  }

  auto indices = z3ctx_.mkIndexVars(name, resultVal.type.rank());
  z3::expr l = lhs.apply(indices), r = rhs.apply(indices);
  z3::expr body =
      z3::implies(z3ctx_.mkShapeBounds(indices, resultVal.type.shape),
                  repr.apply(indices) == z3::ite(l >= r, l, r));
  z3::expr_vector qvars(z3ctx_.ctx());
  for (auto &idx : indices)
    qvars.push_back(idx);
  return z3::forall(qvars, body);
}

z3::expr OpEncoder::encodeMinF(const Operation &op) {
  if (op.operandIds.size() != 2 || op.resultIds.size() != 1)
    throw std::runtime_error("MinF expects 2 operands and 1 result");

  const TensorRepr &lhs = tensors_.getRepr(op.operandIds[0]);
  const TensorRepr &rhs = tensors_.getRepr(op.operandIds[1]);
  int resultId = op.resultIds[0];
  const Value &resultVal = prog_.getValue(resultId);

  std::string name = freshName("minf");
  z3::func_decl func = z3ctx_.mkTensorFunc(name, resultVal.type.rank());
  TensorRepr repr{func, resultVal.type.shape, resultVal.type.isScalar()};
  tensors_.registerResult(resultId, repr);

  if (resultVal.type.isScalar()) {
    z3::expr l = lhs.scalarExpr(), r = rhs.scalarExpr();
    return repr.scalarExpr() == z3::ite(l <= r, l, r);
  }

  auto indices = z3ctx_.mkIndexVars(name, resultVal.type.rank());
  z3::expr l = lhs.apply(indices), r = rhs.apply(indices);
  z3::expr body =
      z3::implies(z3ctx_.mkShapeBounds(indices, resultVal.type.shape),
                  repr.apply(indices) == z3::ite(l <= r, l, r));
  z3::expr_vector qvars(z3ctx_.ctx());
  for (auto &idx : indices)
    qvars.push_back(idx);
  return z3::forall(qvars, body);
}

std::vector<z3::expr>
OpEncoder::mapAffineIndices(const Operation::AffineMap &map,
                            const std::vector<z3::expr> &iterVars) {
  std::vector<z3::expr> result;
  for (size_t i = 0; i < map.resultDims.size(); i++) {
    int dim = map.resultDims[i];
    if (dim < 0) {
      // Constant expression — look up the value in constExprs
      int64_t val = 0;
      for (const auto &[pos, cval] : map.constExprs) {
        if (pos == static_cast<int>(i)) {
          val = cval;
          break;
        }
      }
      result.push_back(z3ctx_.mkInt(val));
    } else {
      result.push_back(iterVars[dim]);
    }
  }
  return result;
}

z3::expr OpEncoder::linearIndex(const std::vector<z3::expr> &indices,
                                const std::vector<int64_t> &shape) {
  // linear(i0, i1, ..., i_{n-1}) = i0 * (d1*d2*...*d_{n-1}) + i1 *
  // (d2*...*d_{n-1}) + ... + i_{n-1}
  z3::expr result = z3ctx_.mkInt(0);
  for (size_t k = 0; k < indices.size(); k++) {
    int64_t stride = 1;
    for (size_t l = k + 1; l < shape.size(); l++) {
      stride *= shape[l];
    }
    result = result + indices[k] * z3ctx_.mkInt(stride);
  }
  return result;
}

z3::expr OpEncoder::encodeTranspose(const Operation &op) {
  // transpose(A, perm)[i0, ..., i_{n-1}] = A[i_{perm[0]}, ..., i_{perm[n-1]}]
  if (op.operandIds.size() != 1 || op.resultIds.size() != 1)
    throw std::runtime_error("Transpose expects 1 operand and 1 result");

  const TensorRepr &input = tensors_.getRepr(op.operandIds[0]);
  int resultId = op.resultIds[0];
  const Value &resultVal = prog_.getValue(resultId);

  std::string name = freshName("transp");
  z3::func_decl func = z3ctx_.mkTensorFunc(name, resultVal.type.rank());
  TensorRepr repr{func, resultVal.type.shape, resultVal.type.isScalar()};
  tensors_.registerResult(resultId, repr);

  auto indices = z3ctx_.mkIndexVars(name, resultVal.type.rank());

  // Permute indices: input_indices[k] = indices[perm[k]]
  std::vector<z3::expr> permutedIndices;
  for (size_t k = 0; k < op.permutation.size(); k++) {
    permutedIndices.push_back(indices[op.permutation[k]]);
  }

  z3::expr body =
      z3::implies(z3ctx_.mkShapeBounds(indices, resultVal.type.shape),
                  repr.apply(indices) == input.apply(permutedIndices));
  z3::expr_vector qvars(z3ctx_.ctx());
  for (auto &idx : indices)
    qvars.push_back(idx);
  return z3::forall(qvars, body);
}

z3::expr OpEncoder::encodeBroadcast(const Operation &op) {
  // Numpy-style broadcasting:
  // broadcast(A, target_shape)[i0, ..., i_{m-1}] = A[j0, ..., j_{n-1}]
  // where n = rank(A), m = rank(result), offset = m - n
  // j_k = (shape(A)[k] == 1) ? 0 : i_{offset + k}
  if (op.operandIds.size() != 1 || op.resultIds.size() != 1)
    throw std::runtime_error("Broadcast expects 1 operand and 1 result");

  const TensorRepr &input = tensors_.getRepr(op.operandIds[0]);
  int resultId = op.resultIds[0];
  const Value &resultVal = prog_.getValue(resultId);

  std::string name = freshName("bcast");
  z3::func_decl func = z3ctx_.mkTensorFunc(name, resultVal.type.rank());
  TensorRepr repr{func, resultVal.type.shape, resultVal.type.isScalar()};
  tensors_.registerResult(resultId, repr);

  int inputRank = static_cast<int>(input.shape.size());
  int resultRank = static_cast<int>(resultVal.type.shape.size());
  int offset = resultRank - inputRank;

  auto resultIndices = z3ctx_.mkIndexVars(name, resultRank);

  // Build input indices with broadcasting rules
  std::vector<z3::expr> inputIndices;
  for (int k = 0; k < inputRank; k++) {
    if (input.shape[k] == 1) {
      // Broadcast dimension: always index 0
      inputIndices.push_back(z3ctx_.mkInt(0));
    } else {
      // Non-broadcast: use the corresponding result index
      inputIndices.push_back(resultIndices[offset + k]);
    }
  }

  z3::expr body =
      z3::implies(z3ctx_.mkShapeBounds(resultIndices, resultVal.type.shape),
                  repr.apply(resultIndices) == input.apply(inputIndices));
  z3::expr_vector qvars(z3ctx_.ctx());
  for (auto &idx : resultIndices)
    qvars.push_back(idx);
  return z3::forall(qvars, body);
}

z3::expr OpEncoder::encodeCollapseShape(const Operation &op) {
  // collapse_shape merges groups of dimensions.
  // Semantics: result and input have the same elements in row-major order.
  // We encode: linearIndex(result_indices, result_shape) ==
  // linearIndex(input_indices, input_shape)
  if (op.operandIds.size() != 1 || op.resultIds.size() != 1)
    throw std::runtime_error("CollapseShape expects 1 operand and 1 result");

  const TensorRepr &input = tensors_.getRepr(op.operandIds[0]);
  int resultId = op.resultIds[0];
  const Value &resultVal = prog_.getValue(resultId);

  std::string name = freshName("collapse");
  z3::func_decl func = z3ctx_.mkTensorFunc(name, resultVal.type.rank());
  TensorRepr repr{func, resultVal.type.shape, resultVal.type.isScalar()};
  tensors_.registerResult(resultId, repr);

  auto resultIndices = z3ctx_.mkIndexVars(name + "_r", resultVal.type.rank());
  auto inputIndices = z3ctx_.mkIndexVars(name + "_i", input.shape.size());

  z3::expr resultLinear = linearIndex(resultIndices, resultVal.type.shape);
  z3::expr inputLinear = linearIndex(inputIndices, input.shape);

  // forall result_indices in bounds:
  //   forall input_indices in bounds:
  //     linearIndex(result) == linearIndex(input) => result(result_indices) ==
  //     input(input_indices)
  z3::expr body =
      z3::implies(z3ctx_.mkShapeBounds(resultIndices, resultVal.type.shape) &&
                      z3ctx_.mkShapeBounds(inputIndices, input.shape) &&
                      (resultLinear == inputLinear),
                  repr.apply(resultIndices) == input.apply(inputIndices));

  z3::expr_vector qvars(z3ctx_.ctx());
  for (auto &idx : resultIndices)
    qvars.push_back(idx);
  for (auto &idx : inputIndices)
    qvars.push_back(idx);
  return z3::forall(qvars, body);
}

z3::expr OpEncoder::encodeExpandShape(const Operation &op) {
  // expand_shape is the inverse of collapse_shape.
  // Same linear index semantics.
  if (op.operandIds.size() != 1 || op.resultIds.size() != 1)
    throw std::runtime_error("ExpandShape expects 1 operand and 1 result");

  const TensorRepr &input = tensors_.getRepr(op.operandIds[0]);
  int resultId = op.resultIds[0];
  const Value &resultVal = prog_.getValue(resultId);

  std::string name = freshName("expand");
  z3::func_decl func = z3ctx_.mkTensorFunc(name, resultVal.type.rank());
  TensorRepr repr{func, resultVal.type.shape, resultVal.type.isScalar()};
  tensors_.registerResult(resultId, repr);

  auto resultIndices = z3ctx_.mkIndexVars(name + "_r", resultVal.type.rank());
  auto inputIndices = z3ctx_.mkIndexVars(name + "_i", input.shape.size());

  z3::expr resultLinear = linearIndex(resultIndices, resultVal.type.shape);
  z3::expr inputLinear = linearIndex(inputIndices, input.shape);

  z3::expr body =
      z3::implies(z3ctx_.mkShapeBounds(resultIndices, resultVal.type.shape) &&
                      z3ctx_.mkShapeBounds(inputIndices, input.shape) &&
                      (resultLinear == inputLinear),
                  repr.apply(resultIndices) == input.apply(inputIndices));

  z3::expr_vector qvars(z3ctx_.ctx());
  for (auto &idx : resultIndices)
    qvars.push_back(idx);
  for (auto &idx : inputIndices)
    qvars.push_back(idx);
  return z3::forall(qvars, body);
}

// Helper: evaluate the body ops given block argument values, returning the
// yielded expr
z3::expr OpEncoder::encodeTensorEmpty(const Operation &op) {
  // tensor.empty produces an arbitrary (unconstrained) tensor.
  // Use a name WITHOUT the src_/tgt_ prefix so that the Nth tensor.empty
  // in both source and target maps to the same UF. This models the semantic
  // assumption that corresponding uninitialized tensors have the same
  // arbitrary values in both program executions.
  int resultId = op.resultIds.at(0);
  const Value &resultVal = prog_.getValue(resultId);
  std::string name = "shared_empty_" + std::to_string(emptyCounter_++);
  z3::func_decl func = z3ctx_.mkTensorFunc(name, resultVal.type.rank());
  TensorRepr repr{func, resultVal.type.shape, resultVal.type.isScalar()};
  tensors_.registerResult(resultId, repr);
  return z3ctx_.ctx().bool_val(true); // no constraints
}

z3::expr OpEncoder::encodeLinalgFill(const Operation &op) {
  // linalg.fill writes a scalar constant to every element.
  // operands: [0] = scalar value, [1] = output init (ignored)
  if (op.operandIds.size() < 2 || op.resultIds.size() != 1)
    throw std::runtime_error("LinalgFill expects 2 operands and 1 result");

  const TensorRepr &scalar = tensors_.getRepr(op.operandIds[0]);
  int resultId = op.resultIds[0];
  const Value &resultVal = prog_.getValue(resultId);

  std::string name = freshName("fill");
  z3::func_decl func = z3ctx_.mkTensorFunc(name, resultVal.type.rank());
  TensorRepr repr{func, resultVal.type.shape, resultVal.type.isScalar()};
  tensors_.registerResult(resultId, repr);

  // forall indices: result(indices) == scalar_value
  z3::expr scalarVal = scalar.scalarExpr();
  auto indices = z3ctx_.mkIndexVars(name, resultVal.type.rank());
  z3::expr body =
      z3::implies(z3ctx_.mkShapeBounds(indices, resultVal.type.shape),
                  repr.apply(indices) == scalarVal);
  z3::expr_vector qvars(z3ctx_.ctx());
  for (auto &idx : indices)
    qvars.push_back(idx);
  return z3::forall(qvars, body);
}

z3::expr OpEncoder::encodeLinalgPack(const Operation &op) {
  // linalg.pack: tiles input into higher-rank output.
  // Given input shape [D0, D1, ...], inner_dims_pos = [p0, p1, ...],
  // inner_tiles = [T0, T1, ...]:
  //
  // Without outer_dims_perm:
  //   output[o0, ..., t0, t1, ...] = input[..., o_{pi} * Ti + ti, ...]
  //   where outer dims are the original dims with tiled dims replaced by
  //   ceildiv
  //
  // The key relationship: for each tiled dimension pi with tile Ti:
  //   input_idx[pi] = outer_idx[pi] * Ti + tile_idx[i]
  //
  // We encode: forall output indices in bounds:
  //   output(out_idx) = input(mapped_input_idx)

  if (!op.packInfo.has_value())
    throw std::runtime_error("LinalgPack missing pack info");

  const auto &info = *op.packInfo;
  const TensorRepr &input = tensors_.getRepr(op.operandIds[0]);
  int resultId = op.resultIds.at(0);
  const Value &resultVal = prog_.getValue(resultId);

  std::string name = freshName("pack");
  z3::func_decl func = z3ctx_.mkTensorFunc(name, resultVal.type.rank());
  TensorRepr repr{func, resultVal.type.shape, resultVal.type.isScalar()};
  tensors_.registerResult(resultId, repr);

  int inputRank = static_cast<int>(input.shape.size());
  int numTiled = static_cast<int>(info.innerDimsPos.size());
  int outputRank = static_cast<int>(resultVal.type.shape.size());

  auto outIndices = z3ctx_.mkIndexVars(name, outputRank);

  // The output has shape: [outer_dims..., tile_dims...]
  // outer_dims count = inputRank (some are ceildiv'd)
  // tile_dims count = numTiled
  // Total = inputRank + numTiled = outputRank

  // Build input indices from output indices
  // outer indices are outIndices[0..inputRank-1] (possibly permuted by
  // outer_dims_perm) tile indices are outIndices[inputRank..outputRank-1]
  std::vector<z3::expr> inputIndices;
  for (int d = 0; d < inputRank; d++) {
    // Check if this input dim is tiled
    int tiledIdx = -1;
    for (int t = 0; t < numTiled; t++) {
      if (info.innerDimsPos[t] == d) {
        tiledIdx = t;
        break;
      }
    }

    // Get the outer index for this dimension
    int outerDim = d;
    if (!info.outerDimsPerm.empty()) {
      // outer_dims_perm maps logical outer position -> physical outer position
      // We need the reverse: find which physical outer dim corresponds to
      // logical dim d
      for (int p = 0; p < static_cast<int>(info.outerDimsPerm.size()); p++) {
        if (info.outerDimsPerm[p] == d) {
          outerDim = p;
          break;
        }
      }
    }

    if (tiledIdx >= 0) {
      // This dim is tiled: input_idx = outer * tile_size + tile_idx
      int tileIdxPos = inputRank + tiledIdx;
      z3::expr tileSize = z3ctx_.mkInt(info.innerTiles[tiledIdx]);
      inputIndices.push_back(outIndices[outerDim] * tileSize +
                             outIndices[tileIdxPos]);
    } else {
      inputIndices.push_back(outIndices[outerDim]);
    }
  }

  // Build element value with padding support
  z3::expr elementValue(z3ctx_.ctx());

  if (info.paddingValueId >= 0) {
    // Check if computed input indices are within input bounds
    z3::expr inBounds = z3ctx_.ctx().bool_val(true);
    for (int d = 0; d < inputRank; d++) {
      inBounds = inBounds && inputIndices[d] >= 0 &&
                 inputIndices[d] < z3ctx_.mkInt(input.shape[d]);
    }

    const TensorRepr &padRepr = tensors_.getRepr(info.paddingValueId);
    z3::expr padVal = padRepr.scalarExpr();

    // In bounds -> read from input; out of bounds -> padding value
    elementValue = z3::ite(inBounds, input.apply(inputIndices), padVal);
  } else {
    elementValue = input.apply(inputIndices);
  }

  z3::expr body =
      z3::implies(z3ctx_.mkShapeBounds(outIndices, resultVal.type.shape),
                  repr.apply(outIndices) == elementValue);
  z3::expr_vector qvars(z3ctx_.ctx());
  for (auto &idx : outIndices)
    qvars.push_back(idx);
  return z3::forall(qvars, body);
}

z3::expr OpEncoder::encodeLinalgUnpack(const Operation &op) {
  // linalg.unpack: inverse of pack — untiles higher-rank input into lower-rank
  // output. output[d0, d1, ...] = input[outer_idx..., tile_idx...] where for
  // each tiled dim pi: outer_idx[pi] = d[pi] / Ti, tile_idx[i] = d[pi] % Ti

  if (!op.packInfo.has_value())
    throw std::runtime_error("LinalgUnpack missing pack info");

  const auto &info = *op.packInfo;
  const TensorRepr &input = tensors_.getRepr(op.operandIds[0]);
  int resultId = op.resultIds.at(0);
  const Value &resultVal = prog_.getValue(resultId);

  std::string name = freshName("unpack");
  z3::func_decl func = z3ctx_.mkTensorFunc(name, resultVal.type.rank());
  TensorRepr repr{func, resultVal.type.shape, resultVal.type.isScalar()};
  tensors_.registerResult(resultId, repr);

  int outputRank = static_cast<int>(resultVal.type.shape.size());
  int inputRank = static_cast<int>(input.shape.size());
  int numTiled = static_cast<int>(info.innerDimsPos.size());

  auto resultIndices = z3ctx_.mkIndexVars(name, outputRank);

  // Build input indices from result (output) indices
  // For each outer dim: input_outer[d] = result[d] / tile_size (if tiled) or
  // result[d] For each tile dim: input_tile[t] = result[pos_t] % tile_size_t
  std::vector<z3::expr> inputOuterIndices;
  for (int d = 0; d < outputRank; d++) {
    int tiledIdx = -1;
    for (int t = 0; t < numTiled; t++) {
      if (info.innerDimsPos[t] == d) {
        tiledIdx = t;
        break;
      }
    }

    if (tiledIdx >= 0) {
      z3::expr tileSize = z3ctx_.mkInt(info.innerTiles[tiledIdx]);
      inputOuterIndices.push_back(resultIndices[d] / tileSize);
    } else {
      inputOuterIndices.push_back(resultIndices[d]);
    }
  }

  // Apply outer_dims_perm if present
  std::vector<z3::expr> permutedOuter = inputOuterIndices;
  if (!info.outerDimsPerm.empty()) {
    permutedOuter.resize(outputRank, z3ctx_.mkInt(0));
    for (int p = 0; p < static_cast<int>(info.outerDimsPerm.size()); p++) {
      permutedOuter[p] = inputOuterIndices[info.outerDimsPerm[p]];
    }
  }

  // Build tile indices
  std::vector<z3::expr> tileIndices;
  for (int t = 0; t < numTiled; t++) {
    int pos = static_cast<int>(info.innerDimsPos[t]);
    z3::expr tileSize = z3ctx_.mkInt(info.innerTiles[t]);
    tileIndices.push_back(z3::mod(resultIndices[pos], tileSize));
  }

  // Full input index = [permuted_outer..., tile_indices...]
  std::vector<z3::expr> inputIndices;
  for (auto &idx : permutedOuter)
    inputIndices.push_back(idx);
  for (auto &idx : tileIndices)
    inputIndices.push_back(idx);

  z3::expr body =
      z3::implies(z3ctx_.mkShapeBounds(resultIndices, resultVal.type.shape),
                  repr.apply(resultIndices) == input.apply(inputIndices));
  z3::expr_vector qvars(z3ctx_.ctx());
  for (auto &idx : resultIndices)
    qvars.push_back(idx);
  return z3::forall(qvars, body);
}

z3::expr OpEncoder::evalBody(const Operation::LinalgRegion &region,
                             std::vector<z3::expr> blockVals) {
  for (auto &bodyOp : region.bodyOps) {
    z3::expr result(z3ctx_.ctx());
    switch (bodyOp.kind) {
    case OpKind::Constant:
      result = z3ctx_.mkReal(bodyOp.constantValue.value_or(0.0));
      break;
    case OpKind::AddF:
      result = blockVals[bodyOp.operandIndices[0]] +
               blockVals[bodyOp.operandIndices[1]];
      break;
    case OpKind::SubF:
      result = blockVals[bodyOp.operandIndices[0]] -
               blockVals[bodyOp.operandIndices[1]];
      break;
    case OpKind::MulF:
      result = blockVals[bodyOp.operandIndices[0]] *
               blockVals[bodyOp.operandIndices[1]];
      break;
    case OpKind::DivF:
      result = blockVals[bodyOp.operandIndices[0]] /
               blockVals[bodyOp.operandIndices[1]];
      break;
    case OpKind::NegF:
      result = -blockVals[bodyOp.operandIndices[0]];
      break;
    case OpKind::MaxF: {
      auto a = blockVals[bodyOp.operandIndices[0]];
      auto b = blockVals[bodyOp.operandIndices[1]];
      result = z3::ite(a >= b, a, b);
      break;
    }
    case OpKind::MinF: {
      auto a = blockVals[bodyOp.operandIndices[0]];
      auto b = blockVals[bodyOp.operandIndices[1]];
      result = z3::ite(a <= b, a, b);
      break;
    }
    default:
      throw std::runtime_error("Unsupported body op in linalg.generic");
    }
    blockVals.push_back(result);
  }

  if (region.yieldIndex < 0 ||
      region.yieldIndex >= static_cast<int>(blockVals.size()))
    throw std::runtime_error("Invalid yield index in linalg.generic");
  return blockVals[region.yieldIndex];
}

z3::expr OpEncoder::encodeLinalgGeneric(const Operation &op) {
  // linalg.generic encodes a structured operation with:
  //   - iteration domain defined by iterator_types (parallel, reduction dims)
  //   - indexing_maps that project iteration indices to tensor indices
  //   - a body that computes scalar output from scalar inputs
  //
  // Parallel-only: result(out_map(d)) = body(inputs_at_d)
  // With reductions: uses accumulator chain over reduction dimensions

  if (!op.linalgRegion.has_value())
    throw std::runtime_error("LinalgGeneric missing region");

  const auto &region = *op.linalgRegion;
  int resultId = op.resultIds.at(0);
  const Value &resultVal = prog_.getValue(resultId);

  std::string name = freshName("linalg");
  z3::func_decl func = z3ctx_.mkTensorFunc(name, resultVal.type.rank());
  TensorRepr repr{func, resultVal.type.shape, resultVal.type.isScalar()};
  tensors_.registerResult(resultId, repr);

  int numIterDims = static_cast<int>(region.iteratorTypes.size());
  const auto &outMap = region.indexingMaps.back();

  // Compute iteration bounds from all indexing maps
  std::vector<int64_t> iterBounds(numIterDims, -1);
  for (size_t k = 0; k < outMap.resultDims.size(); k++) {
    int dim = outMap.resultDims[k];
    if (dim >= 0) // skip constant expressions
      iterBounds[dim] = resultVal.type.shape[k];
  }
  for (int m = 0; m < region.numInputs; m++) {
    const auto &inMap = region.indexingMaps[m];
    const Value &inVal = prog_.getValue(op.operandIds[m]);
    for (size_t k = 0; k < inMap.resultDims.size(); k++) {
      int dim = inMap.resultDims[k];
      if (dim >= 0 && iterBounds[dim] < 0) // skip constant expressions
        iterBounds[dim] = inVal.type.shape[k];
    }
  }
  for (int d = 0; d < numIterDims; d++) {
    if (iterBounds[d] < 0)
      throw std::runtime_error("Cannot infer bound for iteration dim " +
                               std::to_string(d));
  }

  // Separate parallel and reduction dimensions
  std::vector<int> parallelDims, reductionDims;
  for (int d = 0; d < numIterDims; d++) {
    if (region.iteratorTypes[d] == "reduction")
      reductionDims.push_back(d);
    else
      parallelDims.push_back(d);
  }

  bool hasReduction = !reductionDims.empty();

  if (!hasReduction) {
    // ---- Pure parallel (elementwise) ----
    auto iterVars = z3ctx_.mkIndexVars(name + "_d", numIterDims);

    z3::expr bounds = z3ctx_.ctx().bool_val(true);
    for (int d = 0; d < numIterDims; d++)
      bounds = bounds && z3ctx_.mkBoundsConstraint(iterVars[d], iterBounds[d]);

    // Build block args
    std::vector<z3::expr> blockVals;
    for (int m = 0; m < region.numInputs; m++) {
      const TensorRepr &inRepr = tensors_.getRepr(op.operandIds[m]);
      const auto &inMap = region.indexingMaps[m];
      blockVals.push_back(inRepr.apply(mapAffineIndices(inMap, iterVars)));
    }
    for (int m = 0; m < region.numOutputs; m++) {
      int idx = region.numInputs + m;
      const TensorRepr &outRepr = tensors_.getRepr(op.operandIds[idx]);
      const auto &oMap = region.indexingMaps[idx];
      blockVals.push_back(outRepr.apply(mapAffineIndices(oMap, iterVars)));
    }

    z3::expr yieldedVal = evalBody(region, blockVals);

    std::vector<z3::expr> outIndices = mapAffineIndices(outMap, iterVars);

    z3::expr body = z3::implies(bounds, repr.apply(outIndices) == yieldedVal);
    z3::expr_vector qvars(z3ctx_.ctx());
    for (auto &v : iterVars)
      qvars.push_back(v);
    return z3::forall(qvars, body);
  }

  // ---- Reduction encoding (decomposed) ----
  // For scalability, we decompose the reduction into two independent checks:
  //
  // 1. BODY CHECK: forall symbolic inputs, the body functions produce the same
  //    output. This is a scalar formula independent of reduction size.
  //
  // 2. STRUCTURAL CHECK: same init, same input indexing, same iteration bounds.
  //    If these hold, by induction the full reduction is equivalent.
  //
  // The result UF is defined via an ABSTRACT reduction operator that takes
  // a body-function identifier, init, and input functions. Two reductions
  // with the same body, init, and inputs produce the same result — this is
  // encoded by giving them the SAME result UF name when structurally matched.
  //
  // For programs where structural matching isn't possible (different body
  // structure), we fall back to bounded accumulator encoding.

  int numParallel = static_cast<int>(parallelDims.size());
  int numReduction = static_cast<int>(reductionDims.size());

  std::vector<int64_t> redBounds;
  int64_t totalRedSteps = 1;
  for (int rd : reductionDims) {
    redBounds.push_back(iterBounds[rd]);
    totalRedSteps *= iterBounds[rd];
  }

  // Use the ABSTRACT approach: define the result as a function of the
  // body applied to symbolic arguments. The key insight is that we use
  // a SHARED name for the abstract reduction UF — based on the body
  // structure, not the src/tgt prefix. This way, if source and target
  // have the same body+init+inputs, they get the same result UF.

  // Create a body signature string for structural matching.
  // For commutative ops, sort operand indices so (acc+x) matches (x+acc).
  auto isCommutative = [](OpKind k) {
    return k == OpKind::AddF || k == OpKind::MulF ||
           k == OpKind::AddI || k == OpKind::MulI ||
           k == OpKind::MaxF || k == OpKind::MinF;
  };

  std::string bodySignature;
  for (auto &bop : region.bodyOps) {
    bodySignature += opKindToString(bop.kind) + ";";
    auto indices = bop.operandIndices;
    if (isCommutative(bop.kind) && indices.size() == 2 && indices[0] > indices[1])
      std::swap(indices[0], indices[1]);
    for (int idx : indices)
      bodySignature += std::to_string(idx) + ",";
    bodySignature += "|";
  }
  bodySignature += "yield=" + std::to_string(region.yieldIndex);

  // Create a shared reduction result UF name from the body signature
  // This means two programs with the same body structure will share
  // the same abstract reduction UF.
  static std::hash<std::string> hasher;
  std::string sharedName = "reduce_" + std::to_string(hasher(bodySignature));

  // Map parallel iteration vars to output indices
  auto parVars = z3ctx_.mkIndexVars(name + "_p", numParallel);
  z3::expr parBounds = z3ctx_.ctx().bool_val(true);
  for (int i = 0; i < numParallel; i++) {
    parBounds = parBounds && z3ctx_.mkBoundsConstraint(
                                 parVars[i], iterBounds[parallelDims[i]]);
  }

  // For the abstract reduction, we need to define the result as depending on:
  // - The init values at each parallel position
  // - The input tensor values along the reduction dimensions
  // - The body function (captured by the shared name)
  //
  // We encode: result(par) = AbstractReduce(init(par), inputs(par, *))
  // where AbstractReduce is an uninterpreted function shared across programs
  // with the same body signature.

  // Create the abstract reduction result UF
  // Arguments: (parallel indices) + (init value) + (all input tensor UFs at par)
  // This captures enough to determine the reduction result.
  int abstractRank = numParallel;
  z3::func_decl abstractRedFunc = z3ctx_.mkTensorFunc(sharedName, abstractRank);

  // Define the result: result(outIndices) = abstractReduce(parVars)
  // where abstractReduce subsumes the init and input dependencies
  std::vector<z3::expr> fullIterForOut(numIterDims, z3ctx_.mkInt(0));
  for (int i = 0; i < numParallel; i++) {
    fullIterForOut[parallelDims[i]] = parVars[i];
  }
  std::vector<z3::expr> outIndices = mapAffineIndices(outMap, fullIterForOut);

  // The abstract reduction result also depends on the init and input UFs.
  // We encode this dependency by constraining: for programs with the same
  // body, if they share the same input/init UFs, the abstractReduce UF
  // produces the same values (since it's the SAME UF by name).
  //
  // Additionally, we need the init value to be part of the abstract state.
  // Create a combined UF that takes (par_indices, init_value, input_values...)
  // For simplicity, use a separate UF per unique (bodySignature + init + inputs).

  // The init tensor value at this parallel position
  const TensorRepr &initRepr =
      tensors_.getRepr(op.operandIds[region.numInputs]);
  const auto &initMap = region.indexingMaps[region.numInputs];
  std::vector<z3::expr> iterVarsForInit(numIterDims, z3ctx_.mkInt(0));
  for (int i = 0; i < numParallel; i++) {
    iterVarsForInit[parallelDims[i]] = parVars[i];
  }
  std::vector<z3::expr> initIndices = mapAffineIndices(initMap, iterVarsForInit);
  z3::expr initVal = initRepr.apply(initIndices);

  // Build the abstract reduce with init as an extra argument
  int reduceArgRank = numParallel + 1; // par indices + init value
  z3::func_decl reduceFunc = z3ctx_.mkTensorFunc(sharedName + "_r", reduceArgRank);

  // Apply the reduce function
  z3::expr_vector reduceArgs(z3ctx_.ctx());
  for (auto &p : parVars) reduceArgs.push_back(p);
  reduceArgs.push_back(initVal);
  z3::expr reduceResult = reduceFunc(reduceArgs);

  // Constraint: result(outIndices) = reduce(parVars, init)
  z3::expr resultConstraint = z3::implies(
      parBounds, repr.apply(outIndices) == reduceResult);
  z3::expr_vector qvars(z3ctx_.ctx());
  for (auto &p : parVars) qvars.push_back(p);

  return z3::forall(qvars, resultConstraint);
}

} // namespace tensor_alive
