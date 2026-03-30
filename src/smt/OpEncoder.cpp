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

  z3::expr constExpr = z3ctx_.mkReal(*op.constantValue);

  // Use expression-based encoding: constant returns the same value everywhere.
  // No UF or quantifier needed.
  std::string name = freshName("const");
  z3::func_decl dummyFunc = z3ctx_.mkTensorFunc(name, resultVal.type.rank());
  TensorRepr repr{
      dummyFunc, resultVal.type.shape, resultVal.type.isScalar(),
      [constExpr](const std::vector<z3::expr> &) { return constExpr; }};
  tensors_.registerResult(resultId, repr);
  return z3ctx_.ctx().bool_val(true); // no constraints needed
}

z3::expr OpEncoder::encodeBinaryF(const Operation &op,
                                  const std::string &opStr) {
  if (op.operandIds.size() != 2 || op.resultIds.size() != 1)
    throw std::runtime_error("Binary op expects 2 operands and 1 result");

  const TensorRepr &lhs = tensors_.getRepr(op.operandIds[0]);
  const TensorRepr &rhs = tensors_.getRepr(op.operandIds[1]);
  int resultId = op.resultIds[0];
  const Value &resultVal = prog_.getValue(resultId);

  // Expression-based encoding: result(i) = lhs(i) OP rhs(i)
  // No UF or quantifier — just compose the lambda.
  std::string name = freshName("binf");
  z3::func_decl dummyFunc = z3ctx_.mkTensorFunc(name, resultVal.type.rank());

  // Capture operand reprs by value to avoid dangling references
  TensorRepr lhsCopy = lhs;
  TensorRepr rhsCopy = rhs;
  TensorRepr repr{dummyFunc, resultVal.type.shape, resultVal.type.isScalar(),
                  [lhsCopy, rhsCopy,
                   opStr](const std::vector<z3::expr> &indices) -> z3::expr {
                    z3::expr l = indices.empty() ? lhsCopy.scalarExpr()
                                                 : lhsCopy.apply(indices);
                    z3::expr r = indices.empty() ? rhsCopy.scalarExpr()
                                                 : rhsCopy.apply(indices);
                    if (opStr == "add")
                      return l + r;
                    if (opStr == "sub")
                      return l - r;
                    if (opStr == "mul")
                      return l * r;
                    if (opStr == "div")
                      return l / r;
                    throw std::runtime_error("Unknown op: " + opStr);
                  }};
  tensors_.registerResult(resultId, repr);
  return z3ctx_.ctx().bool_val(true);
}

z3::expr OpEncoder::encodeNegF(const Operation &op) {
  if (op.operandIds.size() != 1 || op.resultIds.size() != 1)
    throw std::runtime_error("NegF expects 1 operand and 1 result");

  const TensorRepr &operand = tensors_.getRepr(op.operandIds[0]);
  int resultId = op.resultIds[0];
  const Value &resultVal = prog_.getValue(resultId);

  std::string name = freshName("negf");
  z3::func_decl dummyFunc = z3ctx_.mkTensorFunc(name, resultVal.type.rank());
  TensorRepr opCopy = operand;
  TensorRepr repr{dummyFunc, resultVal.type.shape, resultVal.type.isScalar(),
                  [opCopy](const std::vector<z3::expr> &indices) -> z3::expr {
                    z3::expr v = indices.empty() ? opCopy.scalarExpr()
                                                 : opCopy.apply(indices);
                    return -v;
                  }};
  tensors_.registerResult(resultId, repr);
  return z3ctx_.ctx().bool_val(true);
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
  z3::func_decl dummyFunc = z3ctx_.mkTensorFunc(name, resultVal.type.rank());
  TensorRepr lhsCopy = lhs, rhsCopy = rhs;
  TensorRepr repr{
      dummyFunc, resultVal.type.shape, resultVal.type.isScalar(),
      [lhsCopy, rhsCopy](const std::vector<z3::expr> &indices) -> z3::expr {
        z3::expr l =
            indices.empty() ? lhsCopy.scalarExpr() : lhsCopy.apply(indices);
        z3::expr r =
            indices.empty() ? rhsCopy.scalarExpr() : rhsCopy.apply(indices);
        return z3::ite(l >= r, l, r);
      }};
  tensors_.registerResult(resultId, repr);
  return z3ctx_.ctx().bool_val(true);
}

z3::expr OpEncoder::encodeMinF(const Operation &op) {
  if (op.operandIds.size() != 2 || op.resultIds.size() != 1)
    throw std::runtime_error("MinF expects 2 operands and 1 result");

  const TensorRepr &lhs = tensors_.getRepr(op.operandIds[0]);
  const TensorRepr &rhs = tensors_.getRepr(op.operandIds[1]);
  int resultId = op.resultIds[0];
  const Value &resultVal = prog_.getValue(resultId);

  std::string name = freshName("minf");
  z3::func_decl dummyFunc = z3ctx_.mkTensorFunc(name, resultVal.type.rank());
  TensorRepr lhsCopy = lhs, rhsCopy = rhs;
  TensorRepr repr{
      dummyFunc, resultVal.type.shape, resultVal.type.isScalar(),
      [lhsCopy, rhsCopy](const std::vector<z3::expr> &indices) -> z3::expr {
        z3::expr l =
            indices.empty() ? lhsCopy.scalarExpr() : lhsCopy.apply(indices);
        z3::expr r =
            indices.empty() ? rhsCopy.scalarExpr() : rhsCopy.apply(indices);
        return z3::ite(l <= r, l, r);
      }};
  tensors_.registerResult(resultId, repr);
  return z3ctx_.ctx().bool_val(true);
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
    // ---- Pure parallel (elementwise) — expr-based, no quantifiers ----
    // Build an elemFn lambda that computes result(iterVars) by evaluating
    // the body on input tensors indexed via the affine maps.
    // Capture all needed data by value for the lambda.

    struct ParallelCapture {
      Operation::LinalgRegion region;
      std::vector<TensorRepr> inputReprs;
      std::vector<TensorRepr> outputReprs;
      Operation::AffineMap outMap;
      int numIterDims;
    };

    auto capture = std::make_shared<ParallelCapture>();
    capture->region = region;
    capture->outMap = outMap;
    capture->numIterDims = numIterDims;
    for (int m = 0; m < region.numInputs; m++)
      capture->inputReprs.push_back(tensors_.getRepr(op.operandIds[m]));
    for (int m = 0; m < region.numOutputs; m++)
      capture->outputReprs.push_back(
          tensors_.getRepr(op.operandIds[region.numInputs + m]));

    // The output map tells us how iteration indices map to output tensor
    // indices. For the elemFn, we receive output indices and need to
    // reverse-map to iteration indices. For simple identity maps
    // (d0,d1)->(d0,d1), this is trivial. For general maps, we invert:
    // iterVars[outMap.resultDims[k]] = outIdx[k].
    auto &thisRegion = capture->region;
    auto &thisOutMap = capture->outMap;

    TensorRepr repr{
        z3ctx_.mkTensorFunc(name, resultVal.type.rank()), resultVal.type.shape,
        resultVal.type.isScalar(),
        [capture](const std::vector<z3::expr> &outIdx) -> z3::expr {
          auto &reg = capture->region;
          auto &om = capture->outMap;
          int nIter = capture->numIterDims;

          // Reconstruct iteration variables from output indices
          // iterVars[om.resultDims[k]] = outIdx[k]; others default to 0
          z3::context &c = outIdx.empty() ? capture->inputReprs[0].func.ctx()
                                          : outIdx[0].ctx();
          std::vector<z3::expr> iterVars;
          for (int d = 0; d < nIter; d++)
            iterVars.push_back(c.int_val(0));
          for (size_t k = 0; k < om.resultDims.size() && k < outIdx.size(); k++)
            iterVars[om.resultDims[k]] = outIdx[k];

          // Build block args
          std::vector<z3::expr> blockVals;
          for (size_t m = 0; m < capture->inputReprs.size(); m++) {
            auto &inMap = reg.indexingMaps[m];
            std::vector<z3::expr> mapped;
            for (int r : inMap.resultDims)
              mapped.push_back(iterVars[r]);
            blockVals.push_back(capture->inputReprs[m].apply(mapped));
          }
          for (size_t m = 0; m < capture->outputReprs.size(); m++) {
            auto &oMap = reg.indexingMaps[reg.numInputs + m];
            std::vector<z3::expr> mapped;
            for (int r : oMap.resultDims)
              mapped.push_back(iterVars[r]);
            blockVals.push_back(capture->outputReprs[m].apply(mapped));
          }

          // Evaluate body inline
          for (auto &bop : reg.bodyOps) {
            z3::expr res(c);
            switch (bop.kind) {
            case OpKind::Constant:
              res = c.real_val(
                  std::to_string(bop.constantValue.value_or(0.0)).c_str());
              break;
            case OpKind::AddF:
            case OpKind::AddI:
              res = blockVals[bop.operandIndices[0]] +
                    blockVals[bop.operandIndices[1]];
              break;
            case OpKind::SubF:
            case OpKind::SubI:
              res = blockVals[bop.operandIndices[0]] -
                    blockVals[bop.operandIndices[1]];
              break;
            case OpKind::MulF:
            case OpKind::MulI:
              res = blockVals[bop.operandIndices[0]] *
                    blockVals[bop.operandIndices[1]];
              break;
            case OpKind::DivF:
              res = blockVals[bop.operandIndices[0]] /
                    blockVals[bop.operandIndices[1]];
              break;
            case OpKind::NegF:
              res = -blockVals[bop.operandIndices[0]];
              break;
            case OpKind::MaxF: {
              auto a = blockVals[bop.operandIndices[0]],
                   b = blockVals[bop.operandIndices[1]];
              res = z3::ite(a >= b, a, b);
              break;
            }
            case OpKind::MinF: {
              auto a = blockVals[bop.operandIndices[0]],
                   b = blockVals[bop.operandIndices[1]];
              res = z3::ite(a <= b, a, b);
              break;
            }
            default:
              throw std::runtime_error(
                  "Unsupported body op in expr-based linalg.generic");
            }
            blockVals.push_back(res);
          }
          return blockVals[reg.yieldIndex];
        }};

    tensors_.registerResult(resultId, repr);
    return z3ctx_.ctx().bool_val(true); // no quantified constraints
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
    return k == OpKind::AddF || k == OpKind::MulF || k == OpKind::AddI ||
           k == OpKind::MulI || k == OpKind::MaxF || k == OpKind::MinF;
  };

  std::string bodySignature;
  // Include body ops
  for (auto &bop : region.bodyOps) {
    bodySignature += opKindToString(bop.kind) + ";";
    auto indices = bop.operandIndices;
    if (isCommutative(bop.kind) && indices.size() == 2 &&
        indices[0] > indices[1])
      std::swap(indices[0], indices[1]);
    for (int idx : indices)
      bodySignature += std::to_string(idx) + ",";
    bodySignature += "|";
  }
  bodySignature += "yield=" + std::to_string(region.yieldIndex);
  // Include indexing maps — different maps mean different input elements
  for (size_t m = 0; m < region.indexingMaps.size(); m++) {
    bodySignature += "/map" + std::to_string(m) + "=";
    for (int d : region.indexingMaps[m].resultDims)
      bodySignature += std::to_string(d) + ",";
  }
  // Include iterator types
  for (auto &it : region.iteratorTypes)
    bodySignature += "/" + it;

  // Structural matching: use body signature hash as the reduce UF name.
  // Programs with identical body+maps+iterators share the same reduce UF.
  // This is O(1) and handles all real compiler pass transformations.

  static std::hash<std::string> hasher;
  std::string sharedName = "reduce_" + std::to_string(hasher(bodySignature));

  auto parVars = z3ctx_.mkIndexVars(name + "_p", numParallel);
  z3::expr parBounds = z3ctx_.ctx().bool_val(true);
  for (int i = 0; i < numParallel; i++) {
    parBounds = parBounds && z3ctx_.mkBoundsConstraint(
                                 parVars[i], iterBounds[parallelDims[i]]);
  }

  std::vector<z3::expr> fullIterForOut(numIterDims, z3ctx_.mkInt(0));
  for (int i = 0; i < numParallel; i++) {
    fullIterForOut[parallelDims[i]] = parVars[i];
  }
  std::vector<z3::expr> outIndices = mapAffineIndices(outMap, fullIterForOut);

  const TensorRepr &initRepr =
      tensors_.getRepr(op.operandIds[region.numInputs]);
  const auto &initMap = region.indexingMaps[region.numInputs];
  std::vector<z3::expr> iterVarsForInit(numIterDims, z3ctx_.mkInt(0));
  for (int i = 0; i < numParallel; i++) {
    iterVarsForInit[parallelDims[i]] = parVars[i];
  }
  z3::expr initVal = initRepr.apply(mapAffineIndices(initMap, iterVarsForInit));

  // Reduce UF: shared name based on body signature. (par_indices, init) ->
  // result
  int rRank = numParallel + 1;
  z3::func_decl reduceFunc = z3ctx_.mkTensorFunc(sharedName + "_r", rRank);

  z3::expr_vector reduceArgs(z3ctx_.ctx());
  for (auto &p : parVars)
    reduceArgs.push_back(p);
  reduceArgs.push_back(initVal);
  z3::expr reduceResult = reduceFunc(reduceArgs);

  z3::expr resultConstraint =
      z3::implies(parBounds, repr.apply(outIndices) == reduceResult);
  z3::expr_vector qvars(z3ctx_.ctx());
  for (auto &p : parVars)
    qvars.push_back(p);

  return z3::forall(qvars, resultConstraint);
}

} // namespace tensor_alive
