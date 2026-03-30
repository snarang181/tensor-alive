#pragma once
#include "TensorType.h"
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace tensor_alive {

enum class OpKind {
  // arith
  Constant,
  AddF,
  SubF,
  MulF,
  DivF,
  NegF,
  MaxF,
  MinF,
  AddI,
  SubI,
  MulI,
  // tensor
  Extract,
  FromElements,
  Reshape,
  CollapseShape,
  ExpandShape,
  Transpose,
  Broadcast,
  TensorEmpty,
  // linalg
  LinalgGeneric,
  LinalgMatmul,
  LinalgBatchMatmul,
  LinalgFill,
  LinalgPack,
  LinalgUnpack,
  // meta
  FuncArg,
  FuncReturn,
};

inline std::string opKindToString(OpKind k) {
  switch (k) {
  case OpKind::Constant:
    return "arith.constant";
  case OpKind::AddF:
    return "arith.addf";
  case OpKind::SubF:
    return "arith.subf";
  case OpKind::MulF:
    return "arith.mulf";
  case OpKind::DivF:
    return "arith.divf";
  case OpKind::NegF:
    return "arith.negf";
  case OpKind::MaxF:
    return "arith.maximumf";
  case OpKind::MinF:
    return "arith.minimumf";
  case OpKind::AddI:
    return "arith.addi";
  case OpKind::SubI:
    return "arith.subi";
  case OpKind::MulI:
    return "arith.muli";
  case OpKind::Extract:
    return "tensor.extract";
  case OpKind::FromElements:
    return "tensor.from_elements";
  case OpKind::Reshape:
    return "tensor.reshape";
  case OpKind::CollapseShape:
    return "tensor.collapse_shape";
  case OpKind::ExpandShape:
    return "tensor.expand_shape";
  case OpKind::Transpose:
    return "tensor.transpose"; // actually linalg.transpose in newer MLIR
  case OpKind::Broadcast:
    return "tensor.broadcast";
  case OpKind::TensorEmpty:
    return "tensor.empty";
  case OpKind::LinalgGeneric:
    return "linalg.generic";
  case OpKind::LinalgMatmul:
    return "linalg.matmul";
  case OpKind::LinalgBatchMatmul:
    return "linalg.batch_matmul";
  case OpKind::LinalgFill:
    return "linalg.fill";
  case OpKind::LinalgPack:
    return "linalg.pack";
  case OpKind::LinalgUnpack:
    return "linalg.unpack";
  case OpKind::FuncArg:
    return "func.arg";
  case OpKind::FuncReturn:
    return "func.return";
  }
  return "unknown";
}

using AttrValue =
    std::variant<int64_t, double, std::vector<int64_t>, std::string>;

struct Attribute {
  std::string key;
  AttrValue value;
};

struct Value {
  std::string name;
  TensorType type;
  int id = -1;
};

struct Operation {
  OpKind kind;
  std::vector<int> operandIds; // indices into Program::values
  std::vector<int> resultIds;  // indices into Program::values
  std::vector<Attribute> attrs;

  // For arith.constant
  std::optional<double> constantValue;

  // For transpose: permutation
  std::vector<int64_t> permutation;

  // For reshape: target shape (stored redundantly for convenience)
  std::vector<int64_t> targetShape;

  // For collapse_shape / expand_shape: reassociation indices
  std::vector<std::vector<int64_t>> reassociation;

  // For linalg.generic
  struct AffineMap {
    // An affine map (d0, d1, ...) -> (expr0, expr1, ...)
    // Each expr is an index into the iteration dims (simple projections),
    // or a constant value.
    // e.g., (d0, d1) -> (d1, d0) is stored as resultDims={1, 0}
    // e.g., (d0, d1) -> (d0, 0) is stored as resultDims={0, -1},
    //       constExprs={{1, 0}}  (position 1 has constant value 0)
    std::vector<int> resultDims; // -1 means use constExprs for this position
    std::vector<std::pair<int, int64_t>>
        constExprs; // (position, constant_value) pairs
  };

  struct LinalgRegion {
    std::vector<AffineMap> indexingMaps;    // one per operand (ins + outs)
    std::vector<std::string> iteratorTypes; // "parallel" or "reduction"
    int numInputs = 0;
    int numOutputs = 0;

    // Body: sequence of scalar ops applied to block arguments
    // Block args are named %arg0, %arg1, etc. within the region
    struct BodyOp {
      OpKind kind;
      std::vector<int>
          operandIndices; // indices into block args or prior body results
      // For constants
      std::optional<double> constantValue;
    };
    std::vector<BodyOp> bodyOps;
    int yieldIndex = -1; // which body result to yield
  };

  std::optional<LinalgRegion> linalgRegion;

  // For linalg.pack / linalg.unpack
  struct PackInfo {
    std::vector<int64_t> innerDimsPos;  // which source dims are tiled
    std::vector<int64_t> innerTiles;    // tile sizes
    std::vector<int64_t> outerDimsPerm; // optional permutation of outer dims
    int paddingValueId = -1; // operand ID of the padding scalar (-1 = none)
  };
  std::optional<PackInfo> packInfo;
};

} // namespace tensor_alive
