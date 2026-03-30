#pragma once
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

namespace tensor_alive {

enum class ElemKind { F16, BF16, F32, F64, I1, I8, I16, I32, I64, Index };

inline std::string elemKindToString(ElemKind k) {
  switch (k) {
  case ElemKind::F16:
    return "f16";
  case ElemKind::BF16:
    return "bf16";
  case ElemKind::F32:
    return "f32";
  case ElemKind::F64:
    return "f64";
  case ElemKind::I1:
    return "i1";
  case ElemKind::I8:
    return "i8";
  case ElemKind::I16:
    return "i16";
  case ElemKind::I32:
    return "i32";
  case ElemKind::I64:
    return "i64";
  case ElemKind::Index:
    return "index";
  }
  return "unknown";
}

struct TensorType {
  std::vector<int64_t> shape;
  ElemKind elemKind = ElemKind::F32;

  int64_t rank() const { return static_cast<int64_t>(shape.size()); }

  int64_t numElements() const {
    if (shape.empty())
      return 1;
    return std::accumulate(shape.begin(), shape.end(), int64_t{1},
                           std::multiplies<>());
  }

  bool isScalar() const { return shape.empty(); }

  bool operator==(const TensorType &o) const {
    return shape == o.shape && elemKind == o.elemKind;
  }
  bool operator!=(const TensorType &o) const { return !(*this == o); }

  std::string toString() const {
    if (isScalar())
      return elemKindToString(elemKind);
    std::string s = "tensor<";
    for (auto d : shape)
      s += std::to_string(d) + "x";
    s += elemKindToString(elemKind) + ">";
    return s;
  }
};

} // namespace tensor_alive
