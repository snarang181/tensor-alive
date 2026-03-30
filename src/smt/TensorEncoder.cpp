#include "TensorEncoder.h"
#include <stdexcept>

namespace tensor_alive {

TensorEncoder::TensorEncoder(Z3Ctx &z3ctx, const std::string &prefix)
    : z3ctx_(z3ctx), prefix_(prefix) {}

TensorRepr TensorEncoder::encodeInput(const Value &val) {
  std::string name = prefix_ + "_" + val.name;
  int rank = val.type.rank();
  z3::func_decl func = z3ctx_.mkTensorFunc(name, rank);

  TensorRepr repr{func, val.type.shape, val.type.isScalar()};
  encodings_.insert_or_assign(val.id, repr);
  return repr;
}

void TensorEncoder::registerResult(int valueId, TensorRepr repr) {
  encodings_.insert_or_assign(valueId, repr);
}

const TensorRepr &TensorEncoder::getRepr(int valueId) const {
  auto it = encodings_.find(valueId);
  if (it == encodings_.end())
    throw std::runtime_error("No encoding for value id " +
                             std::to_string(valueId));
  return it->second;
}

bool TensorEncoder::hasRepr(int valueId) const {
  return encodings_.find(valueId) != encodings_.end();
}

} // namespace tensor_alive
