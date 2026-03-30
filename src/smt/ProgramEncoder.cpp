#include "ProgramEncoder.h"
#include <stdexcept>

namespace tensor_alive {

ProgramEncoder::ProgramEncoder(Z3Ctx &z3ctx) : z3ctx_(z3ctx) {}

ProgramEncoding ProgramEncoder::encode(const Program &prog,
                                       const std::string &prefix,
                                       const TensorEncoder *sharedInputs) {
  TensorEncoder tensors(z3ctx_, prefix);
  OpEncoder opEnc(z3ctx_, tensors, prog, prefix);
  z3::expr allConstraints = z3ctx_.ctx().bool_val(true);

  for (auto &op : prog.operations) {
    if (op.kind == OpKind::FuncArg) {
      // Encode function arguments
      int resultId = op.resultIds.at(0);
      const Value &val = prog.getValue(resultId);

      if (sharedInputs && sharedInputs->hasRepr(resultId)) {
        // Use shared input encoding (same UF as source program)
        tensors.registerResult(resultId, sharedInputs->getRepr(resultId));
      } else {
        tensors.encodeInput(val);
      }
    } else if (op.kind == OpKind::FuncReturn) {
      // Return op doesn't need encoding; outputs reference existing values
    } else {
      z3::expr constraint = opEnc.encode(op);
      allConstraints = allConstraints && constraint;
    }
  }

  return ProgramEncoding{std::move(tensors), allConstraints, prog.outputIds};
}

} // namespace tensor_alive
