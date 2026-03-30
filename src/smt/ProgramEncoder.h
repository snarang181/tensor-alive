#pragma once
#include "../ir/Program.h"
#include "OpEncoder.h"
#include "TensorEncoder.h"
#include "Z3Context.h"
#include <vector>

namespace tensor_alive {

struct ProgramEncoding {
  TensorEncoder tensors;
  z3::expr constraints;
  // Output tensor representations (one per return value)
  std::vector<int> outputIds;
};

class ProgramEncoder {
public:
  ProgramEncoder(Z3Ctx &z3ctx);

  // Encode a program. The prefix is used to namespace Z3 names
  // (e.g., "src" or "tgt") so two programs can share input UFs.
  //
  // If sharedInputs is provided, those TensorReprs are used for function
  // arguments instead of creating fresh UFs. This is how source and target
  // programs share the same inputs for equivalence checking.
  ProgramEncoding encode(const Program &prog, const std::string &prefix,
                         const TensorEncoder *sharedInputs = nullptr);

private:
  Z3Ctx &z3ctx_;
};

} // namespace tensor_alive
