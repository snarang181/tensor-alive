#pragma once
#include "../ir/Program.h"
#include "../smt/Z3Context.h"
#include "Result.h"
#include <string>

namespace tensor_alive {

class EquivalenceChecker {
public:
  struct Options {
    unsigned timeoutMs = 30000;
    bool verbose = false;
    bool dumpSmt = false;
  };

  EquivalenceChecker();
  explicit EquivalenceChecker(const Options &opts);

  // Check equivalence between two MLIR files
  CheckResult checkFiles(const std::string &srcFile,
                         const std::string &tgtFile);

  // Check equivalence between two parsed programs
  CheckResult checkPrograms(const Program &src, const Program &tgt);

private:
  Options opts_;
};

} // namespace tensor_alive
