#include "EquivalenceChecker.h"
#include "../parser/Parser.h"
#include "../smt/ProgramEncoder.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

namespace tensor_alive {

std::string Counterexample::prettyPrint() const {
  std::ostringstream oss;
  oss << "  Index: [";
  for (size_t i = 0; i < index.size(); i++) {
    if (i > 0)
      oss << ", ";
    oss << index[i];
  }
  oss << "]\n";
  oss << "  Source output: " << srcValue << "\n";
  oss << "  Target output: " << tgtValue << "\n";
  return oss.str();
}

EquivalenceChecker::EquivalenceChecker() : opts_() {}
EquivalenceChecker::EquivalenceChecker(const Options &opts) : opts_(opts) {}

static std::string readFile(const std::string &path) {
  std::ifstream f(path);
  if (!f.is_open())
    throw std::runtime_error("Cannot open file: " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

CheckResult EquivalenceChecker::checkFiles(const std::string &srcFile,
                                           const std::string &tgtFile) {
  Program srcProg, tgtProg;

  try {
    std::string srcText = readFile(srcFile);
    Parser srcParser(srcText);
    srcProg = srcParser.parse();
  } catch (const std::exception &e) {
    return CheckResult::parseError(std::string("Source: ") + e.what());
  }

  try {
    std::string tgtText = readFile(tgtFile);
    Parser tgtParser(tgtText);
    tgtProg = tgtParser.parse();
  } catch (const std::exception &e) {
    return CheckResult::parseError(std::string("Target: ") + e.what());
  }

  return checkPrograms(srcProg, tgtProg);
}

CheckResult EquivalenceChecker::checkPrograms(const Program &src,
                                              const Program &tgt) {
  // Validate: same number of inputs and outputs
  if (src.inputIds.size() != tgt.inputIds.size()) {
    return CheckResult::shapeMismatch(
        "Different number of inputs: " + std::to_string(src.inputIds.size()) +
        " vs " + std::to_string(tgt.inputIds.size()));
  }
  if (src.outputIds.size() != tgt.outputIds.size()) {
    return CheckResult::shapeMismatch(
        "Different number of outputs: " + std::to_string(src.outputIds.size()) +
        " vs " + std::to_string(tgt.outputIds.size()));
  }

  // Check input types match
  for (size_t i = 0; i < src.inputIds.size(); i++) {
    const auto &srcType = src.getValue(src.inputIds[i]).type;
    const auto &tgtType = tgt.getValue(tgt.inputIds[i]).type;
    if (srcType != tgtType) {
      return CheckResult::shapeMismatch(
          "Input " + std::to_string(i) + " type mismatch: " +
          srcType.toString() + " vs " + tgtType.toString());
    }
  }

  // Check output types match
  for (size_t i = 0; i < src.outputIds.size(); i++) {
    const auto &srcType = src.getValue(src.outputIds[i]).type;
    const auto &tgtType = tgt.getValue(tgt.outputIds[i]).type;
    if (srcType != tgtType) {
      return CheckResult::shapeMismatch(
          "Output " + std::to_string(i) + " type mismatch: " +
          srcType.toString() + " vs " + tgtType.toString());
    }
  }

  // Encode programs
  Z3Ctx z3ctx;
  ProgramEncoder encoder(z3ctx);

  // Encode source program
  auto srcEnc = encoder.encode(src, "src");

  // Encode target program, sharing source's input UFs.
  // We need to map target input IDs to source input IDs.
  // Since both programs have the same function signature, input i in target
  // corresponds to input i in source.
  TensorEncoder sharedInputs(z3ctx, "shared");
  for (size_t i = 0; i < src.inputIds.size(); i++) {
    int srcId = src.inputIds[i];
    int tgtId = tgt.inputIds[i];
    // Register the source's encoding under the target's ID
    sharedInputs.registerResult(tgtId, srcEnc.tensors.getRepr(srcId));
  }
  auto tgtEnc = encoder.encode(tgt, "tgt", &sharedInputs);

  // Build equivalence query
  z3::solver solver(z3ctx.ctx());
  solver.set("timeout", opts_.timeoutMs);

  // Add program constraints
  solver.add(srcEnc.constraints);
  solver.add(tgtEnc.constraints);

  // For each output, assert that there exists an index where they differ
  // (we check one output at a time; for multiple outputs we could OR them)
  for (size_t outIdx = 0; outIdx < src.outputIds.size(); outIdx++) {
    const TensorRepr &srcOut = srcEnc.tensors.getRepr(src.outputIds[outIdx]);
    const TensorRepr &tgtOut = tgtEnc.tensors.getRepr(tgt.outputIds[outIdx]);

    if (srcOut.isScalar) {
      // Scalar: just check src_out != tgt_out
      solver.add(srcOut.scalarExpr() != tgtOut.scalarExpr());
    } else {
      // Tensor: create free index variables, assert bounds and inequality
      auto indices = z3ctx.mkIndexVars("chk", srcOut.shape.size());
      solver.add(z3ctx.mkShapeBounds(indices, srcOut.shape));
      solver.add(srcOut.apply(indices) != tgtOut.apply(indices));
    }
  }

  if (opts_.dumpSmt) {
    std::cout << "--- SMT Formula ---\n" << solver.to_smt2() << "\n---\n";
  }

  // Solve
  auto startTime = std::chrono::high_resolution_clock::now();
  z3::check_result result = solver.check();
  auto endTime = std::chrono::high_resolution_clock::now();
  double ms =
      std::chrono::duration<double, std::milli>(endTime - startTime).count();

  switch (result) {
  case z3::unsat:
    return CheckResult::equivalent(ms);

  case z3::sat: {
    z3::model model = solver.get_model();
    Counterexample cex;

    // Extract index values from model
    for (size_t outIdx = 0; outIdx < src.outputIds.size(); outIdx++) {
      const TensorRepr &srcOut = srcEnc.tensors.getRepr(src.outputIds[outIdx]);
      const TensorRepr &tgtOut = tgtEnc.tensors.getRepr(tgt.outputIds[outIdx]);

      if (srcOut.isScalar) {
        z3::expr srcVal = model.eval(srcOut.scalarExpr(), true);
        z3::expr tgtVal = model.eval(tgtOut.scalarExpr(), true);
        // Try to extract numeric values
        // Z3 may return rationals; convert to double
        cex.srcValue = srcVal.is_numeral() ? srcVal.as_double() : 0.0;
        cex.tgtValue = tgtVal.is_numeral() ? tgtVal.as_double() : 0.0;
      } else {
        auto indices = z3ctx.mkIndexVars("chk", srcOut.shape.size());
        for (auto &idx : indices) {
          z3::expr val = model.eval(idx, true);
          cex.index.push_back(val.is_numeral() ? val.as_int64() : 0);
        }
        z3::expr srcVal = model.eval(srcOut.apply(indices), true);
        z3::expr tgtVal = model.eval(tgtOut.apply(indices), true);
        cex.srcValue = srcVal.is_numeral() ? srcVal.as_double() : 0.0;
        cex.tgtValue = tgtVal.is_numeral() ? tgtVal.as_double() : 0.0;
      }
      break; // Only show first differing output for now
    }

    return CheckResult::notEquivalent(cex, ms);
  }

  case z3::unknown:
    return CheckResult::unknown(solver.reason_unknown(), ms);
  }

  return CheckResult::unknown("unexpected", 0);
}

} // namespace tensor_alive
