#pragma once
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace tensor_alive {

enum class ResultKind {
  Equivalent,
  NotEquivalent,
  Unknown,
  ShapeMismatch,
  ParseError,
};

struct Counterexample {
  std::vector<int64_t> index;
  double srcValue = 0.0;
  double tgtValue = 0.0;

  std::string prettyPrint() const;
};

struct CheckResult {
  ResultKind kind;
  std::optional<Counterexample> counterexample;
  std::string message;
  double solvingTimeMs = 0.0;

  static CheckResult equivalent(double ms) {
    return {ResultKind::Equivalent, std::nullopt,
            "Programs are semantically equivalent", ms};
  }
  static CheckResult notEquivalent(Counterexample cex, double ms) {
    return {ResultKind::NotEquivalent, cex, "Programs are NOT equivalent", ms};
  }
  static CheckResult unknown(const std::string &reason, double ms) {
    return {ResultKind::Unknown, std::nullopt, "Unknown: " + reason, ms};
  }
  static CheckResult shapeMismatch(const std::string &detail) {
    return {ResultKind::ShapeMismatch, std::nullopt,
            "Shape mismatch: " + detail, 0};
  }
  static CheckResult parseError(const std::string &detail) {
    return {ResultKind::ParseError, std::nullopt, "Parse error: " + detail, 0};
  }
};

} // namespace tensor_alive
