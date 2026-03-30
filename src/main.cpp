#include "checker/EquivalenceChecker.h"
#include "parser/Parser.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace tensor_alive;

static void printUsage(const char *prog) {
  std::cerr
      << "Usage: " << prog << " <source.mlir> <target.mlir> [options]\n"
      << "       " << prog << " --parse-only <file.mlir>\n"
      << "\nOptions:\n"
      << "  --timeout <ms>  Z3 solver timeout (default: 30000)\n"
      << "  --verbose       Print intermediate info\n"
      << "  --dump-smt      Dump SMT-LIB2 formula\n"
      << "  --parse-only    Check if a file is parseable (exit 0=yes, 1=no)\n";
}

static std::string readFile(const std::string &path) {
  if (path == "-") {
    std::ostringstream ss;
    ss << std::cin.rdbuf();
    return ss.str();
  }
  std::ifstream f(path);
  if (!f.is_open())
    throw std::runtime_error("Cannot open file: " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printUsage(argv[0]);
    return 1;
  }

  // Handle --parse-only mode
  if (std::strcmp(argv[1], "--parse-only") == 0) {
    if (argc < 3) {
      std::cerr << "Usage: " << argv[0] << " --parse-only <file.mlir>\n";
      return 1;
    }
    try {
      std::string text = readFile(argv[2]);
      if (Parser::canParse(text)) {
        std::cout << "[OK] File is parseable\n";
        return 0;
      } else {
        std::cout << "[FAIL] File is not parseable\n";
        return 1;
      }
    } catch (const std::exception &e) {
      std::cerr << "[ERROR] " << e.what() << "\n";
      return 1;
    }
  }

  if (argc < 3) {
    printUsage(argv[0]);
    return 1;
  }

  std::string srcFile = argv[1];
  std::string tgtFile = argv[2];
  EquivalenceChecker::Options opts;

  for (int i = 3; i < argc; i++) {
    if (std::strcmp(argv[i], "--timeout") == 0 && i + 1 < argc) {
      opts.timeoutMs = std::stoul(argv[++i]);
    } else if (std::strcmp(argv[i], "--verbose") == 0) {
      opts.verbose = true;
    } else if (std::strcmp(argv[i], "--dump-smt") == 0) {
      opts.dumpSmt = true;
    } else {
      std::cerr << "Unknown option: " << argv[i] << "\n";
      printUsage(argv[0]);
      return 1;
    }
  }

  EquivalenceChecker checker(opts);
  CheckResult result = checker.checkFiles(srcFile, tgtFile);

  switch (result.kind) {
  case ResultKind::Equivalent:
    std::cout << "[OK] " << result.message << " (solved in "
              << result.solvingTimeMs << "ms)\n";
    return 0;

  case ResultKind::NotEquivalent:
    std::cout << "[FAIL] " << result.message << "\n";
    if (result.counterexample) {
      std::cout << "Counterexample:\n" << result.counterexample->prettyPrint();
    }
    return 1;

  case ResultKind::Unknown:
    std::cout << "[UNKNOWN] " << result.message << " (" << result.solvingTimeMs
              << "ms)\n";
    return 2;

  case ResultKind::ShapeMismatch:
    std::cerr << "[ERROR] " << result.message << "\n";
    return 3;

  case ResultKind::ParseError:
    std::cerr << "[ERROR] " << result.message << "\n";
    return 4;
  }

  return 0;
}
