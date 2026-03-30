#pragma once
#include "../ir/Program.h"
#include "Lexer.h"
#include <memory>
#include <string>
#include <unordered_map>

namespace tensor_alive {

class Parser {
public:
  explicit Parser(const std::string &source);

  Program parse();

  // Parse-only mode: just check if input is parseable
  static bool canParse(const std::string &source);

private:
  Lexer lexer_;
  Token cur_;

  // Top-level affine_map aliases (#map = affine_map<...>)
  std::unordered_map<std::string, Operation::AffineMap> mapAliases_;

  void advance();
  Token expect(TokenKind kind);
  bool check(TokenKind kind) const;
  bool checkIdent(const std::string &s) const;
  bool match(TokenKind kind);

  // Top-level alias parsing
  void parseTopLevelAliases();

  // Grammar
  void parseModule(Program &prog);
  void parseFunc(Program &prog);
  void parseFuncBody(Program &prog);
  void parseOperation(Program &prog);
  void parseReturnOp(Program &prog);

  // Types
  TensorType parseType();
  ElemKind parseElemKind();

  // Values
  std::string parseValueName();
  std::vector<std::string> parseValueList();

  // Linalg ops
  void parseLinalgGeneric(Program &prog, Operation &op);
  void parseLinalgMatmul(Program &prog, Operation &op, OpKind kind);
  void parseLinalgFill(Program &prog, Operation &op);
  Operation::AffineMap parseAffineMap();

  // Attributes
  double parseConstantAttr();
  std::vector<int64_t> parseIntArrayAttr();
  std::vector<std::vector<int64_t>> parseReassociationAttr();

  // Skip balanced delimiters {...}, [...], <...>
  void skipBalancedBraces();
};

} // namespace tensor_alive
