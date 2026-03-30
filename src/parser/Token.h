#pragma once
#include <string>

namespace tensor_alive {

enum class TokenKind {
  // Symbols
  Percent,
  Equals,
  Colon,
  Comma,
  LParen,
  RParen,
  LAngle,
  RAngle,
  LBrace,
  RBrace,
  LBracket,
  RBracket,
  Arrow, // ->
  At,    // @
  Hash,  // #
  // Literals
  Integer,
  Float,
  StringLit,
  // Identifiers (includes dotted names like "arith.addf")
  Identifier,
  // Keywords
  KW_func,   // func.func
  KW_return, // return
  KW_tensor, // tensor
  KW_module, // module
  // Special
  Newline,
  Eof,
};

struct Token {
  TokenKind kind;
  std::string text;
  int line = 0;
  int col = 0;

  bool is(TokenKind k) const { return kind == k; }
  bool isIdent(const std::string &s) const {
    return kind == TokenKind::Identifier && text == s;
  }
};

} // namespace tensor_alive
