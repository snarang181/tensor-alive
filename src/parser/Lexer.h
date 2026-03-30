#pragma once
#include "Token.h"
#include <string>
#include <vector>

namespace tensor_alive {

class Lexer {
public:
  explicit Lexer(const std::string &source);

  Token next();
  Token peek();
  const std::vector<Token> &tokenize();

private:
  std::string src_;
  size_t pos_ = 0;
  int line_ = 1;
  int col_ = 1;
  std::vector<Token> tokens_;
  bool tokenized_ = false;

  char current() const;
  char advance();
  bool atEnd() const;
  void skipWhitespaceAndComments();
  Token readNumber();
  Token readIdentifierOrKeyword();
  Token readString();
  Token makeToken(TokenKind kind, const std::string &text);
};

} // namespace tensor_alive
