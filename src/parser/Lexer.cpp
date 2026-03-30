#include "Lexer.h"
#include <cctype>
#include <stdexcept>

namespace tensor_alive {

Lexer::Lexer(const std::string &source) : src_(source) {}

char Lexer::current() const {
  if (atEnd())
    return '\0';
  return src_[pos_];
}

char Lexer::advance() {
  char c = src_[pos_++];
  if (c == '\n') {
    line_++;
    col_ = 1;
  } else {
    col_++;
  }
  return c;
}

bool Lexer::atEnd() const { return pos_ >= src_.size(); }

Token Lexer::makeToken(TokenKind kind, const std::string &text) {
  return {kind, text, line_, col_};
}

void Lexer::skipWhitespaceAndComments() {
  while (!atEnd()) {
    char c = current();
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      advance();
    } else if (c == '/' && pos_ + 1 < src_.size() && src_[pos_ + 1] == '/') {
      // Line comment
      while (!atEnd() && current() != '\n')
        advance();
    } else {
      break;
    }
  }
}

Token Lexer::readNumber() {
  size_t start = pos_;
  bool isNeg = false;
  if (current() == '-') {
    advance();
    isNeg = true;
  }

  bool isFloat = false;
  while (!atEnd() && std::isdigit(current()))
    advance();
  if (!atEnd() && current() == '.') {
    isFloat = true;
    advance();
    while (!atEnd() && std::isdigit(current()))
      advance();
  }
  // Scientific notation
  if (!atEnd() && (current() == 'e' || current() == 'E')) {
    isFloat = true;
    advance();
    if (!atEnd() && (current() == '+' || current() == '-'))
      advance();
    while (!atEnd() && std::isdigit(current()))
      advance();
  }

  std::string text = src_.substr(start, pos_ - start);
  return makeToken(isFloat ? TokenKind::Float : TokenKind::Integer, text);
}

Token Lexer::readIdentifierOrKeyword() {
  size_t start = pos_;
  while (!atEnd() && (std::isalnum(current()) || current() == '_' ||
                      current() == '.' || current() == '$')) {
    advance();
  }
  std::string text = src_.substr(start, pos_ - start);

  // Keywords
  if (text == "func.func" || text == "func" ||
      text == "cuda_tile.testing$func" || text == "entry")
    return makeToken(TokenKind::KW_func, text);
  if (text == "return" || text == "func.return")
    return makeToken(TokenKind::KW_return, text);
  if (text == "tensor")
    return makeToken(TokenKind::KW_tensor, text);
  if (text == "module" || text == "cuda_tile.module")
    return makeToken(TokenKind::KW_module, text);

  return makeToken(TokenKind::Identifier, text);
}

Token Lexer::readString() {
  advance(); // skip opening quote
  size_t start = pos_;
  while (!atEnd() && current() != '"') {
    if (current() == '\\')
      advance(); // skip escaped char
    advance();
  }
  std::string text = src_.substr(start, pos_ - start);
  if (!atEnd())
    advance(); // skip closing quote
  return makeToken(TokenKind::StringLit, text);
}

Token Lexer::next() {
  skipWhitespaceAndComments();
  if (atEnd())
    return makeToken(TokenKind::Eof, "");

  char c = current();

  // Single-char tokens
  switch (c) {
  case '%':
    advance();
    return makeToken(TokenKind::Percent, "%");
  case '=':
    advance();
    return makeToken(TokenKind::Equals, "=");
  case ':':
    advance();
    return makeToken(TokenKind::Colon, ":");
  case ',':
    advance();
    return makeToken(TokenKind::Comma, ",");
  case '(':
    advance();
    return makeToken(TokenKind::LParen, "(");
  case ')':
    advance();
    return makeToken(TokenKind::RParen, ")");
  case '<':
    advance();
    return makeToken(TokenKind::LAngle, "<");
  case '>':
    advance();
    return makeToken(TokenKind::RAngle, ">");
  case '{':
    advance();
    return makeToken(TokenKind::LBrace, "{");
  case '}':
    advance();
    return makeToken(TokenKind::RBrace, "}");
  case '[':
    advance();
    return makeToken(TokenKind::LBracket, "[");
  case ']':
    advance();
    return makeToken(TokenKind::RBracket, "]");
  case '@':
    advance();
    return makeToken(TokenKind::At, "@");
  case '#':
    advance();
    return makeToken(TokenKind::Hash, "#");
  case '^': {
    // Block label like ^bb0 - read as identifier
    advance(); // skip ^
    size_t start = pos_;
    while (!atEnd() && (std::isalnum(current()) || current() == '_'))
      advance();
    std::string text = "^" + src_.substr(start, pos_ - start);
    return makeToken(TokenKind::Identifier, text);
  }
  case '?':
    advance();
    return makeToken(TokenKind::Identifier, "?");
  case '!': {
    // Dialect type prefix like !cuda_tile.tile — read as identifier
    advance(); // skip !
    size_t start = pos_;
    while (!atEnd() && (std::isalnum(current()) || current() == '_' ||
                        current() == '.' || current() == '$'))
      advance();
    std::string text = src_.substr(start, pos_ - start);
    // Map !cuda_tile.tile to our tensor keyword
    if (text == "cuda_tile.tile")
      return makeToken(TokenKind::KW_tensor, text);
    return makeToken(TokenKind::Identifier, "!" + text);
  }
  case '"':
    return readString();
  default:
    break;
  }

  // Arrow ->
  if (c == '-' && pos_ + 1 < src_.size() && src_[pos_ + 1] == '>') {
    advance();
    advance();
    return makeToken(TokenKind::Arrow, "->");
  }

  // Numbers (including negative)
  if (std::isdigit(c) ||
      (c == '-' && pos_ + 1 < src_.size() && std::isdigit(src_[pos_ + 1]))) {
    return readNumber();
  }

  // Identifiers / keywords
  if (std::isalpha(c) || c == '_') {
    return readIdentifierOrKeyword();
  }

  throw std::runtime_error(std::string("Unexpected character '") + c +
                           "' at line " + std::to_string(line_) + " col " +
                           std::to_string(col_));
}

Token Lexer::peek() {
  size_t savedPos = pos_;
  int savedLine = line_;
  int savedCol = col_;
  Token t = next();
  pos_ = savedPos;
  line_ = savedLine;
  col_ = savedCol;
  return t;
}

const std::vector<Token> &Lexer::tokenize() {
  if (!tokenized_) {
    while (true) {
      Token t = next();
      tokens_.push_back(t);
      if (t.kind == TokenKind::Eof)
        break;
    }
    tokenized_ = true;
  }
  return tokens_;
}

} // namespace tensor_alive
