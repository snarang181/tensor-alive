#include "Parser.h"
#include <algorithm>
#include <stdexcept>

namespace tensor_alive {

Parser::Parser(const std::string &source) : lexer_(source), cur_{} {
  advance();
}

void Parser::advance() { cur_ = lexer_.next(); }

Token Parser::expect(TokenKind kind) {
  if (cur_.kind != kind) {
    throw std::runtime_error(
        "Expected token kind " + std::to_string(static_cast<int>(kind)) +
        " but got '" + cur_.text + "' at line " + std::to_string(cur_.line));
  }
  Token t = cur_;
  advance();
  return t;
}

bool Parser::check(TokenKind kind) const { return cur_.kind == kind; }
bool Parser::checkIdent(const std::string &s) const { return cur_.isIdent(s); }

bool Parser::match(TokenKind kind) {
  if (cur_.kind == kind) {
    advance();
    return true;
  }
  return false;
}

// ---- Grammar ----

bool Parser::canParse(const std::string &source) {
  try {
    Parser p(source);
    p.parse();
    return true;
  } catch (...) {
    return false;
  }
}

void Parser::parseTopLevelAliases() {
  // Parse #name = affine_map<...> definitions at the top of the file
  while (check(TokenKind::Hash)) {
    advance(); // skip #
    if (!check(TokenKind::Identifier))
      break;
    std::string aliasName = "#" + cur_.text;
    advance(); // skip name
    if (!match(TokenKind::Equals))
      break;

    if (checkIdent("affine_map")) {
      Operation::AffineMap map = parseAffineMap();
      mapAliases_[aliasName] = map;
    } else {
      // Skip unknown alias value (e.g., #trait = {...})
      // Consume until next # at start or module/func keyword
      while (!check(TokenKind::Hash) && !check(TokenKind::KW_module) &&
             !check(TokenKind::KW_func) && !check(TokenKind::Eof)) {
        advance();
      }
    }
  }
}

Program Parser::parse() {
  Program prog;
  parseTopLevelAliases();
  if (cur_.kind == TokenKind::KW_module) {
    parseModule(prog);
  } else if (cur_.kind == TokenKind::KW_func) {
    parseFunc(prog);
  }
  return prog;
}

void Parser::parseModule(Program &prog) {
  expect(TokenKind::KW_module);
  // optional attributes
  while (cur_.kind == TokenKind::Identifier || cur_.kind == TokenKind::Hash)
    advance();
  expect(TokenKind::LBrace);
  parseFunc(prog);
  expect(TokenKind::RBrace);
}

void Parser::parseFunc(Program &prog) {
  // func.func @name(%arg0: type, ...) -> (ret_types) { body }
  expect(TokenKind::KW_func);
  expect(TokenKind::At);
  Token funcName = expect(TokenKind::Identifier);
  (void)funcName;

  // Arguments
  expect(TokenKind::LParen);
  while (!check(TokenKind::RParen)) {
    std::string name = parseValueName();
    expect(TokenKind::Colon);
    TensorType type = parseType();
    int id = prog.addValue(name, type);
    prog.inputIds.push_back(id);

    // Add a FuncArg op
    Operation argOp;
    argOp.kind = OpKind::FuncArg;
    argOp.resultIds.push_back(id);
    prog.operations.push_back(argOp);

    if (!match(TokenKind::Comma))
      break;
  }
  expect(TokenKind::RParen);

  // Return types: -> type  or  -> (type, type)
  if (match(TokenKind::Arrow)) {
    if (check(TokenKind::LParen)) {
      advance(); // skip (
      // We don't store return types separately; they're inferred from return op
      while (!check(TokenKind::RParen)) {
        parseType(); // consume but don't store
        if (!match(TokenKind::Comma))
          break;
      }
      expect(TokenKind::RParen);
    } else {
      parseType(); // single return type
    }
  }

  // Optional attributes dict on function
  if (check(TokenKind::Identifier) && cur_.text == "attributes") {
    advance();
    // skip the attribute dict
    expect(TokenKind::LBrace);
    int depth = 1;
    while (depth > 0 && !check(TokenKind::Eof)) {
      if (check(TokenKind::LBrace))
        depth++;
      if (check(TokenKind::RBrace))
        depth--;
      if (depth > 0)
        advance();
    }
    expect(TokenKind::RBrace);
  }

  // Body
  expect(TokenKind::LBrace);
  parseFuncBody(prog);
  expect(TokenKind::RBrace);
}

void Parser::parseFuncBody(Program &prog) {
  while (!check(TokenKind::RBrace) && !check(TokenKind::Eof)) {
    if (cur_.kind == TokenKind::KW_return || cur_.text == "func.return") {
      parseReturnOp(prog);
    } else if (cur_.kind == TokenKind::Percent) {
      parseOperation(prog);
    } else {
      // Skip unknown tokens
      advance();
    }
  }
}

void Parser::parseOperation(Program &prog) {
  // %result = op_name %operands : type
  // or %result, %result2 = op_name ...

  // Parse result names
  std::vector<std::string> resultNames;
  resultNames.push_back(parseValueName());
  while (match(TokenKind::Comma)) {
    resultNames.push_back(parseValueName());
  }

  expect(TokenKind::Equals);

  // Operation name
  Token opNameTok = expect(TokenKind::Identifier);
  std::string opName = opNameTok.text;

  // Determine OpKind
  OpKind kind;
  if (opName == "arith.constant")
    kind = OpKind::Constant;
  else if (opName == "arith.addf")
    kind = OpKind::AddF;
  else if (opName == "arith.subf")
    kind = OpKind::SubF;
  else if (opName == "arith.mulf")
    kind = OpKind::MulF;
  else if (opName == "arith.divf")
    kind = OpKind::DivF;
  else if (opName == "arith.negf")
    kind = OpKind::NegF;
  else if (opName == "arith.maximumf" || opName == "arith.maxf")
    kind = OpKind::MaxF;
  else if (opName == "arith.minimumf" || opName == "arith.minf")
    kind = OpKind::MinF;
  else if (opName == "arith.addi")
    kind = OpKind::AddI;
  else if (opName == "arith.subi")
    kind = OpKind::SubI;
  else if (opName == "arith.muli")
    kind = OpKind::MulI;
  else if (opName == "tensor.collapse_shape")
    kind = OpKind::CollapseShape;
  else if (opName == "tensor.expand_shape")
    kind = OpKind::ExpandShape;
  else if (opName == "tensor.transpose" || opName == "linalg.transpose")
    kind = OpKind::Transpose;
  else if (opName == "tensor.broadcast")
    kind = OpKind::Broadcast;
  else if (opName == "linalg.broadcast")
    kind = OpKind::Broadcast;
  else if (opName == "linalg.generic")
    kind = OpKind::LinalgGeneric;
  else if (opName == "linalg.matmul")
    kind = OpKind::LinalgMatmul;
  else if (opName == "linalg.batch_matmul")
    kind = OpKind::LinalgBatchMatmul;
  else if (opName == "linalg.fill")
    kind = OpKind::LinalgFill;
  else if (opName == "tensor.empty")
    kind = OpKind::TensorEmpty;
  else if (opName == "linalg.pack")
    kind = OpKind::LinalgPack;
  else if (opName == "linalg.unpack")
    kind = OpKind::LinalgUnpack;
  else
    throw std::runtime_error("Unsupported operation: " + opName + " at line " +
                             std::to_string(opNameTok.line));

  Operation op;
  op.kind = kind;

  if (kind == OpKind::Constant) {
    // arith.constant <value> : type
    // The value is the next token (a number or dense attr)
    if (check(TokenKind::Integer) || check(TokenKind::Float)) {
      op.constantValue = std::stod(cur_.text);
      advance();
    } else if (check(TokenKind::Identifier) && cur_.text == "dense") {
      // dense<value> : type — support splat constants like dense<0.0>
      advance(); // skip "dense"
      expect(TokenKind::LAngle);
      if (check(TokenKind::Integer) || check(TokenKind::Float)) {
        op.constantValue = std::stod(cur_.text);
        advance();
      } else {
        // Complex dense (nested arrays) — skip content between < >
        int depth = 1;
        while (depth > 0 && !check(TokenKind::Eof)) {
          if (check(TokenKind::LAngle))
            depth++;
          else if (check(TokenKind::RAngle)) {
            depth--;
            if (depth == 0)
              break;
          }
          if (depth > 0)
            advance();
        }
        op.constantValue = 0.0; // placeholder for non-splat dense
      }
      expect(TokenKind::RAngle);
    } else if (check(TokenKind::Identifier) && cur_.text == "true") {
      op.constantValue = 1.0;
      advance();
    } else if (check(TokenKind::Identifier) && cur_.text == "false") {
      op.constantValue = 0.0;
      advance();
    } else {
      throw std::runtime_error("Expected constant value at line " +
                               std::to_string(cur_.line));
    }
  } else if (kind == OpKind::NegF) {
    // Unary: %result = arith.negf %operand : type
    std::string operand = parseValueName();
    op.operandIds.push_back(prog.lookupValue(operand));
  } else if (kind == OpKind::LinalgPack || kind == OpKind::LinalgUnpack) {
    // linalg.pack %input [padding_value(...)] [outer_dims_perm=[...]]
    //   inner_dims_pos=[...] inner_tiles=[...] into %output : srcT -> dstT
    // linalg.unpack %input [outer_dims_perm=[...]]
    //   inner_dims_pos=[...] inner_tiles=[...] into %output : srcT -> dstT
    std::string inputName = parseValueName();
    op.operandIds.push_back(prog.lookupValue(inputName));

    Operation::PackInfo info;

    // Parse optional attributes until "into"
    while (!checkIdent("into") && !check(TokenKind::Eof)) {
      if (checkIdent("padding_value")) {
        advance();
        // padding_value(%val : type)
        expect(TokenKind::LParen);
        std::string padName = parseValueName();
        info.paddingValueId = prog.lookupValue(padName);
        expect(TokenKind::Colon);
        parseType(); // consume type
        expect(TokenKind::RParen);
      } else if (checkIdent("outer_dims_perm")) {
        advance();
        expect(TokenKind::Equals);
        info.outerDimsPerm = parseIntArrayAttr();
      } else if (checkIdent("inner_dims_pos")) {
        advance();
        expect(TokenKind::Equals);
        info.innerDimsPos = parseIntArrayAttr();
      } else if (checkIdent("inner_tiles")) {
        advance();
        expect(TokenKind::Equals);
        info.innerTiles = parseIntArrayAttr();
      } else {
        advance(); // skip unknown token
      }
    }

    // "into %output"
    if (checkIdent("into")) {
      advance();
      std::string outName = parseValueName();
      op.operandIds.push_back(prog.lookupValue(outName));
    }

    op.packInfo = info;
  } else if (kind == OpKind::TensorEmpty) {
    // %0 = tensor.empty() : tensor<4x4xf32>
    expect(TokenKind::LParen);
    expect(TokenKind::RParen);
  } else if (kind == OpKind::LinalgFill) {
    parseLinalgFill(prog, op);
  } else if (kind == OpKind::LinalgGeneric) {
    parseLinalgGeneric(prog, op);
  } else if (kind == OpKind::LinalgMatmul ||
             kind == OpKind::LinalgBatchMatmul) {
    // Check if this has {indexing_maps = ...} attribute dict (newer MLIR form)
    if (check(TokenKind::LBrace)) {
      // Skip the attribute dict, then parse like normal matmul
      int depth = 1;
      advance(); // skip {
      while (depth > 0 && !check(TokenKind::Eof)) {
        if (check(TokenKind::LBrace))
          depth++;
        else if (check(TokenKind::RBrace))
          depth--;
        if (depth > 0)
          advance();
      }
      advance(); // skip closing }
    }
    parseLinalgMatmul(prog, op, kind);
  } else if (kind == OpKind::Broadcast) {
    if (checkIdent("ins")) {
      // linalg.broadcast ins(%x : T) outs(%y : T) dimensions = [0]
      advance();
      expect(TokenKind::LParen);
      std::string operand = parseValueName();
      op.operandIds.push_back(prog.lookupValue(operand));
      expect(TokenKind::Colon);
      parseType();
      expect(TokenKind::RParen);
      if (checkIdent("outs")) {
        advance();
        expect(TokenKind::LParen);
        std::string outName = parseValueName();
        op.operandIds.push_back(prog.lookupValue(outName));
        expect(TokenKind::Colon);
        parseType();
        expect(TokenKind::RParen);
      }
      if (checkIdent("dimensions")) {
        advance();
        expect(TokenKind::Equals);
        parseIntArrayAttr(); // consume dimensions list
      }
    } else {
      // tensor.broadcast %input : T -> T
      std::string operand = parseValueName();
      op.operandIds.push_back(prog.lookupValue(operand));
    }
  } else if (kind == OpKind::Transpose) {
    if (checkIdent("ins")) {
      // linalg.transpose ins(%x : T) outs(%y : T) permutation = [1, 0]
      advance(); // skip "ins"
      expect(TokenKind::LParen);
      std::string operand = parseValueName();
      op.operandIds.push_back(prog.lookupValue(operand));
      expect(TokenKind::Colon);
      parseType();
      expect(TokenKind::RParen);
      // outs
      if (checkIdent("outs")) {
        advance();
        expect(TokenKind::LParen);
        std::string outName = parseValueName();
        op.operandIds.push_back(prog.lookupValue(outName));
        expect(TokenKind::Colon);
        parseType();
        expect(TokenKind::RParen);
      }
      // permutation = [...]
      if (checkIdent("permutation")) {
        advance();
        expect(TokenKind::Equals);
        op.permutation = parseIntArrayAttr();
      }
    } else {
      // tensor.transpose %input [1, 0] : T -> T
      std::string operand = parseValueName();
      op.operandIds.push_back(prog.lookupValue(operand));
      op.permutation = parseIntArrayAttr();
    }
  } else if (kind == OpKind::CollapseShape || kind == OpKind::ExpandShape) {
    // tensor.collapse_shape %input [[0, 1], [2]] : tensor<...> into tensor<...>
    // tensor.expand_shape %input [[0, 1], [2]] output_shape [M, N, K] :
    // tensor<...> into tensor<...>
    std::string operand = parseValueName();
    op.operandIds.push_back(prog.lookupValue(operand));
    op.reassociation = parseReassociationAttr();
    // expand_shape may have "output_shape [...]" with values or SSA names
    if (checkIdent("output_shape")) {
      advance();
      expect(TokenKind::LBracket);
      while (!check(TokenKind::RBracket) && !check(TokenKind::Eof)) {
        advance(); // skip values (integers or %ssa names)
        match(TokenKind::Comma);
      }
      expect(TokenKind::RBracket);
    }
  } else {
    // Binary: %result = op %lhs, %rhs : type
    std::string lhs = parseValueName();
    op.operandIds.push_back(prog.lookupValue(lhs));
    expect(TokenKind::Comma);
    std::string rhs = parseValueName();
    op.operandIds.push_back(prog.lookupValue(rhs));
  }

  // Consume type signature
  TensorType resultType;
  // Check if this op uses the linalg-style "-> type" result syntax
  bool isLinalgStyle =
      (kind == OpKind::LinalgGeneric || kind == OpKind::LinalgMatmul ||
       kind == OpKind::LinalgBatchMatmul || kind == OpKind::LinalgFill);
  // Transpose/Broadcast with ins/outs also use -> type
  if (!isLinalgStyle &&
      (kind == OpKind::Transpose || kind == OpKind::Broadcast) &&
      op.operandIds.size() >= 2) {
    isLinalgStyle = true; // was parsed with ins/outs syntax
  }

  if (isLinalgStyle) {
    // linalg ops: result type comes after } -> type or directly after outs() ->
    // type
    if (match(TokenKind::Arrow)) {
      resultType = parseType();
    } else if (!op.operandIds.empty()) {
      // No explicit -> type. For tensor ops, infer from the last operand
      // (outs). For memref ops, this is a void-returning op.
      int lastOpId = op.operandIds.back();
      const auto &lastType = prog.getValue(lastOpId).type;
      if (!lastType.shape.empty() || lastType.elemKind != ElemKind::F32) {
        // Use the outs type as result type
        resultType = lastType;
      } else {
        return; // void-returning
      }
    } else {
      return; // void-returning
    }
  } else if (kind == OpKind::LinalgPack || kind == OpKind::LinalgUnpack) {
    // : srcType -> dstType
    expect(TokenKind::Colon);
    parseType(); // source type (skip)
    expect(TokenKind::Arrow);
    resultType = parseType(); // destination type
  } else {
    expect(TokenKind::Colon);
    resultType = parseType();

    // For ops with "-> result_type" or "into result_type" syntax
    if (kind == OpKind::CollapseShape || kind == OpKind::ExpandShape) {
      if (checkIdent("into")) {
        advance();
        resultType = parseType();
      }
    }
    if (match(TokenKind::Arrow)) {
      resultType = parseType();
    }
  }

  // Register result values
  for (auto &name : resultNames) {
    int id = prog.addValue(name, resultType);
    op.resultIds.push_back(id);
  }

  prog.operations.push_back(op);
}

void Parser::parseReturnOp(Program &prog) {
  advance(); // skip "return" or "func.return"

  Operation op;
  op.kind = OpKind::FuncReturn;

  if (!check(TokenKind::RBrace) && !check(TokenKind::Eof)) {
    // Parse return values
    std::string val = parseValueName();
    op.operandIds.push_back(prog.lookupValue(val));
    while (match(TokenKind::Comma)) {
      val = parseValueName();
      op.operandIds.push_back(prog.lookupValue(val));
    }

    // Consume : type list
    if (match(TokenKind::Colon)) {
      parseType();
      while (match(TokenKind::Comma)) {
        parseType();
      }
    }
  }

  prog.outputIds = op.operandIds;
  prog.operations.push_back(op);
}

// ---- Types ----

static ElemKind stringToElemKind(const std::string &s) {
  if (s == "f16")
    return ElemKind::F16;
  if (s == "bf16")
    return ElemKind::BF16;
  if (s == "f32")
    return ElemKind::F32;
  if (s == "f64")
    return ElemKind::F64;
  if (s == "i1")
    return ElemKind::I1;
  if (s == "i8")
    return ElemKind::I8;
  if (s == "i16")
    return ElemKind::I16;
  if (s == "i32")
    return ElemKind::I32;
  if (s == "i64")
    return ElemKind::I64;
  if (s == "index")
    return ElemKind::Index;
  throw std::runtime_error("Unknown element type: " + s);
}

TensorType Parser::parseType() {
  TensorType t;
  if (cur_.kind == TokenKind::KW_tensor) {
    advance(); // "tensor"
    expect(TokenKind::LAngle);
    // Parse "NxNx...xElemType" pattern.
    // The lexer produces: Integer("4"), Identifier("x4xf32") or similar.
    // We need to handle the case where dimensions and element type are
    // concatenated into a single identifier token like "x4xf32".
    //
    // Strategy: collect the first integer, then handle the Identifier token
    // by splitting on 'x' to extract remaining dims and the element type.
    if (check(TokenKind::Integer)) {
      t.shape.push_back(std::stoll(cur_.text));
      advance();
    }
    // Now we might see an Identifier like "x4xf32" or "xf32"
    if (check(TokenKind::Identifier) && !cur_.text.empty() &&
        cur_.text[0] == 'x') {
      // Split the token: "x4xf32" -> ["4", "f32"] or "xf32" -> ["f32"]
      std::string rest = cur_.text.substr(1); // remove leading 'x'
      advance();                              // consume the identifier token

      // Parse remaining "NxNx...xElemType" from the string
      while (!rest.empty()) {
        // Check if rest starts with a digit
        if (std::isdigit(rest[0])) {
          size_t pos = 0;
          int64_t dim = std::stoll(rest, &pos);
          t.shape.push_back(dim);
          rest = rest.substr(pos);
          // Skip 'x' separator
          if (!rest.empty() && rest[0] == 'x')
            rest = rest.substr(1);
        } else {
          // This is the element type
          t.elemKind = stringToElemKind(rest);
          rest.clear();
        }
      }
    } else if (check(TokenKind::Identifier)) {
      // No dimensions, just element type like tensor<f32>
      t.elemKind = parseElemKind();
    }
    expect(TokenKind::RAngle);
  } else {
    // Bare element type (scalar)
    t.elemKind = parseElemKind();
  }
  return t;
}

ElemKind Parser::parseElemKind() {
  std::string s = cur_.text;
  advance();
  return stringToElemKind(s);
}

// ---- Values ----

std::string Parser::parseValueName() {
  expect(TokenKind::Percent);
  Token name = cur_;
  // SSA names can be identifiers or integers
  if (check(TokenKind::Identifier) || check(TokenKind::Integer)) {
    advance();
    return "%" + name.text;
  }
  throw std::runtime_error("Expected SSA value name after % at line " +
                           std::to_string(cur_.line));
}

std::vector<std::string> Parser::parseValueList() {
  std::vector<std::string> vals;
  vals.push_back(parseValueName());
  while (match(TokenKind::Comma)) {
    vals.push_back(parseValueName());
  }
  return vals;
}

// ---- Linalg.generic ----

Operation::AffineMap Parser::parseAffineMap() {
  // Handle #map alias references
  if (check(TokenKind::Hash)) {
    advance(); // skip #
    std::string aliasName = "#" + cur_.text;
    advance(); // skip name
    auto it = mapAliases_.find(aliasName);
    if (it != mapAliases_.end()) {
      return it->second;
    }
    throw std::runtime_error("Unknown affine_map alias: " + aliasName);
  }

  // affine_map<(d0, d1) -> (d0, d1)>
  // We only support simple projections: each result is a single dimension
  // variable
  Operation::AffineMap map;

  if (!checkIdent("affine_map"))
    throw std::runtime_error("Expected 'affine_map' or #alias at line " +
                             std::to_string(cur_.line));
  advance();
  expect(TokenKind::LAngle);
  expect(TokenKind::LParen);

  // Parse dimension variables: (d0, d1, ...)
  std::vector<std::string> dimVars;
  while (!check(TokenKind::RParen)) {
    Token dim = expect(TokenKind::Identifier);
    dimVars.push_back(dim.text);
    match(TokenKind::Comma);
  }
  expect(TokenKind::RParen);
  expect(TokenKind::Arrow);
  expect(TokenKind::LParen);

  // Parse result expressions: (d1, d0, ...)
  // Supports dimension references and integer constants (e.g., 0 for broadcast)
  int resultPos = 0;
  while (!check(TokenKind::RParen)) {
    if (check(TokenKind::Integer)) {
      // Constant expression (e.g., 0 in broadcast maps)
      int64_t val = std::stoll(cur_.text);
      advance();
      map.resultDims.push_back(-1); // sentinel for constant
      map.constExprs.push_back({resultPos, val});
    } else {
      Token expr = expect(TokenKind::Identifier);
      // Find which dimension this refers to
      int found = -1;
      for (size_t i = 0; i < dimVars.size(); i++) {
        if (dimVars[i] == expr.text) {
          found = static_cast<int>(i);
          break;
        }
      }
      if (found < 0)
        throw std::runtime_error("Unknown dimension variable: " + expr.text);
      map.resultDims.push_back(found);
    }
    resultPos++;
    match(TokenKind::Comma);
  }
  expect(TokenKind::RParen);
  expect(TokenKind::RAngle);

  return map;
}

void Parser::parseLinalgGeneric(Program &prog, Operation &op) {
  Operation::LinalgRegion region;

  // Parse attribute dict: {indexing_maps = [...], iterator_types = [...]}
  expect(TokenKind::LBrace);
  while (!check(TokenKind::RBrace)) {
    Token attrName = expect(TokenKind::Identifier);

    if (attrName.text == "indexing_maps") {
      expect(TokenKind::Equals);
      expect(TokenKind::LBracket);
      while (!check(TokenKind::RBracket)) {
        region.indexingMaps.push_back(parseAffineMap());
        match(TokenKind::Comma);
      }
      expect(TokenKind::RBracket);
    } else if (attrName.text == "iterator_types") {
      expect(TokenKind::Equals);
      expect(TokenKind::LBracket);
      while (!check(TokenKind::RBracket)) {
        Token t = expect(TokenKind::StringLit);
        region.iteratorTypes.push_back(t.text);
        match(TokenKind::Comma);
      }
      expect(TokenKind::RBracket);
    } else {
      // Skip unknown attributes: consume = and value
      if (match(TokenKind::Equals)) {
        // Skip the value (could be anything)
        int depth = 0;
        while (!check(TokenKind::Comma) && !check(TokenKind::RBrace) &&
               !check(TokenKind::Eof)) {
          if (check(TokenKind::LBracket) || check(TokenKind::LBrace) ||
              check(TokenKind::LAngle))
            depth++;
          if (check(TokenKind::RBracket) || check(TokenKind::RBrace) ||
              check(TokenKind::RAngle)) {
            if (depth <= 0)
              break;
            depth--;
          }
          advance();
        }
      }
    }
    match(TokenKind::Comma);
  }
  expect(TokenKind::RBrace);

  // Parse ins(%a, %b : type, type)
  if (checkIdent("ins")) {
    advance();
    expect(TokenKind::LParen);
    std::vector<std::string> insNames;
    while (check(TokenKind::Percent)) {
      insNames.push_back(parseValueName());
      if (!match(TokenKind::Comma))
        break;
      // Check if next is % (more operands) or : (type list)
      if (!check(TokenKind::Percent))
        break;
    }
    expect(TokenKind::Colon);
    for (size_t i = 0; i < insNames.size(); i++) {
      parseType(); // consume type
      if (i + 1 < insNames.size())
        match(TokenKind::Comma);
    }
    expect(TokenKind::RParen);
    for (auto &name : insNames) {
      op.operandIds.push_back(prog.lookupValue(name));
    }
    region.numInputs = static_cast<int>(insNames.size());
  }

  // Parse outs(%c : type)
  if (checkIdent("outs")) {
    advance();
    expect(TokenKind::LParen);
    std::vector<std::string> outsNames;
    while (check(TokenKind::Percent)) {
      outsNames.push_back(parseValueName());
      if (!match(TokenKind::Comma))
        break;
      if (!check(TokenKind::Percent))
        break;
    }
    expect(TokenKind::Colon);
    for (size_t i = 0; i < outsNames.size(); i++) {
      parseType();
      if (i + 1 < outsNames.size())
        match(TokenKind::Comma);
    }
    expect(TokenKind::RParen);
    for (auto &name : outsNames) {
      op.operandIds.push_back(prog.lookupValue(name));
    }
    region.numOutputs = static_cast<int>(outsNames.size());
  }

  // Parse region body: { ^bb0(%a: f32, %b: f32, %c: f32): ... linalg.yield %r :
  // f32 }
  expect(TokenKind::LBrace);

  // Parse block label: ^bb0(%a: type, %b: type, ...):
  // The ^ is not a standard token; it may appear as part of an identifier
  std::vector<std::string> blockArgNames;
  if (cur_.text.size() > 0 && cur_.text[0] == '^') {
    advance(); // skip ^bb0 or similar -- actually lexer won't handle ^
  }
  // Actually ^ will cause a lex error. Let me handle it differently.
  // The block label looks like: ^bb0(%a: f32, %b: f32):
  // Let me skip everything until we see the block args in parens, or handle ^
  // specially.

  // Skip the block label text (^bb0 etc.) - it might be split oddly by the
  // lexer Just consume tokens until we hit ( for the block args
  while (!check(TokenKind::LParen) && !check(TokenKind::Eof)) {
    advance();
  }

  // Parse block arguments
  if (match(TokenKind::LParen)) {
    while (!check(TokenKind::RParen) && !check(TokenKind::Eof)) {
      std::string argName = parseValueName();
      blockArgNames.push_back(argName);
      expect(TokenKind::Colon);
      parseType(); // consume type
      match(TokenKind::Comma);
    }
    expect(TokenKind::RParen);
    expect(TokenKind::Colon);
  }

  // Parse body operations until linalg.yield or }
  // Body ops produce scalar results from block arguments
  // We track block args + results as indices: 0..N-1 are block args, N+ are
  // results
  std::unordered_map<std::string, int> bodyValueMap;
  int bodyIdx = 0;
  for (auto &name : blockArgNames) {
    bodyValueMap[name] = bodyIdx++;
  }

  while (!check(TokenKind::RBrace) && !check(TokenKind::Eof)) {
    if (checkIdent("linalg.yield")) {
      advance();
      // Parse yield value
      std::string yieldVal = parseValueName();
      auto it = bodyValueMap.find(yieldVal);
      if (it == bodyValueMap.end())
        throw std::runtime_error("Unknown yield value: " + yieldVal);
      region.yieldIndex = it->second;
      // Consume : type
      if (match(TokenKind::Colon))
        parseType();
      break;
    }

    // Parse: %result = op %args : type
    if (check(TokenKind::Percent)) {
      std::string resultName = parseValueName();
      expect(TokenKind::Equals);
      Token bodyOpName = expect(TokenKind::Identifier);

      Operation::LinalgRegion::BodyOp bodyOp;

      if (bodyOpName.text == "arith.constant") {
        bodyOp.kind = OpKind::Constant;
        if (check(TokenKind::Integer) || check(TokenKind::Float)) {
          bodyOp.constantValue = std::stod(cur_.text);
          advance();
        }
      } else if (bodyOpName.text == "arith.negf") {
        bodyOp.kind = OpKind::NegF;
        std::string operand = parseValueName();
        auto it = bodyValueMap.find(operand);
        if (it == bodyValueMap.end())
          throw std::runtime_error("Unknown body value: " + operand);
        bodyOp.operandIndices.push_back(it->second);
      } else {
        // Binary op
        if (bodyOpName.text == "arith.addf")
          bodyOp.kind = OpKind::AddF;
        else if (bodyOpName.text == "arith.subf")
          bodyOp.kind = OpKind::SubF;
        else if (bodyOpName.text == "arith.mulf")
          bodyOp.kind = OpKind::MulF;
        else if (bodyOpName.text == "arith.divf")
          bodyOp.kind = OpKind::DivF;
        else if (bodyOpName.text == "arith.maximumf" ||
                 bodyOpName.text == "arith.maxf")
          bodyOp.kind = OpKind::MaxF;
        else if (bodyOpName.text == "arith.minimumf" ||
                 bodyOpName.text == "arith.minf")
          bodyOp.kind = OpKind::MinF;
        else if (bodyOpName.text == "arith.addi")
          bodyOp.kind = OpKind::AddI;
        else if (bodyOpName.text == "arith.subi")
          bodyOp.kind = OpKind::SubI;
        else if (bodyOpName.text == "arith.muli")
          bodyOp.kind = OpKind::MulI;
        else if (bodyOpName.text == "arith.andi")
          bodyOp.kind = OpKind::MulI; // model AND as mul for booleans
        else if (bodyOpName.text == "arith.ori")
          bodyOp.kind = OpKind::AddI; // model OR as add (approx)
        else if (bodyOpName.text == "arith.divsi" ||
                 bodyOpName.text == "arith.divui")
          bodyOp.kind = OpKind::DivF;
        else
          throw std::runtime_error("Unsupported body op: " + bodyOpName.text);

        std::string lhs = parseValueName();
        auto itl = bodyValueMap.find(lhs);
        if (itl == bodyValueMap.end())
          throw std::runtime_error("Unknown body value: " + lhs);
        bodyOp.operandIndices.push_back(itl->second);

        expect(TokenKind::Comma);

        std::string rhs = parseValueName();
        auto itr = bodyValueMap.find(rhs);
        if (itr == bodyValueMap.end())
          throw std::runtime_error("Unknown body value: " + rhs);
        bodyOp.operandIndices.push_back(itr->second);
      }

      // Consume : type
      if (match(TokenKind::Colon))
        parseType();

      bodyValueMap[resultName] = bodyIdx++;
      region.bodyOps.push_back(bodyOp);
    } else {
      advance(); // skip unknown
    }
  }

  expect(TokenKind::RBrace);

  op.linalgRegion = region;
}

void Parser::parseLinalgFill(Program &prog, Operation &op) {
  // linalg.fill ins(%cst : f32) outs(%init : tensor<...>) -> tensor<...>
  // Semantics: fills every element of outs with the scalar from ins

  // Parse ins(%cst : type)
  if (!checkIdent("ins"))
    throw std::runtime_error("Expected 'ins' in linalg.fill");
  advance();
  expect(TokenKind::LParen);
  std::string scalarName = parseValueName();
  op.operandIds.push_back(prog.lookupValue(scalarName));
  expect(TokenKind::Colon);
  parseType(); // consume scalar type
  expect(TokenKind::RParen);

  // Parse outs(%init : tensor<...>)
  if (!checkIdent("outs"))
    throw std::runtime_error("Expected 'outs' in linalg.fill");
  advance();
  expect(TokenKind::LParen);
  std::string outName = parseValueName();
  op.operandIds.push_back(prog.lookupValue(outName));
  expect(TokenKind::Colon);
  parseType(); // consume tensor type
  expect(TokenKind::RParen);
}

void Parser::skipBalancedBraces() {
  int depth = 1;
  while (depth > 0 && !check(TokenKind::Eof)) {
    if (check(TokenKind::LBrace))
      depth++;
    else if (check(TokenKind::RBrace)) {
      depth--;
      if (depth == 0)
        break;
    }
    advance();
  }
  if (check(TokenKind::RBrace))
    advance();
}

void Parser::parseLinalgMatmul(Program &prog, Operation &op, OpKind kind) {
  // linalg.matmul ins(%A, %B : T, T) outs(%C : T) -> T
  // linalg.batch_matmul ins(%A, %B : T, T) outs(%C : T) -> T
  //
  // We synthesize a LinalgRegion equivalent to:
  //   matmul:       (d0,d1,d2) with maps (d0,d2)->(d2,d1)->(d0,d1), iter
  //   [par,par,red] batch_matmul: (d0,d1,d2,d3) with maps
  //   (d0,d1,d3)->(d0,d3,d2)->(d0,d1,d2), iter [par,par,par,red]

  Operation::LinalgRegion region;

  // Parse ins(%A, %B : type, type)
  if (!checkIdent("ins"))
    throw std::runtime_error("Expected 'ins' in linalg.matmul");
  advance();
  expect(TokenKind::LParen);
  std::string nameA = parseValueName();
  expect(TokenKind::Comma);
  std::string nameB = parseValueName();
  expect(TokenKind::Colon);
  TensorType typeA = parseType();
  expect(TokenKind::Comma);
  TensorType typeB = parseType();
  expect(TokenKind::RParen);

  op.operandIds.push_back(prog.lookupValue(nameA));
  op.operandIds.push_back(prog.lookupValue(nameB));
  region.numInputs = 2;

  // Parse outs(%C : type)
  if (!checkIdent("outs"))
    throw std::runtime_error("Expected 'outs' in linalg.matmul");
  advance();
  expect(TokenKind::LParen);
  std::string nameC = parseValueName();
  expect(TokenKind::Colon);
  TensorType typeC = parseType();
  expect(TokenKind::RParen);

  op.operandIds.push_back(prog.lookupValue(nameC));
  region.numOutputs = 1;

  if (kind == OpKind::LinalgMatmul) {
    // matmul: C[i,j] = sum_k A[i,k] * B[k,j]
    // iteration dims: (d0=i, d1=j, d2=k)
    // A map: (d0, d1, d2) -> (d0, d2)
    // B map: (d0, d1, d2) -> (d2, d1)
    // C map: (d0, d1, d2) -> (d0, d1)
    region.indexingMaps = {{{{0, 2}}}, {{{2, 1}}}, {{{0, 1}}}};
    region.iteratorTypes = {"parallel", "parallel", "reduction"};
  } else {
    // batch_matmul: C[b,i,j] = sum_k A[b,i,k] * B[b,k,j]
    // iteration dims: (d0=b, d1=i, d2=j, d3=k)
    // A map: (d0,d1,d2,d3) -> (d0, d1, d3)
    // B map: (d0,d1,d2,d3) -> (d0, d3, d2)
    // C map: (d0,d1,d2,d3) -> (d0, d1, d2)
    region.indexingMaps = {{{{0, 1, 3}}}, {{{0, 3, 2}}}, {{{0, 1, 2}}}};
    region.iteratorTypes = {"parallel", "parallel", "parallel", "reduction"};
  }

  // Body: %mul = arith.mulf %a, %b; %add = arith.addf %c, %mul; yield %add
  // Block args: 0=a (from A), 1=b (from B), 2=c (from C/accumulator)
  Operation::LinalgRegion::BodyOp mulOp;
  mulOp.kind = OpKind::MulF;
  mulOp.operandIndices = {0, 1};

  Operation::LinalgRegion::BodyOp addOp;
  addOp.kind = OpKind::AddF;
  addOp.operandIndices = {2, 3}; // 2=c (init/accum), 3=mul result

  region.bodyOps = {mulOp, addOp};
  region.yieldIndex = 4; // result of addOp (3 block args + 2 body ops - 1)

  op.linalgRegion = region;
  // Override kind to LinalgGeneric so the encoder handles it uniformly
  op.kind = OpKind::LinalgGeneric;
}

// ---- Attributes ----

double Parser::parseConstantAttr() {
  if (check(TokenKind::Float) || check(TokenKind::Integer)) {
    double v = std::stod(cur_.text);
    advance();
    return v;
  }
  throw std::runtime_error("Expected numeric constant");
}

std::vector<int64_t> Parser::parseIntArrayAttr() {
  expect(TokenKind::LBracket);
  std::vector<int64_t> vals;
  while (!check(TokenKind::RBracket)) {
    vals.push_back(std::stoll(expect(TokenKind::Integer).text));
    match(TokenKind::Comma);
  }
  expect(TokenKind::RBracket);
  return vals;
}

std::vector<std::vector<int64_t>> Parser::parseReassociationAttr() {
  expect(TokenKind::LBracket);
  std::vector<std::vector<int64_t>> result;
  while (!check(TokenKind::RBracket)) {
    result.push_back(parseIntArrayAttr());
    match(TokenKind::Comma);
  }
  expect(TokenKind::RBracket);
  return result;
}

} // namespace tensor_alive
