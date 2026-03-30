#pragma once
#include "Operation.h"
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace tensor_alive {

struct Program {
  std::vector<Value> values;
  std::vector<Operation> operations;
  std::vector<int> inputIds;  // func arg value indices
  std::vector<int> outputIds; // return value indices

  std::unordered_map<std::string, int> nameToId;

  int addValue(const std::string &name, const TensorType &type) {
    int id = static_cast<int>(values.size());
    values.push_back({name, type, id});
    nameToId[name] = id;
    return id;
  }

  int lookupValue(const std::string &name) const {
    auto it = nameToId.find(name);
    if (it == nameToId.end())
      throw std::runtime_error("Undefined SSA value: " + name);
    return it->second;
  }

  const Value &getValue(int id) const { return values.at(id); }
};

} // namespace tensor_alive
