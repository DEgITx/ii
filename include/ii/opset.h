// SPDX-License-Identifier: MIT
/// @file
/// Introspection of the operation set (used for documentation and tools such as `ii ops`).
#pragma once

#include <ii/export.h>

#include <optional>
#include <string>
#include <vector>

namespace ii {

/// Description of an attribute of an operation.
struct AttributeInfo {
    std::string name;
    std::string kind;                          ///< "int", "float", "ints", "dims", "dtype", ...
    std::optional<std::string> default_value;  ///< in IR text syntax; empty when the attribute is required
};

/// Description of an operation of the OpSet.
struct OperationInfo {
    std::string name;
    std::string category;  ///< "unary", "binary", "reduction", "composite", ...
    std::string doc;
    std::vector<std::string> inputs;   ///< names; optional inputs end with '?', variadic ones with "..."
    std::vector<std::string> outputs;  ///< names; variadic ones end with "..."
    std::vector<AttributeInfo> attributes;
    bool composite = false;  ///< has a decomposition into simpler operations
    int since_version = 1;
};

/// All operations of the built-in OpSet, in registration order.
II_API std::vector<OperationInfo> list_operations();

}  // namespace ii
