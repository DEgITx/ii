// SPDX-License-Identifier: MIT
#pragma once

#include <ii/builder.h>

#include "ir/builder.h"
#include "ir/graph.h"

#include <memory>
#include <vector>

namespace ii::detail {

struct GraphBuilderImpl {
    std::unique_ptr<ir::Graph> graph = std::make_unique<ir::Graph>();
    std::vector<ir::Value*> values;  ///< Var id -> value
    Status status;

    int add(ir::Value* value) {
        values.push_back(value);
        return static_cast<int>(values.size()) - 1;
    }
};

}  // namespace ii::detail
