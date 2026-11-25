// SPDX-License-Identifier: MIT
#include <ii/opset.h>

#include "ir/op_schema.h"

namespace ii {

std::vector<OperationInfo> list_operations() {
    std::vector<OperationInfo> result;
    for (const auto& schema : ir::default_op_registry().all()) {
        if (schema->has_flag(ir::kOpInternal)) {
            continue;  // implementation details of the compiler, not part of the OpSet
        }
        OperationInfo info;
        info.name = std::string(schema->name);
        info.category = std::string(schema->category);
        info.doc = std::string(schema->doc);
        info.composite = schema->is_composite();
        info.since_version = schema->since_version;
        for (const ir::InputSpec& input : schema->inputs) {
            info.inputs.push_back(std::string(input.name) + (input.variadic ? "..." : (input.optional ? "?" : "")));
        }
        for (const ir::OutputSpec& output : schema->outputs) {
            info.outputs.push_back(std::string(output.name) + (output.variadic ? "..." : ""));
        }
        for (const ir::AttrSpec& attr : schema->attrs) {
            AttributeInfo attribute;
            attribute.name = std::string(attr.name);
            attribute.kind = std::string(to_string(attr.kind));
            if (attr.default_value) {
                attribute.default_value = to_string(*attr.default_value);
            }
            info.attributes.push_back(std::move(attribute));
        }
        result.push_back(std::move(info));
    }
    return result;
}

}  // namespace ii
