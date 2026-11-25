// SPDX-License-Identifier: MIT
#include "backend/backend.h"

#include "core/assert.h"
#include "ir/op_schema.h"

namespace ii::backend {

Status KernelBackend::supports(const ir::Node& node) const {
    const auto it = kernels_.find(node.op_name());
    if (it == kernels_.end()) {
        return errors::unsupported("{}: no kernel for '{}'", name(), node.op_name());
    }
    if (it->second.check != nullptr) {
        if (Status status = it->second.check(node); !status.ok()) {
            return errors::unsupported("{}: {}", name(), status.message());
        }
    }
    return ok_status();
}

Result<std::unique_ptr<Kernel>> KernelBackend::create_kernel(const ir::Node& node) const {
    II_TRY(supports(node));
    return kernels_.at(node.op_name()).factory(node);
}

void KernelBackend::add_kernel(std::string_view op, Factory factory, SupportCheck check) {
    // Store the schema's own name so that the key outlives any temporary string.
    const ir::OpSchema& schema = ir::builtin_op(op);
    II_CHECK(!kernels_.contains(schema.name), "kernel for '{}' registered twice", op);
    kernels_.emplace(schema.name, Entry{factory, check});
}

}  // namespace ii::backend
