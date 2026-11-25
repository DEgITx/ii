// SPDX-License-Identifier: MIT
#include "backends/ref/ref_backend.h"

#include "backends/ref/common.h"

namespace ii::ref {

namespace {

class RefBackend final : public backend::KernelBackend {
public:
    RefBackend() {
        register_elementwise_kernels(*this);
        register_reduction_kernels(*this);
        register_linalg_kernels(*this);
        register_movement_kernels(*this);
        register_image_kernels(*this);
        register_creation_kernels(*this);
        register_composite_kernels(*this);
    }

    std::string_view name() const noexcept override { return kBackendName; }
};

}  // namespace

std::unique_ptr<backend::Backend> create_ref_backend() {
    return std::make_unique<RefBackend>();
}

}  // namespace ii::ref
