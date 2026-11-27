// SPDX-License-Identifier: MIT
#include "backends/ref/ref_backend.h"

#include "backends/ref/common.h"

namespace ii::ref {

namespace {

class RefBackend final : public backend::KernelBackend {
public:
    RefBackend() {
        // Plain single-threaded loops in double precision: far below the machine's peaks. The planner uses ii-ref
        // only for what no faster backend supports.
        info_ = backend::host_device_info(1, 1);
        info_.gflops = 0.5;
        info_.bandwidth_gbs = 2;
        register_elementwise_kernels(*this);
        register_reduction_kernels(*this);
        register_linalg_kernels(*this);
        register_movement_kernels(*this);
        register_image_kernels(*this);
        register_creation_kernels(*this);
        register_composite_kernels(*this);
    }

    std::string_view name() const noexcept override { return kBackendName; }
    const backend::DeviceInfo& device_info() const noexcept override { return info_; }

private:
    backend::DeviceInfo info_;
};

}  // namespace

std::unique_ptr<backend::Backend> create_ref_backend() {
    return std::make_unique<RefBackend>();
}

}  // namespace ii::ref
