// SPDX-License-Identifier: MIT
#include "backend/backend.h"

#include <ii/system_info.h>

#include "core/assert.h"
#include "ir/op_schema.h"

#include <algorithm>

namespace ii::backend {

DeviceInfo host_device_info(int threads, double flops_per_cycle) {
    const CpuInfo& cpu = cpu_info();
    const int cores = std::max(1, cpu.physical_cores > 0 ? cpu.physical_cores : cpu.logical_cpus);
    const int performance = cpu.performance_cores > 0 ? cpu.performance_cores : cores;
    // Efficiency cores count about half; threads beyond the cores add little.
    const double used = std::min<double>(std::max(threads, 1), performance + 0.5 * (cores - performance));
    DeviceInfo info;
    info.device = Device::cpu();
    info.name = cpu.brand.empty() ? std::string("CPU") : cpu.brand;
    info.gflops = used * 3.0 * flops_per_cycle;  // ~3 GHz
    // About 10 GB/s per core up to a typical dual-channel desktop; `ii info --roofline` measures the real value.
    info.bandwidth_gbs = std::min(10.0 * used, 80.0);
    info.launch_us = 1.0;
    return info;
}

double Backend::estimate_us(const ir::CostContext& context) const {
    const DeviceInfo& info = device_info();
    const ir::Cost cost = ir::estimate_cost(context);
    const double compute = cost.flops / (std::max(info.gflops, 1e-3) * 1e3);
    const double memory = cost.bytes / (std::max(info.bandwidth_gbs, 1e-3) * 1e3);
    return std::max(compute, memory) + info.launch_us;
}

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
