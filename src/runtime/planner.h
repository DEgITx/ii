// SPDX-License-Identifier: MIT
/// @file
/// The planner: chooses a backend (and so a device) for every node of a graph.
///
/// Cost model: a node costs what its backend estimates (`Backend::estimate_us`, by default the roofline of the
/// device); a value read on another device than the one it lives on costs one copy per destination device
/// (latency + bytes / link bandwidth; between two devices through the host). Constants cost nothing to move
/// (they are copied once, when the program is compiled) but take memory on every device that reads them.
///
/// Algorithm:
///   1. greedy assignment in topological order: each node goes to the backend with the lowest node time plus
///      copies of its inputs (HEFT-like); states go to the device of their first reader;
///   2. memory limits: while a device holds more than its limit (weights + states + peak activations), the nodes
///      at the front of the graph move to their best other device, one by one, so the graph is cut once
///      (for a transformer: the first layers run on the CPU, the rest on the GPU);
///   3. local search: single nodes, runs of consecutive nodes on one device ("islands") and states move to
///      other devices while that lowers the total estimate and respects the memory limits.
///
/// With all backends on the host there are no copies, so every node simply gets its fastest backend.
#pragma once

#include <ii/status.h>

#include "backend/backend.h"
#include "ir/graph.h"

#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ii::runtime {

/// A device the plan may use. Device 0 is always the host.
struct PlanDevice {
    backend::DeviceInfo info;
    std::uint64_t memory_limit = 0;  ///< bytes the plan may use; 0: unlimited
};

/// A backend the plan may use.
struct PlanBackend {
    const backend::Backend* backend = nullptr;
    int device = 0;  ///< index into the devices
};

struct PlannerOptions {
    /// Values of symbolic dimensions assumed for the estimates; other symbols count as 1.
    std::vector<std::pair<std::string, std::int64_t>> dims;
    /// Fraction of a device's memory limit kept free (fragmentation, allocations outside the plan).
    double memory_reserve = 0.1;
};

/// Estimated memory a plan uses on a device.
struct DeviceUsage {
    std::uint64_t weights = 0;      ///< constants read by nodes on the device
    std::uint64_t states = 0;       ///< states placed on the device
    std::uint64_t activations = 0;  ///< peak of the activation memory (with the assumed dimensions)
    std::uint64_t total() const noexcept { return weights + states + activations; }
};

/// Why backends declined nodes, aggregated.
struct Decline {
    std::string reason;  ///< the backend's message, e.g. "ii-test-remote: no kernel for 'sdpa'"
    int nodes = 0;
};

struct GraphPlan {
    std::vector<int> node_backend;   ///< per node in graph order: index into the backends
    std::vector<double> node_us;     ///< per node: estimated time on its backend
    std::vector<int> state_device;   ///< per graph state: the device holding it
    std::vector<DeviceUsage> usage;  ///< per device
    std::vector<double> device_us;   ///< per device: estimated time of its nodes
    double transfer_us = 0;          ///< estimated time of all copies
    std::uint64_t transfer_bytes = 0;
    int transfers = 0;
    std::vector<std::string> notes;  ///< decisions worth explaining (memory limits)
    std::vector<Decline> declines;

    double total_us() const noexcept;
};

/// Plans `graph` (every node must be supported by at least one backend).
Result<GraphPlan> plan_graph(const ir::Graph& graph, std::span<const PlanBackend> backends,
                             std::span<const PlanDevice> devices, const PlannerOptions& options);

}  // namespace ii::runtime
