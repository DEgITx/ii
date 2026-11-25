// SPDX-License-Identifier: MIT
/// @file
/// Machine roofline measurement: peak floating-point throughput and memory bandwidth.
///
/// Every ii benchmark reports its result relative to these peaks, and the backend planner uses them to estimate the
/// cost of operations.
#pragma once

#include <ii/export.h>
#include <ii/status.h>

#include <cstdint>
#include <string>

namespace ii {

struct RooflineOptions {
    /// Number of threads for the multi-threaded measurements; 0 = all logical CPUs.
    int threads = 0;
    /// Approximate wall time spent on each of the four measurements.
    double seconds_per_measurement = 0.25;
    /// Size of each bandwidth test array in bytes; 0 = chosen automatically to exceed the caches.
    std::uint64_t bandwidth_array_bytes = 0;
};

struct RooflineResult {
    std::string isa;                 ///< Kernel variant used for the FLOPS measurement.
    int threads = 0;                 ///< Threads used for the multi-threaded measurements.
    double f32_gflops_single = 0;    ///< Peak f32 FMA throughput of one thread, GFLOP/s.
    double f32_gflops_multi = 0;     ///< Peak f32 FMA throughput of all threads, GFLOP/s.
    double read_bandwidth_gbs = 0;   ///< Multi-threaded read bandwidth, GB/s (10^9 bytes/s).
    double triad_bandwidth_gbs = 0;  ///< Multi-threaded STREAM triad bandwidth (a = b + s*c), GB/s.
    std::uint64_t array_bytes = 0;   ///< Size of each bandwidth test array.
};

/// Measures the roofline of this machine. Takes about 4 x `seconds_per_measurement`.
/// Results are only meaningful for optimized builds on an otherwise idle machine.
II_API Result<RooflineResult> measure_roofline(const RooflineOptions& options = {});

}  // namespace ii
