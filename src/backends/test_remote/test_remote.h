// SPDX-License-Identifier: MIT
/// @file
/// ii-test-remote: a simulated device for testing multi-device plans on machines without a GPU.
///
/// The device has its own memory domain: memory allocated through its `DeviceApi` (host memory underneath) is
/// tracked, filled with a NaN pattern when allocated, and every kernel checks that all its inputs, outputs and its
/// workspace lie in device memory — a missing `transfer` is reported as an error instead of silently working.
/// Kernels and copies to the host are not executed when they are enqueued but when the queue is synchronized, like
/// asynchronous GPU work: a host step that reads a result without waiting for it sees stale or NaN data, so missing
/// synchronization shows up in differential tests.
///
/// Kernels are those of ii-ref; the operations the device supports, its memory size and its speed (which steer the
/// planner) are configurable. Enabled through `ContextOptions::test_devices` or the `II_TEST_DEVICES` environment
/// variable, e.g. "memory=64M,ops=matmul+linear+add;memory=1G" for two devices.
#pragma once

#include "backend/backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ii::test_remote {

/// Name of the backend of test device 0; device i > 0 is "ii-test-remote.i".
inline constexpr std::string_view kBackendName = "ii-test-remote";

struct TestDeviceOptions {
    int index = 0;
    std::uint64_t memory_bytes = std::uint64_t{1} << 30;
    double gflops = 4000;
    double bandwidth_gbs = 500;
    double launch_us = 4;
    double link_gbs = 25;
    double link_latency_us = 8;
    /// Operations the device supports (each also needs an ii-ref kernel); empty: everything ii-ref supports.
    std::vector<std::string> ops;
};

/// Parses device specifications separated by ';', each a comma-separated list of key=value: memory (bytes, with
/// an optional K/M/G suffix), gflops, bandwidth (GB/s), launch (us), link (GB/s), latency (us), ops (names joined
/// by '+'). An empty specification ("" or "default") describes a device with the defaults.
Result<std::vector<TestDeviceOptions>> parse_test_devices(std::string_view spec);

/// Creates the backend of one simulated device.
std::unique_ptr<backend::Backend> create_test_remote_backend(const TestDeviceOptions& options);

}  // namespace ii::test_remote
