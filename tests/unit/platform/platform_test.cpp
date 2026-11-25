// SPDX-License-Identifier: MIT
#include "core/platform/platform.h"

#include <ii/system_info.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <thread>

namespace ii::platform {
namespace {

TEST(Platform, GetEnv) {
    const auto path = get_env("PATH");
    ASSERT_TRUE(path.has_value());
    EXPECT_FALSE(path->empty());
    EXPECT_FALSE(get_env("II_SURELY_UNDEFINED_VARIABLE_8d2f").has_value());
}

TEST(Platform, Utf8PathRoundTrip) {
    const std::string text = "dir/модель_模型.gguf";
    EXPECT_EQ(path_to_utf8(path_from_utf8(text)), text);
    EXPECT_EQ(path_from_utf8(text).filename(), std::filesystem::path(u8"модель_模型.gguf"));
}

TEST(Platform, CacheDir) {
    const Result<std::filesystem::path> dir = default_cache_dir();
    if (dir.ok()) {
        EXPECT_FALSE(dir->empty());
    } else {
        EXPECT_EQ(dir.status().code(), StatusCode::kUnavailable);
    }
}

TEST(Platform, LibraryPathExists) {
    const Result<std::filesystem::path> path = this_library_path();
    ASSERT_TRUE(path.ok()) << path.status().to_string();
    EXPECT_TRUE(std::filesystem::exists(*path)) << path_to_utf8(*path);
}

TEST(Platform, AlignedAllocation) {
    void* ptr = aligned_allocate(1000, 256);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % 256, 0u);
    aligned_free(ptr);
    aligned_free(nullptr);
}

TEST(Platform, ThreadNameDoesNotFail) {
    std::thread([] { set_current_thread_name("ii-test-thread-with-a-long-name"); }).join();
}

TEST(Platform, ThreadAffinity) {
    const CpuInfo& cpu = cpu_info();
    if (cpu.cores.empty()) {
        // No topology (Apple): affinity is unsupported there.
        const int cpus[] = {0};
        EXPECT_EQ(set_current_thread_affinity(cpus).code(), StatusCode::kUnsupported);
        return;
    }
    const int target = cpu.cores.back().logical_cpus.front();
    Status status;
    int observed = -2;
    std::thread([&] {
        const int cpus[] = {target};
        status = set_current_thread_affinity(cpus);
        std::this_thread::yield();
        observed = current_cpu();
    }).join();
    ASSERT_TRUE(status.ok()) << status.to_string();
    if (observed != -1) {
        EXPECT_EQ(observed, target);
    }
}

TEST(Platform, AffinityRejectsInvalidInput) {
    EXPECT_FALSE(set_current_thread_affinity({}).ok());
}

}  // namespace
}  // namespace ii::platform
