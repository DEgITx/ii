// SPDX-License-Identifier: MIT
#include <ii/system_info.h>
#include <ii/version.h>

#include <gtest/gtest.h>

#include <set>

namespace ii {
namespace {

TEST(CpuInfo, CountsAreConsistent) {
    const CpuInfo& cpu = cpu_info();
    EXPECT_GE(cpu.logical_cpus, 1);
    EXPECT_GE(cpu.physical_cores, 1);
    EXPECT_LE(cpu.physical_cores, cpu.logical_cpus);
    EXPECT_GE(cpu.performance_cores, 1);
    EXPECT_EQ(cpu.performance_cores + cpu.efficiency_cores, cpu.physical_cores);
    EXPECT_EQ(cpu.is_hybrid(), cpu.efficiency_cores > 0);
}

TEST(CpuInfo, TopologyMatchesCounts) {
    const CpuInfo& cpu = cpu_info();
    if (cpu.cores.empty()) {
        GTEST_SKIP() << "the OS does not expose the core topology";
    }
    EXPECT_EQ(static_cast<int>(cpu.cores.size()), cpu.physical_cores);
    std::set<int> seen;
    int logical = 0;
    for (const CpuCore& core : cpu.cores) {
        EXPECT_FALSE(core.logical_cpus.empty());
        for (const int index : core.logical_cpus) {
            EXPECT_GE(index, 0);
            EXPECT_TRUE(seen.insert(index).second) << "logical CPU " << index << " belongs to two cores";
            ++logical;
        }
    }
    EXPECT_EQ(logical, cpu.logical_cpus);
}

TEST(CpuInfo, ArchitectureAndFeatures) {
    const CpuInfo& cpu = cpu_info();
    const std::string_view arch = build_info().target_arch;
    if (arch == "x86_64") {
        EXPECT_EQ(cpu.arch, CpuArch::kX86_64);
        EXPECT_TRUE(cpu.has(CpuFeature::kSse2)) << "SSE2 is part of the x86-64 baseline";
        EXPECT_FALSE(cpu.vendor.empty());
        if (cpu.has(CpuFeature::kAvx2)) {
            EXPECT_TRUE(cpu.has(CpuFeature::kAvx));
        }
        if (cpu.has(CpuFeature::kAvx512Bw)) {
            EXPECT_TRUE(cpu.has(CpuFeature::kAvx512F));
        }
        EXPECT_FALSE(cpu.has(CpuFeature::kNeon));
    } else if (arch == "arm64") {
        EXPECT_EQ(cpu.arch, CpuArch::kArm64);
        EXPECT_TRUE(cpu.has(CpuFeature::kNeon));
        EXPECT_FALSE(cpu.has(CpuFeature::kAvx2));
    }
}

TEST(CpuInfo, CacheSizesArePlausible) {
    const CpuCaches& caches = cpu_info().caches;
    if (caches.l1d != 0) {
        EXPECT_GE(caches.l1d, 4u * 1024);
        EXPECT_LE(caches.l1d, 4u * 1024 * 1024);
    }
    if (caches.l2 != 0 && caches.l1d != 0) {
        EXPECT_GT(caches.l2, caches.l1d);
    }
}

TEST(CpuInfo, FeatureNames) {
    EXPECT_EQ(to_string(CpuFeature::kAvx512F), "avx512f");
    EXPECT_EQ(to_string(CpuFeature::kArmDotProd), "dotprod");
    for (int i = 0; i < static_cast<int>(CpuFeature::kCount); ++i) {
        EXPECT_NE(to_string(static_cast<CpuFeature>(i)), "unknown") << i;
    }
}

TEST(SystemInfo, OsAndMemory) {
    const OsInfo os = os_info();
    EXPECT_FALSE(os.name.empty());
    const MemoryInfo memory = memory_info();
    EXPECT_GT(memory.total_bytes, 0u);
    EXPECT_LE(memory.available_bytes, memory.total_bytes);
}

TEST(SystemInfo, IsaSupport) {
    const std::vector<IsaSupport> isas = isa_support();
    ASSERT_FALSE(isas.empty());
    EXPECT_EQ(isas.front().name, "generic");
    EXPECT_TRUE(isas.front().compiled);
    EXPECT_TRUE(isas.front().supported);
    const Result<std::string> selected = selected_isa();
    ASSERT_TRUE(selected.ok()) << selected.status().to_string();
    EXPECT_FALSE(selected->empty());
}

TEST(BuildInfo, MatchesHeaders) {
    const BuildInfo& build = build_info();
    EXPECT_EQ(build.version, II_VERSION_STRING);
    EXPECT_FALSE(build.compiler.empty());
    EXPECT_FALSE(build.git_hash.empty());
}

}  // namespace
}  // namespace ii
