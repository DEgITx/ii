// SPDX-License-Identifier: MIT
#include "core/isa.h"

#include <gtest/gtest.h>

namespace ii {
namespace {

TEST(Isa, NamesRoundTrip) {
    for (const Isa isa : {Isa::kGeneric, Isa::kAvx2, Isa::kAvx512, Isa::kNeon}) {
        const auto parsed = parse_isa(isa_name(isa));
        ASSERT_TRUE(parsed.has_value());
        EXPECT_EQ(*parsed, isa);
    }
    EXPECT_FALSE(parse_isa("sse9").has_value());
}

TEST(Isa, TargetListStartsWithGeneric) {
    ASSERT_FALSE(target_isas().empty());
    EXPECT_EQ(target_isas().front(), Isa::kGeneric);
    EXPECT_TRUE(isa_compiled(Isa::kGeneric));
}

TEST(Isa, GenericIsAlwaysUsable) {
    const CpuInfo no_features;
    const std::vector<Isa> usable = usable_isas(no_features);
    ASSERT_EQ(usable.size(), 1u);
    EXPECT_EQ(usable.front(), Isa::kGeneric);

    const Result<Isa> selected = select_isa(no_features, std::nullopt);
    ASSERT_TRUE(selected.ok());
    EXPECT_EQ(*selected, Isa::kGeneric);
}

TEST(Isa, ForcedGeneric) {
    const Result<Isa> selected = select_isa(cpu_info(), "generic");
    ASSERT_TRUE(selected.ok());
    EXPECT_EQ(*selected, Isa::kGeneric);
}

TEST(Isa, ForcedUnknownFails) {
    const Result<Isa> selected = select_isa(cpu_info(), "bogus");
    ASSERT_FALSE(selected.ok());
    EXPECT_EQ(selected.status().code(), StatusCode::kInvalidArgument);
}

TEST(Isa, ForcedUnsupportedFails) {
    const CpuInfo no_features;
    for (const Isa isa : target_isas()) {
        if (isa == Isa::kGeneric) {
            continue;
        }
        const Result<Isa> selected = select_isa(no_features, isa_name(isa));
        ASSERT_FALSE(selected.ok()) << isa_name(isa);
        EXPECT_EQ(selected.status().code(), isa_compiled(isa) ? StatusCode::kUnsupported : StatusCode::kUnavailable);
    }
}

TEST(Isa, SelectsMostAdvancedUsable) {
    const std::vector<Isa> usable = usable_isas(cpu_info());
    ASSERT_FALSE(usable.empty());
    const Result<Isa> selected = select_isa(cpu_info(), std::nullopt);
    ASSERT_TRUE(selected.ok());
    EXPECT_EQ(*selected, usable.back());
}

TEST(Isa, FeatureRequirements) {
    CpuInfo cpu;
    cpu.features.set(CpuFeature::kAvx2);
    EXPECT_FALSE(isa_supported(Isa::kAvx2, cpu)) << "AVX2 kernels also need FMA and F16C";
    cpu.features.set(CpuFeature::kFma);
    cpu.features.set(CpuFeature::kF16c);
    EXPECT_TRUE(isa_supported(Isa::kAvx2, cpu));
    cpu.features.set(CpuFeature::kAvx512F);
    EXPECT_FALSE(isa_supported(Isa::kAvx512, cpu));
    cpu.features.set(CpuFeature::kAvx512Bw);
    cpu.features.set(CpuFeature::kAvx512Vl);
    cpu.features.set(CpuFeature::kAvx512Dq);
    EXPECT_TRUE(isa_supported(Isa::kAvx512, cpu));
}

}  // namespace
}  // namespace ii
