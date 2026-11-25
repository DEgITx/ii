// SPDX-License-Identifier: MIT
#include "runtime/memory_planner.h"

#include <gtest/gtest.h>

#include <random>

namespace ii::runtime {
namespace {

/// Checks that no two simultaneously live requests overlap in memory and that everything fits.
void expect_valid(std::span<const MemoryRequest> requests, const MemoryPlan& plan, std::size_t alignment) {
    ASSERT_EQ(plan.offsets.size(), requests.size());
    for (std::size_t i = 0; i < requests.size(); ++i) {
        EXPECT_EQ(plan.offsets[i] % alignment, 0u);
        EXPECT_LE(plan.offsets[i] + requests[i].size, plan.total);
        for (std::size_t j = i + 1; j < requests.size(); ++j) {
            const bool live_together = requests[i].first <= requests[j].last && requests[j].first <= requests[i].last;
            const bool overlap = plan.offsets[i] < plan.offsets[j] + requests[j].size &&
                                 plan.offsets[j] < plan.offsets[i] + requests[i].size;
            EXPECT_FALSE(live_together && overlap && requests[i].size > 0 && requests[j].size > 0)
                << "requests " << i << " and " << j << " overlap";
        }
    }
}

TEST(MemoryPlannerTest, ReusesMemoryOfDeadBuffers) {
    // A chain a -> b -> c: a and c can share memory.
    const MemoryRequest requests[] = {{1000, 0, 1}, {1000, 1, 2}, {1000, 2, 3}};
    const MemoryPlan plan = plan_memory(requests);
    expect_valid(requests, plan, 64);
    EXPECT_EQ(plan.offsets[0], plan.offsets[2]);
    EXPECT_EQ(plan.total, 2 * 1024u);
}

TEST(MemoryPlannerTest, FillsGaps) {
    const MemoryRequest requests[] = {{4096, 0, 10}, {4096, 0, 2}, {4096, 0, 10}, {2048, 3, 5}, {1024, 6, 7}};
    const MemoryPlan plan = plan_memory(requests);
    expect_valid(requests, plan, 64);
    EXPECT_EQ(plan.total, 3 * 4096u);  // the small late buffers reuse the middle block
}

TEST(MemoryPlannerTest, RandomizedPlansAreValid) {
    std::mt19937 rng(7);
    for (int trial = 0; trial < 50; ++trial) {
        std::vector<MemoryRequest> requests;
        const int count = 1 + static_cast<int>(rng() % 40);
        for (int i = 0; i < count; ++i) {
            const int first = static_cast<int>(rng() % 30);
            requests.push_back(MemoryRequest{rng() % 5000, first, first + static_cast<int>(rng() % 10)});
        }
        const MemoryPlan plan = plan_memory(requests, 128);
        expect_valid(requests, plan, 128);
        std::size_t sum = 0;
        for (const MemoryRequest& r : requests) {
            sum += (std::max<std::size_t>(r.size, 1) + 127) / 128 * 128;
        }
        EXPECT_LE(plan.total, sum);
    }
}

}  // namespace
}  // namespace ii::runtime
