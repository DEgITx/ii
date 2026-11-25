// SPDX-License-Identifier: MIT
#include "runtime/memory_planner.h"

#include <algorithm>
#include <numeric>

namespace ii::runtime {

MemoryPlan plan_memory(std::span<const MemoryRequest> requests, std::size_t alignment) {
    MemoryPlan plan;
    plan.offsets.assign(requests.size(), 0);
    auto align = [alignment](std::size_t value) { return (value + alignment - 1) / alignment * alignment; };

    std::vector<std::size_t> order(requests.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return requests[a].size > requests[b].size; });

    struct Placed {
        std::size_t offset;
        std::size_t size;
        int first;
        int last;
    };
    std::vector<Placed> placed;
    std::vector<Placed> conflicts;
    for (const std::size_t index : order) {
        const MemoryRequest& request = requests[index];
        const std::size_t size = align(std::max<std::size_t>(request.size, 1));
        conflicts.clear();
        for (const Placed& p : placed) {
            if (p.first <= request.last && request.first <= p.last) {
                conflicts.push_back(p);
            }
        }
        std::sort(conflicts.begin(), conflicts.end(),
                  [](const Placed& a, const Placed& b) { return a.offset < b.offset; });
        // The lowest gap between live buffers that is large enough.
        std::size_t offset = 0;
        for (const Placed& p : conflicts) {
            if (p.offset >= offset + size) {
                break;
            }
            offset = std::max(offset, p.offset + p.size);
        }
        placed.push_back(Placed{offset, size, request.first, request.last});
        plan.offsets[index] = offset;
        plan.total = std::max(plan.total, offset + size);
    }
    return plan;
}

}  // namespace ii::runtime
