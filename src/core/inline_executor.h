// SPDX-License-Identifier: MIT
#pragma once

#include <ii/executor.h>

namespace ii {

/// An executor that runs everything on the calling thread.
class InlineExecutor final : public Executor {
public:
    int concurrency() const noexcept override { return 1; }

    void parallel_for(std::int64_t n, std::int64_t grain,
                      FunctionRef<void(std::int64_t begin, std::int64_t end)> fn) override {
        (void)grain;
        if (n > 0) {
            fn(0, n);
        }
    }
};

}  // namespace ii
