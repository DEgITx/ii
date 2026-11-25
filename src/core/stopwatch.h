// SPDX-License-Identifier: MIT
#pragma once

#include <chrono>

namespace ii {

/// Measures elapsed wall time with a monotonic clock.
class Stopwatch {
public:
    using Clock = std::chrono::steady_clock;

    Stopwatch() noexcept : start_(Clock::now()) {}

    void restart() noexcept { start_ = Clock::now(); }
    Clock::duration elapsed() const noexcept { return Clock::now() - start_; }
    double elapsed_seconds() const noexcept { return std::chrono::duration<double>(elapsed()).count(); }

private:
    Clock::time_point start_;
};

}  // namespace ii
