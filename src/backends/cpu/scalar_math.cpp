// SPDX-License-Identifier: MIT
// Scalar fallbacks of ii-cpu for functions that are rare in neural networks (compiled without ISA flags).
#include "backends/cpu/kernel_table.h"

#include <cmath>

namespace ii::cpu {

float scalar_pow(float a, float b) noexcept {
    return static_cast<float>(std::pow(static_cast<double>(a), static_cast<double>(b)));
}

float scalar_fmod(float a, float b) noexcept {
    return std::fmod(a, b);  // exact
}

float scalar_mod(float a, float b) noexcept {
    double r = std::fmod(static_cast<double>(a), static_cast<double>(b));
    if (r != 0 && ((r < 0) != (b < 0))) {
        r += b;
    }
    return static_cast<float>(r);
}

float scalar_sin(float a) noexcept {
    return static_cast<float>(std::sin(static_cast<double>(a)));
}

float scalar_cos(float a) noexcept {
    return static_cast<float>(std::cos(static_cast<double>(a)));
}

}  // namespace ii::cpu
