// SPDX-License-Identifier: MIT
/// @file
/// The counter-based Philox4x32-10 generator (Salmon et al., "Parallel random numbers: as easy as 1, 2, 3", SC'11)
/// and the exact definition of `random_uniform` / `random_normal` values. Every backend must produce bit-identical
/// uniform words, so this header is the specification.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>

namespace ii::ops {

/// Four 32-bit random words for `counter` under `key`.
constexpr std::array<std::uint32_t, 4> philox4x32_10(std::uint64_t counter, std::uint64_t key) noexcept {
    constexpr std::uint32_t kM0 = 0xD2511F53u;
    constexpr std::uint32_t kM1 = 0xCD9E8D57u;
    constexpr std::uint32_t kW0 = 0x9E3779B9u;
    constexpr std::uint32_t kW1 = 0xBB67AE85u;
    std::uint32_t c0 = static_cast<std::uint32_t>(counter);
    std::uint32_t c1 = static_cast<std::uint32_t>(counter >> 32);
    std::uint32_t c2 = 0;
    std::uint32_t c3 = 0;
    std::uint32_t k0 = static_cast<std::uint32_t>(key);
    std::uint32_t k1 = static_cast<std::uint32_t>(key >> 32);
    for (int round = 0; round < 10; ++round) {
        const std::uint64_t p0 = static_cast<std::uint64_t>(kM0) * c0;
        const std::uint64_t p1 = static_cast<std::uint64_t>(kM1) * c2;
        const auto hi0 = static_cast<std::uint32_t>(p0 >> 32);
        const auto lo0 = static_cast<std::uint32_t>(p0);
        const auto hi1 = static_cast<std::uint32_t>(p1 >> 32);
        const auto lo1 = static_cast<std::uint32_t>(p1);
        const std::uint32_t n0 = hi1 ^ c1 ^ k0;
        const std::uint32_t n2 = hi0 ^ c3 ^ k1;
        c0 = n0;
        c1 = lo1;
        c2 = n2;
        c3 = lo0;
        k0 += kW0;
        k1 += kW1;
    }
    return {c0, c1, c2, c3};
}

/// Maps a random word to [0, 1) with 24 bits of precision (exactly representable in every float type).
constexpr double unit_interval(std::uint32_t word) noexcept {
    return static_cast<double>(word >> 8) * (1.0 / 16777216.0);
}

/// Element `index` of `random_uniform` with seed `seed`, in [0, 1): word index % 4 of block index / 4.
constexpr double uniform_element(std::uint64_t index, std::uint64_t seed) noexcept {
    return unit_interval(philox4x32_10(index / 4, seed)[index % 4]);
}

/// Element `index` of a standard normal `random_normal` with seed `seed` (Box-Muller): block index / 2 provides two
/// pairs of words; element index uses pair index % 2.
inline double normal_element(std::uint64_t index, std::uint64_t seed) noexcept {
    const std::array<std::uint32_t, 4> words = philox4x32_10(index / 2, seed);
    const std::size_t pair = 2 * (index % 2);
    const double u1 = 1.0 - unit_interval(words[pair]);  // (0, 1]
    const double u2 = unit_interval(words[pair + 1]);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * std::numbers::pi * u2);
}

}  // namespace ii::ops
