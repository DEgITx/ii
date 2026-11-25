// SPDX-License-Identifier: MIT
// Fuzz target: the .npy / .npz readers (including DEFLATE decompression) must reject malformed input without
// crashing.
#include <ii/io.h>

#include <cstddef>
#include <cstdint>
#include <span>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    const std::span<const std::byte> bytes = std::as_bytes(std::span<const std::uint8_t>(data, size));
    (void)ii::parse_npy(bytes);
    (void)ii::parse_npz(bytes);
    return 0;
}
