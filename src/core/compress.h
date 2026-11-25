// SPDX-License-Identifier: MIT
/// @file
/// DEFLATE decompression (RFC 1951) and CRC-32, for ZIP archives (`.npz`) and other compressed formats.
///
/// The decoder is a small, careful implementation in the style of zlib's `puff`: it favours simplicity and robustness
/// against malicious input over speed. Every read is bounds-checked and the output size is limited by the caller.
#pragma once

#include <ii/status.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ii {

/// Decompresses a raw DEFLATE stream (no zlib or gzip header). Fails on malformed data or when the output would
/// exceed `max_output` bytes.
Result<std::vector<std::byte>> inflate(std::span<const std::byte> data, std::size_t max_output);

/// CRC-32 (IEEE 802.3, as used by ZIP and PNG), continuing from `crc` (0 to start).
std::uint32_t crc32(std::span<const std::byte> data, std::uint32_t crc = 0) noexcept;

}  // namespace ii
