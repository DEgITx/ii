// SPDX-License-Identifier: MIT
/// @file
/// Reading and writing tensors in NumPy formats: `.npy` (one array) and `.npz` (a ZIP archive of named arrays, as
/// written by `numpy.savez` and `numpy.savez_compressed`).
///
/// @code
/// ii::Tensor x = ii::load_npy("input.npy").value();
/// ii::save_npz("outputs.npz", {{"logits", logits}}).value();
/// @endcode
///
/// Supported element types: float16/32/64, signed and unsigned 8-64 bit integers and bool (little-endian). bfloat16
/// has no NumPy type and is rejected; convert it to f32 first.
#pragma once

#include <ii/export.h>
#include <ii/status.h>
#include <ii/tensor.h>

#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <utility>
#include <vector>

II_BEGIN_DLL_EXPORTS

namespace ii {

/// A named tensor (an entry of an `.npz` archive, a model input or output).
using NamedTensor = std::pair<std::string, Tensor>;

/// Parses the contents of an `.npy` file. The data is copied.
II_API Result<Tensor> parse_npy(std::span<const std::byte> data);

/// Serializes a tensor as an `.npy` file (format version 1.0, C order).
II_API Result<std::vector<std::byte>> serialize_npy(const Tensor& tensor);

/// Reads an `.npy` file.
II_API Result<Tensor> load_npy(const std::filesystem::path& path);

/// Writes an `.npy` file.
II_API Status save_npy(const std::filesystem::path& path, const Tensor& tensor);

/// Reads every array of an `.npz` archive (stored or deflate-compressed entries), in archive order. Entry names
/// lose their `.npy` suffix.
II_API Result<std::vector<NamedTensor>> load_npz(const std::filesystem::path& path);

/// Parses the contents of an `.npz` archive (see `load_npz`). The data is copied.
II_API Result<std::vector<NamedTensor>> parse_npz(std::span<const std::byte> data);

/// Writes an uncompressed `.npz` archive (like `numpy.savez`).
II_API Status save_npz(const std::filesystem::path& path, std::span<const NamedTensor> arrays);

}  // namespace ii

II_END_DLL_EXPORTS
