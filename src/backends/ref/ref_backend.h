// SPDX-License-Identifier: MIT
/// @file
/// ii-ref: the reference backend.
///
/// Implements every OpSet operation in plain, single-threaded C++ that is easy to audit. Floating-point math is
/// done in double precision and rounded once per output element, so ii-ref is the oracle against which every
/// other backend is tested, and the last-resort fallback for nodes no faster backend supports.
#pragma once

#include "backend/backend.h"

#include <memory>

namespace ii::ref {

/// Name of the reference backend.
inline constexpr std::string_view kBackendName = "ii-ref";

/// Creates the reference backend.
std::unique_ptr<backend::Backend> create_ref_backend();

}  // namespace ii::ref
