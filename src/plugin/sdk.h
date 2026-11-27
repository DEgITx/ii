// SPDX-License-Identifier: MIT
/// @file
/// The plugin SDK: turns C++ backends (`ii::backend::Backend`) into a backend plugin (`ii/plugin.h`).
///
/// A plugin built with the ii sources links the static library `ii_plugin_sdk` (which links `ii_internal`, so the
/// plugin carries its own copy of the IR, the operation set and whatever else it uses) and defines its entry point:
///
/// @code
/// Result<std::vector<std::unique_ptr<ii::backend::Backend>>> create(const ii::plugin::PluginHost& host) {
///     if (!driver_present()) {
///         return ii::errors::unavailable("no CUDA driver");
///     }
///     ...
/// }
/// constexpr ii::plugin::PluginDefinition kPlugin{"ii-backend-cuda", "1.0", &create};
/// II_DEFINE_BACKEND_PLUGIN(kPlugin)
/// @endcode
///
/// The SDK parses the nodes ii passes as text, wraps constant data without copying, forwards kernels, device memory
/// and queues, and converts errors and exceptions into status codes. Nothing of the plugin's C++ runtime crosses
/// the boundary, so the plugin may be built with another compiler than the ii library that loads it.
#pragma once

#include <ii/log.h>
#include <ii/plugin.h>
#include <ii/status.h>

#include "backend/backend.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ii::plugin {

/// What the loading ii provides to `PluginDefinition::create_backends`.
class PluginHost {
public:
    explicit PluginHost(const ii_plugin_host& host) noexcept : host_(host) {}

    /// A configuration value (`ContextOptions::plugin_options`; "test_devices"), empty if unset.
    std::string option(std::string_view key) const;

    /// A logger writing to the logger of the ii context; valid while the backends live.
    Logger logger() const;

    /// Version of the ii library that loads the plugin.
    std::string_view ii_version() const noexcept { return host_.ii_version != nullptr ? host_.ii_version : ""; }

private:
    const ii_plugin_host& host_;
};

using BackendList = std::vector<std::unique_ptr<backend::Backend>>;

struct PluginDefinition {
    const char* name = nullptr;     ///< e.g. "ii-backend-cuda"
    const char* version = nullptr;  ///< version of the plugin
    /// Creates the backends for one context; an `kUnavailable` error when the plugin cannot work on this machine.
    Result<BackendList> (*create_backends)(const PluginHost& host) = nullptr;
    /// Forward `Backend::estimate_us` to ii; otherwise ii estimates with the roofline of the device info (the
    /// default of `Backend::estimate_us`) without asking the plugin.
    bool custom_estimates = false;
};

/// The function table of the plugin defined by `definition` (a plugin has one definition; it must outlive the
/// table). Used by `II_DEFINE_BACKEND_PLUGIN`, and by tests that load a plugin without a shared library.
const ii_plugin_v1* make_plugin_table(const PluginDefinition& definition);

}  // namespace ii::plugin

/// Defines the entry point of a backend plugin.
#define II_DEFINE_BACKEND_PLUGIN(definition)                                                                           \
    extern "C" II_PLUGIN_EXPORT const ii_plugin_v1* ii_backend_plugin_v1(uint32_t host_abi_version) {                  \
        if (host_abi_version != II_PLUGIN_ABI_VERSION) {                                                               \
            return nullptr;                                                                                            \
        }                                                                                                              \
        return ::ii::plugin::make_plugin_table(definition);                                                            \
    }
