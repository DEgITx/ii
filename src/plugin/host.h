// SPDX-License-Identifier: MIT
/// @file
/// Loading backend plugins (ii/plugin.h) and adapting their C interface to `ii::backend::Backend`.
///
/// Plugin libraries are loaded once per process and never unloaded (kernels, buffers and threads of a plugin may
/// live until the process exits). Every context creates its own backends from the loaded plugins.
#pragma once

#include <ii/context.h>
#include <ii/log.h>
#include <ii/plugin.h>
#include <ii/status.h>

#include "backend/backend.h"

#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ii::detail {
struct ContextImpl;
}  // namespace ii::detail

namespace ii::plugin {

/// Configuration a context passes to the plugins.
struct HostConfig {
    const Logger* logger = nullptr;  ///< must outlive the backends
    std::vector<std::pair<std::string, std::string>> options;
};

/// Checks a function table returned by a plugin's entry point.
Status validate_table(const ii_plugin_v1* table);

/// Loads the plugin library at `path` (once per process) and returns its validated function table.
Result<const ii_plugin_v1*> load_library(const std::filesystem::path& path);

/// Creates the backends of a plugin for one context. An `kUnavailable` error: the plugin cannot work here.
Result<std::vector<std::unique_ptr<backend::Backend>>> create_backends(const ii_plugin_v1& table,
                                                                       const HostConfig& config);

/// True if the file name looks like a backend plugin of this platform ("ii-backend-*.dll", "libii-backend-*.so").
bool is_plugin_file_name(const std::filesystem::path& path);

/// Plugin files to load, in order: `options.plugin_paths` (files, or directories to scan), then — if
/// `options.search_plugins` — the directories in `II_PLUGIN_PATH` and the directory of the ii library. The flag is
/// true for paths named explicitly (a failure to load them is an error).
std::vector<std::pair<std::filesystem::path, bool>> find_plugins(const ContextOptions& options);

/// Loads the plugins of a context: adds their backends to `context.backends` and describes every plugin in
/// `context.plugins`. Fails only for plugins named explicitly in `options.plugin_paths`.
Status load_plugins(detail::ContextImpl& context, const ContextOptions& options, std::string_view test_devices);

/// Adds the backends of an already loaded table (tests: plugins linked into the test program).
Status add_plugin(detail::ContextImpl& context, const ii_plugin_v1& table, std::string path,
                  std::span<const std::pair<std::string, std::string>> options);

}  // namespace ii::plugin
