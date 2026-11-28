// SPDX-License-Identifier: MIT
#include "backends/tensorrt/library.h"

#include "core/platform/dynlib.h"
#include "core/platform/platform.h"

#include <filesystem>
#include <format>
#include <mutex>
#include <optional>
#include <vector>

namespace ii::tensorrt {

namespace {

namespace fs = std::filesystem;

#define II_TENSORRT_STRINGIFY2(x) #x
#define II_TENSORRT_STRINGIFY(x) II_TENSORRT_STRINGIFY2(x)

#ifdef _WIN32
constexpr char kPathSeparator = ';';
const std::string kName = "nvinfer_" II_TENSORRT_STRINGIFY(II_TENSORRT_MAJOR) ".dll";
#else
constexpr char kPathSeparator = ':';
const std::string kName = "libnvinfer.so." II_TENSORRT_STRINGIFY(II_TENSORRT_MAJOR);
#endif

void add_root(std::vector<fs::path>& directories, const fs::path& root) {
    directories.push_back(root);
    directories.push_back(root / "bin");
    directories.push_back(root / "lib");
}

std::vector<fs::path> search_directories(std::string_view directory) {
    std::vector<fs::path> directories;
    if (!directory.empty()) {
        directories.push_back(platform::path_from_utf8(directory));
    }
    if (Result<fs::path> self = platform::this_library_path(); self.ok()) {
        directories.push_back(self->parent_path());
    }
    if (const std::optional<std::string> root = platform::get_env("TENSORRT_ROOT")) {
        add_root(directories, platform::path_from_utf8(*root));
    }
    add_root(directories, platform::path_from_utf8(II_TENSORRT_DEFAULT_DIR));
#ifdef _WIN32
    const std::string list = platform::get_env("PATH").value_or("");
#else
    const std::string list = platform::get_env("LD_LIBRARY_PATH").value_or("");
#endif
    for (std::string_view rest = list; !rest.empty();) {
        const std::size_t end = rest.find(kPathSeparator);
        if (const std::string_view item = rest.substr(0, end); !item.empty()) {
            directories.push_back(platform::path_from_utf8(item));
        }
        rest = end == std::string_view::npos ? std::string_view() : rest.substr(end + 1);
    }
#ifndef _WIN32
    for (const char* standard : {"/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu", "/usr/lib64", "/usr/lib"}) {
        directories.emplace_back(standard);
    }
#endif
    return directories;
}

Result<const Library*> load(std::string_view directory) {
    const std::vector<fs::path> directories = search_directories(directory);
    std::string tried;
    for (const fs::path& dir : directories) {
        std::error_code ec;
        const fs::path path = dir / platform::path_from_utf8(kName);
        if (!fs::is_regular_file(path, ec)) {
            continue;
        }
        Result<platform::DynLib> opened = platform::DynLib::open(path);
        if (!opened.ok()) {
            tried += std::format("; {}", opened.status().message());
            continue;
        }
        // Loaded for the life of the process, like the plugin itself.
        auto* library = new platform::DynLib(std::move(*opened));  // NOLINT(cppcoreguidelines-owning-memory)
        auto* result = new Library();                              // NOLINT(cppcoreguidelines-owning-memory)
        result->path = platform::path_to_utf8(path);
        II_ASSIGN_OR_RETURN(result->create_builder,
                            library->function<void* (*)(void*, std::int32_t)>("createInferBuilder_INTERNAL"));
        II_ASSIGN_OR_RETURN(result->create_runtime,
                            library->function<void* (*)(void*, std::int32_t)>("createInferRuntime_INTERNAL"));
        II_ASSIGN_OR_RETURN(auto get_version, library->function<std::int32_t (*)()>("getInferLibVersion"));
        result->version = get_version();
        if (result->version / 10000 != NV_TENSORRT_MAJOR) {
            return errors::unavailable("{} is TensorRT {}, the plugin was built for {}", result->path,
                                       result->version_string(), NV_TENSORRT_MAJOR);
        }
        return result;
    }
    std::string searched;
    for (const fs::path& dir : directories) {
        searched += (searched.empty() ? "" : ", ") + platform::path_to_utf8(dir);
    }
    return errors::unavailable("{} not found (searched {}){}; set the plugin option tensorrt.library_path or "
                               "TENSORRT_ROOT",
                               kName, searched, tried);
}

}  // namespace

std::string Library::version_string() const {
    return std::format("{}.{}.{}", version / 10000, version / 100 % 100, version % 100);
}

Result<const Library*> load_library(std::string_view directory) {
    static std::mutex mutex;
    static std::optional<Result<const Library*>> loaded;
    const std::lock_guard lock(mutex);
    if (!loaded) {
        loaded = load(directory);
    }
    return *loaded;
}

}  // namespace ii::tensorrt
