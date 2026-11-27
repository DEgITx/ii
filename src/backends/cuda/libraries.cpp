// SPDX-License-Identifier: MIT
#include "backends/cuda/libraries.h"

#include "core/platform/dynlib.h"
#include "core/platform/platform.h"

#include <filesystem>
#include <format>
#include <mutex>
#include <optional>
#include <vector>

namespace ii::cuda {

namespace {

namespace fs = std::filesystem;

#define II_CUDA_STRINGIFY2(x) #x
#define II_CUDA_STRINGIFY(x) II_CUDA_STRINGIFY2(x)
constexpr const char* kMajor = II_CUDA_STRINGIFY(II_CUDA_VERSION_MAJOR);

#ifdef _WIN32
constexpr char kPathSeparator = ';';
std::string cublas_lt_name() {
    return std::format("cublasLt64_{}.dll", kMajor);
}
std::string nvrtc_name() {
    return std::format("nvrtc64_{}0_0.dll", kMajor);
}
std::string nvrtc_builtins_name(int major, int minor) {
    return std::format("nvrtc-builtins64_{}{}.dll", major, minor);
}
#else
constexpr char kPathSeparator = ':';
std::string cublas_lt_name() {
    return std::format("libcublasLt.so.{}", kMajor);
}
std::string nvrtc_name() {
    return std::format("libnvrtc.so.{}", kMajor);
}
std::string nvrtc_builtins_name(int major, int minor) {
    return std::format("libnvrtc-builtins.so.{}.{}", major, minor);
}
#endif

void add_list(std::vector<fs::path>& directories, std::string_view list) {
    while (!list.empty()) {
        const std::size_t end = list.find(kPathSeparator);
        const std::string_view item = list.substr(0, end);
        if (!item.empty()) {
            directories.push_back(platform::path_from_utf8(item));
#ifdef _WIN32
            directories.push_back(platform::path_from_utf8(item) / "x64");  // CUDA 13 keeps its DLLs in bin\x64
#endif
        }
        list = end == std::string_view::npos ? std::string_view() : list.substr(end + 1);
    }
}

std::vector<fs::path> search_directories(std::string_view directory) {
    std::vector<fs::path> directories;
    if (!directory.empty()) {
        directories.push_back(platform::path_from_utf8(directory));
    }
    if (Result<fs::path> self = platform::this_library_path(); self.ok()) {
        directories.push_back(self->parent_path());
    }
    if (const std::optional<std::string> cuda_path = platform::get_env("CUDA_PATH")) {
        const fs::path root = platform::path_from_utf8(*cuda_path);
#ifdef _WIN32
        directories.push_back(root / "bin" / "x64");
        directories.push_back(root / "bin");
#else
        directories.push_back(root / "lib64");
        directories.push_back(root / "lib");
#endif
    }
#ifdef _WIN32
    directories.push_back(platform::path_from_utf8(II_CUDA_TOOLKIT_BIN_DIR) / "x64");
    directories.push_back(platform::path_from_utf8(II_CUDA_TOOLKIT_BIN_DIR));
    add_list(directories, platform::get_env("PATH").value_or(""));
#else
    directories.push_back(platform::path_from_utf8(II_CUDA_TOOLKIT_LIBRARY_DIR));
    add_list(directories, platform::get_env("LD_LIBRARY_PATH").value_or(""));
    for (const char* standard : {"/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu",
                                 "/usr/lib64", "/usr/lib"}) {
        directories.emplace_back(standard);
    }
#endif
    return directories;
}

struct Loaded {
    platform::DynLib library;
    fs::path directory;
};

Result<Loaded> open_first(const std::vector<fs::path>& directories, const std::string& name) {
    std::string tried;
    for (const fs::path& directory : directories) {
        std::error_code ec;
        const fs::path path = directory / platform::path_from_utf8(name);
        if (!fs::is_regular_file(path, ec)) {
            continue;
        }
        Result<platform::DynLib> library = platform::DynLib::open(path);
        if (library.ok()) {
            return Loaded{std::move(*library), directory};
        }
        tried += std::format("; {}", library.status().message());
    }
    std::string searched;
    for (const fs::path& directory : directories) {
        searched += (searched.empty() ? "" : ", ") + platform::path_to_utf8(directory);
    }
    return errors::unavailable("{} not found (searched {}){}; set the plugin option cuda.library_path or CUDA_PATH",
                               name, searched, tried);
}

template <class Fn>
Status bind(const platform::DynLib& library, Fn& function, const char* name) {
    II_ASSIGN_OR_RETURN(function, library.function<Fn>(name));
    return ok_status();
}

Result<Libraries> load(std::string_view directory) {
    const std::vector<fs::path> directories = search_directories(directory);
    // The libraries stay loaded for the life of the process (like the plugin itself).
    II_ASSIGN_OR_RETURN(Loaded lt, open_first(directories, cublas_lt_name()));
    II_ASSIGN_OR_RETURN(Loaded rtc, open_first(directories, nvrtc_name()));
    auto* lt_library = new platform::DynLib(std::move(lt.library));    // NOLINT(cppcoreguidelines-owning-memory)
    auto* rtc_library = new platform::DynLib(std::move(rtc.library));  // NOLINT(cppcoreguidelines-owning-memory)

    Libraries l;
    II_TRY(bind(*lt_library, l.lt_create, "cublasLtCreate"));
    II_TRY(bind(*lt_library, l.lt_destroy, "cublasLtDestroy"));
    II_TRY(bind(*lt_library, l.lt_version, "cublasLtGetVersion"));
    II_TRY(bind(*lt_library, l.lt_status_string, "cublasLtGetStatusString"));
    II_TRY(bind(*lt_library, l.matmul_desc_create, "cublasLtMatmulDescCreate"));
    II_TRY(bind(*lt_library, l.matmul_desc_destroy, "cublasLtMatmulDescDestroy"));
    II_TRY(bind(*lt_library, l.matmul_desc_set, "cublasLtMatmulDescSetAttribute"));
    II_TRY(bind(*lt_library, l.layout_create, "cublasLtMatrixLayoutCreate"));
    II_TRY(bind(*lt_library, l.layout_destroy, "cublasLtMatrixLayoutDestroy"));
    II_TRY(bind(*lt_library, l.layout_set, "cublasLtMatrixLayoutSetAttribute"));
    II_TRY(bind(*lt_library, l.preference_create, "cublasLtMatmulPreferenceCreate"));
    II_TRY(bind(*lt_library, l.preference_destroy, "cublasLtMatmulPreferenceDestroy"));
    II_TRY(bind(*lt_library, l.preference_set, "cublasLtMatmulPreferenceSetAttribute"));
    II_TRY(bind(*lt_library, l.algo_heuristic, "cublasLtMatmulAlgoGetHeuristic"));
    II_TRY(bind(*lt_library, l.matmul, "cublasLtMatmul"));
    II_TRY(bind(*lt_library, l.matmul_desc_init, "cublasLtMatmulDescInit_internal"));
    II_TRY(bind(*lt_library, l.layout_init, "cublasLtMatrixLayoutInit_internal"));

    II_TRY(bind(*rtc_library, l.nvrtc_version, "nvrtcVersion"));
    II_TRY(bind(*rtc_library, l.nvrtc_error_string, "nvrtcGetErrorString"));
    II_TRY(bind(*rtc_library, l.nvrtc_create, "nvrtcCreateProgram"));
    II_TRY(bind(*rtc_library, l.nvrtc_destroy, "nvrtcDestroyProgram"));
    II_TRY(bind(*rtc_library, l.nvrtc_compile, "nvrtcCompileProgram"));
    II_TRY(bind(*rtc_library, l.nvrtc_log_size, "nvrtcGetProgramLogSize"));
    II_TRY(bind(*rtc_library, l.nvrtc_log, "nvrtcGetProgramLog"));
    II_TRY(bind(*rtc_library, l.nvrtc_cubin_size, "nvrtcGetCUBINSize"));
    II_TRY(bind(*rtc_library, l.nvrtc_cubin, "nvrtcGetCUBIN"));
    II_TRY(bind(*rtc_library, l.nvrtc_supported_archs, "nvrtcGetSupportedArchs"));
    II_TRY(bind(*rtc_library, l.nvrtc_num_supported_archs, "nvrtcGetNumSupportedArchs"));

    int major = 0;
    int minor = 0;
    if (l.nvrtc_version(&major, &minor) != NVRTC_SUCCESS) {
        return errors::unavailable("nvrtcVersion failed");
    }
    // NVRTC loads its builtins library by name, which finds it only on the system's search path; load it from
    // NVRTC's directory first so that the loader finds it already in the process.
    const fs::path builtins = rtc.directory / platform::path_from_utf8(nvrtc_builtins_name(major, minor));
    std::error_code ec;
    if (fs::is_regular_file(builtins, ec)) {
        if (Result<platform::DynLib> library = platform::DynLib::open(builtins); library.ok()) {
            new platform::DynLib(std::move(*library));  // NOLINT(cppcoreguidelines-owning-memory)
        }
    }
    const std::size_t lt_version = l.lt_version();
    l.description = std::format("cuBLASLt {}.{}.{} ({}), NVRTC {}.{} ({})", lt_version / 10000, lt_version / 100 % 100,
                                lt_version % 100, platform::path_to_utf8(lt.directory), major, minor,
                                platform::path_to_utf8(rtc.directory));
    return l;
}

}  // namespace

Result<const Libraries*> load_libraries(std::string_view directory) {
    static std::mutex mutex;
    static std::optional<Result<Libraries>> loaded;
    const std::lock_guard lock(mutex);
    if (!loaded) {
        loaded.emplace(load(directory));
    }
    if (!loaded->ok()) {
        return loaded->status();
    }
    return &**loaded;
}

}  // namespace ii::cuda
