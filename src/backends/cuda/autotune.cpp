// SPDX-License-Identifier: MIT
#include "backends/cuda/autotune.h"

#include "backends/cuda/cuda_device.h"

#include "core/json.h"
#include "core/platform/platform.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <format>
#include <fstream>
#include <sstream>
#include <system_error>
#include <vector>

namespace ii::cuda {

namespace {

constexpr int kFormat = 1;

/// A JSON string literal (the keys and values are ASCII, but escape anyway).
std::string json_string(std::string_view text) {
    std::string out = "\"";
    for (const char c : text) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (static_cast<unsigned char>(c) < 0x20) {
            out += std::format("\\u{:04x}", static_cast<int>(c));
        } else {
            out += c;
        }
    }
    out += '"';
    return out;
}

}  // namespace

TuningCache::TuningCache(std::filesystem::path path, std::string identity)
    : path_(std::move(path)), identity_(std::move(identity)) {
    if (path_.empty()) {
        return;
    }
    std::ifstream file(path_, std::ios::binary);
    if (!file) {
        return;
    }
    std::stringstream text;
    text << file.rdbuf();
    const Result<Json> json = Json::parse(text.str());
    if (!json.ok() || !json->is_object()) {
        return;
    }
    const Json* format = json->find("format");
    const Json* recorded = json->find("identity");
    const Json* entries = json->find("entries");
    if (format == nullptr || !format->is_number() || format->as_double() != kFormat || recorded == nullptr ||
        !recorded->is_string() || recorded->as_string() != identity_ || entries == nullptr || !entries->is_object()) {
        return;  // another device, library or ii version: start over
    }
    for (const auto& [key, value] : entries->as_object()) {
        if (value.is_string()) {
            entries_.emplace(key, value.as_string());
        }
    }
}

std::optional<std::string> TuningCache::find(std::string_view key) const {
    const std::lock_guard lock(mutex_);
    const auto it = entries_.find(key);
    if (it == entries_.end()) {
        return std::nullopt;
    }
    return it->second;
}

void TuningCache::store(const std::string& key, const std::string& value) {
    const std::lock_guard lock(mutex_);
    entries_[key] = value;
    save();
}

std::size_t TuningCache::size() const {
    const std::lock_guard lock(mutex_);
    return entries_.size();
}

void TuningCache::save() const {
    if (path_.empty()) {
        return;
    }
    std::string text =
        std::format("{{\n  \"format\": {},\n  \"identity\": {},\n  \"entries\": {{", kFormat, json_string(identity_));
    bool first = true;
    for (const auto& [key, value] : entries_) {
        text += first ? "\n    " : ",\n    ";
        text += json_string(key) + ": " + json_string(value);
        first = false;
    }
    text += "\n  }\n}\n";
    std::error_code error;
    std::filesystem::create_directories(path_.parent_path(), error);
    // Write a temporary file, then replace: another process never reads half a file.
    std::filesystem::path temporary = path_;
    temporary += std::format(".{:x}.tmp", std::chrono::steady_clock::now().time_since_epoch().count() ^
                                              reinterpret_cast<std::uintptr_t>(this));
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (!file) {
            return;
        }
        file << text;
        if (!file) {
            return;
        }
    }
    std::filesystem::rename(temporary, path_, error);
    if (error) {
        std::filesystem::remove(temporary, error);
    }
}

std::string DeviceFigures::to_string() const {
    return std::format("launch={:.3f};bandwidth={:.1f};link={:.2f};latency={:.2f}", launch_us, bandwidth_gbs, link_gbs,
                       link_latency_us);
}

std::optional<DeviceFigures> DeviceFigures::parse(std::string_view text) {
    DeviceFigures figures;
    int found = 0;
    while (!text.empty()) {
        const std::size_t semicolon = text.find(';');
        const std::string_view item = text.substr(0, semicolon);
        text = semicolon == std::string_view::npos ? std::string_view() : text.substr(semicolon + 1);
        const std::size_t eq = item.find('=');
        if (eq == std::string_view::npos) {
            return std::nullopt;
        }
        const std::string_view key = item.substr(0, eq);
        const std::string_view value = item.substr(eq + 1);
        double number = 0;
        const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
        if (error != std::errc() || end != value.data() + value.size() || !(number > 0)) {
            return std::nullopt;
        }
        if (key == "launch") {
            figures.launch_us = number;
        } else if (key == "bandwidth") {
            figures.bandwidth_gbs = number;
        } else if (key == "link") {
            figures.link_gbs = number;
        } else if (key == "latency") {
            figures.link_latency_us = number;
        } else {
            return std::nullopt;
        }
        ++found;
    }
    if (found != 4) {
        return std::nullopt;
    }
    return figures;
}

namespace {

/// Frees what a measurement allocated, whatever happens.
struct Scratch {
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    void* a = nullptr;
    void* b = nullptr;
    ~Scratch() {
        if (stream != nullptr) {
            cudaStreamSynchronize(stream);
        }
        if (exec != nullptr) {
            cudaGraphExecDestroy(exec);
        }
        if (graph != nullptr) {
            cudaGraphDestroy(graph);
        }
        cudaFree(a);
        cudaFree(b);
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
        cudaGetLastError();
    }
};

}  // namespace

Result<DeviceFigures> measure_device() {
    Scratch s;
    II_CUDA_TRY(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
    II_CUDA_TRY(cudaEventCreate(&s.start));
    II_CUDA_TRY(cudaEventCreate(&s.stop));
    // The best of a few repetitions of `enqueue`, in milliseconds.
    auto time = [&](auto&& enqueue, int repeats) -> Result<double> {
        double best = 1e30;
        for (int r = 0; r < repeats; ++r) {
            II_CUDA_TRY(cudaEventRecord(s.start, s.stream));
            II_TRY(enqueue());
            II_CUDA_TRY(cudaEventRecord(s.stop, s.stream));
            II_CUDA_TRY(cudaEventSynchronize(s.stop));
            float ms = 0;
            II_CUDA_TRY(cudaEventElapsedTime(&ms, s.start, s.stop));
            best = std::min(best, static_cast<double>(ms));
        }
        return best;
    };
    DeviceFigures figures;

    // Launch: the kernels of a recorded run follow each other.
    constexpr int kKernels = 200;
    II_CUDA_TRY(cudaStreamBeginCapture(s.stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < kKernels; ++i) {
        launch_empty(s.stream);
    }
    II_CUDA_TRY(cudaStreamEndCapture(s.stream, &s.graph));
    II_CUDA_TRY(cudaGraphInstantiate(&s.exec, s.graph, 0));
    II_CUDA_TRY(cudaGraphLaunch(s.exec, s.stream));
    II_ASSIGN_OR_RETURN(const double graph_ms, time(
                                                   [&]() -> Status {
                                                       II_CUDA_TRY(cudaGraphLaunch(s.exec, s.stream));
                                                       return ok_status();
                                                   },
                                                   5));
    // An empty kernel finishes sooner than one that reads memory: count the latency of a memory round trip too.
    figures.launch_us = graph_ms * 1000.0 / kKernels + 0.5;

    // Device memory: a copy reads and writes every byte.
    constexpr std::size_t kBytes = std::size_t{128} << 20;
    II_CUDA_TRY(cudaMalloc(&s.a, kBytes));
    II_CUDA_TRY(cudaMalloc(&s.b, kBytes));
    II_CUDA_TRY(cudaMemsetAsync(s.a, 0, kBytes, s.stream));
    II_ASSIGN_OR_RETURN(const double copy_ms,
                        time(
                            [&]() -> Status {
                                II_CUDA_TRY(cudaMemcpyAsync(s.b, s.a, kBytes, cudaMemcpyDeviceToDevice, s.stream));
                                return ok_status();
                            },
                            5));
    figures.bandwidth_gbs = 2.0 * static_cast<double>(kBytes) / (copy_ms * 1e6);

    // The link, as the runtime uses it: copies from and to pageable host memory.
    constexpr std::size_t kLinkBytes = std::size_t{32} << 20;
    std::vector<std::byte> host(kLinkBytes);
    II_ASSIGN_OR_RETURN(const double up_ms, time(
                                                [&]() -> Status {
                                                    II_CUDA_TRY(cudaMemcpyAsync(s.a, host.data(), kLinkBytes,
                                                                                cudaMemcpyHostToDevice, s.stream));
                                                    return ok_status();
                                                },
                                                3));
    II_ASSIGN_OR_RETURN(const double down_ms, time(
                                                  [&]() -> Status {
                                                      II_CUDA_TRY(cudaMemcpyAsync(host.data(), s.a, kLinkBytes,
                                                                                  cudaMemcpyDeviceToHost, s.stream));
                                                      return ok_status();
                                                  },
                                                  3));
    figures.link_gbs = static_cast<double>(kLinkBytes) / (std::max(up_ms, down_ms) * 1e6);

    // Latency: a small copy to the host and the wait for it, as the host would see it.
    constexpr int kSmall = 20;
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < kSmall; ++i) {
        II_CUDA_TRY(cudaMemcpyAsync(host.data(), s.a, 64, cudaMemcpyDeviceToHost, s.stream));
        II_CUDA_TRY(cudaStreamSynchronize(s.stream));
    }
    figures.link_latency_us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count() / kSmall;
    return figures;
}

std::filesystem::path tuning_cache_path(std::string_view option, std::string_view device_key) {
    if (option == "none") {
        return {};
    }
    if (!option.empty()) {
        return platform::path_from_utf8(std::string(option));
    }
    const Result<std::filesystem::path> directory = platform::default_cache_dir();
    if (!directory.ok()) {
        return {};
    }
    return *directory / std::format("cuda-{}.json", device_key);
}

std::string to_hex(const void* data, std::size_t size) {
    static constexpr char kDigits[] = "0123456789abcdef";
    std::string out;
    out.reserve(2 * size);
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        out += kDigits[bytes[i] >> 4];
        out += kDigits[bytes[i] & 15];
    }
    return out;
}

bool from_hex(std::string_view text, void* data, std::size_t size) {
    if (text.size() != 2 * size) {
        return false;
    }
    auto digit = [](char c) -> int {
        if (c >= '0' && c <= '9') {
            return c - '0';
        }
        if (c >= 'a' && c <= 'f') {
            return c - 'a' + 10;
        }
        return -1;
    };
    auto* bytes = static_cast<unsigned char*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        const int high = digit(text[2 * i]);
        const int low = digit(text[2 * i + 1]);
        if (high < 0 || low < 0) {
            return false;
        }
        bytes[i] = static_cast<unsigned char>(high * 16 + low);
    }
    return true;
}

}  // namespace ii::cuda
