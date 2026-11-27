// SPDX-License-Identifier: MIT
// `ii info`: describes the machine as ii sees it.
#include <ii/ii.h>

#include "args.h"
#include "commands.h"
#include "json_writer.h"

#include <cstdint>
#include <cstdio>
#include <format>
#include <optional>
#include <string>
#include <vector>

namespace ii::cli {

namespace {

constexpr const char* kUsage = R"(Usage: ii info [options]

Describes the operating system, CPU, memory and the kernel variants ii uses.

Options:
  --roofline          Measure peak f32 FLOPS and memory bandwidth (takes a few seconds)
  --threads N         Threads for multi-threaded roofline measurements (default: all logical CPUs)
  --seconds S         Time per roofline measurement in seconds (default: 0.25)
  --json              Print JSON instead of text
  --help              Show this help
)";

std::string format_bytes(std::uint64_t bytes) {
    if (bytes == 0) {
        return "unknown";
    }
    constexpr const char* kUnits[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double value = static_cast<double>(bytes);
    int unit = 0;
    while (value >= 1024.0 && unit < 4) {
        value /= 1024.0;
        ++unit;
    }
    if (value == static_cast<double>(static_cast<std::uint64_t>(value))) {
        return std::format("{} {}", static_cast<std::uint64_t>(value), kUnits[unit]);
    }
    return std::format("{:.1f} {}", value, kUnits[unit]);
}

std::string feature_list(const CpuInfo& cpu) {
    std::string result;
    for (int i = 0; i < static_cast<int>(CpuFeature::kCount); ++i) {
        const auto feature = static_cast<CpuFeature>(i);
        if (cpu.has(feature)) {
            if (!result.empty()) {
                result += ' ';
            }
            result += to_string(feature);
        }
    }
    return result.empty() ? "none detected" : result;
}

void print_text(const OsInfo& os, const CpuInfo& cpu, const MemoryInfo& memory, const std::vector<IsaSupport>& isas,
                const Result<std::string>& selected, const Result<RooflineResult>* roofline) {
    const BuildInfo& build = build_info();
    std::fputs(std::format("ii {} (git {}, {}, {}, {})\n", build.version, build.git_hash, build.config, build.compiler,
                           build.shared ? "shared" : "static")
                   .c_str(),
               stdout);
    std::printf("\n");
    std::printf("OS        %s (%s)\n", os.name.c_str(), os.version.c_str());
    std::printf("CPU       %s\n", cpu.brand.empty() ? "unknown" : cpu.brand.c_str());
    std::printf("          %s, %s\n", std::string(to_string(cpu.arch)).c_str(),
                cpu.vendor.empty() ? "unknown vendor" : cpu.vendor.c_str());
    if (cpu.is_hybrid()) {
        std::printf("Cores     %d physical (%d performance + %d efficiency), %d logical\n", cpu.physical_cores,
                    cpu.performance_cores, cpu.efficiency_cores, cpu.logical_cpus);
    } else {
        std::printf("Cores     %d physical, %d logical\n", cpu.physical_cores, cpu.logical_cpus);
    }
    std::printf("Caches    L1d %s, L2 %s, L3 %s%s\n", format_bytes(cpu.caches.l1d).c_str(),
                format_bytes(cpu.caches.l2).c_str(), cpu.caches.l3 ? format_bytes(cpu.caches.l3).c_str() : "none",
                cpu.is_hybrid() ? " (performance core)" : "");
    std::printf("Features  %s\n", feature_list(cpu).c_str());
    std::printf("Memory    %s total, %s available\n", format_bytes(memory.total_bytes).c_str(),
                format_bytes(memory.available_bytes).c_str());

    std::string kernels;
    for (const IsaSupport& isa : isas) {
        if (!kernels.empty()) {
            kernels += ", ";
        }
        kernels += isa.name;
        if (!isa.compiled) {
            kernels += " (not compiled)";
        } else if (!isa.supported) {
            kernels += " (unsupported)";
        }
    }
    std::printf("Kernels   %s\n", kernels.c_str());
    if (selected.ok()) {
        std::printf("Selected  %s\n", selected->c_str());
    } else {
        std::printf("Selected  error: %s\n", selected.status().to_string().c_str());
    }
    if (const Result<Context> context = Context::create(); context.ok()) {
        for (const DeviceDescription& device : context->devices()) {
            std::string backends;
            for (const std::string& backend : device.backends) {
                backends += (backends.empty() ? "" : ", ") + backend;
            }
            std::string line = std::format("{:<8} {}", device.id, device.name);
            if (device.memory_bytes != 0) {
                line += ", " + format_bytes(device.memory_bytes);
            }
            std::printf("Device    %s (%s)\n", line.c_str(), backends.c_str());
        }
    } else {
        std::printf("Devices   error: %s\n", context.status().to_string().c_str());
    }

    if (roofline != nullptr) {
        std::printf("\n");
        if (!roofline->ok()) {
            std::printf("Roofline  error: %s\n", roofline->status().to_string().c_str());
            return;
        }
        const RooflineResult& r = **roofline;
        std::printf("Roofline  (%s kernels, %d threads)\n", r.isa.c_str(), r.threads);
        std::printf("  f32 FMA peak    %.1f GFLOP/s single thread, %.1f GFLOP/s all threads\n", r.f32_gflops_single,
                    r.f32_gflops_multi);
        std::printf("  memory          %.1f GB/s read, %.1f GB/s triad (%s arrays)\n", r.read_bandwidth_gbs,
                    r.triad_bandwidth_gbs, format_bytes(r.array_bytes).c_str());
        if (build.config == "Debug") {
            std::printf("  note: this is a Debug build; the numbers are not representative\n");
        }
    }
}

void print_json(const OsInfo& os, const CpuInfo& cpu, const MemoryInfo& memory, const std::vector<IsaSupport>& isas,
                const Result<std::string>& selected, const Result<RooflineResult>* roofline) {
    const BuildInfo& build = build_info();
    JsonWriter json;
    json.begin_object();

    json.key("build");
    json.begin_object();
    json.field("version", build.version);
    json.field("git_hash", build.git_hash);
    json.field("compiler", build.compiler);
    json.field("config", build.config);
    json.field("target_arch", build.target_arch);
    json.field("shared", build.shared);
    json.end_object();

    json.key("os");
    json.begin_object();
    json.field("name", os.name);
    json.field("version", os.version);
    json.end_object();

    json.key("cpu");
    json.begin_object();
    json.field("arch", to_string(cpu.arch));
    json.field("vendor", cpu.vendor);
    json.field("brand", cpu.brand);
    json.field("logical_cpus", cpu.logical_cpus);
    json.field("physical_cores", cpu.physical_cores);
    json.field("performance_cores", cpu.performance_cores);
    json.field("efficiency_cores", cpu.efficiency_cores);
    json.key("caches");
    json.begin_object();
    json.field("l1d", cpu.caches.l1d);
    json.field("l2", cpu.caches.l2);
    json.field("l3", cpu.caches.l3);
    json.end_object();
    json.key("features");
    json.begin_array();
    for (int i = 0; i < static_cast<int>(CpuFeature::kCount); ++i) {
        if (cpu.has(static_cast<CpuFeature>(i))) {
            json.value(to_string(static_cast<CpuFeature>(i)));
        }
    }
    json.end_array();
    json.end_object();

    json.key("memory");
    json.begin_object();
    json.field("total_bytes", memory.total_bytes);
    json.field("available_bytes", memory.available_bytes);
    json.end_object();

    json.key("kernels");
    json.begin_object();
    json.key("variants");
    json.begin_array();
    for (const IsaSupport& isa : isas) {
        json.begin_object();
        json.field("name", isa.name);
        json.field("compiled", isa.compiled);
        json.field("supported", isa.supported);
        json.end_object();
    }
    json.end_array();
    if (selected.ok()) {
        json.field("selected", *selected);
    } else {
        json.field("error", selected.status().to_string());
    }
    json.end_object();

    json.key("devices");
    json.begin_array();
    if (const Result<Context> context = Context::create(); context.ok()) {
        for (const DeviceDescription& device : context->devices()) {
            json.begin_object();
            json.field("id", device.id);
            json.field("name", device.name);
            json.field("memory_bytes", device.memory_bytes);
            json.key("backends");
            json.begin_array();
            for (const std::string& backend : device.backends) {
                json.value(backend);
            }
            json.end_array();
            json.end_object();
        }
    }
    json.end_array();

    if (roofline != nullptr) {
        json.key("roofline");
        json.begin_object();
        if (roofline->ok()) {
            const RooflineResult& r = **roofline;
            json.field("isa", r.isa);
            json.field("threads", r.threads);
            json.field("f32_gflops_single", r.f32_gflops_single);
            json.field("f32_gflops_multi", r.f32_gflops_multi);
            json.field("read_bandwidth_gbs", r.read_bandwidth_gbs);
            json.field("triad_bandwidth_gbs", r.triad_bandwidth_gbs);
            json.field("array_bytes", r.array_bytes);
        } else {
            json.field("error", roofline->status().to_string());
        }
        json.end_object();
    }

    json.end_object();
    std::printf("%s\n", json.str().c_str());
}

}  // namespace

int run_info(const std::vector<std::string>& args) {
    auto parsed = Args::parse(args, {"roofline", "json", "help"}, {"threads", "seconds"});
    if (!parsed.ok()) {
        std::fprintf(stderr, "ii info: %s\n\n%s", std::string(parsed.status().message()).c_str(), kUsage);
        return 2;
    }
    if (parsed->flag("help")) {
        std::printf("%s", kUsage);
        return 0;
    }
    if (!parsed->positionals().empty()) {
        std::fprintf(stderr, "ii info: unexpected argument '%s'\n", parsed->positionals().front().c_str());
        return 2;
    }

    const OsInfo os = os_info();
    const CpuInfo& cpu = cpu_info();
    const MemoryInfo memory = memory_info();
    const std::vector<IsaSupport> isas = isa_support();
    const Result<std::string> selected = selected_isa();

    std::optional<Result<RooflineResult>> roofline;
    if (parsed->flag("roofline")) {
        const Result<int> threads = parsed->int_option("threads", 0);
        const Result<double> seconds = parsed->double_option("seconds", 0.25);
        if (!threads.ok() || !seconds.ok()) {
            const Status& error = !threads.ok() ? threads.status() : seconds.status();
            std::fprintf(stderr, "ii info: %s\n", std::string(error.message()).c_str());
            return 2;
        }
        roofline = measure_roofline({.threads = *threads, .seconds_per_measurement = *seconds});
    }

    const Result<RooflineResult>* roofline_ptr = roofline ? &*roofline : nullptr;
    if (parsed->flag("json")) {
        print_json(os, cpu, memory, isas, selected, roofline_ptr);
    } else {
        print_text(os, cpu, memory, isas, selected, roofline_ptr);
    }
    return (roofline && !roofline->ok()) ? 1 : 0;
}

}  // namespace ii::cli
