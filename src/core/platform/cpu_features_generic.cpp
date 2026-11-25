// SPDX-License-Identifier: MIT
// CPU feature detection for architectures without specialized kernels: nothing to detect.
#include "core/platform/platform.h"

namespace ii::platform {

void detect_cpu_features(CpuInfo& info) {
    info.arch = CpuArch::kUnknown;
}

}  // namespace ii::platform
