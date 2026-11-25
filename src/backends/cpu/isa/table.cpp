// SPDX-License-Identifier: MIT
// The kernel table of one ISA variant.
#include "backends/cpu/isa/isa_kernels.h"
#include "backends/cpu/isa/simd.h"

namespace ii::cpu::II_ISA_NS {

const KernelTable& kernel_table() noexcept {
    static const KernelTable table{
        kIsaName,   ew_run, convert, gemm_mr(),  gemm_nr(), gemm_pack_a,   gemm_pack_b,
        gemm_macro, reduce, softmax, layer_norm, rms_norm,  attention_row,
    };
    return table;
}

}  // namespace ii::cpu::II_ISA_NS
