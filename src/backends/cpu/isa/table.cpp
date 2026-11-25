// SPDX-License-Identifier: MIT
// The kernel table of one ISA variant.
#include "backends/cpu/isa/isa_kernels.h"
#include "backends/cpu/isa/simd.h"

namespace ii::cpu::II_ISA_NS {

const KernelTable& kernel_table() noexcept {
    static const KernelTable table{
        kIsaName,
        ew_run,
        convert,
        gemm_mr(),
        gemm_nr(),
        gemm_pack_a,
        gemm_pack_b,
        gemm_macro,
        reduce,
        softmax,
        layer_norm,
        rms_norm,
        attention_row,
        axpy,
        max_inplace,
        depthwise,
        quantize_q8x32,
        quantize_q8k,
        {qdot_q4_0, qdot_q4_1, qdot_q5_0, qdot_q5_1, qdot_q8_0, qdot_q2_k, qdot_q3_k, qdot_q4_k, qdot_q5_k, qdot_q6_k},
        {kQGemm8[0], kQGemm8[1], kQGemm8[2], kQGemm8[3], kQGemm8[4], kQGemm8[5], kQGemm8[6], kQGemm8[7], kQGemm8[8],
         kQGemm8[9]},
    };
    return table;
}

}  // namespace ii::cpu::II_ISA_NS
