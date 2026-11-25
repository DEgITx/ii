// SPDX-License-Identifier: MIT
// ii-cpu products with block-quantized weights: linear (w [N, K], transpose_w) and matmul (b [N, K], transpose_b) of
// f32 activations. The activation rows are quantized to 8 bits once (q8x32 or q8_K blocks, by weight type), then
// tiles of (activation rows x weight rows) are distributed over the threads; every weight row is decoded once per four
// activation rows by the ISA kernel.
#include <ii/half.h>

#include "backends/cpu/cpu_backend.h"
#include "core/aligned_buffer.h"
#include "ir/graph.h"
#include "ir/op_schema.h"

#include <algorithm>
#include <cstring>

namespace ii::cpu {

namespace {

std::optional<QType> qtype_of(DType dtype) noexcept {
    switch (dtype) {
    case DType::kQ4_0:
        return QType::kQ4_0;
    case DType::kQ4_1:
        return QType::kQ4_1;
    case DType::kQ5_0:
        return QType::kQ5_0;
    case DType::kQ5_1:
        return QType::kQ5_1;
    case DType::kQ8_0:
        return QType::kQ8_0;
    case DType::kQ2_K:
        return QType::kQ2_K;
    case DType::kQ3_K:
        return QType::kQ3_K;
    case DType::kQ4_K:
        return QType::kQ4_K;
    case DType::kQ5_K:
        return QType::kQ5_K;
    case DType::kQ6_K:
        return QType::kQ6_K;
    default:
        return std::nullopt;
    }
}

bool k_type(QType type) noexcept {
    return type == QType::kQ2_K || type == QType::kQ3_K || type == QType::kQ4_K || type == QType::kQ5_K ||
           type == QType::kQ6_K;
}

/// Bytes of one quantized activation row of k values.
std::int64_t act_row_bytes(QType type, std::int64_t k) noexcept {
    return k_type(type) ? k / 256 * 292 : k / 32 * 40;
}

constexpr std::int64_t kRowBlock = 64;  ///< activation rows per work item

// ------------------------------------------------------------------------------------------------------------------
// Panels: 8 weight rows repacked for the GEMM kernels (layouts in kernel_table.h)

float half_at(const std::uint8_t* p) {
    std::uint16_t bits = 0;
    std::memcpy(&bits, p, 2);
    return f16::from_bits(bits).to_float();
}

/// Values per block and bytes of one panel block (8 rows), or 0 when the type has no panel layout.
std::int64_t panel_block_values(QType type) {
    return type == QType::kQ4_K || type == QType::kQ6_K ? 256 : 32;
}

std::int64_t panel_block_bytes(QType type) {
    switch (type) {
    case QType::kQ4_0:
        return 160;
    case QType::kQ5_0:
        return 192;
    case QType::kQ8_0:
        return 288;
    case QType::kQ4_K:
        return 1216;
    case QType::kQ6_K:
        return 1696;
    default:
        return 0;
    }
}

/// The integer codes of one weight block (unsigned for the 4/5/6-bit types, signed bytes for q8_0) and its scales.
struct BlockCodes {
    std::uint8_t code[256];
    float d = 0;
    float dmin = 0;
    std::uint8_t scale[16];  ///< q4_K: 6-bit scales; q6_K: int8 scales (as bytes)
    std::uint8_t minimum[8];
};

void unpack_block(QType type, const std::uint8_t* b, BlockCodes& out) {
    switch (type) {
    case QType::kQ4_0:
        out.d = half_at(b);
        for (int i = 0; i < 16; ++i) {
            out.code[i] = b[2 + i] & 0x0F;
            out.code[i + 16] = b[2 + i] >> 4;
        }
        break;
    case QType::kQ5_0: {
        out.d = half_at(b);
        std::uint32_t qh = 0;
        std::memcpy(&qh, b + 2, 4);
        for (int i = 0; i < 16; ++i) {
            out.code[i] = static_cast<std::uint8_t>((b[6 + i] & 0x0F) | (((qh >> i) & 1u) << 4));
            out.code[i + 16] = static_cast<std::uint8_t>((b[6 + i] >> 4) | (((qh >> (i + 16)) & 1u) << 4));
        }
        break;
    }
    case QType::kQ8_0:
        out.d = half_at(b);
        std::memcpy(out.code, b + 2, 32);
        break;
    case QType::kQ4_K: {
        out.d = half_at(b);
        out.dmin = half_at(b + 2);
        const std::uint8_t* q = b + 4;
        for (int j = 0; j < 8; ++j) {
            if (j < 4) {
                out.scale[j] = q[j] & 63;
                out.minimum[j] = q[j + 4] & 63;
            } else {
                out.scale[j] = static_cast<std::uint8_t>((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
                out.minimum[j] = static_cast<std::uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
            }
        }
        for (int v = 0; v < 256; ++v) {
            const int chunk = v / 64;
            const int within = v % 64;
            const std::uint8_t byte = b[16 + 32 * chunk + within % 32];
            out.code[v] = within < 32 ? byte & 0x0F : byte >> 4;
        }
        break;
    }
    case QType::kQ6_K: {
        out.d = half_at(b + 208);
        std::memcpy(out.scale, b + 192, 16);
        for (int v = 0; v < 256; ++v) {
            const int n = v / 128;
            const int region = (v % 128) / 32;
            const int i = v % 32;
            const std::uint8_t* ql = b + 64 * n;
            const std::uint8_t qh = b[128 + 32 * n + i];
            const std::uint8_t low = region == 0   ? ql[i] & 0x0F
                                     : region == 1 ? ql[32 + i] & 0x0F
                                     : region == 2 ? ql[i] >> 4
                                                   : ql[32 + i] >> 4;
            out.code[v] = static_cast<std::uint8_t>(low | (((qh >> (2 * region)) & 3) << 4));
        }
        break;
    }
    default:
        break;
    }
}

/// Packs rows [0, rows) (a multiple of 8) of a weight [N, K] into panels.
Result<AlignedBuffer> pack_panels(QType type, DType dtype, const std::uint8_t* w, std::int64_t rows, std::int64_t k) {
    const std::int64_t values = panel_block_values(type);
    const std::int64_t blocks = k / values;
    const std::int64_t block_bytes = panel_block_bytes(type);
    const std::int64_t row_bytes = static_cast<std::int64_t>(storage_size(dtype, k));
    const std::int64_t source_block = static_cast<std::int64_t>(dtype_info(dtype).block_bytes);
    II_ASSIGN_OR_RETURN(AlignedBuffer buffer, AlignedBuffer::allocate(static_cast<std::size_t>(
                                                  rows / kPanelRows * blocks * block_bytes + 64)));
    auto* out = static_cast<std::uint8_t*>(buffer.data());
    BlockCodes codes[kPanelRows];
    for (std::int64_t group = 0; group < rows / kPanelRows; ++group) {
        for (std::int64_t b = 0; b < blocks; ++b) {
            std::uint8_t* p = out + (group * blocks + b) * block_bytes;
            std::memset(p, 0, static_cast<std::size_t>(block_bytes));
            for (int c = 0; c < kPanelRows; ++c) {
                unpack_block(type, w + (group * kPanelRows + c) * row_bytes + b * source_block, codes[c]);
            }
            auto put_f32 = [&](std::int64_t offset, int c, float value) { std::memcpy(p + offset + 4 * c, &value, 4); };
            // Byte j of a nibble step: row j / 4 (low half) and row 4 + j / 4 (high half), value 4 * step + j % 4.
            auto nibble_step = [&](std::uint8_t* dst, int first) {
                for (int j = 0; j < 16; ++j) {
                    const int value = first + j % 4;
                    dst[j] = static_cast<std::uint8_t>((codes[j / 4].code[value] & 0x0F) |
                                                       ((codes[4 + j / 4].code[value] & 0x0F) << 4));
                }
            };
            switch (type) {
            case QType::kQ4_0:
            case QType::kQ5_0:
                for (int c = 0; c < kPanelRows; ++c) {
                    put_f32(0, c, codes[c].d);
                }
                for (int s = 0; s < 8; ++s) {
                    nibble_step(p + 32 + 16 * s, 4 * s);
                    if (type == QType::kQ5_0) {
                        std::uint32_t bits = 0;
                        for (int i = 0; i < 32; ++i) {
                            bits |= static_cast<std::uint32_t>((codes[i / 4].code[4 * s + i % 4] >> 4) & 1) << i;
                        }
                        std::memcpy(p + 160 + 4 * s, &bits, 4);
                    }
                }
                break;
            case QType::kQ8_0:
                for (int c = 0; c < kPanelRows; ++c) {
                    put_f32(0, c, codes[c].d);
                }
                for (int s = 0; s < 8; ++s) {
                    for (int i = 0; i < 32; ++i) {
                        p[32 + 32 * s + i] = codes[i / 4].code[4 * s + i % 4];
                    }
                }
                break;
            case QType::kQ4_K:
                for (int c = 0; c < kPanelRows; ++c) {
                    put_f32(0, c, codes[c].d);
                    put_f32(32, c, codes[c].dmin);
                    for (int sub = 0; sub < 8; ++sub) {
                        p[64 + 8 * sub + c] = codes[c].scale[sub];
                        p[128 + 8 * sub + c] = codes[c].minimum[sub];
                    }
                }
                for (int step = 0; step < 64; ++step) {
                    nibble_step(p + 192 + 16 * step, 4 * step);
                }
                break;
            case QType::kQ6_K:
                for (int c = 0; c < kPanelRows; ++c) {
                    put_f32(0, c, codes[c].d);
                    for (int g = 0; g < 16; ++g) {
                        p[32 + 8 * g + c] = codes[c].scale[g];
                    }
                }
                for (int g = 0; g < 16; ++g) {
                    std::uint8_t* group_bytes = p + 160 + 96 * g;
                    for (int t = 0; t < 4; ++t) {
                        nibble_step(group_bytes + 16 * t, 16 * g + 4 * t);
                    }
                    for (int i = 0; i < 32; ++i) {
                        std::uint8_t high = 0;
                        for (int t = 0; t < 4; ++t) {
                            high = static_cast<std::uint8_t>(
                                high | (((codes[i / 4].code[16 * g + 4 * t + i % 4] >> 4) & 3) << (2 * t)));
                        }
                        group_bytes[64 + i] = high;
                    }
                }
                break;
            default:
                break;
            }
        }
    }
    return buffer;
}

class QuantMatmulKernel final : public Kernel {
public:
    QuantMatmulKernel(const CpuConfig& config, DType wtype, bool has_bias)
        : config_(config), wtype_(wtype), qtype_(*qtype_of(wtype)), has_bias_(has_bias) {}

    /// Repacks a constant weight into panels when the ISA variant has a panel kernel for its type.
    Status prepack(const Tensor& w) {
        gemm8_ = config_.table->qgemm8[static_cast<int>(qtype_)];
        const std::int64_t n = w.shape()[0];
        const std::int64_t k = w.shape()[1];
        if (gemm8_ == nullptr || panel_block_bytes(qtype_) == 0 || n < kPanelRows ||
            k % panel_block_values(qtype_) != 0) {
            gemm8_ = nullptr;
            return ok_status();
        }
        II_ASSIGN_OR_RETURN(const Tensor contiguous, w.contiguous());
        panel_rows_ = n / kPanelRows * kPanelRows;
        II_ASSIGN_OR_RETURN(
            panels_, pack_panels(qtype_, wtype_, static_cast<const std::uint8_t*>(contiguous.data()), panel_rows_, k));
        panel_stride_ = k / panel_block_values(qtype_) * panel_block_bytes(qtype_);
        return ok_status();
    }

    std::size_t workspace_size(std::span<const Shape> inputs, std::span<const Shape>) const override {
        const Shape& x = inputs[0];
        const std::int64_t k = x[x.rank() - 1];
        const std::int64_t rows = k == 0 ? 0 : x.numel() / k;
        return static_cast<std::size_t>(rows * act_row_bytes(qtype_, k) + 64);
    }

    Status run(const KernelContext& context) const override {
        const TensorView& x = context.input(0);
        const TensorView& w = context.input(1);
        const TensorView& y = context.output(0);
        const std::int64_t k = x.shape[x.rank() - 1];
        const std::int64_t n = w.shape[0];
        const std::int64_t rows = k == 0 ? 0 : x.numel() / k;
        if (rows == 0 || n == 0) {
            return ok_status();
        }
        const KernelTable& table = *config_.table;
        const std::int64_t a_stride = act_row_bytes(qtype_, k);
        const std::int64_t w_stride = static_cast<std::int64_t>(storage_size(wtype_, k));
        auto* act = reinterpret_cast<std::uint8_t*>(context.workspace.data());
        const float* px = x.as<float>();
        const QuantizeActFn quantize = k_type(qtype_) ? table.quantize_q8k : table.quantize_q8x32;
        parallel_range(context, rows, static_cast<double>(k) * 8.0, [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t r = begin; r < end; ++r) {
                quantize(px + r * k, act + r * a_stride, k);
            }
        });

        // Work items: blocks of activation rows x blocks of weight rows, several per thread.
        const std::int64_t row_blocks = (rows + kRowBlock - 1) / kRowBlock;
        const std::int64_t wanted = 4 * static_cast<std::int64_t>(config_.threads);
        std::int64_t col_block = std::max<std::int64_t>(8, (n * row_blocks + wanted - 1) / wanted);
        col_block = std::min<std::int64_t>((col_block + 7) / 8 * 8, std::max<std::int64_t>(n, 8));
        const std::int64_t col_blocks = (n + col_block - 1) / col_block;
        const QDotFn dot = table.qdot[static_cast<int>(qtype_)];
        const auto* pw = static_cast<const std::uint8_t*>(w.data);
        float* py = y.as<float>();
        const float* bias = has_bias_ ? context.input(2).as<float>() : nullptr;
        const double cost = static_cast<double>(std::min(rows, kRowBlock)) * static_cast<double>(col_block * k);
        parallel_chunks(
            context, config_.threads, row_blocks * col_blocks, cost, [&](int, std::int64_t begin, std::int64_t end) {
                for (std::int64_t item = begin; item < end; ++item) {
                    const std::int64_t rb = item % row_blocks;
                    const std::int64_t cb = item / row_blocks;
                    const std::int64_t r0 = rb * kRowBlock;
                    const std::int64_t r1 = std::min(rows, r0 + kRowBlock);
                    const std::int64_t c0 = cb * col_block;
                    const std::int64_t c1 = std::min(n, c0 + col_block);
                    std::int64_t c = c0;
                    if (gemm8_ != nullptr) {
                        // Panels of 8 columns (c0 is a multiple of 8).
                        const auto* panels = static_cast<const std::uint8_t*>(panels_.data());
                        for (; c + kPanelRows <= std::min(c1, panel_rows_); c += kPanelRows) {
                            const std::uint8_t* panel = panels + c / kPanelRows * panel_stride_;
                            for (std::int64_t r = r0; r < r1; r += 4) {
                                const int count = static_cast<int>(std::min<std::int64_t>(4, r1 - r));
                                gemm8_(k, panel, act + r * a_stride, a_stride, count, py + r * n + c, n);
                            }
                        }
                    }
                    for (; c < c1; ++c) {
                        for (std::int64_t r = r0; r < r1; r += 4) {
                            const int count = static_cast<int>(std::min<std::int64_t>(4, r1 - r));
                            dot(k, pw + c * w_stride, act + r * a_stride, a_stride, count, py + r * n + c, n);
                        }
                    }
                    if (bias != nullptr) {
                        for (std::int64_t r = r0; r < r1; ++r) {
                            for (std::int64_t col = c0; col < c1; ++col) {
                                py[r * n + col] += bias[col];
                            }
                        }
                    }
                }
            });
        return ok_status();
    }

private:
    CpuConfig config_;
    DType wtype_;
    QType qtype_;
    bool has_bias_;
    QGemm8Fn gemm8_ = nullptr;
    AlignedBuffer panels_;
    std::int64_t panel_rows_ = 0;    ///< leading weight rows held in panels (a multiple of 8)
    std::int64_t panel_stride_ = 0;  ///< bytes per panel
};

}  // namespace

Result<AlignedBuffer> pack_quantized_panels(QType type, DType dtype, const void* w, std::int64_t rows, std::int64_t k) {
    if (panel_block_bytes(type) == 0 || rows % kPanelRows != 0 || k % panel_block_values(type) != 0) {
        return errors::unsupported("no panel layout for these weights");
    }
    return pack_panels(type, dtype, static_cast<const std::uint8_t*>(w), rows, k);
}

std::int64_t quantized_panel_stride(QType type, std::int64_t k) {
    return k / panel_block_values(type) * panel_block_bytes(type);
}

Status check_quantized_matmul(const ir::Node& node) {
    const ir::Value* x = node.input(0);
    const ir::Value* w = node.input(1);
    const bool linear = node.op_name() == "linear";
    if (x->dtype() != DType::kF32 || node.output()->dtype() != DType::kF32) {
        return errors::unsupported("block-quantized '{}' is accelerated for f32 activations", node.op_name());
    }
    if (!qtype_of(w->dtype())) {
        return errors::unsupported("weights of type {}", to_string(w->dtype()));
    }
    if (w->rank() != 2 || !w->shape()[1].is_constant() ||
        w->shape()[1].value() % dtype_info(w->dtype()).block_size != 0) {
        return errors::unsupported("weight rows must be whole blocks");
    }
    if (linear) {
        if (!node.attrs().get_bool("transpose_w")) {
            return errors::unsupported("linear with block-quantized weights needs transpose_w");
        }
        if (node.input(2) != nullptr && node.input(2)->dtype() != DType::kF32) {
            return errors::unsupported("bias must be f32");
        }
    } else if (!node.attrs().get_bool("transpose_b") || node.attrs().get_bool("transpose_a") || x->rank() < 2) {
        return errors::unsupported("matmul with block-quantized weights needs transpose_b and a matrix a");
    }
    return ok_status();
}

Result<std::unique_ptr<Kernel>> create_quantized_matmul(const ir::Node& node, const CpuConfig& config) {
    const bool has_bias = node.op_name() == "linear" && node.input(2) != nullptr;
    auto kernel = std::make_unique<QuantMatmulKernel>(config, node.input(1)->dtype(), has_bias);
    if (node.input(1)->kind() == ir::ValueKind::kConstant) {
        II_TRY(kernel->prepack(node.input(1)->data()));
    }
    return std::unique_ptr<Kernel>(std::move(kernel));
}

}  // namespace ii::cpu
