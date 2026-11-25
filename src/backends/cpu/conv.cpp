// SPDX-License-Identifier: MIT
// ii-cpu convolution, pooling and resize (NCHW, f32, 1-D and 2-D).
//
// Convolution is a GEMM per image and group: Y[M_g, OH*OW] = W[M_g, C_g*KH*KW] x Col[C_g*KH*KW, OH*OW] + bias.
// The im2col matrix is never materialized: the GEMM driver asks for one KC x NC block of it at a time, which is
// packed straight from the image into micro-kernel panels (zero padding included). A 1x1 convolution with stride 1
// is a plain GEMM over the image. Depthwise convolutions (one input channel per group) are direct stencils.
#include "backends/cpu/gemm_driver.h"
#include "core/aligned_buffer.h"
#include "ir/op_schema.h"
#include "ops/fused.h"
#include "ops/infer_util.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace ii::cpu {

namespace {

std::int64_t round_up64(std::int64_t bytes) {
    return (bytes + 63) / 64 * 64;
}

/// Concrete 2-D window geometry (1-D windows have height 1).
struct Window2d {
    std::int64_t h = 1, w = 1;    // input
    std::int64_t oh = 1, ow = 1;  // output
    std::int64_t kh = 1, kw = 1;
    std::int64_t sh = 1, sw = 1;
    std::int64_t dh = 1, dw = 1;
    std::int64_t ph = 0, pw = 0;  // leading padding
    std::int64_t ph_end = 0, pw_end = 0;
};

/// Resolves the window of a node for concrete input / output shapes ([N, C, (H,) W]).
Window2d resolve(const ops::WindowAttrs& attrs, int spatial, const Shape& x, const Shape& y, std::int64_t kh,
                 std::int64_t kw) {
    Window2d g;
    const auto dim = [&](int d, std::int64_t kernel) {
        return ops::resolve_window(
            x[2 + d], kernel, attrs.strides[static_cast<std::size_t>(d)], attrs.dilations[static_cast<std::size_t>(d)],
            attrs.pads[static_cast<std::size_t>(d)], attrs.pads[static_cast<std::size_t>(d + spatial)], attrs.auto_pad);
    };
    if (spatial == 2) {
        const ops::WindowGeometry gh = dim(0, kh);
        g.h = x[2];
        g.oh = y[2];
        g.kh = kh;
        g.sh = gh.stride;
        g.dh = gh.dilation;
        g.ph = gh.pad_begin;
        g.ph_end = gh.pad_end;
    }
    const ops::WindowGeometry gw = dim(spatial - 1, kw);
    g.w = x[x.rank() - 1];
    g.ow = y[y.rank() - 1];
    g.kw = kw;
    g.sw = gw.stride;
    g.dw = gw.dilation;
    g.pw = gw.pad_begin;
    g.pw_end = gw.pad_end;
    return g;
}

// --------------------------------------------------------------------------------------------------------------------
// Convolution

class ConvKernel final : public Kernel {
public:
    ConvKernel(const CpuConfig& config, ops::WindowAttrs attrs, int spatial, std::int64_t groups, bool has_bias)
        : config_(config), attrs_(std::move(attrs)), spatial_(spatial), groups_(groups), has_bias_(has_bias) {}

    /// Sets the elementwise epilogue of a fused_conv (input 0 is the convolution result).
    void set_epilogue(CompiledElementwise epilogue) {
        epilogue_ = std::move(epilogue);
        program_ =
            EwProgram{epilogue_.code.data(), static_cast<int>(epilogue_.code.size()), epilogue_.result, Elem::kF32};
        has_epilogue_ = true;
    }

    /// Packs constant weights [M, C/groups, kernel...] into GEMM panels once (per group, M/groups x K matrices).
    Status prepack(const Tensor& weights) {
        II_ASSIGN_OR_RETURN(const Tensor w, weights.contiguous());
        const std::int64_t m = w.shape()[0];
        const std::int64_t k = w.numel() / std::max<std::int64_t>(m, 1);
        const std::int64_t mg = m / groups_;
        if (mg * groups_ != m || (k == 1 && groups_ > 1)) {
            return ok_status();  // depthwise or malformed: no GEMM
        }
        packed_kc_ = gemm_depth_block(config_, mg, k);
        packed_stride_ = packed_a_size(*config_.table, mg, k);
        II_ASSIGN_OR_RETURN(packed_weights_, AlignedBuffer::allocate(static_cast<std::size_t>(
                                                 std::max<std::int64_t>(packed_stride_ * groups_ * 4, 64))));
        auto* out = static_cast<float*>(packed_weights_.data());
        for (std::int64_t g = 0; g < groups_; ++g) {
            prepack_a(*config_.table, w.data_as<float>() + g * mg * k, k, 1, mg, k, packed_kc_,
                      out + g * packed_stride_);
        }
        return ok_status();
    }

    std::size_t workspace_size(std::span<const Shape> inputs, std::span<const Shape> outputs) const override {
        const Geometry g = geometry(inputs[0], inputs[1], outputs[0]);
        if (g.depthwise) {
            return static_cast<std::size_t>(std::max(config_.threads, 1)) *
                   static_cast<std::size_t>(round_up64(depthwise_scratch(g) * 4));
        }
        const GemmPlan plan = gemm_plan(g);
        return static_cast<std::size_t>(plan.slots) * plan.slot_bytes() + tables_bytes() + packed_bytes(g);
    }

    Status run(const KernelContext& context) const override {
        const TensorView& x = context.input(0);
        const TensorView& w = context.input(1);
        const TensorView& bias = context.input(2);
        const TensorView& y = context.output(0);
        if (y.numel() == 0) {
            return ok_status();
        }
        const Geometry g = geometry(x.shape, w.shape, y.shape);
        const float* bias_data = has_bias_ && bias.present() ? bias.as<float>() : nullptr;
        Epilogue epilogue;
        if (has_epilogue_) {
            prepare_epilogue(context, epilogue);
        }
        if (g.depthwise) {
            run_depthwise(context, g, x.as<float>(), w.as<float>(), bias_data, y.as<float>(), epilogue);
            return ok_status();
        }
        run_gemm_conv(context, g, x.as<float>(), w.as<float>(), bias_data, y.as<float>(), epilogue);
        return ok_status();
    }

private:
    static constexpr int kMaxExtras = 15;

    /// Extra inputs of the epilogue as seen from the output's (image, channel, pixel) index space.
    struct Epilogue {
        int extras = 0;
        const std::uint8_t* data[kMaxExtras] = {};
        Elem elem[kMaxExtras] = {};
        std::int64_t image_stride[kMaxExtras] = {};    // elements per image (0: broadcast)
        std::int64_t channel_stride[kMaxExtras] = {};  // elements per channel (0: broadcast)
        bool per_pixel[kMaxExtras] = {};               // varies along the pixels (else one value per row)
    };

    void prepare_epilogue(const KernelContext& context, Epilogue& e) const {
        const TensorView& y = context.output(0);
        const int rank = y.rank();
        e.extras = static_cast<int>(context.inputs.size()) - 3;
        for (int i = 0; i < e.extras; ++i) {
            const TensorView& v = context.input(3 + i);
            e.data[i] = static_cast<const std::uint8_t*>(v.data);
            e.elem[i] = epilogue_.input_elems[static_cast<std::size_t>(i + 1)];
            // Strides of the extra broadcast to y's shape.
            std::int64_t strides[kMaxRank] = {};
            const int offset = rank - v.rank();
            std::int64_t stride = 1;
            for (int d = v.rank() - 1; d >= 0; --d) {
                strides[d + offset] = v.shape[d] == 1 && y.shape[d + offset] != 1 ? 0 : stride;
                stride *= v.shape[d];
            }
            e.image_stride[i] = strides[0];
            e.channel_stride[i] = strides[1];
            bool pixels = false;
            for (int d = 2; d < rank; ++d) {
                pixels = pixels || strides[d] != 0;
            }
            e.per_pixel[i] = pixels;
        }
    }

    /// Applies the epilogue to `cols` pixels starting at pixel `j0` of (image, channel), in place at `c`.
    void apply_epilogue(const Epilogue& e, std::int64_t image, std::int64_t channel, std::int64_t j0, std::int64_t cols,
                        float* c) const {
        EwInput inputs[kMaxExtras + 1];
        inputs[0] = EwInput{c, Elem::kF32, false};
        for (int i = 0; i < e.extras; ++i) {
            const std::int64_t size = e.elem[i] == Elem::kF32 ? 4 : e.elem[i] == Elem::kBool ? 1 : 2;
            const std::int64_t element =
                image * e.image_stride[i] + channel * e.channel_stride[i] + (e.per_pixel[i] ? j0 : 0);
            inputs[i + 1] = EwInput{e.data[i] + element * size, e.elem[i], !e.per_pixel[i]};
        }
        config_.table->ew_run(program_, inputs, c, cols);
    }

    struct Geometry {
        std::int64_t n = 1, c = 1, m = 1;
        std::int64_t cg = 1, mg = 1;  // channels per group (input, output)
        std::int64_t k = 1;           // GEMM depth: cg * kh * kw
        std::int64_t batches = 1;     // n * groups
        bool depthwise = false;
        bool pointwise = false;  // 1x1, stride 1, no padding: the image is the B matrix
        Window2d window;
    };

    Geometry geometry(const Shape& x, const Shape& w, const Shape& y) const {
        Geometry g;
        g.n = x[0];
        g.c = x[1];
        g.m = y[1];
        g.cg = g.c / groups_;
        g.mg = g.m / groups_;
        const std::int64_t kh = spatial_ == 2 ? w[2] : 1;
        const std::int64_t kw = w[w.rank() - 1];
        g.window = resolve(attrs_, spatial_, x, y, kh, kw);
        g.k = g.cg * kh * kw;
        g.batches = g.n * groups_;
        g.depthwise = g.cg == 1 && groups_ > 1;
        const Window2d& win = g.window;
        g.pointwise = kh == 1 && kw == 1 && win.sh == 1 && win.sw == 1 && win.ph == 0 && win.pw == 0 &&
                      win.oh == win.h && win.ow == win.w;
        return g;
    }

    GemmPlan gemm_plan(const Geometry& g) const {
        return plan_gemm(config_, groups_, g.mg, g.window.oh * g.window.ow, g.k, true, true);
    }

    std::size_t tables_bytes() const {
        return static_cast<std::size_t>(round_up64(2 * groups_ * 8));  // a offsets, b matrix indices
    }

    /// Stride (in floats) of one group's packed im2col matrix: K rows of every NR-column panel.
    std::int64_t packed_stride(const Geometry& g) const {
        const std::int64_t nr = config_.table->gemm_nr;
        const std::int64_t pixels = g.window.oh * g.window.ow;
        return (pixels + nr - 1) / nr * nr * g.k;
    }

    std::size_t packed_bytes(const Geometry& g) const {
        return static_cast<std::size_t>(round_up64(groups_ * packed_stride(g) * 4));
    }

    /// Per image: packs the im2col matrix of every group into GEMM panels once (in parallel), then multiplies the
    /// weights by it with the panels used in place.
    void run_gemm_conv(const KernelContext& context, const Geometry& g, const float* x, const float* w,
                       const float* bias, float* y, const Epilogue& epilogue) const {
        const Window2d& win = g.window;
        const std::int64_t pixels = win.oh * win.ow;
        const std::int64_t plane = win.h * win.w;
        const KernelTable& table = *config_.table;
        const std::int64_t nr = table.gemm_nr;
        const std::int64_t panels = (pixels + nr - 1) / nr;
        const std::int64_t stride = packed_stride(g);
        const GemmPlan plan = gemm_plan(g);
        std::byte* cursor = context.workspace.data() + static_cast<std::size_t>(plan.slots) * plan.slot_bytes();
        auto* a_offsets = reinterpret_cast<std::int64_t*>(cursor);
        auto* b_offsets = a_offsets + groups_;
        auto* packed = reinterpret_cast<float*>(cursor + tables_bytes());
        for (std::int64_t group = 0; group < groups_; ++group) {
            a_offsets[group] = group * g.mg * g.k;
            b_offsets[group] = group;
        }
        const std::int64_t kernel_area = win.kh * win.kw;

        for (std::int64_t image = 0; image < g.n; ++image) {
            const float* x_image = x + image * g.c * plane;
            // Packs panel q (output pixels [q * NR, q * NR + NR)) of the im2col matrix of one group: K rows (input
            // channel, kernel row, kernel column) of NR values each.
            const auto pack_panel = [&](std::int64_t group, std::int64_t q) {
                const float* source_group = x_image + group * g.cg * plane;
                float* panel = packed + group * stride + q * g.k * nr;
                const std::int64_t first = q * nr;
                const std::int64_t cols = std::min(nr, pixels - first);
                if (g.pointwise) {
                    table.gemm_pack_b(source_group + first, Elem::kF32, plane, 1, g.k, cols, panel);
                    return;
                }
                const std::int64_t oh0 = first / win.ow;
                const std::int64_t ow0 = first % win.ow;
                for (std::int64_t p = 0; p < g.k; ++p) {
                    const std::int64_t channel = p / kernel_area;
                    const std::int64_t r = p % kernel_area;
                    const std::int64_t ky = r / win.kw;
                    const std::int64_t kx = r % win.kw;
                    const float* source = source_group + channel * plane;
                    float* row = panel + p * nr;
                    std::int64_t oh = oh0;
                    std::int64_t ow = ow0;
                    for (std::int64_t lane = 0; lane < cols;) {
                        const std::int64_t run = std::min(cols - lane, win.ow - ow);
                        const std::int64_t ih = oh * win.sh - win.ph + ky * win.dh;
                        float* to = row + lane;
                        if (ih < 0 || ih >= win.h) {
                            std::fill(to, to + run, 0.0f);
                        } else {
                            const float* from = source + ih * win.w;
                            const std::int64_t iw0 = ow * win.sw - win.pw + kx * win.dw;
                            if (win.sw == 1) {
                                const std::int64_t lo = std::clamp<std::int64_t>(-iw0, 0, run);
                                const std::int64_t hi = std::clamp<std::int64_t>(win.w - iw0, lo, run);
                                std::fill(to, to + lo, 0.0f);
                                std::memcpy(to + lo, from + iw0 + lo, static_cast<std::size_t>(hi - lo) * 4);
                                std::fill(to + hi, to + run, 0.0f);
                            } else {
                                for (std::int64_t t = 0; t < run; ++t) {
                                    const std::int64_t iw = iw0 + t * win.sw;
                                    to[t] = iw >= 0 && iw < win.w ? from[iw] : 0.0f;
                                }
                            }
                        }
                        lane += run;
                        ow += run;
                        if (ow == win.ow) {
                            ow = 0;
                            ++oh;
                        }
                    }
                    std::fill(row + cols, row + nr, 0.0f);
                }
            };
            parallel_range(context, groups_ * panels, static_cast<double>(g.k * nr) * 4,
                           [&](std::int64_t begin, std::int64_t end) {
                               for (std::int64_t item = begin; item < end; ++item) {
                                   pack_panel(item / panels, item % panels);
                               }
                           });

            GemmCall call{};
            call.a = Operand{w, Elem::kF32, g.k, 1};
            if (packed_weights_.data() != nullptr && plan.kc == packed_kc_) {
                call.packed_a = static_cast<const float*>(packed_weights_.data());
                call.packed_a_stride = packed_stride_;
                call.a_offsets = b_offsets;  // matrix index = group
            } else {
                call.a_offsets = a_offsets;
            }
            call.b.elem = Elem::kF32;
            call.packed_b = packed;
            call.packed_b_stride = stride;
            call.c = y + image * g.m * pixels;
            call.c_elem = Elem::kF32;
            call.row_bias = bias;
            call.row_bias_stride = g.mg;
            call.b_offsets = b_offsets;
            const auto apply = [&](std::int64_t group, std::int64_t i0, std::int64_t rows, std::int64_t j0,
                                   std::int64_t cols, float* c, std::int64_t ldc) {
                for (std::int64_t r = 0; r < rows; ++r) {
                    apply_epilogue(epilogue, image, group * g.mg + i0 + r, j0, cols, c + r * ldc);
                }
            };
            const EpilogueFn apply_ref(apply);
            if (has_epilogue_) {
                call.epilogue = &apply_ref;
            }
            run_gemm(context, config_, plan, call);
        }
    }

    std::int64_t depthwise_scratch(const Geometry& g) const {
        const Window2d& win = g.window;
        return depthwise_scratch_floats(win.oh, win.ow, win.kh, win.kw, win.sh, win.sw, win.dh, win.dw);
    }

    void run_depthwise(const KernelContext& context, const Geometry& g, const float* x, const float* w,
                       const float* bias, float* y, const Epilogue& epilogue) const {
        const Window2d& win = g.window;
        const KernelTable& table = *config_.table;
        const std::int64_t planes = g.n * g.m;
        const std::int64_t scratch_floats = round_up64(depthwise_scratch(g) * 4) / 4;
        auto* scratch = reinterpret_cast<float*>(context.workspace.data());
        const double cost = static_cast<double>(win.oh * win.ow * win.kh * win.kw) * 2;
        parallel_chunks(context, std::max(config_.threads, 1), planes, cost,
                        [&](int slot, std::int64_t begin, std::int64_t end) {
                            for (std::int64_t p = begin; p < end; ++p) {
                                const std::int64_t image = p / g.m;
                                const std::int64_t m = p % g.m;
                                const std::int64_t channel = m / g.mg;  // the input channel of the group
                                DepthwiseArgs args{};
                                args.in = x + (image * g.c + channel) * win.h * win.w;
                                args.h = win.h;
                                args.w = win.w;
                                args.filter = w + m * win.kh * win.kw;
                                args.kh = win.kh;
                                args.kw = win.kw;
                                args.bias = bias != nullptr ? bias[m] : 0.0f;
                                args.out = y + p * win.oh * win.ow;
                                args.oh = win.oh;
                                args.ow = win.ow;
                                args.sh = win.sh;
                                args.sw = win.sw;
                                args.dh = win.dh;
                                args.dw = win.dw;
                                args.ph = win.ph;
                                args.pw = win.pw;
                                args.scratch = scratch + slot * scratch_floats;
                                table.depthwise(args);
                                if (has_epilogue_) {
                                    apply_epilogue(epilogue, image, m, 0, win.oh * win.ow, args.out);
                                }
                            }
                        });
    }

    CpuConfig config_;
    ops::WindowAttrs attrs_;
    int spatial_;
    std::int64_t groups_;
    bool has_bias_;
    AlignedBuffer packed_weights_;
    std::int64_t packed_kc_ = 0;
    std::int64_t packed_stride_ = 0;
    bool has_epilogue_ = false;
    CompiledElementwise epilogue_;
    EwProgram program_{};
};

Status check_window_node(const ir::Node& node) {
    if (node.output()->dtype() != DType::kF32) {
        return errors::unsupported("'{}' is accelerated for f32 only", node.op_name());
    }
    const int rank = node.input(0)->rank();
    if (rank != 3 && rank != 4) {
        return errors::unsupported("'{}' is accelerated for 1-D and 2-D windows only", node.op_name());
    }
    return ok_status();
}

Status check_conv(const ir::Node& node) {
    II_TRY(check_window_node(node));
    for (const ir::Value* input : node.inputs()) {
        if (input != nullptr && input->dtype() != DType::kF32) {
            return errors::unsupported("'conv' with non-f32 operands");
        }
    }
    return ok_status();
}

/// The compiled epilogue of a fused_conv node (f32 in and out, at most 15 extra inputs).
Result<CompiledElementwise> compile_epilogue(const ir::Node& node) {
    II_ASSIGN_OR_RETURN(const std::unique_ptr<ir::Graph> program, ops::parse_epilogue_program(node));
    II_ASSIGN_OR_RETURN(CompiledElementwise compiled, compile_elementwise_graph(*program));
    if (compiled.input_elems.size() > 16 || compiled.input_elems[0] != Elem::kF32 || compiled.out_elem != Elem::kF32) {
        return errors::unsupported("'fused_conv' epilogue with these inputs");
    }
    return compiled;
}

Status check_fused_conv(const ir::Node& node) {
    II_TRY(check_window_node(node));
    for (int i = 0; i < 3; ++i) {
        if (node.input(i) != nullptr && node.input(i)->dtype() != DType::kF32) {
            return errors::unsupported("'fused_conv' with non-f32 operands");
        }
    }
    II_ASSIGN_OR_RETURN(const CompiledElementwise compiled, compile_epilogue(node));
    (void)compiled;
    return ok_status();
}

Result<std::unique_ptr<Kernel>> create_conv(const ir::Node& node, const CpuConfig& config) {
    const int spatial = node.input(0)->rank() - 2;
    II_ASSIGN_OR_RETURN(ops::WindowAttrs attrs, ops::parse_window_attrs(node.attrs(), spatial));
    auto kernel = std::make_unique<ConvKernel>(config, std::move(attrs), spatial, node.attrs().get_int("groups", 1),
                                               node.input(2) != nullptr);
    if (node.op_name() == ops::kFusedConv) {
        II_ASSIGN_OR_RETURN(CompiledElementwise epilogue, compile_epilogue(node));
        kernel->set_epilogue(std::move(epilogue));
    }
    if (node.input(1)->kind() == ir::ValueKind::kConstant) {
        II_TRY(kernel->prepack(node.input(1)->data()));
    }
    return std::unique_ptr<Kernel>(std::move(kernel));
}

// --------------------------------------------------------------------------------------------------------------------
// Pooling

template <bool kMax>
class PoolKernel final : public Kernel {
public:
    PoolKernel(const CpuConfig& config, ops::WindowAttrs attrs, int spatial, std::vector<std::int64_t> kernel,
               bool count_include_pad)
        : config_(config),
          attrs_(std::move(attrs)),
          spatial_(spatial),
          kernel_(std::move(kernel)),
          count_include_pad_(count_include_pad) {}

    std::size_t workspace_size(std::span<const Shape> inputs, std::span<const Shape> outputs) const override {
        (void)inputs;
        // Per thread: one strided-gather row and the per-column divisors.
        const std::int64_t ow = outputs[0][outputs[0].rank() - 1];
        return static_cast<std::size_t>(config_.threads) * static_cast<std::size_t>(round_up64(ow * 8));
    }

    Status run(const KernelContext& context) const override {
        const TensorView& x = context.input(0);
        const TensorView& y = context.output(0);
        if (y.numel() == 0) {
            return ok_status();
        }
        const Window2d win =
            resolve(attrs_, spatial_, x.shape, y.shape, spatial_ == 2 ? kernel_[0] : 1, kernel_.back());
        const KernelTable& table = *config_.table;
        const std::int64_t planes = x.shape[0] * x.shape[1];
        const float* px = x.as<float>();
        float* py = y.as<float>();
        const auto row_bytes = static_cast<std::size_t>(round_up64(win.ow * 8));
        std::byte* workspace = context.workspace.data();
        // Number of window positions inside the input (or the explicit padding) per output column / row.
        const auto count_in = [&](std::int64_t o, std::int64_t stride, std::int64_t pad, std::int64_t pad_end,
                                  std::int64_t dilation, std::int64_t kernel, std::int64_t size) {
            std::int64_t count = 0;
            for (std::int64_t k = 0; k < kernel; ++k) {
                const std::int64_t pos = o * stride - pad + k * dilation;
                const bool counted = count_include_pad_ ? pos >= -pad && pos < size + pad_end : pos >= 0 && pos < size;
                count += counted ? 1 : 0;
            }
            return count;
        };
        const double cost = static_cast<double>(win.oh * win.ow * win.kh * win.kw);
        parallel_chunks(
            context, std::max(config_.threads, 1), planes, cost, [&](int slot, std::int64_t begin, std::int64_t end) {
                for (std::int64_t p = begin; p < end; ++p) {
                    auto* scratch = reinterpret_cast<float*>(workspace + static_cast<std::size_t>(slot) * row_bytes);
                    const float* in = px + p * win.h * win.w;
                    float* out = py + p * win.oh * win.ow;
                    for (std::int64_t oh = 0; oh < win.oh; ++oh) {
                        float* out_row = out + oh * win.ow;
                        std::fill(out_row, out_row + win.ow, kMax ? -std::numeric_limits<float>::infinity() : 0.0f);
                        for (std::int64_t ky = 0; ky < win.kh; ++ky) {
                            const std::int64_t ih = oh * win.sh - win.ph + ky * win.dh;
                            if (ih < 0 || ih >= win.h) {
                                continue;
                            }
                            const float* in_row = in + ih * win.w;
                            for (std::int64_t kx = 0; kx < win.kw; ++kx) {
                                const std::int64_t iw0 = -win.pw + kx * win.dw;
                                const std::int64_t lo = iw0 >= 0 ? 0 : (-iw0 + win.sw - 1) / win.sw;
                                const std::int64_t hi =
                                    std::min(win.ow, win.w - iw0 <= 0 ? 0 : (win.w - iw0 + win.sw - 1) / win.sw);
                                if (lo >= hi) {
                                    continue;
                                }
                                const float* source = in_row + iw0 + lo;
                                if (win.sw != 1) {
                                    for (std::int64_t t = lo; t < hi; ++t) {
                                        scratch[t - lo] = in_row[iw0 + t * win.sw];
                                    }
                                    source = scratch;
                                }
                                if constexpr (kMax) {
                                    table.max_inplace(out_row + lo, source, hi - lo);
                                } else {
                                    table.axpy(out_row + lo, source, 1.0f, hi - lo);
                                }
                            }
                        }
                        if constexpr (!kMax) {
                            const std::int64_t rows = count_in(oh, win.sh, win.ph, win.ph_end, win.dh, win.kh, win.h);
                            for (std::int64_t ow = 0; ow < win.ow; ++ow) {
                                const std::int64_t cols =
                                    count_in(ow, win.sw, win.pw, win.pw_end, win.dw, win.kw, win.w);
                                const std::int64_t divisor = rows * cols;
                                out_row[ow] = divisor == 0 ? std::numeric_limits<float>::quiet_NaN()
                                                           : out_row[ow] / static_cast<float>(divisor);
                            }
                        }
                    }
                }
            });
        return ok_status();
    }

private:
    CpuConfig config_;
    ops::WindowAttrs attrs_;
    int spatial_;
    std::vector<std::int64_t> kernel_;
    bool count_include_pad_;
};

template <bool kMax>
Result<std::unique_ptr<Kernel>> create_pool(const ir::Node& node, const CpuConfig& config) {
    const int spatial = node.input(0)->rank() - 2;
    II_ASSIGN_OR_RETURN(ops::WindowAttrs attrs, ops::parse_window_attrs(node.attrs(), spatial));
    return std::unique_ptr<Kernel>(
        std::make_unique<PoolKernel<kMax>>(config, std::move(attrs), spatial, node.attrs().get_ints("kernel"),
                                           !kMax && node.attrs().get_bool("count_include_pad")));
}

// --------------------------------------------------------------------------------------------------------------------
// Resize (the last two dimensions; nearest and linear)

class ResizeKernel final : public Kernel {
public:
    ResizeKernel(const CpuConfig& config, std::vector<double> scales, std::string mode, std::string coordinate_mode,
                 std::string nearest_mode)
        : config_(config),
          scales_(std::move(scales)),
          mode_(std::move(mode)),
          coordinate_mode_(std::move(coordinate_mode)),
          nearest_mode_(std::move(nearest_mode)) {}

    std::size_t workspace_size(std::span<const Shape> inputs, std::span<const Shape> outputs) const override {
        (void)inputs;
        const Shape& y = outputs[0];
        const int taps = ops::resize_tap_count(mode_);
        const std::int64_t positions = y[y.rank() - 1] + (y.rank() >= 2 ? y[y.rank() - 2] : 1);
        // Indices and weights of both axes, then one f32 row per thread for the linear blend.
        return static_cast<std::size_t>(round_up64(positions * taps * 16) +
                                        std::max(config_.threads, 1) * round_up64(y[y.rank() - 1] * 4));
    }

    Status run(const KernelContext& context) const override {
        const TensorView& x = context.input(0);
        const TensorView& y = context.output(0);
        if (y.numel() == 0) {
            return ok_status();
        }
        const int rank = x.rank();
        const std::int64_t in_w = x.shape[rank - 1];
        const std::int64_t in_h = rank >= 2 ? x.shape[rank - 2] : 1;
        const std::int64_t out_w = y.shape[rank - 1];
        const std::int64_t out_h = rank >= 2 ? y.shape[rank - 2] : 1;
        const int taps = ops::resize_tap_count(mode_);
        const ops::ResizeMode mode{mode_, coordinate_mode_, nearest_mode_, -0.75, false};
        auto* x_index = reinterpret_cast<std::int64_t*>(context.workspace.data());
        auto* y_index = x_index + out_w * taps;
        auto* x_weight = reinterpret_cast<double*>(y_index + out_h * taps);
        auto* y_weight = x_weight + out_w * taps;
        const auto scale_of = [&](int d) { return scales_.empty() ? 0.0 : scales_[static_cast<std::size_t>(d)]; };
        ops::resize_taps_into(in_w, out_w, scale_of(rank - 1), mode, x_index, x_weight);
        if (rank >= 2) {
            ops::resize_taps_into(in_h, out_h, scale_of(rank - 2), mode, y_index, y_weight);
        } else {
            y_index[0] = 0;
            y_weight[0] = 1;
        }
        auto* rows = reinterpret_cast<float*>(context.workspace.data() + round_up64((out_w + out_h) * taps * 16));
        const std::int64_t row_stride = round_up64(out_w * 4) / 4;
        const std::int64_t planes = x.numel() / (in_h * in_w);
        const float* px = x.as<float>();
        float* py = y.as<float>();
        const double cost = static_cast<double>(out_h * out_w * taps * taps);
        parallel_chunks(
            context, std::max(config_.threads, 1), planes, cost, [&](int slot, std::int64_t begin, std::int64_t end) {
                for (std::int64_t p = begin; p < end; ++p) {
                    const float* in = px + p * in_h * in_w;
                    float* out = py + p * out_h * out_w;
                    float* blend = rows + slot * row_stride;
                    for (std::int64_t oy = 0; oy < out_h; ++oy) {
                        float* out_row = out + oy * out_w;
                        if (taps == 1) {
                            const float* in_row = in + y_index[oy] * in_w;
                            for (std::int64_t ox = 0; ox < out_w; ++ox) {
                                out_row[ox] = in_row[x_index[ox]];
                            }
                            continue;
                        }
                        std::fill(out_row, out_row + out_w, 0.0f);
                        for (int ty = 0; ty < taps; ++ty) {
                            const auto wy = static_cast<float>(y_weight[oy * taps + ty]);
                            if (wy == 0) {
                                continue;
                            }
                            const float* in_row = in + y_index[oy * taps + ty] * in_w;
                            for (std::int64_t ox = 0; ox < out_w; ++ox) {
                                float v = 0;
                                for (int tx = 0; tx < taps; ++tx) {
                                    v += static_cast<float>(x_weight[ox * taps + tx]) * in_row[x_index[ox * taps + tx]];
                                }
                                blend[ox] = v;
                            }
                            config_.table->axpy(out_row, blend, wy, out_w);
                        }
                    }
                }
            });
        return ok_status();
    }

private:
    CpuConfig config_;
    std::vector<double> scales_;
    std::string mode_;
    std::string coordinate_mode_;
    std::string nearest_mode_;
};

Status check_resize(const ir::Node& node) {
    const ir::Value* x = node.input(0);
    const ir::Value* y = node.output();
    if (x->dtype() != DType::kF32) {
        return errors::unsupported("'resize' is accelerated for f32 only");
    }
    const std::string_view mode = node.attrs().get_string("mode");
    if (mode != "nearest" && mode != "linear") {
        return errors::unsupported("'resize' mode '{}'", mode);
    }
    // Only the last two dimensions may change.
    for (int d = 0; d + 2 < x->rank(); ++d) {
        const std::vector<double> scales = node.attrs().get_floats("scales");
        const bool scaled =
            !scales.empty() && scales[static_cast<std::size_t>(d)] != 0 && scales[static_cast<std::size_t>(d)] != 1;
        if (!(x->shape()[static_cast<std::size_t>(d)] == y->shape()[static_cast<std::size_t>(d)]) || scaled) {
            return errors::unsupported("'resize' of other than the last two dimensions");
        }
    }
    return ok_status();
}

Result<std::unique_ptr<Kernel>> create_resize(const ir::Node& node, const CpuConfig& config) {
    return std::unique_ptr<Kernel>(std::make_unique<ResizeKernel>(
        config, node.attrs().get_floats("scales"), std::string(node.attrs().get_string("mode")),
        std::string(node.attrs().get_string("coordinate_mode")), std::string(node.attrs().get_string("nearest_mode"))));
}

}  // namespace

void register_conv_kernels(CpuBackend& backend) {
    backend.add_kernel("conv", create_conv, check_conv);
    backend.add_kernel(ops::kFusedConv, create_conv, check_fused_conv);
    backend.add_kernel("max_pool", create_pool<true>, check_window_node);
    backend.add_kernel("avg_pool", create_pool<false>, check_window_node);
    backend.add_kernel("resize", create_resize, check_resize);
}

}  // namespace ii::cpu
