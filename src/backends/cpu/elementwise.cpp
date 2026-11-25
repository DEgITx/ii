// SPDX-License-Identifier: MIT
// ii-cpu elementwise operations: single nodes and fused_elementwise subgraphs compile to an `EwProgram` that the ISA
// interpreter evaluates tile by tile; this driver handles broadcasting and threading.
#include "backends/cpu/cpu_backend.h"
#include "core/assert.h"
#include "ir/op_schema.h"
#include "ops/fused.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace ii::cpu {

namespace {

constexpr int kMaxInputs = 16;
constexpr int kMaxRank = 8;

bool is_float_type(DType dtype) noexcept {
    return dtype == DType::kF32 || dtype == DType::kF16 || dtype == DType::kBF16;
}

/// The instruction an operation translates to.
struct Translation {
    EwOp op = EwOp::kLoad;
    int arity = 1;
    float imm0 = 0;
    float imm1 = 0;
    bool identity = false;  ///< the result is the first operand (casts between f32 and wider-or-equal types)
};

/// Translates one elementwise operation with the given input and output dtypes, or explains why ii-cpu cannot.
Result<Translation> translate(std::string_view op, const Attributes& attrs, std::span<const DType> in, DType out) {
    for (const DType dtype : in) {
        if (!elem_of(dtype)) {
            return errors::unsupported("dtype {} is not accelerated", to_string(dtype));
        }
    }
    if (!elem_of(out)) {
        return errors::unsupported("dtype {} is not accelerated", to_string(out));
    }
    auto all_float = [&] {
        for (const DType dtype : in) {
            if (!is_float_type(dtype)) {
                return false;
            }
        }
        return true;
    };
    auto all_bool = [&] {
        for (const DType dtype : in) {
            if (dtype != DType::kBool) {
                return false;
            }
        }
        return true;
    };
    static const std::unordered_map<std::string_view, EwOp> kUnary = {
        {"neg", EwOp::kNeg},     {"abs", EwOp::kAbs},   {"sign", EwOp::kSign},   {"exp", EwOp::kExp},
        {"log", EwOp::kLog},     {"sqrt", EwOp::kSqrt}, {"rsqrt", EwOp::kRsqrt}, {"recip", EwOp::kRecip},
        {"sin", EwOp::kSin},     {"cos", EwOp::kCos},   {"tanh", EwOp::kTanh},   {"sigmoid", EwOp::kSigmoid},
        {"erf", EwOp::kErf},     {"relu", EwOp::kRelu}, {"floor", EwOp::kFloor}, {"ceil", EwOp::kCeil},
        {"round", EwOp::kRound}, {"silu", EwOp::kSilu},
    };
    static const std::unordered_map<std::string_view, EwOp> kBinary = {
        {"add", EwOp::kAdd}, {"sub", EwOp::kSub}, {"mul", EwOp::kMul}, {"div", EwOp::kDiv},
        {"pow", EwOp::kPow}, {"max", EwOp::kMax}, {"min", EwOp::kMin}, {"lt", EwOp::kLt},
        {"le", EwOp::kLe},   {"gt", EwOp::kGt},   {"ge", EwOp::kGe},
    };
    Translation t;
    if (const auto it = kUnary.find(op); it != kUnary.end()) {
        if (!all_float()) {
            return errors::unsupported("'{}' is accelerated for float types only", op);
        }
        t.op = it->second;
        return t;
    }
    if (const auto it = kBinary.find(op); it != kBinary.end()) {
        if (!all_float()) {
            return errors::unsupported("'{}' is accelerated for float types only", op);
        }
        t.op = it->second;
        t.arity = 2;
        return t;
    }
    if (op == "gelu") {
        if (!all_float()) {
            return errors::unsupported("'gelu' is accelerated for float types only");
        }
        t.op = attrs.get_string("approximate", "none") == "tanh" ? EwOp::kGeluTanh : EwOp::kGelu;
        return t;
    }
    if (op == "mod") {
        if (!all_float()) {
            return errors::unsupported("'mod' is accelerated for float types only");
        }
        t.op = attrs.get_bool("fmod") ? EwOp::kFmod : EwOp::kMod;
        t.arity = 2;
        return t;
    }
    if (op == "eq" || op == "ne") {
        if (!all_float() && !all_bool()) {
            return errors::unsupported("'{}' is accelerated for float and bool types only", op);
        }
        t.op = op == "eq" ? EwOp::kEq : EwOp::kNe;
        t.arity = 2;
        return t;
    }
    if (op == "and" || op == "or" || op == "xor" || op == "not") {
        if (!all_bool()) {
            return errors::unsupported("'{}' is accelerated for bool only", op);
        }
        t.op = op == "and" ? EwOp::kAnd : op == "or" ? EwOp::kOr : op == "xor" ? EwOp::kXor : EwOp::kNot;
        t.arity = op == "not" ? 1 : 2;
        return t;
    }
    if (op == "where") {
        if (in.size() != 3 || in[0] != DType::kBool) {
            return errors::unsupported("'where' needs a bool condition");
        }
        t.op = EwOp::kWhere;
        t.arity = 3;
        return t;
    }
    if (op == "clamp") {
        if (!all_float()) {
            return errors::unsupported("'clamp' is accelerated for float types only");
        }
        t.op = EwOp::kClamp;
        const double low = attrs.get_float("min", -INFINITY);
        const double high = attrs.get_float("max", INFINITY);
        t.imm0 = static_cast<float>(low);
        t.imm1 = static_cast<float>(high);
        return t;
    }
    if (op == "cast") {
        switch (out) {
        case DType::kBool:
            t.op = EwOp::kToBool;
            break;
        case DType::kF16:
            if (in[0] == DType::kF16 || in[0] == DType::kBool) {
                t.identity = true;
            } else {
                t.op = EwOp::kRoundF16;
            }
            break;
        case DType::kBF16:
            if (in[0] == DType::kBF16 || in[0] == DType::kBool) {
                t.identity = true;
            } else {
                t.op = EwOp::kRoundBF16;
            }
            break;
        default:
            t.identity = true;  // to f32: every f16, bf16 and bool value is exact
            break;
        }
        return t;
    }
    return errors::unsupported("no elementwise kernel for '{}'", op);
}

/// One operation of a program source: value ids [0, num_inputs) are program inputs, then node outputs.
struct SourceNode {
    std::string_view op;
    const Attributes* attrs;
    std::vector<int> inputs;
    std::vector<DType> input_dtypes;
    DType out_dtype;
};

struct ProgramSource {
    std::vector<DType> input_dtypes;
    std::vector<SourceNode> nodes;  ///< node i defines value num_inputs + i
    int result = 0;                 ///< value id of the result
    DType out_dtype = DType::kInvalid;
};

struct CompiledProgram {
    std::vector<EwInstr> code;
    int result = 0;
};

/// Translates a program source into instructions with register allocation (registers are recycled after the last
/// use of a value).
Result<CompiledProgram> compile_program(const ProgramSource& source) {
    const int num_inputs = static_cast<int>(source.input_dtypes.size());
    const int num_values = num_inputs + static_cast<int>(source.nodes.size());
    std::vector<int> remaining_uses(static_cast<std::size_t>(num_values), 0);
    for (const SourceNode& node : source.nodes) {
        for (const int v : node.inputs) {
            ++remaining_uses[static_cast<std::size_t>(v)];
        }
    }
    ++remaining_uses[static_cast<std::size_t>(source.result)];

    CompiledProgram program;
    std::vector<int> reg_of(static_cast<std::size_t>(num_values), -1);
    std::vector<int> reg_refs(kEwMaxRegisters, 0);  // values currently held by each register
    auto allocate = [&]() -> Result<int> {
        for (int r = 0; r < kEwMaxRegisters; ++r) {
            if (reg_refs[static_cast<std::size_t>(r)] == 0) {
                return r;
            }
        }
        return errors::unsupported("elementwise program needs more than {} registers", kEwMaxRegisters);
    };
    auto release = [&](int value) {
        if (--remaining_uses[static_cast<std::size_t>(value)] == 0) {
            --reg_refs[static_cast<std::size_t>(reg_of[static_cast<std::size_t>(value)])];
        }
    };
    auto operand = [&](int value) -> Result<int> {
        if (reg_of[static_cast<std::size_t>(value)] < 0) {
            II_CHECK(value < num_inputs, "elementwise program uses a value before its definition");
            II_ASSIGN_OR_RETURN(const int r, allocate());
            reg_of[static_cast<std::size_t>(value)] = r;
            reg_refs[static_cast<std::size_t>(r)] = 1;
            program.code.push_back(
                EwInstr{EwOp::kLoad, static_cast<std::uint8_t>(r), static_cast<std::uint8_t>(value), 0, 0, 0.0f, 0.0f});
        }
        return reg_of[static_cast<std::size_t>(value)];
    };

    for (std::size_t i = 0; i < source.nodes.size(); ++i) {
        const SourceNode& node = source.nodes[i];
        const int value = num_inputs + static_cast<int>(i);
        II_ASSIGN_OR_RETURN(const Translation t, translate(node.op, *node.attrs, node.input_dtypes, node.out_dtype));
        if (static_cast<int>(node.inputs.size()) != (t.identity ? 1 : t.arity)) {
            return errors::unsupported("'{}' has an unexpected number of inputs", node.op);
        }
        std::array<int, 3> regs{0, 0, 0};
        for (std::size_t k = 0; k < node.inputs.size(); ++k) {
            II_ASSIGN_OR_RETURN(regs[k], operand(node.inputs[k]));
        }
        if (t.identity) {
            reg_of[static_cast<std::size_t>(value)] = regs[0];
            ++reg_refs[static_cast<std::size_t>(regs[0])];
            for (const int v : node.inputs) {
                release(v);
            }
        } else {
            // Release the operands first so the result may reuse one of their registers (in-place is safe: each
            // instruction reads a tile element before writing the same element).
            for (const int v : node.inputs) {
                release(v);
            }
            II_ASSIGN_OR_RETURN(const int dst, allocate());
            reg_of[static_cast<std::size_t>(value)] = dst;
            reg_refs[static_cast<std::size_t>(dst)] = 1;
            program.code.push_back(EwInstr{t.op, static_cast<std::uint8_t>(dst), static_cast<std::uint8_t>(regs[0]),
                                           static_cast<std::uint8_t>(regs[1]), static_cast<std::uint8_t>(regs[2]),
                                           t.imm0, t.imm1});
        }
        if (remaining_uses[static_cast<std::size_t>(value)] == 0) {
            --reg_refs[static_cast<std::size_t>(reg_of[static_cast<std::size_t>(value)])];  // dead value
        }
    }
    II_ASSIGN_OR_RETURN(program.result, operand(source.result));
    return program;
}

/// The program source of a single elementwise node or of a fused_elementwise node.
Result<ProgramSource> source_of(const ir::Node& node, std::unique_ptr<ir::Graph>& storage) {
    ProgramSource source;
    if (node.op_name() != ops::kFusedElementwise) {
        SourceNode single{node.op_name(), &node.attrs(), {}, {}, node.output()->dtype()};
        for (int i = 0; i < node.num_inputs(); ++i) {
            if (node.input(i) == nullptr) {
                return errors::unsupported("absent inputs are not supported");
            }
            source.input_dtypes.push_back(node.input(i)->dtype());
            single.inputs.push_back(i);
            single.input_dtypes.push_back(node.input(i)->dtype());
        }
        source.nodes.push_back(std::move(single));
        source.result = node.num_inputs();
        source.out_dtype = node.output()->dtype();
        return source;
    }
    II_ASSIGN_OR_RETURN(storage, ops::parse_fused_program(node));
    const ir::Graph& graph = *storage;
    std::unordered_map<const ir::Value*, int> ids;
    for (const ir::Value* input : graph.inputs()) {
        ids.emplace(input, static_cast<int>(source.input_dtypes.size()));
        source.input_dtypes.push_back(input->dtype());
    }
    for (const auto& inner : graph.nodes()) {
        SourceNode item{inner->op_name(), &inner->attrs(), {}, {}, inner->output()->dtype()};
        for (const ir::Value* value : inner->inputs()) {
            if (value == nullptr || !ids.contains(value)) {
                return errors::unsupported("fused program uses constants or absent inputs");
            }
            item.inputs.push_back(ids.at(value));
            item.input_dtypes.push_back(value->dtype());
        }
        ids.emplace(inner->output(), static_cast<int>(source.input_dtypes.size() + source.nodes.size()));
        source.nodes.push_back(std::move(item));
    }
    source.result = ids.at(graph.outputs()[0].value);
    source.out_dtype = node.output()->dtype();
    return source;
}

class ElementwiseKernel final : public Kernel {
public:
    ElementwiseKernel(const KernelTable& table, CompiledProgram program, std::vector<Elem> input_elems, Elem out)
        : table_(&table), code_(std::move(program.code)), input_elems_(std::move(input_elems)), out_elem_(out) {
        program_ = EwProgram{code_.data(), static_cast<int>(code_.size()), program.result, out};
    }

    bool can_run_in_place(int input) const override {
        return elem_size(input_elems_[static_cast<std::size_t>(input)]) == elem_size(out_elem_);
    }

    Status run(const KernelContext& context) const override {
        const TensorView& y = context.output(0);
        const int rank = y.rank();
        const int num_inputs = static_cast<int>(input_elems_.size());
        if (y.numel() == 0) {
            return ok_status();
        }

        // Element strides of every input in the output's index space (0 along broadcast dimensions).
        std::int64_t strides[kMaxInputs][kMaxRank] = {};
        for (int i = 0; i < num_inputs; ++i) {
            const Shape& shape = context.input(i).shape;
            const int offset = rank - shape.rank();
            std::int64_t stride = 1;
            for (int d = shape.rank() - 1; d >= 0; --d) {
                strides[i][d + offset] = (shape[d] == 1 && y.shape[d + offset] != 1) ? 0 : stride;
                stride *= shape[d];
            }
        }
        // Collapse dimensions that are contiguous for every operand; drop size-1 dimensions.
        std::int64_t sizes[kMaxRank];
        std::int64_t cstrides[kMaxInputs][kMaxRank];
        int crank = 0;
        for (int d = 0; d < rank; ++d) {
            const std::int64_t size = y.shape[d];
            if (size == 1) {
                continue;
            }
            bool merge = crank > 0;
            for (int i = 0; i < num_inputs && merge; ++i) {
                merge = cstrides[i][crank - 1] == strides[i][d] * size;
            }
            if (merge) {
                sizes[crank - 1] *= size;
                for (int i = 0; i < num_inputs; ++i) {
                    cstrides[i][crank - 1] = strides[i][d];
                }
            } else {
                sizes[crank] = size;
                for (int i = 0; i < num_inputs; ++i) {
                    cstrides[i][crank] = strides[i][d];
                }
                ++crank;
            }
        }
        const std::int64_t inner = crank > 0 ? sizes[crank - 1] : 1;
        const std::int64_t rows = y.numel() / inner;
        constexpr std::int64_t kChunk = 16384;
        const std::int64_t chunks_per_row = (inner + kChunk - 1) / kChunk;
        const std::int64_t chunk = (inner + chunks_per_row - 1) / chunks_per_row;
        const std::int64_t out_size = elem_size(out_elem_);

        auto work = [&](std::int64_t begin, std::int64_t end) {
            EwInput ins[kMaxInputs];
            for (std::int64_t item = begin; item < end; ++item) {
                const std::int64_t row = item / chunks_per_row;
                const std::int64_t c0 = (item % chunks_per_row) * chunk;
                const std::int64_t count = std::min(chunk, inner - c0);
                // Unravel the row index over the outer dimensions.
                std::int64_t offsets[kMaxInputs] = {};
                std::int64_t rest = row;
                for (int d = crank - 2; d >= 0; --d) {
                    const std::int64_t index = rest % sizes[d];
                    rest /= sizes[d];
                    for (int i = 0; i < num_inputs; ++i) {
                        offsets[i] += index * cstrides[i][d];
                    }
                }
                for (int i = 0; i < num_inputs; ++i) {
                    const std::int64_t inner_stride = crank > 0 ? cstrides[i][crank - 1] : 0;
                    const std::int64_t element = offsets[i] + c0 * inner_stride;
                    const Elem elem = input_elems_[static_cast<std::size_t>(i)];
                    ins[i] =
                        EwInput{static_cast<const std::uint8_t*>(context.input(i).data) + element * elem_size(elem),
                                elem, inner_stride == 0};
                }
                table_->ew_run(program_, ins, static_cast<std::uint8_t*>(y.data) + (row * inner + c0) * out_size,
                               count);
            }
        };
        const double cost = static_cast<double>(chunk) * static_cast<double>(code_.size() + 1);
        parallel_range(context, rows * chunks_per_row, cost, work);
        return ok_status();
    }

private:
    static std::int64_t elem_size(Elem elem) noexcept { return elem == Elem::kF32 ? 4 : elem == Elem::kBool ? 1 : 2; }

    const KernelTable* table_;
    std::vector<EwInstr> code_;
    EwProgram program_{};
    std::vector<Elem> input_elems_;
    Elem out_elem_;
};

Status check_elementwise(const ir::Node& node) {
    if (node.num_inputs() > kMaxInputs || node.output()->rank() > kMaxRank) {
        return errors::unsupported("too many inputs or dimensions");
    }
    std::unique_ptr<ir::Graph> storage;
    II_ASSIGN_OR_RETURN(const ProgramSource source, source_of(node, storage));
    II_ASSIGN_OR_RETURN(const CompiledProgram program, compile_program(source));
    (void)program;
    return ok_status();
}

Result<std::unique_ptr<Kernel>> create_elementwise(const ir::Node& node, const CpuConfig& config) {
    std::unique_ptr<ir::Graph> storage;
    II_ASSIGN_OR_RETURN(const ProgramSource source, source_of(node, storage));
    II_ASSIGN_OR_RETURN(CompiledProgram program, compile_program(source));
    std::vector<Elem> input_elems;
    for (const DType dtype : source.input_dtypes) {
        input_elems.push_back(*elem_of(dtype));
    }
    return std::unique_ptr<Kernel>(std::make_unique<ElementwiseKernel>(
        *config.table, std::move(program), std::move(input_elems), *elem_of(source.out_dtype)));
}

}  // namespace

Result<CompiledElementwise> compile_elementwise_graph(const ir::Graph& graph) {
    ProgramSource source;
    std::unordered_map<const ir::Value*, int> ids;
    for (const ir::Value* input : graph.inputs()) {
        ids.emplace(input, static_cast<int>(source.input_dtypes.size()));
        source.input_dtypes.push_back(input->dtype());
    }
    for (const auto& inner : graph.nodes()) {
        SourceNode item{inner->op_name(), &inner->attrs(), {}, {}, inner->output()->dtype()};
        for (const ir::Value* value : inner->inputs()) {
            if (value == nullptr || !ids.contains(value)) {
                return errors::unsupported("elementwise program uses constants or absent inputs");
            }
            item.inputs.push_back(ids.at(value));
            item.input_dtypes.push_back(value->dtype());
        }
        ids.emplace(inner->output(), static_cast<int>(source.input_dtypes.size() + source.nodes.size()));
        source.nodes.push_back(std::move(item));
    }
    if (graph.outputs().size() != 1 || static_cast<int>(source.input_dtypes.size()) > kMaxInputs) {
        return errors::unsupported("elementwise program with {} outputs and {} inputs", graph.outputs().size(),
                                   source.input_dtypes.size());
    }
    source.result = ids.at(graph.outputs()[0].value);
    source.out_dtype = graph.outputs()[0].value->dtype();
    II_ASSIGN_OR_RETURN(CompiledProgram program, compile_program(source));
    CompiledElementwise compiled;
    compiled.code = std::move(program.code);
    compiled.result = program.result;
    for (const DType dtype : source.input_dtypes) {
        const std::optional<Elem> elem = elem_of(dtype);
        if (!elem) {
            return errors::unsupported("elementwise program input of type {}", to_string(dtype));
        }
        compiled.input_elems.push_back(*elem);
    }
    const std::optional<Elem> out = elem_of(source.out_dtype);
    if (!out) {
        return errors::unsupported("elementwise program result of type {}", to_string(source.out_dtype));
    }
    compiled.out_elem = *out;
    return compiled;
}

void register_elementwise_kernels(CpuBackend& backend) {
    for (const std::string_view op :
         {"neg",  "abs",     "sign", "exp",  "log",   "sqrt",  "rsqrt", "recip", "sin",  "cos",
          "tanh", "sigmoid", "erf",  "relu", "floor", "ceil",  "round", "not",   "add",  "sub",
          "mul",  "div",     "pow",  "max",  "min",   "mod",   "eq",    "ne",    "lt",   "le",
          "gt",   "ge",      "and",  "or",   "xor",   "where", "clamp", "cast",  "gelu", "silu"}) {
        backend.add_kernel(op, create_elementwise, check_elementwise);
    }
    backend.add_kernel(ops::kFusedElementwise, create_elementwise, check_elementwise);
}

}  // namespace ii::cpu
