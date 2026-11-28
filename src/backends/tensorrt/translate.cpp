// SPDX-License-Identifier: MIT
#include "backends/tensorrt/translate.h"

#include <ii/half.h>

#include "compiler/passes.h"
#include "ir/op_schema.h"
#include "ops/fused.h"
#include "tensor/dim_poly.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace ii::tensorrt {

namespace {

namespace nv = nvinfer1;

constexpr int kMaxRank = nv::Dims::MAX_DIMS;
constexpr std::int64_t kMaxTopK = 3840;  // TensorRT's limit for ITopKLayer

std::optional<nv::DataType> trt_type(DType dtype) {
    switch (dtype) {
    case DType::kF32:
        return nv::DataType::kFLOAT;
    case DType::kF16:
        return nv::DataType::kHALF;
    case DType::kBF16:
        return nv::DataType::kBF16;
    case DType::kI32:
        return nv::DataType::kINT32;
    case DType::kI64:
        return nv::DataType::kINT64;
    case DType::kBool:
        return nv::DataType::kBOOL;
    case DType::kU8:
        return nv::DataType::kUINT8;
    default:
        return std::nullopt;
    }
}

bool is_index(DType dtype) {
    return dtype == DType::kI32 || dtype == DType::kI64;
}

const std::unordered_map<std::string_view, nv::UnaryOperation>& unary_ops() {
    static const std::unordered_map<std::string_view, nv::UnaryOperation> map = {
        {"neg", nv::UnaryOperation::kNEG},     {"abs", nv::UnaryOperation::kABS},
        {"sign", nv::UnaryOperation::kSIGN},   {"exp", nv::UnaryOperation::kEXP},
        {"log", nv::UnaryOperation::kLOG},     {"sqrt", nv::UnaryOperation::kSQRT},
        {"recip", nv::UnaryOperation::kRECIP}, {"sin", nv::UnaryOperation::kSIN},
        {"cos", nv::UnaryOperation::kCOS},     {"erf", nv::UnaryOperation::kERF},
        {"floor", nv::UnaryOperation::kFLOOR}, {"ceil", nv::UnaryOperation::kCEIL},
        {"round", nv::UnaryOperation::kROUND}, {"rsqrt", nv::UnaryOperation::kSQRT},
    };
    return map;
}

const std::unordered_map<std::string_view, nv::ActivationType>& activations() {
    static const std::unordered_map<std::string_view, nv::ActivationType> map = {
        {"relu", nv::ActivationType::kRELU},
        {"sigmoid", nv::ActivationType::kSIGMOID},
        {"tanh", nv::ActivationType::kTANH},
    };
    return map;
}

const std::unordered_map<std::string_view, nv::ElementWiseOperation>& binary_ops() {
    static const std::unordered_map<std::string_view, nv::ElementWiseOperation> map = {
        {"add", nv::ElementWiseOperation::kSUM},    {"sub", nv::ElementWiseOperation::kSUB},
        {"mul", nv::ElementWiseOperation::kPROD},   {"div", nv::ElementWiseOperation::kDIV},
        {"pow", nv::ElementWiseOperation::kPOW},    {"max", nv::ElementWiseOperation::kMAX},
        {"min", nv::ElementWiseOperation::kMIN},    {"eq", nv::ElementWiseOperation::kEQUAL},
        {"lt", nv::ElementWiseOperation::kLESS},    {"gt", nv::ElementWiseOperation::kGREATER},
        {"and", nv::ElementWiseOperation::kAND},    {"or", nv::ElementWiseOperation::kOR},
        {"xor", nv::ElementWiseOperation::kXOR},    {"ne", nv::ElementWiseOperation::kEQUAL},
        {"le", nv::ElementWiseOperation::kGREATER}, {"ge", nv::ElementWiseOperation::kLESS},
    };
    return map;
}

const std::unordered_map<std::string_view, nv::ReduceOperation>& reductions() {
    static const std::unordered_map<std::string_view, nv::ReduceOperation> map = {
        {"reduce_sum", nv::ReduceOperation::kSUM},   {"reduce_mean", nv::ReduceOperation::kAVG},
        {"reduce_max", nv::ReduceOperation::kMAX},   {"reduce_min", nv::ReduceOperation::kMIN},
        {"reduce_prod", nv::ReduceOperation::kPROD},
    };
    return map;
}

/// Composite operations translated through their decomposition.
bool decomposed(std::string_view op) {
    return op == "rms_norm" || op == "log_softmax" || op == "sdpa";
}

int normalize_axis(std::int64_t axis, int rank) {
    return static_cast<int>(axis < 0 ? axis + rank : axis);
}

Status unsupported(const ir::Node& node, std::string_view why) {
    return errors::unsupported("'{}' {}", node.op_name(), why);
}

bool all_constant(const Dims& dims) {
    return std::all_of(dims.begin(), dims.end(), [](const Dim& d) { return d.is_constant(); });
}

// --- Support checks ---------------------------------------------------------------------------------------------

Status supports_impl(const ir::Node& node, int depth);

Status check_program(const ir::Graph& graph, int depth) {
    for (const auto& inner : graph.nodes()) {
        II_TRY(supports_impl(*inner, depth + 1));
    }
    return ok_status();
}

/// Decomposes a copy of the node and checks the result.
Status supports_decomposed(const ir::Node& node, int depth) {
    ir::Graph graph;
    std::vector<ir::Value*> inputs;
    for (const ir::Value* value : node.inputs()) {
        if (value == nullptr) {
            inputs.push_back(nullptr);
        } else if (value->kind() == ir::ValueKind::kConstant) {
            inputs.push_back(graph.add_constant(value->data()));
        } else {
            inputs.push_back(graph.add_input(std::format("x{}", inputs.size()), value->type()));
        }
    }
    std::vector<ir::TensorType> types;
    for (const ir::Value* output : node.outputs()) {
        types.push_back(output->type());
    }
    ir::Node* copy = graph.create_node(node.op(), std::move(inputs), node.attrs(), types);
    for (int o = 0; o < copy->num_outputs(); ++o) {
        graph.add_output(std::format("y{}", o), copy->output(o));
    }
    II_TRY(compiler::decompose_node(graph, *copy));
    return check_program(graph, depth);
}

Status check_types(const ir::Node& node) {
    auto check = [&](const ir::Value* value) -> Status {
        if (value == nullptr) {
            return ok_status();
        }
        if (!trt_type(value->dtype())) {
            return unsupported(node, std::format("with {}", to_string(value->dtype())));
        }
        if (value->rank() > kMaxRank) {
            return unsupported(node, std::format("of rank {}", value->rank()));
        }
        return ok_status();
    };
    for (const ir::Value* value : node.inputs()) {
        II_TRY(check(value));
    }
    for (const ir::Value* value : node.outputs()) {
        II_TRY(check(value));
    }
    return ok_status();
}

Status supports_impl(const ir::Node& node, int depth) {
    if (depth > 8) {
        return unsupported(node, "nests too deeply");
    }
    const std::string_view op = node.op_name();
    if (op == ops::kSubgraph) {
        // With the node's constants as constants of the program (weights of convolutions must be constants).
        II_ASSIGN_OR_RETURN(const ops::SubgraphProgram program, ops::subgraph_program(node));
        return check_program(*program.graph, depth);
    }
    II_TRY(check_types(node));
    const ir::Value* x = node.input(0);
    const DType dtype = x != nullptr ? x->dtype() : node.output()->dtype();
    if (op == ops::kFusedElementwise) {
        II_ASSIGN_OR_RETURN(const std::unique_ptr<ir::Graph> program, ops::parse_fused_program(node));
        return check_program(*program, depth);
    }
    if (op == ops::kFusedConv || op == "conv" || op == "conv_transpose") {
        if (op == "conv_transpose" && node.input(1)->kind() != ir::ValueKind::kConstant) {
            return unsupported(node, "with weights computed by the model");
        }
        if (!node.input(1)->type().is_static()) {
            return unsupported(node, "with weights of a symbolic shape");
        }
        if (x->rank() < 3 || x->rank() > 5 || !is_float(dtype) || node.input(1)->dtype() != dtype) {
            return unsupported(node, std::format("on {}", x->type().to_string()));
        }
        if (op == "conv_transpose") {
            const std::vector<std::int64_t> output_padding = node.attrs().get_ints("output_padding");
            if (std::any_of(output_padding.begin(), output_padding.end(), [](std::int64_t p) { return p != 0; }) ||
                node.attrs().get_string("auto_pad", "none") != "none" || x->rank() == 3) {
                return unsupported(node, "with output padding, automatic padding or one spatial dimension");
            }
        }
        if (op == ops::kFusedConv) {
            II_ASSIGN_OR_RETURN(const std::unique_ptr<ir::Graph> program, ops::parse_epilogue_program(node));
            return check_program(*program, depth);
        }
        return ok_status();
    }
    if (decomposed(op)) {
        if (op == "sdpa" && ((node.input(4) != nullptr) || node.attrs().get_int("window") != 0 ||
                             node.attrs().get_float("softcap") != 0)) {
            return unsupported(node, "with a KV cache length, a window or a soft cap (language models run on ii-cuda)");
        }
        return supports_decomposed(node, depth);
    }
    if (const auto it = unary_ops().find(op); it != unary_ops().end()) {
        return is_float(dtype) ? ok_status() : unsupported(node, "on integers");
    }
    if (activations().contains(op) || op == "clamp" || op == "gelu" || op == "silu" || op == "softmax") {
        if (op == "gelu" && node.attrs().get_string("approximate", "none") != "none" &&
            node.attrs().get_string("approximate") != "tanh") {
            return unsupported(node, "with this approximation");
        }
        return is_float(dtype) ? ok_status() : unsupported(node, "on integers");
    }
    if (op == "not") {
        return ok_status();
    }
    if (const auto it = binary_ops().find(op); it != binary_ops().end()) {
        if (op == "and" || op == "or" || op == "xor") {
            return ok_status();
        }
        if ((op == "div" || op == "pow") && !is_float(dtype)) {
            return unsupported(node, "on integers");
        }
        return is_float(dtype) || is_index(dtype) ? ok_status() : unsupported(node, "on this type");
    }
    if (op == "where" || op == "cast" || op == "reshape" || op == "permute" || op == "concat" || op == "tile" ||
        op == "cumsum" || op == "split") {
        return ok_status();
    }
    if (reductions().contains(op)) {
        return is_float(dtype) ? ok_status() : unsupported(node, "on integers");
    }
    if (op == "argmax" || op == "argmin" || op == "topk") {
        if (!is_float(dtype) || node.attrs().get_bool("select_last_index")) {
            return unsupported(node, "on integers or with select_last_index");
        }
        if (op == "topk" && node.attrs().get_int("k") > kMaxTopK) {
            return unsupported(node, std::format("with k > {}", kMaxTopK));
        }
        return ok_status();
    }
    if (op == "matmul" || op == "linear") {
        if (!is_float(dtype) || node.input(1)->dtype() != dtype || (op == "linear" && node.input(1)->rank() != 2)) {
            return unsupported(node,
                               std::format("of {} and {}", x->type().to_string(), node.input(1)->type().to_string()));
        }
        return ok_status();
    }
    if (op == "max_pool" || op == "avg_pool") {
        const std::vector<std::int64_t> dilations = node.attrs().get_ints("dilations");
        if (std::any_of(dilations.begin(), dilations.end(), [](std::int64_t d) { return d != 1; }) || x->rank() < 4 ||
            x->rank() > 5 || !is_float(dtype)) {
            return unsupported(node, "with dilations or not 2-d / 3-d");
        }
        if (op == "avg_pool" && node.attrs().get_bool("ceil_mode") && node.attrs().get_bool("count_include_pad")) {
            return unsupported(node, "with ceil_mode and count_include_pad");
        }
        return ok_status();
    }
    if (op == "resize") {
        const std::string_view mode = node.attrs().get_string("mode", "nearest");
        const std::string_view coordinates = node.attrs().get_string("coordinate_mode", "half_pixel");
        const std::string_view nearest = node.attrs().get_string("nearest_mode", "round_prefer_floor");
        if ((mode != "nearest" && mode != "linear" && mode != "cubic") ||
            (coordinates != "half_pixel" && coordinates != "align_corners" && coordinates != "asymmetric") ||
            (nearest != "round_prefer_floor" && nearest != "round_prefer_ceil" && nearest != "floor" &&
             nearest != "ceil") ||
            !is_float(dtype)) {
            return unsupported(node, std::format("with mode {}, coordinate_mode {}", mode, coordinates));
        }
        return ok_status();
    }
    if (op == "slice") {
        const std::vector<std::int64_t> axes = node.attrs().get_ints("axes");
        const std::vector<std::int64_t> starts = node.attrs().get_ints("starts");
        const std::vector<std::int64_t> steps = node.attrs().get_ints("steps");
        for (std::size_t i = 0; i < axes.size(); ++i) {
            const int axis = normalize_axis(axes[i], x->rank());
            if (axis < 0 || axis >= x->rank()) {
                return unsupported(node, "with an invalid axis");
            }
            const std::int64_t step = i < steps.size() ? steps[i] : 1;
            if (!x->shape()[static_cast<std::size_t>(axis)].is_constant() && (step != 1 || starts[i] < 0)) {
                return unsupported(node, "of a symbolic dimension with a step or a negative start");
            }
        }
        return ok_status();
    }
    if (op == "expand") {
        const Dims& in = x->shape();
        const Dims& out = node.output()->shape();
        for (std::size_t i = 0; i < in.size(); ++i) {
            const Dim& a = in[in.size() - 1 - i];
            const Dim& b = out[out.size() - 1 - i];
            if (!(a == b) && !(a.is_constant() && a.value() == 1)) {
                return unsupported(node, "of a symbolic dimension");
            }
        }
        return ok_status();
    }
    if (op == "pad") {
        const std::string_view mode = node.attrs().get_string("mode", "constant");
        return mode == "constant" || mode == "reflect" || mode == "edge" ? ok_status()
                                                                         : unsupported(node, "in this mode");
    }
    if (op == "gather" || op == "scatter") {
        if (op == "scatter" && node.attrs().get_string("reduction", "none") != "none") {
            return unsupported(node, "with a reduction");
        }
        return is_index(node.input(1)->dtype()) ? ok_status() : unsupported(node, "with these indices");
    }
    if (op == "flip") {
        for (const std::int64_t axis : node.attrs().get_ints("axes")) {
            const int a = normalize_axis(axis, x->rank());
            if (a < 0 || a >= x->rank() || !x->shape()[static_cast<std::size_t>(a)].is_constant()) {
                return unsupported(node, "of a symbolic dimension");
            }
        }
        return ok_status();
    }
    if (op == "iota" || op == "fill") {
        return all_constant(node.output()->shape()) && node.output()->dtype() != DType::kU8
                   ? ok_status()
                   : unsupported(node, "of a symbolic shape");
    }
    if (op == "layer_norm") {
        const int rank = x->rank();
        const int axis = normalize_axis(node.attrs().get_int("axis", -1), rank);
        if (!is_float(dtype) || axis < 0 || axis >= rank) {
            return unsupported(node, "on this input");
        }
        if (node.input(1) == nullptr || node.input(2) == nullptr) {
            for (int d = axis; d < rank; ++d) {
                if (!x->shape()[static_cast<std::size_t>(d)].is_constant()) {
                    return unsupported(node, "without scale or bias over symbolic dimensions");
                }
            }
        }
        return ok_status();
    }
    return errors::unsupported("no TensorRT translation of '{}'", op);
}

// --- Translation -------------------------------------------------------------------------------------------------

class Translator {
public:
    Translator(nv::INetworkDefinition& network, NetworkStorage& storage) : net_(network), storage_(storage) {}

    Status run(const ir::Graph& program) {
        for (std::size_t j = 0; j < program.inputs().size(); ++j) {
            const ir::Value* input = program.inputs()[j];
            II_ASSIGN_OR_RETURN(const nv::DataType type, type_of(input->dtype()));
            nv::Dims dims{};
            dims.nbDims = input->rank();
            for (int d = 0; d < input->rank(); ++d) {
                const Dim& dim = input->shape()[static_cast<std::size_t>(d)];
                dims.d[d] = dim.is_constant() ? dim.value() : -1;
            }
            const std::string name = std::format("x{}", j);
            nv::ITensor* tensor = net_.addInput(name.c_str(), type, dims);
            if (tensor == nullptr) {
                return errors::internal("TensorRT rejected input {} ({})", name, input->type().to_string());
            }
            values_[input] = tensor;
            for (int d = 0; d < input->rank(); ++d) {
                const std::string_view symbol = input->shape()[static_cast<std::size_t>(d)].symbol_name();
                if (!symbol.empty() && !sources_.contains(std::string(symbol))) {
                    sources_.emplace(std::string(symbol), std::make_pair(tensor, d));
                }
            }
        }
        for (const auto& node : program.nodes()) {
            if (Status status = translate_node(*node); !status.ok()) {
                return status.add_context(std::format("translating '{}' to TensorRT", node->op_name()));
            }
        }
        std::unordered_set<nv::ITensor*> marked;
        for (std::size_t k = 0; k < program.outputs().size(); ++k) {
            II_ASSIGN_OR_RETURN(nv::ITensor * tensor, value(program.outputs()[k].value));
            if (tensor->isNetworkInput() || marked.contains(tensor) || constants_.contains(tensor)) {
                II_ASSIGN_OR_RETURN(tensor, out(net_.addIdentity(*tensor), "identity"));
            }
            marked.insert(tensor);
            const std::string name = std::format("y{}", k);
            tensor->setName(name.c_str());
            net_.markOutput(*tensor);
        }
        return ok_status();
    }

private:
    // --- Basics ---

    static Result<nv::DataType> type_of(DType dtype) {
        const std::optional<nv::DataType> type = trt_type(dtype);
        if (!type) {
            return errors::unsupported("TensorRT has no {} tensors here", to_string(dtype));
        }
        return *type;
    }

    template <class Layer>
    static Result<nv::ITensor*> out(Layer* layer, std::string_view what, int index = 0) {
        if (layer == nullptr || layer->getOutput(index) == nullptr) {
            return errors::internal("TensorRT could not add a {} layer", what);
        }
        return layer->getOutput(index);
    }

    static nv::Dims to_dims(std::span<const std::int64_t> values) {
        nv::Dims dims{};
        dims.nbDims = static_cast<int32_t>(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) {
            dims.d[i] = values[i];
        }
        return dims;
    }

    static int rank(nv::ITensor* tensor) { return tensor->getDimensions().nbDims; }

    Result<nv::ITensor*> value(const ir::Value* v) {
        if (const auto it = values_.find(v); it != values_.end()) {
            return it->second;
        }
        if (v->kind() != ir::ValueKind::kConstant) {
            return errors::internal("value {} has not been translated", v->id());
        }
        II_ASSIGN_OR_RETURN(nv::ITensor * tensor, constant(v->data()));
        values_[v] = tensor;
        return tensor;
    }

    Result<nv::ITensor*> input(const ir::Node& node, int index) { return value(node.input(index)); }

    void set(const ir::Node& node, nv::ITensor* tensor, int index = 0) { values_[node.output(index)] = tensor; }

    /// Host data the network refers to, kept until the engine is built.
    Result<const void*> keep(const Tensor& data) {
        if (data.is_contiguous() && data.device().is_host()) {
            storage_.tensors.push_back(data);
        } else {
            II_ASSIGN_OR_RETURN(Tensor copy, data.contiguous());
            storage_.tensors.push_back(std::move(copy));
        }
        return storage_.tensors.back().data();
    }

    Result<nv::Weights> weights(const Tensor& data) {
        II_ASSIGN_OR_RETURN(const nv::DataType type, type_of(data.dtype()));
        II_ASSIGN_OR_RETURN(const void* bytes, keep(data));
        return nv::Weights{type, bytes, data.numel()};
    }

    Result<nv::ITensor*> constant(const Tensor& data) {
        II_ASSIGN_OR_RETURN(const nv::Weights w, weights(data));
        const Shape& shape = data.shape();
        nv::Dims dims = to_dims(std::span<const std::int64_t>(shape.begin(), shape.end()));
        II_ASSIGN_OR_RETURN(nv::ITensor * tensor, out(net_.addConstant(dims, w), "constant"));
        constants_.insert(tensor);
        return tensor;
    }

    /// A constant from bytes made here.
    Result<nv::ITensor*> constant_bytes(nv::DataType type, std::vector<std::byte> bytes, std::int64_t count,
                                        const nv::Dims& dims) {
        storage_.bytes.push_back(std::move(bytes));
        II_ASSIGN_OR_RETURN(
            nv::ITensor * tensor,
            out(net_.addConstant(dims, nv::Weights{type, storage_.bytes.back().data(), count}), "constant"));
        constants_.insert(tensor);
        return tensor;
    }

    /// `value` converted to `dtype`, as bytes.
    static std::vector<std::byte> scalar_bytes(DType dtype, double value) {
        std::vector<std::byte> bytes(storage_size(dtype, 1));
        auto store = [&](auto v) { std::memcpy(bytes.data(), &v, sizeof(v)); };
        switch (dtype) {
        case DType::kF32:
            store(static_cast<float>(value));
            break;
        case DType::kF16:
            store(f16(static_cast<float>(value)).bits);
            break;
        case DType::kBF16:
            store(bf16(static_cast<float>(value)).bits);
            break;
        case DType::kI32:
            store(static_cast<std::int32_t>(value));
            break;
        case DType::kI64:
            store(static_cast<std::int64_t>(value));
            break;
        case DType::kBool:
            store(static_cast<std::uint8_t>(value != 0));
            break;
        case DType::kU8:
            store(static_cast<std::uint8_t>(value));
            break;
        default:
            break;
        }
        return bytes;
    }

    /// A tensor of `rank` dimensions of size 1 holding `value` (rank 0: a scalar).
    Result<nv::ITensor*> scalar(DType dtype, double value, int rank) {
        II_ASSIGN_OR_RETURN(const nv::DataType type, type_of(dtype));
        nv::Dims dims{};
        dims.nbDims = rank;
        for (int d = 0; d < rank; ++d) {
            dims.d[d] = 1;
        }
        return constant_bytes(type, scalar_bytes(dtype, value), 1, dims);
    }

    /// A 1-d i64 constant.
    Result<nv::ITensor*> ints(const std::vector<std::int64_t>& values) {
        std::vector<std::byte> bytes(values.size() * sizeof(std::int64_t));
        std::memcpy(bytes.data(), values.data(), bytes.size());
        const std::array<std::int64_t, 1> size{static_cast<std::int64_t>(values.size())};
        return constant_bytes(nv::DataType::kINT64, std::move(bytes), static_cast<std::int64_t>(values.size()),
                              to_dims(size));
    }

    /// `tensor` with dimensions of size 1 prepended up to `target` dimensions (numpy broadcasting).
    Result<nv::ITensor*> align_rank(nv::ITensor* tensor, int target) {
        const int r = rank(tensor);
        if (r >= target) {
            return tensor;
        }
        std::vector<std::int64_t> axes;
        for (int i = 0; i < target - r; ++i) {
            axes.push_back(i);
        }
        II_ASSIGN_OR_RETURN(nv::ITensor * list, ints(axes));
        return out(net_.addUnsqueeze(*tensor, *list), "unsqueeze");
    }

    Result<nv::ITensor*> squeeze(nv::ITensor* tensor, std::vector<std::int64_t> axes) {
        II_ASSIGN_OR_RETURN(nv::ITensor * list, ints(axes));
        return out(net_.addSqueeze(*tensor, *list), "squeeze");
    }

    Result<nv::ITensor*> binary(nv::ITensor* a, nv::ITensor* b, nv::ElementWiseOperation op) {
        const int r = std::max(rank(a), rank(b));
        II_ASSIGN_OR_RETURN(a, align_rank(a, r));
        II_ASSIGN_OR_RETURN(b, align_rank(b, r));
        return out(net_.addElementWise(*a, *b, op), "elementwise");
    }

    Result<nv::ITensor*> unary(nv::ITensor* a, nv::UnaryOperation op) { return out(net_.addUnary(*a, op), "unary"); }

    // --- Shapes computed in the network ---

    Result<nv::ITensor*> symbol(const std::string& name) {
        if (const auto it = symbol_tensors_.find(name); it != symbol_tensors_.end()) {
            return it->second;
        }
        const auto source = sources_.find(name);
        if (source == sources_.end()) {
            return errors::unsupported("the symbol {} is not a dimension of an input", name);
        }
        II_ASSIGN_OR_RETURN(nv::ITensor * shape, out(net_.addShape(*source->second.first), "shape"));
        const std::array<std::int64_t, 1> start{source->second.second};
        const std::array<std::int64_t, 1> one{1};
        II_ASSIGN_OR_RETURN(nv::ITensor * dim,
                            out(net_.addSlice(*shape, to_dims(start), to_dims(one), to_dims(one)), "slice"));
        symbol_tensors_.emplace(name, dim);
        return dim;
    }

    Result<nv::ITensor*> poly(const detail::DimPoly& polynomial) {
        nv::ITensor* sum = nullptr;
        for (const detail::DimTerm& term : polynomial.terms) {
            nv::ITensor* product = nullptr;
            if (term.coefficient != 1 || term.factors.empty()) {
                II_ASSIGN_OR_RETURN(product, ints({term.coefficient}));
            }
            for (const detail::DimFactor& factor : term.factors) {
                nv::ITensor* atom = nullptr;
                if (factor.atom->kind == detail::DimAtom::Kind::kSymbol) {
                    II_ASSIGN_OR_RETURN(atom, symbol(factor.atom->name));
                } else {
                    II_ASSIGN_OR_RETURN(nv::ITensor * numerator, poly(*factor.atom->numerator));
                    II_ASSIGN_OR_RETURN(nv::ITensor * divisor, ints({factor.atom->divisor}));
                    II_ASSIGN_OR_RETURN(atom, binary(numerator, divisor, nv::ElementWiseOperation::kFLOOR_DIV));
                }
                for (int p = 0; p < factor.power; ++p) {
                    if (product == nullptr) {
                        product = atom;
                    } else {
                        II_ASSIGN_OR_RETURN(product, binary(product, atom, nv::ElementWiseOperation::kPROD));
                    }
                }
            }
            if (sum == nullptr) {
                sum = product;
            } else {
                II_ASSIGN_OR_RETURN(sum, binary(sum, product, nv::ElementWiseOperation::kSUM));
            }
        }
        if (sum == nullptr) {
            return ints({0});
        }
        return sum;
    }

    /// A 1-element i64 tensor with the value of `dim`.
    Result<nv::ITensor*> dim_tensor(const Dim& dim) {
        if (dim.is_constant()) {
            return ints({dim.value()});
        }
        return poly(*dim.poly());
    }

    /// A 1-d i64 tensor with the values of `dims`.
    Result<nv::ITensor*> shape_tensor(const Dims& dims) {
        if (all_constant(dims)) {
            std::vector<std::int64_t> values;
            for (const Dim& d : dims) {
                values.push_back(d.value());
            }
            return ints(values);
        }
        std::vector<nv::ITensor*> parts;
        for (const Dim& d : dims) {
            II_ASSIGN_OR_RETURN(nv::ITensor * part, dim_tensor(d));
            parts.push_back(part);
        }
        nv::IConcatenationLayer* concat = net_.addConcatenation(parts.data(), static_cast<int32_t>(parts.size()));
        if (concat != nullptr) {
            concat->setAxis(0);
        }
        return out(concat, "concatenation");
    }

    static nv::Dims static_or_zero(const Dims& dims) {
        nv::Dims out{};
        out.nbDims = static_cast<int32_t>(dims.size());
        for (std::size_t i = 0; i < dims.size(); ++i) {
            out.d[i] = dims[i].is_constant() ? dims[i].value() : 0;
        }
        return out;
    }

    /// A slice layer with a static start and stride and the size `size` (computed in the network when symbolic).
    Result<nv::ISliceLayer*> slice(nv::ITensor* x, const nv::Dims& start, const Dims& size, const nv::Dims& stride) {
        nv::ISliceLayer* layer = net_.addSlice(*x, start, static_or_zero(size), stride);
        if (layer == nullptr) {
            return errors::internal("TensorRT could not add a slice layer");
        }
        if (!all_constant(size)) {
            II_ASSIGN_OR_RETURN(nv::ITensor * shape, shape_tensor(size));
            layer->setInput(2, *shape);
        }
        return layer;
    }

    // --- Operations ---

    Status translate_program(const ir::Graph& program, std::span<nv::ITensor* const> inputs,
                             std::vector<nv::ITensor*>& outputs) {
        for (std::size_t i = 0; i < program.inputs().size(); ++i) {
            values_[program.inputs()[i]] = inputs[i];
        }
        for (const auto& inner : program.nodes()) {
            II_TRY(translate_node(*inner));
        }
        outputs.clear();
        for (const ir::GraphOutput& output : program.outputs()) {
            II_ASSIGN_OR_RETURN(nv::ITensor * tensor, value(output.value));
            outputs.push_back(tensor);
        }
        return ok_status();
    }

    Status translate_node(const ir::Node& node) {
        const std::string_view op = node.op_name();
        if (const auto it = unary_ops().find(op); it != unary_ops().end()) {
            II_ASSIGN_OR_RETURN(nv::ITensor * x, input(node, 0));
            II_ASSIGN_OR_RETURN(nv::ITensor * y, unary(x, it->second));
            if (op == "rsqrt") {
                II_ASSIGN_OR_RETURN(y, unary(y, nv::UnaryOperation::kRECIP));
            }
            set(node, y);
            return ok_status();
        }
        if (const auto it = activations().find(op); it != activations().end()) {
            II_ASSIGN_OR_RETURN(nv::ITensor * x, input(node, 0));
            II_ASSIGN_OR_RETURN(nv::ITensor * y, out(net_.addActivation(*x, it->second), "activation"));
            set(node, y);
            return ok_status();
        }
        if (const auto it = binary_ops().find(op); it != binary_ops().end()) {
            II_ASSIGN_OR_RETURN(nv::ITensor * a, input(node, 0));
            II_ASSIGN_OR_RETURN(nv::ITensor * b, input(node, 1));
            II_ASSIGN_OR_RETURN(nv::ITensor * y, binary(a, b, it->second));
            if (op == "ne" || op == "le" || op == "ge") {  // not(eq), not(gt), not(lt) except for NaN: or with eq
                if (op == "ne") {
                    II_ASSIGN_OR_RETURN(y, unary(y, nv::UnaryOperation::kNOT));
                } else {
                    II_ASSIGN_OR_RETURN(nv::ITensor * strict, binary(a, b,
                                                                     op == "le" ? nv::ElementWiseOperation::kLESS
                                                                                : nv::ElementWiseOperation::kGREATER));
                    II_ASSIGN_OR_RETURN(nv::ITensor * equal, binary(a, b, nv::ElementWiseOperation::kEQUAL));
                    II_ASSIGN_OR_RETURN(y, binary(strict, equal, nv::ElementWiseOperation::kOR));
                }
            }
            set(node, y);
            return ok_status();
        }
        if (const auto it = reductions().find(op); it != reductions().end()) {
            return translate_reduce(node, it->second);
        }
        if (op == "not") {
            II_ASSIGN_OR_RETURN(nv::ITensor * x, input(node, 0));
            II_ASSIGN_OR_RETURN(nv::ITensor * y, unary(x, nv::UnaryOperation::kNOT));
            set(node, y);
            return ok_status();
        }
        if (op == "where") {
            II_ASSIGN_OR_RETURN(nv::ITensor * c, input(node, 0));
            II_ASSIGN_OR_RETURN(nv::ITensor * a, input(node, 1));
            II_ASSIGN_OR_RETURN(nv::ITensor * b, input(node, 2));
            const int r = std::max({rank(c), rank(a), rank(b)});
            II_ASSIGN_OR_RETURN(c, align_rank(c, r));
            II_ASSIGN_OR_RETURN(a, align_rank(a, r));
            II_ASSIGN_OR_RETURN(b, align_rank(b, r));
            II_ASSIGN_OR_RETURN(nv::ITensor * y, out(net_.addSelect(*c, *a, *b), "select"));
            set(node, y);
            return ok_status();
        }
        if (op == "cast") {
            II_ASSIGN_OR_RETURN(nv::ITensor * x, input(node, 0));
            if (node.input(0)->dtype() == node.output()->dtype()) {
                set(node, x);
                return ok_status();
            }
            II_ASSIGN_OR_RETURN(const nv::DataType to, type_of(node.output()->dtype()));
            II_ASSIGN_OR_RETURN(nv::ITensor * y, out(net_.addCast(*x, to), "cast"));
            set(node, y);
            return ok_status();
        }
        if (op == "clamp") {
            II_ASSIGN_OR_RETURN(nv::ITensor * y, input(node, 0));
            const double low = node.attrs().get_float("min", -std::numeric_limits<double>::infinity());
            const double high = node.attrs().get_float("max", std::numeric_limits<double>::infinity());
            const DType dtype = node.output()->dtype();
            if (std::isfinite(low)) {
                II_ASSIGN_OR_RETURN(nv::ITensor * bound, scalar(dtype, low, rank(y)));
                II_ASSIGN_OR_RETURN(y, binary(y, bound, nv::ElementWiseOperation::kMAX));
            }
            if (std::isfinite(high)) {
                II_ASSIGN_OR_RETURN(nv::ITensor * bound, scalar(dtype, high, rank(y)));
                II_ASSIGN_OR_RETURN(y, binary(y, bound, nv::ElementWiseOperation::kMIN));
            }
            set(node, y);
            return ok_status();
        }
        if (op == "gelu") {
            II_ASSIGN_OR_RETURN(nv::ITensor * x, input(node, 0));
            const bool tanh = node.attrs().get_string("approximate", "none") == "tanh";
            II_ASSIGN_OR_RETURN(nv::ITensor * y, out(net_.addActivation(*x, tanh ? nv::ActivationType::kGELU_TANH
                                                                                 : nv::ActivationType::kGELU_ERF),
                                                     "activation"));
            set(node, y);
            return ok_status();
        }
        if (op == "silu") {
            II_ASSIGN_OR_RETURN(nv::ITensor * x, input(node, 0));
            II_ASSIGN_OR_RETURN(nv::ITensor * s, out(net_.addActivation(*x, nv::ActivationType::kSIGMOID), "sigmoid"));
            II_ASSIGN_OR_RETURN(nv::ITensor * y, binary(x, s, nv::ElementWiseOperation::kPROD));
            set(node, y);
            return ok_status();
        }
        if (op == "softmax") {
            II_ASSIGN_OR_RETURN(nv::ITensor * x, input(node, 0));
            const int r = rank(x);
            int axis = normalize_axis(node.attrs().get_int("axis", -1), r);
            if (r < 2) {  // TensorRT's softmax needs two dimensions
                II_ASSIGN_OR_RETURN(x, align_rank(x, 2));
                axis += 2 - r;
            }
            nv::ISoftMaxLayer* layer = net_.addSoftMax(*x);
            if (layer != nullptr) {
                layer->setAxes(1u << static_cast<unsigned>(axis));
            }
            II_ASSIGN_OR_RETURN(nv::ITensor * y, out(layer, "softmax"));
            if (r < 2) {
                II_ASSIGN_OR_RETURN(
                    y, squeeze(y, r == 1 ? std::vector<std::int64_t>{0} : std::vector<std::int64_t>{0, 1}));
            }
            set(node, y);
            return ok_status();
        }
        if (op == "layer_norm") {
            return translate_layer_norm(node);
        }
        if (op == "argmax" || op == "argmin" || op == "topk") {
            return translate_topk(node);
        }
        if (op == "matmul" || op == "linear") {
            return translate_matmul(node);
        }
        if (op == "conv" || op == ops::kFusedConv) {
            return translate_conv(node);
        }
        if (op == "conv_transpose") {
            return translate_deconv(node);
        }
        if (op == "max_pool" || op == "avg_pool") {
            return translate_pool(node);
        }
        if (op == "resize") {
            return translate_resize(node);
        }
        if (op == "reshape") {
            II_ASSIGN_OR_RETURN(nv::ITensor * x, input(node, 0));
            nv::IShuffleLayer* layer = net_.addShuffle(*x);
            if (layer == nullptr) {
                return errors::internal("TensorRT could not add a shuffle layer");
            }
            layer->setZeroIsPlaceholder(false);
            const Dims& shape = node.output()->shape();
            if (all_constant(shape)) {
                layer->setReshapeDimensions(static_or_zero(shape));
            } else {
                II_ASSIGN_OR_RETURN(nv::ITensor * target, shape_tensor(shape));
                layer->setInput(1, *target);
            }
            II_ASSIGN_OR_RETURN(nv::ITensor * y, out(layer, "reshape"));
            set(node, y);
            return ok_status();
        }
        if (op == "permute") {
            II_ASSIGN_OR_RETURN(nv::ITensor * x, input(node, 0));
            const std::vector<std::int64_t> perm = node.attrs().get_ints("perm");
            nv::Permutation permutation{};
            for (std::size_t i = 0; i < perm.size(); ++i) {
                permutation.order[i] = static_cast<int32_t>(perm[i]);
            }
            nv::IShuffleLayer* layer = net_.addShuffle(*x);
            if (layer != nullptr) {
                layer->setFirstTranspose(permutation);
            }
            II_ASSIGN_OR_RETURN(nv::ITensor * y, out(layer, "transpose"));
            set(node, y);
            return ok_status();
        }
        if (op == "slice") {
            return translate_slice(node);
        }
        if (op == "expand" || op == "tile") {
            II_ASSIGN_OR_RETURN(nv::ITensor * x, input(node, 0));
            const Dims& shape = node.output()->shape();
            const int r = static_cast<int>(shape.size());
            II_ASSIGN_OR_RETURN(x, align_rank(x, r));
            const Dims in = node.input(0)->shape();
            nv::Dims start{};
            nv::Dims stride{};
            start.nbDims = stride.nbDims = r;
            for (int d = 0; d < r; ++d) {
                const int from = d - (r - static_cast<int>(in.size()));
                const bool one = from < 0 || (in[static_cast<std::size_t>(from)].is_constant() &&
                                              in[static_cast<std::size_t>(from)].value() == 1);
                start.d[d] = 0;
                stride.d[d] = op == "expand" && one ? 0 : 1;
            }
            II_ASSIGN_OR_RETURN(nv::ISliceLayer * layer, slice(x, start, shape, stride));
            if (op == "tile") {
                layer->setMode(nv::SampleMode::kWRAP);
            }
            II_ASSIGN_OR_RETURN(nv::ITensor * y, out(layer, op));
            set(node, y);
            return ok_status();
        }
        if (op == "concat") {
            std::vector<nv::ITensor*> parts;
            for (int i = 0; i < node.num_inputs(); ++i) {
                II_ASSIGN_OR_RETURN(nv::ITensor * part, input(node, i));
                parts.push_back(part);
            }
            nv::IConcatenationLayer* layer = net_.addConcatenation(parts.data(), static_cast<int32_t>(parts.size()));
            if (layer != nullptr) {
                layer->setAxis(normalize_axis(node.attrs().get_int("axis"), node.output()->rank()));
            }
            II_ASSIGN_OR_RETURN(nv::ITensor * y, out(layer, "concatenation"));
            set(node, y);
            return ok_status();
        }
        if (op == "split") {
            II_ASSIGN_OR_RETURN(nv::ITensor * x, input(node, 0));
            const int r = rank(x);
            const int axis = normalize_axis(node.attrs().get_int("axis"), r);
            const std::vector<std::int64_t> sizes = node.attrs().get_ints("sizes");
            std::int64_t offset = 0;
            for (int o = 0; o < node.num_outputs(); ++o) {
                nv::Dims start{};
                nv::Dims stride{};
                start.nbDims = stride.nbDims = r;
                for (int d = 0; d < r; ++d) {
                    start.d[d] = d == axis ? offset : 0;
                    stride.d[d] = 1;
                }
                II_ASSIGN_OR_RETURN(nv::ISliceLayer * layer, slice(x, start, node.output(o)->shape(), stride));
                II_ASSIGN_OR_RETURN(nv::ITensor * y, out(layer, "slice"));
                set(node, y, o);
                offset += sizes[static_cast<std::size_t>(o)];
            }
            return ok_status();
        }
        if (op == "pad") {
            return translate_pad(node);
        }
        if (op == "flip") {
            II_ASSIGN_OR_RETURN(nv::ITensor * x, input(node, 0));
            const int r = rank(x);
            const Dims& shape = node.input(0)->shape();
            nv::Dims start{};
            nv::Dims stride{};
            start.nbDims = stride.nbDims = r;
            for (int d = 0; d < r; ++d) {
                start.d[d] = 0;
                stride.d[d] = 1;
            }
            for (const std::int64_t axis : node.attrs().get_ints("axes")) {
                const int a = normalize_axis(axis, r);
                start.d[a] = shape[static_cast<std::size_t>(a)].value() - 1;
                stride.d[a] = -1;
            }
            II_ASSIGN_OR_RETURN(nv::ISliceLayer * layer, slice(x, start, shape, stride));
            II_ASSIGN_OR_RETURN(nv::ITensor * y, out(layer, "flip"));
            set(node, y);
            return ok_status();
        }
        if (op == "gather" || op == "scatter") {
            return translate_gather(node);
        }
        if (op == "cumsum") {
            II_ASSIGN_OR_RETURN(nv::ITensor * x, input(node, 0));
            std::vector<std::byte> bytes(sizeof(std::int32_t));
            const auto axis = static_cast<std::int32_t>(normalize_axis(node.attrs().get_int("axis"), rank(x)));
            std::memcpy(bytes.data(), &axis, sizeof(axis));
            II_ASSIGN_OR_RETURN(nv::ITensor * axis_tensor,
                                constant_bytes(nv::DataType::kINT32, std::move(bytes), 1, nv::Dims{}));
            II_ASSIGN_OR_RETURN(nv::ITensor * y, out(net_.addCumulative(*x, *axis_tensor, nv::CumulativeOperation::kSUM,
                                                                        node.attrs().get_bool("exclusive"),
                                                                        node.attrs().get_bool("reverse")),
                                                     "cumulative"));
            set(node, y);
            return ok_status();
        }
        if (op == "iota" || op == "fill") {
            return translate_creation(node);
        }
        if (op == ops::kFusedElementwise) {
            II_ASSIGN_OR_RETURN(std::unique_ptr<ir::Graph> program, ops::parse_fused_program(node));
            std::vector<nv::ITensor*> inputs;
            for (int i = 0; i < node.num_inputs(); ++i) {
                II_ASSIGN_OR_RETURN(nv::ITensor * t, input(node, i));
                inputs.push_back(t);
            }
            std::vector<nv::ITensor*> outputs;
            II_TRY(translate_program(*program, inputs, outputs));
            set(node, outputs.front());
            programs_.push_back(std::shared_ptr<const ir::Graph>(program.release()));  // values_ refers to its values
            return ok_status();
        }
        return errors::unsupported("no TensorRT translation of '{}'", op);
    }

    Status translate_reduce(const ir::Node& node, nv::ReduceOperation operation) {
        II_ASSIGN_OR_RETURN(nv::ITensor * x, input(node, 0));
        const int r = rank(x);
        std::vector<std::int64_t> axes = node.attrs().get_ints("axes");
        std::uint32_t mask = 0;
        if (axes.empty()) {
            mask = r == 0 ? 0 : (1u << static_cast<unsigned>(r)) - 1;
        }
        for (const std::int64_t axis : axes) {
            mask |= 1u << static_cast<unsigned>(normalize_axis(axis, r));
        }
        if (mask == 0) {
            set(node, x);
            return ok_status();
        }
        // A reduction over several axes runs as two layers, the last of its axes first: TensorRT 10.16 fuses a
        // softmax followed by a reduction of all elements of a small tensor into a kernel that computes both wrong
        // (reproduced with trtexec on a two-node ONNX model, [4, 4]).
        const unsigned last = 31u - static_cast<unsigned>(std::countl_zero(mask));
        if ((mask & ~(1u << last)) != 0) {
            II_ASSIGN_OR_RETURN(x, out(net_.addReduce(*x, operation, 1u << last, true), "reduce"));
        }
        II_ASSIGN_OR_RETURN(nv::ITensor * y,
                            out(net_.addReduce(*x, operation, mask, node.attrs().get_bool("keepdims")), "reduce"));
        set(node, y);
        return ok_status();
    }

    Status translate_topk(const ir::Node& node) {
        II_ASSIGN_OR_RETURN(nv::ITensor * x, input(node, 0));
        const std::string_view op = node.op_name();
        const int r = rank(x);
        const int axis = normalize_axis(node.attrs().get_int("axis", op == "topk" ? -1 : 0), r);
        const bool largest = op == "argmax" || (op == "topk" && node.attrs().get_bool("largest", true));
        const auto k = static_cast<int32_t>(op == "topk" ? node.attrs().get_int("k") : 1);
        nv::ITopKLayer* layer = net_.addTopK(*x, largest ? nv::TopKOperation::kMAX : nv::TopKOperation::kMIN, k,
                                             1u << static_cast<unsigned>(axis), nv::DataType::kINT64);
        if (op == "topk") {
            II_ASSIGN_OR_RETURN(nv::ITensor * values, out(layer, "topk", 0));
            II_ASSIGN_OR_RETURN(nv::ITensor * indices, out(layer, "topk", 1));
            set(node, values, 0);
            set(node, indices, 1);
            return ok_status();
        }
        II_ASSIGN_OR_RETURN(nv::ITensor * indices, out(layer, "topk", 1));
        if (!node.attrs().get_bool("keepdims")) {
            II_ASSIGN_OR_RETURN(indices, squeeze(indices, {axis}));
        }
        set(node, indices);
        return ok_status();
    }

    Status translate_matmul(const ir::Node& node) {
        II_ASSIGN_OR_RETURN(nv::ITensor * a, input(node, 0));
        II_ASSIGN_OR_RETURN(nv::ITensor * b, input(node, 1));
        const bool linear = node.op_name() == "linear";
        const bool transpose_a = !linear && node.attrs().get_bool("transpose_a");
        const bool transpose_b =
            linear ? node.attrs().get_bool("transpose_w", true) : node.attrs().get_bool("transpose_b");
        // numpy semantics: a vector operand becomes a matrix ([1, K] or [K, 1]) whose added dimension is removed
        // from the result; batch dimensions broadcast.
        const bool vector_a = rank(a) == 1;
        const bool vector_b = rank(b) == 1;
        if (vector_a) {
            II_ASSIGN_OR_RETURN(nv::ITensor * axes, ints({0}));
            II_ASSIGN_OR_RETURN(a, out(net_.addUnsqueeze(*a, *axes), "unsqueeze"));
        }
        if (vector_b) {
            II_ASSIGN_OR_RETURN(nv::ITensor * axes, ints({1}));
            II_ASSIGN_OR_RETURN(b, out(net_.addUnsqueeze(*b, *axes), "unsqueeze"));
        }
        const int r = std::max(rank(a), rank(b));
        II_ASSIGN_OR_RETURN(a, align_rank(a, r));
        II_ASSIGN_OR_RETURN(b, align_rank(b, r));
        auto op = [](bool transpose) {
            return transpose ? nv::MatrixOperation::kTRANSPOSE : nv::MatrixOperation::kNONE;
        };
        II_ASSIGN_OR_RETURN(nv::ITensor * y, out(net_.addMatrixMultiply(*a, op(transpose_a && !vector_a), *b,
                                                                        op(transpose_b && !vector_b)),
                                                 "matrix multiply"));
        if (vector_a || vector_b) {
            std::vector<std::int64_t> axes;
            if (vector_a) {
                axes.push_back(r - 2);
            }
            if (vector_b) {
                axes.push_back(r - 1);
            }
            II_ASSIGN_OR_RETURN(y, squeeze(y, axes));
        }
        if (linear && node.input(2) != nullptr) {
            II_ASSIGN_OR_RETURN(nv::ITensor * bias, input(node, 2));
            II_ASSIGN_OR_RETURN(y, binary(y, bias, nv::ElementWiseOperation::kSUM));
        }
        set(node, y);
        return ok_status();
    }

    struct Window {
        nv::Dims strides{};
        nv::Dims dilations{};
        nv::Dims pre{};
        nv::Dims post{};
        std::optional<nv::PaddingMode> mode;
    };

    static Window window(const Attributes& attrs, int spatial) {
        Window w;
        const std::vector<std::int64_t> strides = attrs.get_ints("strides");
        const std::vector<std::int64_t> dilations = attrs.get_ints("dilations");
        const std::vector<std::int64_t> pads = attrs.get_ints("pads");
        w.strides.nbDims = w.dilations.nbDims = w.pre.nbDims = w.post.nbDims = spatial;
        for (int i = 0; i < spatial; ++i) {
            const auto s = static_cast<std::size_t>(i);
            w.strides.d[i] = s < strides.size() ? strides[s] : 1;
            w.dilations.d[i] = s < dilations.size() ? dilations[s] : 1;
            w.pre.d[i] = s < pads.size() ? pads[s] : 0;
            w.post.d[i] =
                s + static_cast<std::size_t>(spatial) < pads.size() ? pads[s + static_cast<std::size_t>(spatial)] : 0;
        }
        const std::string_view auto_pad = attrs.get_string("auto_pad", "none");
        if (auto_pad == "same_upper") {
            w.mode = nv::PaddingMode::kSAME_UPPER;
        } else if (auto_pad == "same_lower") {
            w.mode = nv::PaddingMode::kSAME_LOWER;
        } else if (auto_pad == "valid") {
            for (int i = 0; i < spatial; ++i) {
                w.pre.d[i] = w.post.d[i] = 0;
            }
        }
        return w;
    }

    Status translate_conv(const ir::Node& node) {
        II_ASSIGN_OR_RETURN(nv::ITensor * x, input(node, 0));
        const ir::Value* w_value = node.input(1);
        const bool constant_weights = w_value->kind() == ir::ValueKind::kConstant;
        Shape w_shape;
        for (const Dim& d : w_value->shape()) {
            w_shape.push_back(d.value());
        }
        const int r = rank(x);
        const bool one_d = r == 3;
        if (one_d) {  // [N, C, L] as [N, C, L, 1]
            II_ASSIGN_OR_RETURN(nv::ITensor * axes, ints({3}));
            II_ASSIGN_OR_RETURN(x, out(net_.addUnsqueeze(*x, *axes), "unsqueeze"));
        }
        const int spatial = rank(x) - 2;
        nv::Dims kernel{};
        kernel.nbDims = spatial;
        for (int i = 0; i < spatial; ++i) {
            kernel.d[i] = i + 2 < w_shape.rank() ? w_shape[i + 2] : 1;
        }
        Attributes attrs = node.op_name() == ops::kFusedConv ? ops::conv_attributes(node.attrs()) : node.attrs();
        if (one_d) {  // the extra dimension: stride 1, no padding
            auto extend = [&](std::string_view name, std::int64_t fill, bool pads) {
                std::vector<std::int64_t> values = attrs.get_ints(name);
                if (values.empty()) {
                    return;
                }
                if (pads) {
                    values = {values[0], 0, values.size() > 1 ? values[1] : 0, 0};
                } else {
                    values.push_back(fill);
                }
                attrs.set(name, values);
            };
            extend("strides", 1, false);
            extend("dilations", 1, false);
            extend("pads", 0, true);
        }
        const Window win = window(attrs, spatial);
        II_ASSIGN_OR_RETURN(const nv::DataType w_type, type_of(w_value->dtype()));
        nv::Weights kernel_weights{w_type, nullptr, 0};
        if (constant_weights) {
            II_ASSIGN_OR_RETURN(kernel_weights, weights(w_value->data()));
        }
        nv::Weights bias_weights{w_type, nullptr, 0};
        const ir::Value* bias = node.input(2);
        if (bias != nullptr && bias->kind() == ir::ValueKind::kConstant) {
            II_ASSIGN_OR_RETURN(bias_weights, weights(bias->data()));
        }
        nv::IConvolutionLayer* layer = net_.addConvolutionNd(*x, w_shape[0], kernel, kernel_weights, bias_weights);
        if (layer == nullptr) {
            return errors::internal("TensorRT could not add a convolution layer");
        }
        if (!constant_weights) {  // weights computed by the model: an input of the layer
            II_ASSIGN_OR_RETURN(nv::ITensor * w, value(w_value));
            if (one_d) {
                II_ASSIGN_OR_RETURN(nv::ITensor * axes, ints({3}));
                II_ASSIGN_OR_RETURN(w, out(net_.addUnsqueeze(*w, *axes), "unsqueeze"));
            }
            layer->setInput(1, *w);
        }
        layer->setStrideNd(win.strides);
        layer->setDilationNd(win.dilations);
        layer->setPrePadding(win.pre);
        layer->setPostPadding(win.post);
        if (win.mode) {
            layer->setPaddingMode(*win.mode);
        }
        layer->setNbGroups(attrs.get_int("groups", 1));
        II_ASSIGN_OR_RETURN(nv::ITensor * y, out(layer, "convolution"));
        if (bias != nullptr && bias->kind() != ir::ValueKind::kConstant) {
            II_ASSIGN_OR_RETURN(nv::ITensor * b, value(bias));
            // bias [M] -> [M, 1, 1(, 1)] broadcast over the spatial dimensions
            std::vector<std::int64_t> spatial_axes;
            for (int i = 0; i < spatial; ++i) {
                spatial_axes.push_back(i + 1);
            }
            II_ASSIGN_OR_RETURN(nv::ITensor * list, ints(spatial_axes));
            II_ASSIGN_OR_RETURN(b, out(net_.addUnsqueeze(*b, *list), "unsqueeze"));
            II_ASSIGN_OR_RETURN(y, binary(y, b, nv::ElementWiseOperation::kSUM));
        }
        if (one_d) {
            II_ASSIGN_OR_RETURN(y, squeeze(y, {3}));
        }
        if (node.op_name() == ops::kFusedConv) {
            II_ASSIGN_OR_RETURN(std::unique_ptr<ir::Graph> program, ops::parse_epilogue_program(node));
            std::vector<nv::ITensor*> inputs = {y};
            for (int i = 3; i < node.num_inputs(); ++i) {
                II_ASSIGN_OR_RETURN(nv::ITensor * extra, input(node, i));
                inputs.push_back(extra);
            }
            std::vector<nv::ITensor*> outputs;
            II_TRY(translate_program(*program, inputs, outputs));
            y = outputs.front();
            programs_.push_back(std::shared_ptr<const ir::Graph>(program.release()));
        }
        set(node, y);
        return ok_status();
    }

    Status translate_deconv(const ir::Node& node) {
        II_ASSIGN_OR_RETURN(nv::ITensor * x, input(node, 0));
        const Tensor& w = node.input(1)->data();  // [C, M / groups, kernel...]
        const int spatial = rank(x) - 2;
        nv::Dims kernel{};
        kernel.nbDims = spatial;
        for (int i = 0; i < spatial; ++i) {
            kernel.d[i] = w.shape()[i + 2];
        }
        const Window win = window(node.attrs(), spatial);
        const std::int64_t groups = node.attrs().get_int("groups", 1);
        II_ASSIGN_OR_RETURN(const nv::Weights kernel_weights, weights(w));
        nv::Weights bias_weights{kernel_weights.type, nullptr, 0};
        const ir::Value* bias = node.input(2);
        if (bias != nullptr && bias->kind() == ir::ValueKind::kConstant) {
            II_ASSIGN_OR_RETURN(bias_weights, weights(bias->data()));
        }
        nv::IDeconvolutionLayer* layer =
            net_.addDeconvolutionNd(*x, w.shape()[1] * groups, kernel, kernel_weights, bias_weights);
        if (layer == nullptr) {
            return errors::internal("TensorRT could not add a deconvolution layer");
        }
        layer->setStrideNd(win.strides);
        layer->setDilationNd(win.dilations);
        layer->setPrePadding(win.pre);
        layer->setPostPadding(win.post);
        layer->setNbGroups(groups);
        II_ASSIGN_OR_RETURN(nv::ITensor * y, out(layer, "deconvolution"));
        if (bias != nullptr && bias->kind() != ir::ValueKind::kConstant) {
            II_ASSIGN_OR_RETURN(nv::ITensor * b, value(bias));
            std::vector<std::int64_t> spatial_axes;
            for (int i = 0; i < spatial; ++i) {
                spatial_axes.push_back(i + 1);
            }
            II_ASSIGN_OR_RETURN(nv::ITensor * list, ints(spatial_axes));
            II_ASSIGN_OR_RETURN(b, out(net_.addUnsqueeze(*b, *list), "unsqueeze"));
            II_ASSIGN_OR_RETURN(y, binary(y, b, nv::ElementWiseOperation::kSUM));
        }
        set(node, y);
        return ok_status();
    }

    Status translate_pool(const ir::Node& node) {
        II_ASSIGN_OR_RETURN(nv::ITensor * x, input(node, 0));
        const int spatial = rank(x) - 2;
        const std::vector<std::int64_t> kernel = node.attrs().get_ints("kernel");
        const bool max = node.op_name() == "max_pool";
        nv::IPoolingLayer* layer =
            net_.addPoolingNd(*x, max ? nv::PoolingType::kMAX : nv::PoolingType::kAVERAGE, to_dims(kernel));
        if (layer == nullptr) {
            return errors::internal("TensorRT could not add a pooling layer");
        }
        const Window win = window(node.attrs(), spatial);
        layer->setStrideNd(win.strides);
        layer->setPrePadding(win.pre);
        layer->setPostPadding(win.post);
        if (win.mode) {
            layer->setPaddingMode(*win.mode);
        } else if (node.attrs().get_bool("ceil_mode")) {
            layer->setPaddingMode(nv::PaddingMode::kEXPLICIT_ROUND_UP);
        }
        if (!max) {
            layer->setAverageCountExcludesPadding(!node.attrs().get_bool("count_include_pad"));
        }
        II_ASSIGN_OR_RETURN(nv::ITensor * y, out(layer, "pooling"));
        set(node, y);
        return ok_status();
    }

    Status translate_resize(const ir::Node& node) {
        II_ASSIGN_OR_RETURN(nv::ITensor * x, input(node, 0));
        nv::IResizeLayer* layer = net_.addResize(*x);
        if (layer == nullptr) {
            return errors::internal("TensorRT could not add a resize layer");
        }
        const Attributes& attrs = node.attrs();
        const std::vector<double> scales = attrs.get_floats("scales");
        const bool use_scales =
            !scales.empty() && std::all_of(scales.begin(), scales.end(), [](double s) { return s > 0; });
        if (use_scales) {
            std::vector<std::byte> bytes(scales.size() * sizeof(float));
            for (std::size_t i = 0; i < scales.size(); ++i) {
                const auto s = static_cast<float>(scales[i]);
                std::memcpy(bytes.data() + i * sizeof(float), &s, sizeof(float));
            }
            storage_.bytes.push_back(std::move(bytes));
            layer->setScales(reinterpret_cast<const float*>(storage_.bytes.back().data()),
                             static_cast<int32_t>(scales.size()));
        } else {
            const Dims& shape = node.output()->shape();
            if (all_constant(shape)) {
                layer->setOutputDimensions(static_or_zero(shape));
            } else {
                II_ASSIGN_OR_RETURN(nv::ITensor * target, shape_tensor(shape));
                layer->setInput(1, *target);
            }
        }
        const std::string_view mode = attrs.get_string("mode", "nearest");
        layer->setResizeMode(mode == "nearest"  ? nv::InterpolationMode::kNEAREST
                             : mode == "linear" ? nv::InterpolationMode::kLINEAR
                                                : nv::InterpolationMode::kCUBIC);
        const std::string_view coordinates = attrs.get_string("coordinate_mode", "half_pixel");
        layer->setCoordinateTransformation(
            coordinates == "align_corners" ? nv::ResizeCoordinateTransformation::kALIGN_CORNERS
            : coordinates == "asymmetric"  ? nv::ResizeCoordinateTransformation::kASYMMETRIC
                                           : nv::ResizeCoordinateTransformation::kHALF_PIXEL);
        layer->setSelectorForSinglePixel(nv::ResizeSelector::kFORMULA);
        const std::string_view nearest = attrs.get_string("nearest_mode", "round_prefer_floor");
        layer->setNearestRounding(nearest == "round_prefer_ceil" ? nv::ResizeRoundMode::kHALF_UP
                                  : nearest == "floor"           ? nv::ResizeRoundMode::kFLOOR
                                  : nearest == "ceil"            ? nv::ResizeRoundMode::kCEIL
                                                                 : nv::ResizeRoundMode::kHALF_DOWN);
        layer->setCubicCoeff(static_cast<float>(attrs.get_float("cubic_coeff_a", -0.75)));
        layer->setExcludeOutside(attrs.get_bool("exclude_outside"));
        II_ASSIGN_OR_RETURN(nv::ITensor * y, out(layer, "resize"));
        set(node, y);
        return ok_status();
    }

    Status translate_slice(const ir::Node& node) {
        II_ASSIGN_OR_RETURN(nv::ITensor * x, input(node, 0));
        const Dims& in = node.input(0)->shape();
        const int r = static_cast<int>(in.size());
        const std::vector<std::int64_t> axes = node.attrs().get_ints("axes");
        const std::vector<std::int64_t> starts = node.attrs().get_ints("starts");
        const std::vector<std::int64_t> steps = node.attrs().get_ints("steps");
        nv::Dims start{};
        nv::Dims stride{};
        start.nbDims = stride.nbDims = r;
        for (int d = 0; d < r; ++d) {
            start.d[d] = 0;
            stride.d[d] = 1;
        }
        for (std::size_t i = 0; i < axes.size(); ++i) {
            const int axis = normalize_axis(axes[i], r);
            const std::int64_t step = i < steps.size() ? steps[i] : 1;
            stride.d[axis] = step;
            const Dim& dim = in[static_cast<std::size_t>(axis)];
            if (!dim.is_constant()) {
                start.d[axis] = starts[i];  // checked: non-negative, step 1
                continue;
            }
            const std::int64_t size = dim.value();
            std::int64_t s = starts[i];
            if (s < 0) {
                s += size;
            }
            s = step > 0 ? std::clamp<std::int64_t>(s, 0, size) : std::clamp<std::int64_t>(s, 0, size - 1);
            start.d[axis] = s;  // the size comes from the output type
        }
        II_ASSIGN_OR_RETURN(nv::ISliceLayer * layer, slice(x, start, node.output()->shape(), stride));
        II_ASSIGN_OR_RETURN(nv::ITensor * y, out(layer, "slice"));
        set(node, y);
        return ok_status();
    }

    Status translate_pad(const ir::Node& node) {
        II_ASSIGN_OR_RETURN(nv::ITensor * x, input(node, 0));
        const int r = rank(x);
        const std::vector<std::int64_t> pads = node.attrs().get_ints("pads");
        nv::Dims start{};
        nv::Dims stride{};
        start.nbDims = stride.nbDims = r;
        for (int d = 0; d < r; ++d) {
            start.d[d] = -(static_cast<std::size_t>(d) < pads.size() ? pads[static_cast<std::size_t>(d)] : 0);
            stride.d[d] = 1;
        }
        II_ASSIGN_OR_RETURN(nv::ISliceLayer * layer, slice(x, start, node.output()->shape(), stride));
        const std::string_view mode = node.attrs().get_string("mode", "constant");
        if (mode == "constant") {
            layer->setMode(nv::SampleMode::kFILL);
            II_ASSIGN_OR_RETURN(nv::ITensor * fill, scalar(node.input(0)->dtype(), node.attrs().get_float("value"), 0));
            layer->setInput(4, *fill);
        } else {
            layer->setMode(mode == "reflect" ? nv::SampleMode::kREFLECT : nv::SampleMode::kCLAMP);
        }
        II_ASSIGN_OR_RETURN(nv::ITensor * y, out(layer, "pad"));
        set(node, y);
        return ok_status();
    }

    /// Indices with negative values counted from the back of `dim`.
    Result<nv::ITensor*> wrap_indices(const ir::Value* indices, const Dim& dim) {
        if (indices->kind() == ir::ValueKind::kConstant && dim.is_constant()) {
            II_ASSIGN_OR_RETURN(Tensor data, indices->data().contiguous());
            Tensor copy = data;
            if (data.data() == indices->data().data()) {
                II_ASSIGN_OR_RETURN(copy, Tensor::from_bytes(data.dtype(), data.shape(), data.data(), data.nbytes()));
            }
            const std::int64_t n = copy.numel();
            if (copy.dtype() == DType::kI64) {
                auto* p = static_cast<std::int64_t*>(copy.data());
                for (std::int64_t i = 0; i < n; ++i) {
                    p[i] = p[i] < 0 ? p[i] + dim.value() : p[i];
                }
            } else {
                auto* p = static_cast<std::int32_t*>(copy.data());
                for (std::int64_t i = 0; i < n; ++i) {
                    p[i] = p[i] < 0 ? static_cast<std::int32_t>(p[i] + dim.value()) : p[i];
                }
            }
            return constant(copy);
        }
        II_ASSIGN_OR_RETURN(nv::ITensor * idx, value(indices));
        II_ASSIGN_OR_RETURN(nv::ITensor * zero, scalar(indices->dtype(), 0, rank(idx)));
        II_ASSIGN_OR_RETURN(nv::ITensor * negative, binary(idx, zero, nv::ElementWiseOperation::kLESS));
        II_ASSIGN_OR_RETURN(nv::ITensor * size, dim_tensor(dim));  // [1] i64
        if (indices->dtype() != DType::kI64) {
            II_ASSIGN_OR_RETURN(size, out(net_.addCast(*size, nv::DataType::kINT32), "cast"));
        }
        if (rank(idx) == 0) {
            II_ASSIGN_OR_RETURN(size, squeeze(size, {0}));
        }
        II_ASSIGN_OR_RETURN(nv::ITensor * shifted, binary(idx, size, nv::ElementWiseOperation::kSUM));
        II_ASSIGN_OR_RETURN(negative, align_rank(negative, rank(shifted)));
        II_ASSIGN_OR_RETURN(idx, align_rank(idx, rank(shifted)));
        return out(net_.addSelect(*negative, *shifted, *idx), "select");
    }

    Status translate_gather(const ir::Node& node) {
        II_ASSIGN_OR_RETURN(nv::ITensor * data, input(node, 0));
        const int r = rank(data);
        const int axis = normalize_axis(node.attrs().get_int("axis"), r);
        const Dim& dim = node.input(0)->shape()[static_cast<std::size_t>(axis)];
        II_ASSIGN_OR_RETURN(nv::ITensor * indices, wrap_indices(node.input(1), dim));
        if (node.op_name() == "gather") {
            II_ASSIGN_OR_RETURN(nv::ITensor * y, out(net_.addGather(*data, *indices, axis), "gather"));
            set(node, y);
            return ok_status();
        }
        II_ASSIGN_OR_RETURN(nv::ITensor * updates, input(node, 2));
        nv::IScatterLayer* layer = net_.addScatter(*data, *indices, *updates, nv::ScatterMode::kELEMENT);
        if (layer != nullptr) {
            layer->setAxis(axis);
        }
        II_ASSIGN_OR_RETURN(nv::ITensor * y, out(layer, "scatter"));
        set(node, y);
        return ok_status();
    }

    Status translate_layer_norm(const ir::Node& node) {
        II_ASSIGN_OR_RETURN(nv::ITensor * x, input(node, 0));
        const int original = rank(x);
        const int axis_in = normalize_axis(node.attrs().get_int("axis", -1), original);
        if (original < 2) {  // TensorRT normalizes tensors of two dimensions or more
            II_ASSIGN_OR_RETURN(x, align_rank(x, 2));
        }
        const int r = rank(x);
        const int axis = axis_in + (r - original);
        std::uint32_t mask = 0;
        for (int d = axis; d < r; ++d) {
            mask |= 1u << static_cast<unsigned>(d);
        }
        const DType dtype = node.input(0)->dtype();
        auto operand = [&](int index, double fill) -> Result<nv::ITensor*> {
            nv::ITensor* t = nullptr;
            if (node.input(index) != nullptr) {
                II_ASSIGN_OR_RETURN(t, input(node, index));
            } else {
                Shape shape;
                for (int d = axis_in; d < original; ++d) {
                    shape.push_back(node.input(0)->shape()[static_cast<std::size_t>(d)].value());
                }
                std::vector<std::byte> one = scalar_bytes(dtype, fill);
                std::vector<std::byte> bytes;
                for (std::int64_t i = 0; i < shape.numel(); ++i) {
                    bytes.insert(bytes.end(), one.begin(), one.end());
                }
                II_ASSIGN_OR_RETURN(const nv::DataType type, type_of(dtype));
                II_ASSIGN_OR_RETURN(t,
                                    constant_bytes(type, std::move(bytes), shape.numel(),
                                                   to_dims(std::span<const std::int64_t>(shape.begin(), shape.end()))));
            }
            return align_rank(t, r);
        };
        II_ASSIGN_OR_RETURN(nv::ITensor * scale, operand(1, 1.0));
        II_ASSIGN_OR_RETURN(nv::ITensor * bias, operand(2, 0.0));
        nv::INormalizationLayer* layer = net_.addNormalizationV2(*x, *scale, *bias, mask);
        if (layer == nullptr) {
            return errors::internal("TensorRT could not add a normalization layer");
        }
        layer->setEpsilon(static_cast<float>(node.attrs().get_float("epsilon", 1e-5)));
        II_ASSIGN_OR_RETURN(nv::ITensor * y, out(layer, "normalization"));
        if (original < 2) {
            II_ASSIGN_OR_RETURN(
                y, squeeze(y, original == 1 ? std::vector<std::int64_t>{0} : std::vector<std::int64_t>{0, 1}));
        }
        set(node, y);
        return ok_status();
    }

    Status translate_creation(const ir::Node& node) {
        const ir::Value* output = node.output();
        Shape shape;
        for (const Dim& d : output->shape()) {
            shape.push_back(d.value());
        }
        const DType dtype = output->dtype();
        std::vector<std::byte> bytes;
        bytes.reserve(storage_size(dtype, shape.numel()));
        const bool iota = node.op_name() == "iota";
        const int axis = iota ? normalize_axis(node.attrs().get_int("axis"), shape.rank()) : 0;
        std::int64_t inner = 1;
        for (int d = axis + 1; d < shape.rank(); ++d) {
            inner *= shape[d];
        }
        const std::int64_t extent = shape.rank() == 0 ? 1 : shape[axis];
        for (std::int64_t i = 0; i < shape.numel(); ++i) {
            double v = node.attrs().get_float("value");
            if (iota) {
                const std::int64_t index = extent == 0 ? 0 : (i / inner) % extent;
                v = node.attrs().get_float("start") + node.attrs().get_float("step", 1.0) * static_cast<double>(index);
            }
            const std::vector<std::byte> one = scalar_bytes(dtype, std::isnan(v) && !is_float(dtype) ? 0.0 : v);
            bytes.insert(bytes.end(), one.begin(), one.end());
        }
        II_ASSIGN_OR_RETURN(const nv::DataType type, type_of(dtype));
        II_ASSIGN_OR_RETURN(nv::ITensor * y,
                            constant_bytes(type, std::move(bytes), shape.numel(),
                                           to_dims(std::span<const std::int64_t>(shape.begin(), shape.end()))));
        set(node, y);
        return ok_status();
    }

    nv::INetworkDefinition& net_;
    NetworkStorage& storage_;
    std::unordered_map<const ir::Value*, nv::ITensor*> values_;
    std::unordered_set<nv::ITensor*> constants_;
    std::map<std::string, std::pair<nv::ITensor*, int>> sources_;  // symbol -> (input, axis)
    std::map<std::string, nv::ITensor*> symbol_tensors_;
    std::vector<std::shared_ptr<const ir::Graph>> programs_;  // inner programs of fused nodes
};

}  // namespace

Status supports_node(const ir::Node& node) {
    return supports_impl(node, 0);
}

Status prepare_program(ir::Graph& graph) {
    for (int round = 0; round < 8; ++round) {
        std::vector<ir::Node*> nodes;
        for (const auto& node : graph.nodes()) {
            if (decomposed(node->op_name())) {
                nodes.push_back(node.get());
            }
        }
        if (nodes.empty()) {
            return ok_status();
        }
        for (ir::Node* node : nodes) {
            II_TRY(compiler::decompose_node(graph, *node));
        }
        compiler::eliminate_dead_code(graph);
    }
    return errors::internal("decomposition for TensorRT did not converge");
}

Status translate(const ir::Graph& program, nvinfer1::INetworkDefinition& network, NetworkStorage& storage) {
    return Translator(network, storage).run(program);
}

}  // namespace ii::tensorrt
