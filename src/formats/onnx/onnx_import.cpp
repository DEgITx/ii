// SPDX-License-Identifier: MIT
#include "formats/onnx/onnx_import.h"

#include "compiler/passes.h"
#include "formats/protobuf.h"
#include "ir/op_schema.h"

#include <algorithm>
#include <cmath>
#include <format>

namespace ii::formats::onnx {

namespace {

/// Function calls and subgraphs nest; deeper nesting is treated as malicious (or recursive).
constexpr int kMaxNesting = 32;

/// Contents of small integer constants are tracked symbolically up to this many elements.
constexpr std::int64_t kMaxKnownElements = 64;

/// A valid ii symbol name derived from an ONNX dim_param ("batch size" -> "batch_size").
std::string sanitize_symbol(std::string_view name) {
    std::string result;
    for (const char c : name) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
        result += ok ? c : '_';
    }
    if (!result.empty() && result.front() >= '0' && result.front() <= '9') {
        result.insert(result.begin(), '_');
    }
    return result;
}

std::vector<std::int64_t> tensor_ints(const Tensor& tensor) {
    std::vector<std::int64_t> result;
    const Result<Tensor> contiguous = tensor.contiguous();
    if (!contiguous.ok()) {
        return result;
    }
    const void* data = contiguous->data();
    const auto n = static_cast<std::size_t>(contiguous->numel());
    result.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        switch (tensor.dtype()) {
        case DType::kI64:
            result.push_back(static_cast<const std::int64_t*>(data)[i]);
            break;
        case DType::kI32:
            result.push_back(static_cast<const std::int32_t*>(data)[i]);
            break;
        case DType::kI16:
            result.push_back(static_cast<const std::int16_t*>(data)[i]);
            break;
        case DType::kI8:
            result.push_back(static_cast<const std::int8_t*>(data)[i]);
            break;
        case DType::kU8:
        case DType::kBool:
            result.push_back(static_cast<const std::uint8_t*>(data)[i]);
            break;
        case DType::kU16:
            result.push_back(static_cast<const std::uint16_t*>(data)[i]);
            break;
        case DType::kU32:
            result.push_back(static_cast<const std::uint32_t*>(data)[i]);
            break;
        case DType::kU64:
            result.push_back(static_cast<std::int64_t>(static_cast<const std::uint64_t*>(data)[i]));
            break;
        default:
            return {};
        }
    }
    return result;
}

std::vector<double> tensor_doubles(const Tensor& tensor) {
    std::vector<double> result;
    const Result<Tensor> contiguous = tensor.contiguous();
    if (!contiguous.ok()) {
        return result;
    }
    const void* data = contiguous->data();
    const auto n = static_cast<std::size_t>(contiguous->numel());
    switch (tensor.dtype()) {
    case DType::kF32:
        for (std::size_t i = 0; i < n; ++i) {
            result.push_back(static_cast<const float*>(data)[i]);
        }
        return result;
    case DType::kF64:
        for (std::size_t i = 0; i < n; ++i) {
            result.push_back(static_cast<const double*>(data)[i]);
        }
        return result;
    case DType::kF16:
        for (std::size_t i = 0; i < n; ++i) {
            result.push_back(static_cast<const f16*>(data)[i].to_float());
        }
        return result;
    case DType::kBF16:
        for (std::size_t i = 0; i < n; ++i) {
            result.push_back(static_cast<const bf16*>(data)[i].to_float());
        }
        return result;
    default:
        for (const std::int64_t v : tensor_ints(tensor)) {
            result.push_back(static_cast<double>(v));
        }
        return result;
    }
}

/// A tensor of `dtype` holding constant `values` (rank 0 when `scalar`).
Result<Tensor> make_int_tensor(std::span<const std::int64_t> values, bool scalar, DType dtype) {
    Shape shape;
    if (!scalar) {
        shape.push_back(static_cast<std::int64_t>(values.size()));
    }
    II_ASSIGN_OR_RETURN(Tensor tensor, Tensor::empty(dtype, shape));
    void* data = tensor.data();
    for (std::size_t i = 0; i < values.size(); ++i) {
        const std::int64_t v = values[i];
        switch (dtype) {
        case DType::kI64:
            static_cast<std::int64_t*>(data)[i] = v;
            break;
        case DType::kI32:
            static_cast<std::int32_t*>(data)[i] = static_cast<std::int32_t>(v);
            break;
        case DType::kF32:
            static_cast<float*>(data)[i] = static_cast<float>(v);
            break;
        case DType::kF64:
            static_cast<double*>(data)[i] = static_cast<double>(v);
            break;
        case DType::kBool:
            static_cast<std::uint8_t*>(data)[i] = v != 0 ? 1 : 0;
            break;
        default:
            return errors::unsupported("known contents of type {}", to_string(dtype));
        }
    }
    return tensor;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// NodeContext

Result<Entry*> NodeContext::entry(int index) {
    if (!has_input(index)) {
        return errors::invalid_argument("'{}' requires input {}", node_.op_type, index);
    }
    const std::string& name = node_.inputs[static_cast<std::size_t>(index)];
    Entry* found = importer_.find(name);
    if (found == nullptr) {
        return errors::data_corrupted("input '{}' is not defined before its use", name);
    }
    return found;
}

Result<ir::Value*> NodeContext::input(int index) {
    II_ASSIGN_OR_RETURN(Entry * found, entry(index));
    return importer_.materialize(node_.inputs[static_cast<std::size_t>(index)], *found);
}

Result<ir::Value*> NodeContext::optional_input(int index) {
    if (!has_input(index)) {
        return static_cast<ir::Value*>(nullptr);
    }
    return input(index);
}

Result<ir::TensorType> NodeContext::input_type(int index) {
    II_ASSIGN_OR_RETURN(Entry * found, entry(index));
    if (found->value != nullptr) {
        return found->value->type();
    }
    ir::TensorType type{found->dtype, {}};
    if (!found->scalar) {
        type.shape.emplace_back(static_cast<std::int64_t>(found->known->size()));
    }
    return type;
}

Result<Dims> NodeContext::dims_input(int index) {
    II_ASSIGN_OR_RETURN(Entry * found, entry(index));
    if (found->known) {
        return *found->known;
    }
    II_ASSIGN_OR_RETURN(ir::Value * folded, importer_.try_fold(found->value));
    if (folded == nullptr) {
        return errors::unsupported("input {} ('{}') of '{}' must be computable when the model is loaded, but it "
                                   "depends on run-time data",
                                   index, node_.inputs[static_cast<std::size_t>(index)], node_.op_type);
    }
    if (folded->rank() > 1) {
        return errors::invalid_argument("input {} of '{}' must have rank 0 or 1", index, node_.op_type);
    }
    Dims dims;
    for (const std::int64_t v : tensor_ints(folded->data())) {
        dims.emplace_back(v);
    }
    if (static_cast<std::int64_t>(dims.size()) != folded->data().numel()) {
        return errors::invalid_argument("input {} of '{}' must hold integers, got {}", index, node_.op_type,
                                        to_string(folded->dtype()));
    }
    return dims;
}

Result<std::vector<std::int64_t>> NodeContext::ints_input(int index) {
    II_ASSIGN_OR_RETURN(const Dims dims, dims_input(index));
    std::vector<std::int64_t> values;
    for (const Dim& dim : dims) {
        if (!dim.is_constant()) {
            return errors::unsupported("input {} of '{}' holds the symbolic values {}; fix the model's input "
                                       "dimensions (e.g. --dims) to import it",
                                       index, node_.op_type, to_string(dims));
        }
        values.push_back(dim.value());
    }
    return values;
}

Result<std::vector<double>> NodeContext::floats_input(int index) {
    II_ASSIGN_OR_RETURN(Entry * found, entry(index));
    if (found->known) {
        II_ASSIGN_OR_RETURN(const std::vector<std::int64_t> ints, ints_input(index));
        std::vector<double> values;
        for (const std::int64_t v : ints) {
            values.push_back(static_cast<double>(v));
        }
        return values;
    }
    II_ASSIGN_OR_RETURN(const Tensor tensor, constant_input(index));
    return tensor_doubles(tensor);
}

Result<Tensor> NodeContext::constant_input(int index) {
    II_ASSIGN_OR_RETURN(ir::Value * value, input(index));
    II_ASSIGN_OR_RETURN(ir::Value * folded, importer_.try_fold(value));
    if (folded == nullptr) {
        return errors::unsupported("input {} ('{}') of '{}' must be a constant", index,
                                   node_.inputs[static_cast<std::size_t>(index)], node_.op_type);
    }
    return folded->data();
}

bool NodeContext::is_constant_input(int index) {
    const Result<Entry*> found = entry(index);
    if (!found.ok()) {
        return false;
    }
    if ((*found)->known) {
        return std::ranges::all_of(*(*found)->known, [](const Dim& d) { return d.is_constant(); });
    }
    const Result<ir::Value*> folded = importer_.try_fold((*found)->value);
    return folded.ok() && *folded != nullptr;
}

std::optional<KnownContents> NodeContext::known_input(int index) {
    const Result<Entry*> found = entry(index);
    if (!found.ok()) {
        return std::nullopt;
    }
    const Entry& e = **found;
    if (e.known) {
        return KnownContents{*e.known, e.scalar, e.dtype};
    }
    const ir::Value* value = e.value;
    if (value == nullptr || (value->dtype() != DType::kI64 && value->dtype() != DType::kI32) || value->rank() > 1 ||
        !value->type().is_static() || value->type().static_shape()->numel() > kMaxKnownElements) {
        return std::nullopt;
    }
    const Result<ir::Value*> folded = importer_.try_fold(e.value);
    if (!folded.ok() || *folded == nullptr) {
        return std::nullopt;
    }
    KnownContents contents;
    contents.scalar = (*folded)->rank() == 0;
    contents.dtype = (*folded)->dtype();
    for (const std::int64_t v : tensor_ints((*folded)->data())) {
        contents.values.emplace_back(v);
    }
    return contents;
}

std::int64_t NodeContext::int_attr(std::string_view name, std::int64_t fallback) const noexcept {
    const OnnxAttribute* attr = attribute(name);
    return attr != nullptr ? attr->i : fallback;
}

double NodeContext::float_attr(std::string_view name, double fallback) const noexcept {
    const OnnxAttribute* attr = attribute(name);
    if (attr == nullptr) {
        return fallback;
    }
    return attr->type == kAttrInt ? static_cast<double>(attr->i) : static_cast<double>(attr->f);
}

std::string NodeContext::string_attr(std::string_view name, std::string_view fallback) const {
    const OnnxAttribute* attr = attribute(name);
    return attr != nullptr ? attr->s : std::string(fallback);
}

std::optional<std::vector<std::int64_t>> NodeContext::ints_attr(std::string_view name) const {
    const OnnxAttribute* attr = attribute(name);
    if (attr == nullptr) {
        return std::nullopt;
    }
    if (attr->type == kAttrInt) {
        return std::vector<std::int64_t>{attr->i};
    }
    return attr->ints;
}

std::optional<std::vector<double>> NodeContext::floats_attr(std::string_view name) const {
    const OnnxAttribute* attr = attribute(name);
    if (attr == nullptr) {
        return std::nullopt;
    }
    if (attr->type == kAttrFloat) {
        return std::vector<double>{attr->f};
    }
    return std::vector<double>(attr->floats.begin(), attr->floats.end());
}

Result<Tensor> NodeContext::tensor_attr(std::string_view name) const {
    const OnnxAttribute* attr = attribute(name);
    if (attr == nullptr || !attr->t) {
        return errors::not_found("'{}' has no tensor attribute '{}'", node_.op_type, name);
    }
    return importer_.tensor_of(*attr->t);
}

Result<std::vector<ir::Value*>> NodeContext::op_outputs(std::string_view name, std::vector<ir::Value*> inputs,
                                                        Attributes attrs) {
    II_ASSIGN_OR_RETURN(ir::Node * created, importer_.builder().create_node(name, std::move(inputs), std::move(attrs)));
    if (!node_.name.empty()) {
        created->set_name(node_.name);
    }
    return std::vector<ir::Value*>(created->outputs().begin(), created->outputs().end());
}

Result<ir::Value*> NodeContext::op(std::string_view name, std::vector<ir::Value*> inputs, Attributes attrs) {
    II_ASSIGN_OR_RETURN(std::vector<ir::Value*> outputs, op_outputs(name, std::move(inputs), std::move(attrs)));
    return outputs.front();
}

ir::Value* NodeContext::constant(Tensor tensor) {
    return importer_.graph().add_constant(std::move(tensor));
}

Result<ir::Value*> NodeContext::scalar(double value, DType dtype) {
    II_ASSIGN_OR_RETURN(Tensor tensor, ir::make_scalar_tensor(value, dtype));
    return constant(std::move(tensor));
}

Result<ir::Value*> NodeContext::cast(ir::Value* value, DType dtype) {
    if (value->dtype() == dtype) {
        return value;
    }
    return op("cast", {value}, {{"to", dtype}});
}

Result<ir::Value*> NodeContext::reshape(ir::Value* value, Dims dims) {
    if (value->shape() == dims) {
        return value;
    }
    return op("reshape", {value}, {{"shape", std::move(dims)}});
}

Status NodeContext::set_output(int index, ir::Value* value) {
    if (!has_output(index)) {
        return ok_status();
    }
    const std::string& name = node_.outputs[static_cast<std::size_t>(index)];
    if (value->kind() == ir::ValueKind::kNode && value->name().empty()) {
        value->set_name(name);
    }
    return importer_.bind(name, Entry{value, std::nullopt, false, value->dtype()});
}

Status NodeContext::set_known_output(int index, Dims contents, bool scalar, DType dtype) {
    if (!has_output(index)) {
        return ok_status();
    }
    if (scalar && contents.size() != 1) {
        return errors::internal("scalar known contents with {} elements", contents.size());
    }
    return importer_.bind(node_.outputs[static_cast<std::size_t>(index)],
                          Entry{nullptr, std::move(contents), scalar, dtype});
}

Status NodeContext::unsupported(std::string_view what) const {
    return errors::unsupported("{}", what);
}

// ---------------------------------------------------------------------------------------------------------------
// Importer

Importer::Importer(const OnnxModel& model, std::shared_ptr<const void> keep_alive, const ImportOptions& options)
    : model_(model),
      keep_alive_(std::move(keep_alive)),
      options_(options),
      graph_(std::make_unique<ir::Graph>()),
      builder_(*graph_),
      external_(options.base_dir) {}

Entry* Importer::find(const std::string& name) {
    const auto it = values_.find(name);
    return it != values_.end() ? &it->second : nullptr;
}

Status Importer::bind(const std::string& name, Entry entry) {
    if (name.empty()) {
        return ok_status();
    }
    values_[name] = std::move(entry);
    return ok_status();
}

Result<Tensor> Importer::tensor_of(const OnnxTensor& proto) {
    return to_tensor(proto, keep_alive_, options_.base_dir.empty() ? nullptr : &external_);
}

std::string Importer::fresh_symbol() {
    while (true) {
        std::string name = std::format("unk{}", next_symbol_++);
        bool used = false;
        for (const ir::Value* input : graph_->inputs()) {
            for (const Dim& dim : input->shape()) {
                if (dim.symbol_name() == name) {
                    used = true;
                }
            }
        }
        if (!used) {
            return name;
        }
    }
}

Result<ir::Value*> Importer::materialize(const std::string& name, Entry& entry) {
    if (entry.value != nullptr) {
        return entry.value;
    }
    if (!entry.known) {
        return errors::internal("value '{}' has neither an ii value nor known contents", name);
    }
    const Dims& known = *entry.known;
    const bool all_constant = std::ranges::all_of(known, [](const Dim& d) { return d.is_constant(); });
    if (all_constant) {
        std::vector<std::int64_t> values;
        for (const Dim& d : known) {
            values.push_back(d.value());
        }
        II_ASSIGN_OR_RETURN(Tensor tensor, make_int_tensor(values, entry.scalar, entry.dtype));
        entry.value = graph_->add_constant(std::move(tensor), name);
        return entry.value;
    }
    // Symbolic contents become a run-time computation: a symbolic dimension d is the sum of d ones.
    std::vector<ir::Value*> parts;
    std::vector<std::int64_t> pending;
    auto flush = [&]() -> Status {
        if (!pending.empty()) {
            II_ASSIGN_OR_RETURN(Tensor tensor, make_int_tensor(pending, false, DType::kI64));
            parts.push_back(graph_->add_constant(std::move(tensor)));
            pending.clear();
        }
        return ok_status();
    };
    for (const Dim& d : known) {
        if (d.is_constant()) {
            pending.push_back(d.value());
            continue;
        }
        II_TRY(flush());
        II_ASSIGN_OR_RETURN(
            ir::Node * ones,
            builder_.create_node("fill", {}, {{"shape", Dims{d}}, {"dtype", DType::kI64}, {"value", 1.0}}));
        II_ASSIGN_OR_RETURN(ir::Node * sum,
                            builder_.create_node("reduce_sum", {ones->output()},
                                                 {{"axes", std::vector<std::int64_t>{0}}, {"keepdims", true}}));
        parts.push_back(sum->output());
    }
    II_TRY(flush());
    ir::Value* value = parts.front();
    if (parts.size() > 1) {
        II_ASSIGN_OR_RETURN(ir::Node * concat, builder_.create_node("concat", parts, {{"axis", std::int64_t{0}}}));
        value = concat->output();
    }
    if (entry.dtype != DType::kI64) {
        II_ASSIGN_OR_RETURN(ir::Node * cast, builder_.create_node("cast", {value}, {{"to", entry.dtype}}));
        value = cast->output();
    }
    if (entry.scalar) {
        II_ASSIGN_OR_RETURN(ir::Node * reshape, builder_.create_node("reshape", {value}, {{"shape", Dims{}}}));
        value = reshape->output();
    }
    value->set_name(name);
    entry.value = value;
    return value;
}

Result<ir::Value*> Importer::try_fold(ir::Value* value, int depth) {
    if (value == nullptr) {
        return static_cast<ir::Value*>(nullptr);
    }
    if (value->kind() == ir::ValueKind::kConstant) {
        return value;
    }
    if (value->kind() != ir::ValueKind::kNode || depth > 256 || !value->type().is_static()) {
        return static_cast<ir::Value*>(nullptr);
    }
    ir::Node* node = value->producer();
    if (node->op().has_flag(ir::kOpRandom)) {
        return static_cast<ir::Value*>(nullptr);
    }
    for (int i = 0; i < node->num_inputs(); ++i) {
        ir::Value* input = node->input(i);
        if (input == nullptr) {
            continue;
        }
        II_ASSIGN_OR_RETURN(ir::Value * folded, try_fold(input, depth + 1));
        if (folded == nullptr) {
            return static_cast<ir::Value*>(nullptr);
        }
    }
    // Never let a (possibly malicious) model allocate huge constants while it is being imported.
    std::size_t input_bytes = 0;
    for (const ir::Value* input : node->inputs()) {
        if (input != nullptr) {
            input_bytes += input->data().nbytes();
        }
    }
    for (const ir::Value* output : node->outputs()) {
        if (!output->type().is_static()) {
            return static_cast<ir::Value*>(nullptr);
        }
        const std::size_t bytes = storage_size(output->dtype(), output->type().static_shape()->numel());
        if (bytes > (std::size_t{64} << 20) && bytes > input_bytes) {
            return static_cast<ir::Value*>(nullptr);
        }
    }
    Result<std::vector<Tensor>> results = compiler::evaluate_node(*node, *options_.evaluator);
    if (!results.ok()) {
        Status status = results.status();
        return status.add_context(std::format("evaluating '{}' while importing", node->op_name()));
    }
    ir::Value* replacement = nullptr;
    for (int i = 0; i < node->num_outputs(); ++i) {
        ir::Value* output = node->output(i);
        ir::Value* constant = graph_->add_constant(std::move((*results)[static_cast<std::size_t>(i)]), output->name());
        graph_->replace_all_uses(output, constant);
        for (auto& [name, entry] : values_) {
            if (entry.value == output) {
                entry.value = constant;
            }
        }
        if (output == value) {
            replacement = constant;
        }
    }
    ir::Node* doomed[] = {node};
    graph_->erase_nodes(doomed);
    return replacement;
}

Status Importer::import_graph_inputs() {
    for (const OnnxValueInfo& input : model_.graph.inputs) {
        if (find(input.name) != nullptr) {
            continue;  // an initializer (IR < 4 lists initializers among the inputs)
        }
        const auto constant = std::ranges::find_if(options_.constant_inputs,
                                                   [&](const auto& entry) { return entry.first == input.name; });
        if (constant != options_.constant_inputs.end()) {
            const DType declared = dtype_from_onnx(input.elem_type);
            if (declared != constant->second.dtype()) {
                return errors::invalid_argument("constant for input '{}' has dtype {}, the model declares {}",
                                                input.name, to_string(constant->second.dtype()),
                                                onnx_type_name(input.elem_type));
            }
            ir::Value* value = graph_->add_constant(constant->second, input.name);
            II_TRY(bind(input.name, Entry{value, std::nullopt, false, value->dtype()}));
            continue;
        }
        if (!input.is_tensor) {
            return errors::unsupported("input '{}' is not a tensor (sequences, maps and optionals are not supported)",
                                       input.name);
        }
        const DType dtype = dtype_from_onnx(input.elem_type);
        if (dtype == DType::kInvalid) {
            return errors::unsupported("input '{}' has element type {}, which ii does not support", input.name,
                                       onnx_type_name(input.elem_type));
        }
        if (!input.has_shape) {
            return errors::unsupported("input '{}' has no shape (inputs of unknown rank are not supported)",
                                       input.name);
        }
        if (input.shape.size() > static_cast<std::size_t>(kMaxRank)) {
            return errors::unsupported("input '{}' has rank {}; ii supports at most {}", input.name, input.shape.size(),
                                       kMaxRank);
        }
        Dims shape;
        for (const OnnxDim& dim : input.shape) {
            std::optional<std::int64_t> fixed;
            if (!dim.param.empty()) {
                for (const auto& [name, value] : options_.dims) {
                    if (name == dim.param || name == sanitize_symbol(dim.param)) {
                        fixed = value;
                    }
                }
            }
            if (fixed) {
                shape.emplace_back(*fixed);
            } else if (dim.has_value && dim.value >= 0) {
                shape.emplace_back(dim.value);
            } else if (const std::string symbol = sanitize_symbol(dim.param); !symbol.empty()) {
                shape.emplace_back(Dim::symbol(symbol));
            } else {
                shape.emplace_back(Dim::symbol(fresh_symbol()));
            }
        }
        ir::Value* value = graph_->add_input(input.name, ir::TensorType{dtype, std::move(shape)});
        II_TRY(bind(input.name, Entry{value, std::nullopt, false, dtype}));
    }
    return ok_status();
}

Status Importer::import_function_call(const OnnxNode& node, const OnnxFunction& function, int depth) {
    const std::string prefix = std::format("{}::{}#{}/", function.domain, function.name, next_inline_++);
    auto rename = [&](const std::string& name) -> std::string {
        if (name.empty()) {
            return name;
        }
        for (std::size_t i = 0; i < function.inputs.size(); ++i) {
            if (function.inputs[i] == name) {
                return i < node.inputs.size() ? node.inputs[i] : std::string();
            }
        }
        for (std::size_t i = 0; i < function.outputs.size(); ++i) {
            if (function.outputs[i] == name) {
                return i < node.outputs.size() ? node.outputs[i] : prefix + name;
            }
        }
        return prefix + name;
    };
    std::int64_t opset = model_.default_opset();
    for (const OnnxOpset& import : function.opset_import) {
        if (import.domain.empty() || import.domain == "ai.onnx") {
            opset = import.version;
        }
    }
    for (const OnnxNode& inner : function.nodes) {
        OnnxNode call = inner;
        for (std::string& name : call.inputs) {
            name = rename(name);
        }
        for (std::string& name : call.outputs) {
            name = rename(name);
        }
        call.attributes.clear();
        for (const OnnxAttribute& attr : inner.attributes) {
            if (attr.ref_attr_name.empty()) {
                call.attributes.push_back(attr);
                continue;
            }
            const OnnxAttribute* given = node.find_attribute(attr.ref_attr_name);
            if (given == nullptr) {
                for (const OnnxAttribute& fallback : function.attribute_defaults) {
                    if (fallback.name == attr.ref_attr_name) {
                        given = &fallback;
                    }
                }
            }
            if (given != nullptr) {
                OnnxAttribute copy = *given;
                copy.name = attr.name;
                call.attributes.push_back(std::move(copy));
            }
        }
        if (call.name.empty()) {
            call.name = node.name;
        }
        II_TRY(import_node(call, opset, depth + 1));
    }
    return ok_status();
}

Status Importer::import_node(const OnnxNode& node, std::int64_t opset, int depth) {
    if (depth > kMaxNesting) {
        return errors::data_corrupted("functions or subgraphs are nested more than {} levels deep", kMaxNesting);
    }
    auto with_context = [&](Status status) {
        return status.add_context(std::format("ONNX node '{}' ({}{}{})", node.name, node.domain,
                                              node.domain.empty() ? "" : ".", node.op_type));
    };
    for (const OnnxFunction& function : model_.functions) {
        if (function.name == node.op_type && function.domain == node.domain) {
            if (Status status = import_function_call(node, function, depth); !status.ok()) {
                return with_context(std::move(status));
            }
            return ok_status();
        }
    }
    if (!node.domain.empty() && node.domain != "ai.onnx") {
        return with_context(errors::unsupported("operators of domain '{}' are not supported", node.domain));
    }
    NodeContext context(*this, node, opset);
    Result<bool> known = try_known_contents(context);
    if (!known.ok()) {
        return with_context(known.status());
    }
    if (!*known) {
        const OpHandler handler = find_op_handler(node.op_type);
        if (handler == nullptr) {
            return with_context(
                errors::unsupported("ONNX operator '{}' (opset {}) is not supported", node.op_type, opset));
        }
        if (Status status = handler(context); !status.ok()) {
            return with_context(std::move(status));
        }
    }
    for (const std::string& output : node.outputs) {
        if (!output.empty() && find(output) == nullptr) {
            return with_context(errors::unsupported("output '{}' is not supported", output));
        }
    }
    return ok_status();
}

Status Importer::import_nodes(std::span<const OnnxNode> nodes, std::int64_t opset, int depth) {
    for (const OnnxNode& node : nodes) {
        II_TRY(import_node(node, opset, depth));
    }
    return ok_status();
}

Result<std::vector<Entry>> Importer::inline_subgraph(const OnnxGraph& subgraph) {
    for (const OnnxTensor& initializer : subgraph.initializers) {
        II_ASSIGN_OR_RETURN(Tensor tensor, tensor_of(initializer));
        ir::Value* value = graph_->add_constant(std::move(tensor), initializer.name);
        II_TRY(bind(initializer.name, Entry{value, std::nullopt, false, value->dtype()}));
    }
    if (!subgraph.inputs.empty()) {
        return errors::unsupported("subgraphs with inputs");
    }
    II_TRY(import_nodes(subgraph.nodes, model_.default_opset(), 1));
    std::vector<Entry> outputs;
    for (const OnnxValueInfo& output : subgraph.outputs) {
        Entry* found = find(output.name);
        if (found == nullptr) {
            return errors::data_corrupted("subgraph output '{}' is not defined", output.name);
        }
        outputs.push_back(*found);
    }
    return outputs;
}

Result<std::unique_ptr<ir::Graph>> Importer::run() {
    if (options_.evaluator == nullptr) {
        return errors::internal("ONNX import needs an evaluator");
    }
    const std::int64_t opset = model_.default_opset();
    if (opset <= 0) {
        return errors::unsupported("the model does not import the default ONNX operator set");
    }
    graph_->set_name(model_.graph.name);
    if (model_.graph.sparse_initializers != 0) {
        return errors::unsupported("sparse initializers are not supported");
    }
    for (const OnnxTensor& initializer : model_.graph.initializers) {
        if (find(initializer.name) != nullptr) {
            return errors::data_corrupted("initializer '{}' is defined twice", initializer.name);
        }
        Result<Tensor> tensor = tensor_of(initializer);
        if (!tensor.ok()) {
            Status status = tensor.status();
            return status.add_context(std::format("initializer '{}'", initializer.name));
        }
        ir::Value* value = graph_->add_constant(std::move(*tensor), initializer.name);
        II_TRY(bind(initializer.name, Entry{value, std::nullopt, false, value->dtype()}));
    }
    II_TRY(import_graph_inputs());
    II_TRY(import_nodes(model_.graph.nodes, opset, 0));
    for (const OnnxValueInfo& output : model_.graph.outputs) {
        Entry* found = find(output.name);
        if (found == nullptr) {
            return errors::data_corrupted("graph output '{}' is not defined", output.name);
        }
        II_ASSIGN_OR_RETURN(ir::Value * value, materialize(output.name, *found));
        const DType declared = output.is_tensor ? dtype_from_onnx(output.elem_type) : DType::kInvalid;
        if (declared != DType::kInvalid && declared != value->dtype()) {
            II_ASSIGN_OR_RETURN(ir::Node * cast, builder_.create_node("cast", {value}, {{"to", declared}}));
            value = cast->output();
        }
        if (graph_->find_output(output.name) != nullptr) {
            return errors::data_corrupted("graph output '{}' is listed twice", output.name);
        }
        graph_->add_output(output.name, value);
    }
    compiler::eliminate_dead_code(*graph_);
    if (Status status = graph_->verify(); !status.ok()) {
        return status.add_context("the imported graph is invalid");
    }
    return std::move(graph_);
}

// ---------------------------------------------------------------------------------------------------------------

Result<std::unique_ptr<ir::Graph>> import_model(std::span<const std::byte> bytes,
                                                std::shared_ptr<const void> keep_alive, const ImportOptions& options) {
    II_ASSIGN_OR_RETURN(const OnnxModel model, parse_model(bytes));
    Importer importer(model, std::move(keep_alive), options);
    return importer.run();
}

bool looks_like_onnx(std::span<const std::byte> bytes) noexcept {
    // A ModelProto starts with small fields (ir_version = 1, producer_name = 2, ...) and contains a graph (7).
    pb::Reader reader(bytes.first(std::min<std::size_t>(bytes.size(), std::size_t{1} << 20)));
    pb::Field field;
    int fields = 0;
    while (reader.next(field) && fields < 64) {
        ++fields;
        if (field.number == 1 && field.wire == pb::WireType::kVarint) {
            return field.scalar >= 1 && field.scalar <= 64;
        }
        if (field.number > 25) {
            return false;
        }
    }
    return false;
}

}  // namespace ii::formats::onnx
