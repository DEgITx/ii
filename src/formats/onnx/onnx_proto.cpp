// SPDX-License-Identifier: MIT
#include "formats/onnx/onnx_proto.h"

#include <ii/half.h>

#include "core/platform/mapped_file.h"
#include "formats/protobuf.h"

#include <cstring>
#include <format>
#include <limits>

namespace ii::formats::onnx {

namespace {

/// Subgraphs (If / Loop bodies) nest; deeper nesting than this is treated as malicious.
constexpr int kMaxGraphDepth = 32;

Status parse_tensor_into(std::span<const std::byte> bytes, OnnxTensor& tensor);
Status parse_graph_into(std::span<const std::byte> bytes, OnnxGraph& graph, int depth);

Status parse_string_entry(std::span<const std::byte> bytes, std::pair<std::string, std::string>& entry) {
    pb::Reader reader(bytes);
    pb::Field field;
    while (reader.next(field)) {
        if (field.wire != pb::WireType::kLength) {
            continue;
        }
        if (field.number == 1) {
            entry.first = field.string();
        } else if (field.number == 2) {
            entry.second = field.string();
        }
    }
    return reader.status();
}

Status parse_tensor_into(std::span<const std::byte> bytes, OnnxTensor& tensor) {
    pb::Reader reader(bytes);
    pb::Field field;
    while (reader.next(field)) {
        switch (field.number) {
        case 1:
            II_TRY(pb::append_varints(field, tensor.dims));
            break;
        case 2:
            tensor.data_type = field.int32();
            break;
        case 3:
            tensor.has_segment = true;
            break;
        case 4:
            II_TRY(pb::append_floats(field, tensor.float_data));
            break;
        case 5:
            II_TRY(pb::append_varints(field, tensor.int32_data));
            break;
        case 6:
            ++tensor.string_count;
            break;
        case 7:
            II_TRY(pb::append_varints(field, tensor.int64_data));
            break;
        case 8:
            tensor.name = field.string();
            break;
        case 9:
            tensor.has_raw_data = true;
            tensor.raw_data = field.bytes;
            break;
        case 10:
            II_TRY(pb::append_doubles(field, tensor.double_data));
            break;
        case 11:
            II_TRY(pb::append_varints(field, tensor.uint64_data));
            break;
        case 13: {
            std::pair<std::string, std::string> entry;
            II_TRY(parse_string_entry(field.bytes, entry));
            tensor.external_data.push_back(std::move(entry));
            break;
        }
        case 14:
            tensor.external = field.scalar == 1;
            break;
        default:
            break;
        }
    }
    return reader.status();
}

Status parse_shape(std::span<const std::byte> bytes, std::vector<OnnxDim>& shape) {
    pb::Reader reader(bytes);
    pb::Field field;
    while (reader.next(field)) {
        if (field.number != 1 || field.wire != pb::WireType::kLength) {
            continue;
        }
        OnnxDim dim;
        pb::Reader dim_reader(field.bytes);
        pb::Field dim_field;
        while (dim_reader.next(dim_field)) {
            if (dim_field.number == 1 && dim_field.wire == pb::WireType::kVarint) {
                dim.has_value = true;
                dim.value = dim_field.int64();
            } else if (dim_field.number == 2 && dim_field.wire == pb::WireType::kLength) {
                dim.param = dim_field.string();
            }
        }
        II_TRY(dim_reader.status());
        shape.push_back(std::move(dim));
    }
    return reader.status();
}

Status parse_value_info(std::span<const std::byte> bytes, OnnxValueInfo& info) {
    pb::Reader reader(bytes);
    pb::Field field;
    while (reader.next(field)) {
        if (field.number == 1 && field.wire == pb::WireType::kLength) {
            info.name = field.string();
        } else if (field.number == 2 && field.wire == pb::WireType::kLength) {
            // TypeProto: tensor_type = 1 (other kinds are not tensors).
            pb::Reader type_reader(field.bytes);
            pb::Field type_field;
            while (type_reader.next(type_field)) {
                if (type_field.number != 1 || type_field.wire != pb::WireType::kLength) {
                    continue;
                }
                info.is_tensor = true;
                pb::Reader tensor_reader(type_field.bytes);
                pb::Field tensor_field;
                while (tensor_reader.next(tensor_field)) {
                    if (tensor_field.number == 1 && tensor_field.wire == pb::WireType::kVarint) {
                        info.elem_type = tensor_field.int32();
                    } else if (tensor_field.number == 2 && tensor_field.wire == pb::WireType::kLength) {
                        info.has_shape = true;
                        II_TRY(parse_shape(tensor_field.bytes, info.shape));
                    }
                }
                II_TRY(tensor_reader.status());
            }
            II_TRY(type_reader.status());
        }
    }
    return reader.status();
}

Status parse_attribute(std::span<const std::byte> bytes, OnnxAttribute& attr, int depth) {
    pb::Reader reader(bytes);
    pb::Field field;
    while (reader.next(field)) {
        switch (field.number) {
        case 1:
            attr.name = field.string();
            break;
        case 2:
            attr.f = field.float32();
            break;
        case 3:
            attr.i = field.int64();
            break;
        case 4:
            attr.s = field.string();
            break;
        case 5:
            attr.t = std::make_shared<OnnxTensor>();
            II_TRY(parse_tensor_into(field.bytes, *attr.t));
            break;
        case 6:
            attr.g = std::make_shared<OnnxGraph>();
            II_TRY(parse_graph_into(field.bytes, *attr.g, depth + 1));
            break;
        case 7:
            II_TRY(pb::append_floats(field, attr.floats));
            break;
        case 8:
            II_TRY(pb::append_varints(field, attr.ints));
            break;
        case 9:
            attr.strings.push_back(field.string());
            break;
        case 10:
            attr.tensors.emplace_back();
            II_TRY(parse_tensor_into(field.bytes, attr.tensors.back()));
            break;
        case 11:
            attr.graphs.push_back(std::make_shared<OnnxGraph>());
            II_TRY(parse_graph_into(field.bytes, *attr.graphs.back(), depth + 1));
            break;
        case 20:
            attr.type = field.int32();
            break;
        case 21:
            attr.ref_attr_name = field.string();
            break;
        default:
            break;
        }
    }
    II_TRY(reader.status());
    if (attr.type == kAttrUndefined) {
        // Very old files (IR version < 2) omit the type: infer it from the populated field.
        if (attr.g) {
            attr.type = kAttrGraph;
        } else if (attr.t) {
            attr.type = kAttrTensor;
        } else if (!attr.floats.empty()) {
            attr.type = kAttrFloats;
        } else if (!attr.ints.empty()) {
            attr.type = kAttrInts;
        } else if (!attr.strings.empty()) {
            attr.type = kAttrStrings;
        } else if (!attr.s.empty()) {
            attr.type = kAttrString;
        }
    }
    return ok_status();
}

Status parse_node(std::span<const std::byte> bytes, OnnxNode& node, int depth) {
    pb::Reader reader(bytes);
    pb::Field field;
    while (reader.next(field)) {
        switch (field.number) {
        case 1:
            node.inputs.push_back(field.string());
            break;
        case 2:
            node.outputs.push_back(field.string());
            break;
        case 3:
            node.name = field.string();
            break;
        case 4:
            node.op_type = field.string();
            break;
        case 5:
            node.attributes.emplace_back();
            II_TRY(parse_attribute(field.bytes, node.attributes.back(), depth));
            break;
        case 7:
            node.domain = field.string();
            break;
        default:
            break;
        }
    }
    return reader.status();
}

Status parse_graph_into(std::span<const std::byte> bytes, OnnxGraph& graph, int depth) {
    if (depth > kMaxGraphDepth) {
        return errors::data_corrupted("subgraphs are nested more than {} levels deep", kMaxGraphDepth);
    }
    pb::Reader reader(bytes);
    pb::Field field;
    while (reader.next(field)) {
        if (field.wire != pb::WireType::kLength) {
            continue;
        }
        switch (field.number) {
        case 1:
            graph.nodes.emplace_back();
            II_TRY(parse_node(field.bytes, graph.nodes.back(), depth));
            break;
        case 2:
            graph.name = field.string();
            break;
        case 5:
            graph.initializers.emplace_back();
            II_TRY(parse_tensor_into(field.bytes, graph.initializers.back()));
            break;
        case 11:
            graph.inputs.emplace_back();
            II_TRY(parse_value_info(field.bytes, graph.inputs.back()));
            break;
        case 12:
            graph.outputs.emplace_back();
            II_TRY(parse_value_info(field.bytes, graph.outputs.back()));
            break;
        case 13:
            graph.value_info.emplace_back();
            II_TRY(parse_value_info(field.bytes, graph.value_info.back()));
            break;
        case 15:
            ++graph.sparse_initializers;
            break;
        default:
            break;
        }
    }
    return reader.status();
}

Status parse_opset(std::span<const std::byte> bytes, OnnxOpset& opset) {
    pb::Reader reader(bytes);
    pb::Field field;
    while (reader.next(field)) {
        if (field.number == 1 && field.wire == pb::WireType::kLength) {
            opset.domain = field.string();
        } else if (field.number == 2 && field.wire == pb::WireType::kVarint) {
            opset.version = field.int64();
        }
    }
    return reader.status();
}

Status parse_function(std::span<const std::byte> bytes, OnnxFunction& function) {
    pb::Reader reader(bytes);
    pb::Field field;
    while (reader.next(field)) {
        if (field.wire != pb::WireType::kLength) {
            continue;
        }
        switch (field.number) {
        case 1:
            function.name = field.string();
            break;
        case 4:
            function.inputs.push_back(field.string());
            break;
        case 5:
            function.outputs.push_back(field.string());
            break;
        case 6:
            function.attributes.push_back(field.string());
            break;
        case 7:
            function.nodes.emplace_back();
            II_TRY(parse_node(field.bytes, function.nodes.back(), 1));
            break;
        case 9:
            function.opset_import.emplace_back();
            II_TRY(parse_opset(field.bytes, function.opset_import.back()));
            break;
        case 10:
            function.domain = field.string();
            break;
        case 11:
            function.attribute_defaults.emplace_back();
            II_TRY(parse_attribute(field.bytes, function.attribute_defaults.back(), 1));
            break;
        case 13:
            function.overload = field.string();
            break;
        default:
            break;
        }
    }
    return reader.status();
}

}  // namespace

DType dtype_from_onnx(std::int32_t data_type) noexcept {
    switch (data_type) {
    case kOnnxFloat:
        return DType::kF32;
    case kOnnxUint8:
        return DType::kU8;
    case kOnnxInt8:
        return DType::kI8;
    case kOnnxUint16:
        return DType::kU16;
    case kOnnxInt16:
        return DType::kI16;
    case kOnnxInt32:
        return DType::kI32;
    case kOnnxInt64:
        return DType::kI64;
    case kOnnxBool:
        return DType::kBool;
    case kOnnxFloat16:
        return DType::kF16;
    case kOnnxDouble:
        return DType::kF64;
    case kOnnxUint32:
        return DType::kU32;
    case kOnnxUint64:
        return DType::kU64;
    case kOnnxBfloat16:
        return DType::kBF16;
    default:
        return DType::kInvalid;
    }
}

std::int32_t dtype_to_onnx(DType dtype) noexcept {
    for (std::int32_t t = kOnnxFloat; t <= kOnnxBfloat16; ++t) {
        if (dtype_from_onnx(t) == dtype && dtype != DType::kInvalid) {
            return t;
        }
    }
    return kOnnxUndefined;
}

std::string onnx_type_name(std::int32_t data_type) {
    static constexpr const char* kNames[] = {"UNDEFINED", "FLOAT",  "UINT8",     "INT8",       "UINT16",  "INT16",
                                             "INT32",     "INT64",  "STRING",    "BOOL",       "FLOAT16", "DOUBLE",
                                             "UINT32",    "UINT64", "COMPLEX64", "COMPLEX128", "BFLOAT16"};
    if (data_type >= 0 && data_type < static_cast<std::int32_t>(std::size(kNames))) {
        return kNames[data_type];
    }
    switch (data_type) {
    case 17:
    case 18:
    case 19:
    case 20:
        return "FLOAT8";
    case 21:
        return "UINT4";
    case 22:
        return "INT4";
    case 23:
        return "FLOAT4E2M1";
    default:
        return std::format("type {}", data_type);
    }
}

const OnnxAttribute* OnnxNode::find_attribute(std::string_view attr_name) const noexcept {
    for (const OnnxAttribute& attr : attributes) {
        if (attr.name == attr_name) {
            return &attr;
        }
    }
    return nullptr;
}

std::int64_t OnnxModel::default_opset() const noexcept {
    for (const OnnxOpset& opset : opset_import) {
        if (opset.domain.empty() || opset.domain == "ai.onnx") {
            return opset.version;
        }
    }
    return 0;
}

Result<OnnxModel> parse_model(std::span<const std::byte> bytes) {
    OnnxModel model;
    bool has_graph = false;
    pb::Reader reader(bytes);
    pb::Field field;
    while (reader.next(field)) {
        switch (field.number) {
        case 1:
            model.ir_version = field.int64();
            break;
        case 2:
            model.producer_name = field.string();
            break;
        case 3:
            model.producer_version = field.string();
            break;
        case 4:
            model.domain = field.string();
            break;
        case 5:
            model.model_version = field.int64();
            break;
        case 7:
            if (field.wire != pb::WireType::kLength) {
                return errors::data_corrupted("ONNX model: malformed graph field");
            }
            has_graph = true;
            II_TRY(parse_graph_into(field.bytes, model.graph, 0));
            break;
        case 8:
            model.opset_import.emplace_back();
            II_TRY(parse_opset(field.bytes, model.opset_import.back()));
            break;
        case 14: {
            std::pair<std::string, std::string> entry;
            II_TRY(parse_string_entry(field.bytes, entry));
            model.metadata.push_back(std::move(entry));
            break;
        }
        case 25:
            model.functions.emplace_back();
            II_TRY(parse_function(field.bytes, model.functions.back()));
            break;
        default:
            break;
        }
    }
    II_TRY(reader.status());
    if (!has_graph) {
        return errors::data_corrupted("not an ONNX model (no graph)");
    }
    return model;
}

Result<OnnxTensor> parse_tensor(std::span<const std::byte> bytes) {
    OnnxTensor tensor;
    II_TRY(parse_tensor_into(bytes, tensor));
    return tensor;
}

// ---------------------------------------------------------------------------------------------------------------
// External data

struct ExternalData::File {
    platform::MappedFile mapping;
};

Result<std::pair<std::span<const std::byte>, std::shared_ptr<const void>>>
ExternalData::read(std::string_view location, std::int64_t offset, std::int64_t length) {
    // The location must stay inside the model directory: no absolute paths, no "..".
    const std::filesystem::path relative = std::filesystem::path(std::u8string(location.begin(), location.end()));
    if (location.empty() || relative.is_absolute() || relative.has_root_name() || relative.has_root_directory()) {
        return errors::data_corrupted("external data location '{}' must be a relative path", location);
    }
    for (const auto& part : relative) {
        if (part == "..") {
            return errors::data_corrupted("external data location '{}' leaves the model directory", location);
        }
    }
    std::shared_ptr<File> file;
    for (const auto& [name, open] : files_) {
        if (name == location) {
            file = open;
        }
    }
    if (!file) {
        file = std::make_shared<File>();
        Result<platform::MappedFile> mapping = platform::MappedFile::open(base_dir_ / relative);
        if (!mapping.ok()) {
            Status status = mapping.status();
            return status.add_context(std::format("external data '{}'", location));
        }
        file->mapping = std::move(*mapping);
        files_.emplace_back(std::string(location), file);
    }
    const std::size_t size = file->mapping.size();
    if (offset < 0 || static_cast<std::uint64_t>(offset) > size) {
        return errors::data_corrupted("external data '{}': offset {} is outside of the file ({} bytes)", location,
                                      offset, size);
    }
    const std::size_t available = size - static_cast<std::size_t>(offset);
    const std::size_t count = length < 0 ? available : static_cast<std::size_t>(length);
    if (length >= 0 && static_cast<std::uint64_t>(length) > available) {
        return errors::data_corrupted("external data '{}': {} bytes at offset {} exceed the file ({} bytes)", location,
                                      length, offset, size);
    }
    return std::pair{file->mapping.bytes().subspan(static_cast<std::size_t>(offset), count),
                     std::static_pointer_cast<const void>(file)};
}

// ---------------------------------------------------------------------------------------------------------------
// Tensor conversion

namespace {

/// Number of elements of `dims`, or nullopt on negative dimensions or overflow.
std::optional<std::int64_t> element_count(std::span<const std::int64_t> dims) {
    std::int64_t count = 1;
    for (const std::int64_t d : dims) {
        if (d < 0) {
            return std::nullopt;
        }
        if (d != 0 && count > std::numeric_limits<std::int64_t>::max() / d) {
            return std::nullopt;
        }
        count *= d;
    }
    return count;
}

template <class To, class From>
void convert_values(const std::vector<From>& from, void* out) {
    auto* dst = static_cast<To*>(out);
    for (std::size_t i = 0; i < from.size(); ++i) {
        if constexpr (std::is_same_v<To, f16> || std::is_same_v<To, bf16>) {
            dst[i] = To::from_bits(static_cast<std::uint16_t>(from[i]));
        } else if constexpr (std::is_same_v<To, bool>) {
            dst[i] = from[i] != 0;
        } else {
            dst[i] = static_cast<To>(from[i]);
        }
    }
}

}  // namespace

Result<Tensor> to_tensor(const OnnxTensor& proto, const std::shared_ptr<const void>& keep_alive,
                         ExternalData* external) {
    const DType dtype = dtype_from_onnx(proto.data_type);
    if (dtype == DType::kInvalid) {
        return errors::unsupported("tensor '{}' has element type {}, which ii does not support", proto.name,
                                   onnx_type_name(proto.data_type));
    }
    if (proto.has_segment) {
        return errors::unsupported("tensor '{}' is segmented", proto.name);
    }
    if (proto.dims.size() > static_cast<std::size_t>(kMaxRank)) {
        return errors::unsupported("tensor '{}' has rank {}; ii supports at most {}", proto.name, proto.dims.size(),
                                   kMaxRank);
    }
    const std::optional<std::int64_t> count = element_count(proto.dims);
    if (!count || *count > (std::int64_t{1} << 40)) {
        return errors::data_corrupted("tensor '{}' has invalid dimensions", proto.name);
    }
    Shape shape;
    for (const std::int64_t d : proto.dims) {
        shape.push_back(d);
    }
    const std::size_t nbytes = storage_size(dtype, *count);

    // Raw bytes: inline raw_data or external data.
    std::span<const std::byte> raw;
    std::shared_ptr<const void> owner = keep_alive;
    bool has_raw = proto.has_raw_data;
    if (proto.external) {
        if (external == nullptr) {
            return errors::unsupported("tensor '{}' uses external data, which needs the model's file path", proto.name);
        }
        std::string location;
        std::int64_t offset = 0;
        std::int64_t length = -1;
        for (const auto& [key, value] : proto.external_data) {
            if (key == "location") {
                location = value;
            } else if (key == "offset" || key == "length") {
                const std::optional<std::int64_t> parsed = [&]() -> std::optional<std::int64_t> {
                    std::int64_t v = 0;
                    if (value.empty()) {
                        return std::nullopt;
                    }
                    for (const char c : value) {
                        if (c < '0' || c > '9' || v > (std::numeric_limits<std::int64_t>::max() - 9) / 10) {
                            return std::nullopt;
                        }
                        v = v * 10 + (c - '0');
                    }
                    return v;
                }();
                if (!parsed) {
                    return errors::data_corrupted("tensor '{}': bad external data {} '{}'", proto.name, key, value);
                }
                (key == "offset" ? offset : length) = *parsed;
            }
        }
        II_ASSIGN_OR_RETURN(auto data, external->read(location, offset, length));
        raw = data.first;
        owner = std::move(data.second);
        has_raw = true;
    } else if (proto.has_raw_data) {
        raw = proto.raw_data;
    }

    if (has_raw) {
        if (raw.size() != nbytes) {
            return errors::data_corrupted("tensor '{}' ({}{}) has {} bytes of data, expected {}", proto.name,
                                          to_string(dtype), shape.to_string(), raw.size(), nbytes);
        }
        const auto address = reinterpret_cast<std::uintptr_t>(raw.data());
        if (owner != nullptr && address % kDefaultAlignment == 0 && nbytes != 0) {
            auto buffer = Buffer::wrap(const_cast<std::byte*>(raw.data()), nbytes, owner);
            return Tensor::view(std::move(buffer), dtype, shape, contiguous_strides(shape));
        }
        return Tensor::from_bytes(dtype, shape, raw.data(), raw.size());
    }

    II_ASSIGN_OR_RETURN(Tensor tensor, Tensor::zeros(dtype, shape));
    const auto expect = [&](std::size_t size) -> Status {
        if (size != static_cast<std::size_t>(*count)) {
            return errors::data_corrupted("tensor '{}' ({}{}) has {} values, expected {}", proto.name, to_string(dtype),
                                          shape.to_string(), size, *count);
        }
        return ok_status();
    };
    void* out = tensor.data();
    switch (proto.data_type) {
    case kOnnxFloat:
        II_TRY(expect(proto.float_data.size()));
        convert_values<float>(proto.float_data, out);
        break;
    case kOnnxDouble:
        II_TRY(expect(proto.double_data.size()));
        convert_values<double>(proto.double_data, out);
        break;
    case kOnnxInt64:
        II_TRY(expect(proto.int64_data.size()));
        convert_values<std::int64_t>(proto.int64_data, out);
        break;
    case kOnnxUint32:
        II_TRY(expect(proto.uint64_data.size()));
        convert_values<std::uint32_t>(proto.uint64_data, out);
        break;
    case kOnnxUint64:
        II_TRY(expect(proto.uint64_data.size()));
        convert_values<std::uint64_t>(proto.uint64_data, out);
        break;
    case kOnnxInt32:
        II_TRY(expect(proto.int32_data.size()));
        convert_values<std::int32_t>(proto.int32_data, out);
        break;
    case kOnnxInt16:
        II_TRY(expect(proto.int32_data.size()));
        convert_values<std::int16_t>(proto.int32_data, out);
        break;
    case kOnnxInt8:
        II_TRY(expect(proto.int32_data.size()));
        convert_values<std::int8_t>(proto.int32_data, out);
        break;
    case kOnnxUint16:
        II_TRY(expect(proto.int32_data.size()));
        convert_values<std::uint16_t>(proto.int32_data, out);
        break;
    case kOnnxUint8:
        II_TRY(expect(proto.int32_data.size()));
        convert_values<std::uint8_t>(proto.int32_data, out);
        break;
    case kOnnxBool:
        II_TRY(expect(proto.int32_data.size()));
        convert_values<bool>(proto.int32_data, out);
        break;
    case kOnnxFloat16:
        II_TRY(expect(proto.int32_data.size()));
        convert_values<f16>(proto.int32_data, out);
        break;
    case kOnnxBfloat16:
        II_TRY(expect(proto.int32_data.size()));
        convert_values<bf16>(proto.int32_data, out);
        break;
    default:
        return errors::unsupported("tensor '{}': element type {}", proto.name, onnx_type_name(proto.data_type));
    }
    return tensor;
}

}  // namespace ii::formats::onnx
