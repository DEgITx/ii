// SPDX-License-Identifier: MIT
/// @file
/// The subset of the ONNX protobuf schema (onnx/onnx.proto) that ii reads, decoded with the own wire-format reader.
/// Field numbers follow onnx.proto; unknown fields are skipped, so newer files load as long as they use known
/// constructs.
#pragma once

#include <ii/dtype.h>
#include <ii/status.h>
#include <ii/tensor.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ii::formats::onnx {

/// TensorProto.DataType.
enum OnnxDataType : std::int32_t {
    kOnnxUndefined = 0,
    kOnnxFloat = 1,
    kOnnxUint8 = 2,
    kOnnxInt8 = 3,
    kOnnxUint16 = 4,
    kOnnxInt16 = 5,
    kOnnxInt32 = 6,
    kOnnxInt64 = 7,
    kOnnxString = 8,
    kOnnxBool = 9,
    kOnnxFloat16 = 10,
    kOnnxDouble = 11,
    kOnnxUint32 = 12,
    kOnnxUint64 = 13,
    kOnnxComplex64 = 14,
    kOnnxComplex128 = 15,
    kOnnxBfloat16 = 16,
};

/// The ii dtype of an ONNX element type, or kInvalid when ii has none (strings, complex, float8, int4, ...).
DType dtype_from_onnx(std::int32_t data_type) noexcept;
/// The ONNX element type of an ii dtype (kOnnxUndefined for block-quantized types).
std::int32_t dtype_to_onnx(DType dtype) noexcept;
/// "FLOAT", "INT64", ... for messages.
std::string onnx_type_name(std::int32_t data_type);

struct OnnxTensor {
    std::string name;
    std::int32_t data_type = kOnnxUndefined;
    std::vector<std::int64_t> dims;
    bool has_raw_data = false;
    std::span<const std::byte> raw_data;  ///< points into the parsed file
    std::vector<float> float_data;
    std::vector<std::int32_t> int32_data;
    std::vector<std::int64_t> int64_data;
    std::vector<double> double_data;
    std::vector<std::uint64_t> uint64_data;
    std::size_t string_count = 0;
    bool external = false;  ///< data_location == EXTERNAL
    std::vector<std::pair<std::string, std::string>> external_data;
    bool has_segment = false;
};

struct OnnxDim {
    bool has_value = false;
    std::int64_t value = 0;
    std::string param;  ///< symbolic name (may be empty: unknown dimension)
};

struct OnnxValueInfo {
    std::string name;
    bool is_tensor = false;  ///< a tensor type (sequences, maps, optionals are not)
    std::int32_t elem_type = kOnnxUndefined;
    bool has_shape = false;
    std::vector<OnnxDim> shape;
};

struct OnnxGraph;

/// AttributeProto.AttributeType.
enum OnnxAttributeType : std::int32_t {
    kAttrUndefined = 0,
    kAttrFloat = 1,
    kAttrInt = 2,
    kAttrString = 3,
    kAttrTensor = 4,
    kAttrGraph = 5,
    kAttrFloats = 6,
    kAttrInts = 7,
    kAttrStrings = 8,
    kAttrTensors = 9,
    kAttrGraphs = 10,
    kAttrSparseTensor = 11,
    kAttrSparseTensors = 12,
    kAttrTypeProto = 13,
    kAttrTypeProtos = 14,
};

struct OnnxAttribute {
    std::string name;
    std::string ref_attr_name;  ///< inside functions: refers to an attribute of the calling node
    std::int32_t type = kAttrUndefined;
    float f = 0;
    std::int64_t i = 0;
    std::string s;
    std::shared_ptr<OnnxTensor> t;
    std::shared_ptr<OnnxGraph> g;
    std::vector<float> floats;
    std::vector<std::int64_t> ints;
    std::vector<std::string> strings;
    std::vector<OnnxTensor> tensors;
    std::vector<std::shared_ptr<OnnxGraph>> graphs;
};

struct OnnxNode {
    std::string name;
    std::string op_type;
    std::string domain;
    std::vector<std::string> inputs;   ///< "" for an omitted optional input
    std::vector<std::string> outputs;  ///< "" for an omitted optional output
    std::vector<OnnxAttribute> attributes;

    const OnnxAttribute* find_attribute(std::string_view attr_name) const noexcept;
};

struct OnnxGraph {
    std::string name;
    std::vector<OnnxNode> nodes;
    std::vector<OnnxTensor> initializers;
    std::size_t sparse_initializers = 0;
    std::vector<OnnxValueInfo> inputs;
    std::vector<OnnxValueInfo> outputs;
    std::vector<OnnxValueInfo> value_info;
};

struct OnnxOpset {
    std::string domain;  ///< "" (or "ai.onnx") for the default operator set
    std::int64_t version = 0;
};

/// A model-local function (FunctionProto); calls are inlined when importing.
struct OnnxFunction {
    std::string name;
    std::string domain;
    std::string overload;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<std::string> attributes;            ///< attribute names without defaults
    std::vector<OnnxAttribute> attribute_defaults;  ///< attributes with defaults
    std::vector<OnnxNode> nodes;
    std::vector<OnnxOpset> opset_import;
};

struct OnnxModel {
    std::int64_t ir_version = 0;
    std::vector<OnnxOpset> opset_import;
    std::string producer_name;
    std::string producer_version;
    std::string domain;
    std::int64_t model_version = 0;
    OnnxGraph graph;
    std::vector<OnnxFunction> functions;
    std::vector<std::pair<std::string, std::string>> metadata;

    /// The version of the default (ai.onnx) operator set, or 0 if not imported.
    std::int64_t default_opset() const noexcept;
};

/// Parses a serialized ModelProto. Tensor `raw_data` refers into `bytes`, which must outlive the result.
Result<OnnxModel> parse_model(std::span<const std::byte> bytes);

/// Parses a serialized TensorProto (e.g. the `input_0.pb` files of ONNX test data).
Result<OnnxTensor> parse_tensor(std::span<const std::byte> bytes);

/// Resolves external tensor data: maps `location` (relative to the model directory) and returns its bytes. The
/// returned keep-alive owns the memory.
class ExternalData {
public:
    explicit ExternalData(std::filesystem::path base_dir) : base_dir_(std::move(base_dir)) {}

    /// The bytes [offset, offset + length) of `location`; length < 0 means "to the end of the file".
    Result<std::pair<std::span<const std::byte>, std::shared_ptr<const void>>>
    read(std::string_view location, std::int64_t offset, std::int64_t length);

private:
    struct File;
    std::filesystem::path base_dir_;
    std::vector<std::pair<std::string, std::shared_ptr<File>>> files_;
};

/// Converts a TensorProto to a tensor. Raw data is referenced without copying when it is suitably aligned
/// (`keep_alive` then keeps the memory alive); otherwise it is copied. `external` resolves external data (may be
/// null: external tensors then fail).
Result<Tensor> to_tensor(const OnnxTensor& proto, const std::shared_ptr<const void>& keep_alive,
                         ExternalData* external);

}  // namespace ii::formats::onnx
