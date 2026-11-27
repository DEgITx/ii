// SPDX-License-Identifier: MIT
// ii-cuda elementwise operations: a single elementwise node or a fused_elementwise subgraph becomes a CUDA C++
// kernel (one expression per operation, evaluated in registers) compiled with NVRTC when the kernel is created.
// Identical programs share one compiled module. Floats compute in f32 (f64 in f64), integers in 64 bits; results
// follow the `cast` rules of the reference backend.
#include "backends/cuda/cuda_backend.h"

#include "ops/fused.h"

#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ii::cuda {

namespace {

constexpr int kMaxInputs = 16;

/// Compute types of the generated code.
enum class Kind : std::uint8_t { kFloat, kDouble, kBool, kSigned, kUnsigned };

std::optional<Kind> kind_of(DType dtype) noexcept {
    switch (dtype) {
    case DType::kF32:
    case DType::kF16:
    case DType::kBF16:
        return Kind::kFloat;
    case DType::kF64:
        return Kind::kDouble;
    case DType::kBool:
        return Kind::kBool;
    case DType::kI8:
    case DType::kI16:
    case DType::kI32:
    case DType::kI64:
        return Kind::kSigned;
    case DType::kU8:
    case DType::kU16:
    case DType::kU32:
    case DType::kU64:
        return Kind::kUnsigned;
    default:
        return std::nullopt;
    }
}

const char* compute_type(Kind kind) noexcept {
    switch (kind) {
    case Kind::kFloat:
        return "float";
    case Kind::kDouble:
        return "double";
    case Kind::kBool:
        return "bool";
    case Kind::kSigned:
        return "long long";
    case Kind::kUnsigned:
        return "unsigned long long";
    }
    return "";
}

const char* storage_type(DType dtype) noexcept {
    switch (dtype) {
    case DType::kF32:
        return "float";
    case DType::kF16:
    case DType::kBF16:
    case DType::kU16:
        return "unsigned short";
    case DType::kF64:
        return "double";
    case DType::kI8:
        return "signed char";
    case DType::kU8:
    case DType::kBool:
        return "unsigned char";
    case DType::kI16:
        return "short";
    case DType::kI32:
        return "int";
    case DType::kU32:
        return "unsigned int";
    case DType::kI64:
        return "long long";
    case DType::kU64:
        return "unsigned long long";
    default:
        return "";
    }
}

/// The compute value of a stored element.
std::string load_expr(DType dtype, const std::string& raw) {
    switch (dtype) {
    case DType::kF16:
        return std::format("h2f({})", raw);
    case DType::kBF16:
        return std::format("bf2f({})", raw);
    case DType::kBool:
        return std::format("({} != 0)", raw);
    default:
        return std::format("(({}){})", compute_type(*kind_of(dtype)), raw);
    }
}

/// Saturation bounds of integer storage types, as written in the generated code.
struct IntRange {
    const char* low;
    const char* high;
    const char* low_double;
    const char* high_double;
};

IntRange int_range(DType dtype) {
    switch (dtype) {
    case DType::kI8:
        return {"-128", "127", "-128.0", "127.0"};
    case DType::kU8:
        return {"0", "255", "0.0", "255.0"};
    case DType::kI16:
        return {"-32768", "32767", "-32768.0", "32767.0"};
    case DType::kU16:
        return {"0", "65535", "0.0", "65535.0"};
    case DType::kI32:
        return {"(-2147483647 - 1)", "2147483647", "-2147483648.0", "2147483647.0"};
    case DType::kU32:
        return {"0u", "4294967295u", "0.0", "4294967295.0"};
    case DType::kI64:
        return {"(-9223372036854775807ll - 1)", "9223372036854775807ll", "-9223372036854775808.0",
                "9223372036854775808.0"};
    default:  // kU64
        return {"0ull", "18446744073709551615ull", "0.0", "18446744073709551616.0"};
    }
}

/// Converts a compute value of `kind` to storage dtype `dtype` (the `cast` rules).
std::string store_expr(DType dtype, Kind kind, const std::string& value) {
    const std::string type = storage_type(dtype);
    switch (dtype) {
    case DType::kF32:
        return kind == Kind::kBool ? std::format("({} ? 1.0f : 0.0f)", value) : std::format("((float){})", value);
    case DType::kF16:
        return std::format("f2h({})", store_expr(DType::kF32, kind, value));
    case DType::kBF16:
        return std::format("f2bf({})", store_expr(DType::kF32, kind, value));
    case DType::kF64:
        return kind == Kind::kBool ? std::format("({} ? 1.0 : 0.0)", value) : std::format("((double){})", value);
    case DType::kBool:
        return std::format("(unsigned char)(({}) != 0)", value);
    default:
        break;
    }
    if (kind == Kind::kFloat || kind == Kind::kDouble) {
        const IntRange range = int_range(dtype);
        return std::format("sat<{}>((double)({}), {}, {}, ({}){}, ({}){})", type, value, range.low_double,
                           range.high_double, type, range.low, type, range.high);
    }
    return std::format("(({})({}))", type, value);  // integers wrap around
}

/// `value` (of `kind`) converted to dtype `to` and back to the compute type of `to`.
std::string cast_expr(Kind kind, const std::string& value, DType to) {
    if (to == DType::kF32 && kind == Kind::kFloat) {
        return value;
    }
    return load_expr(to, store_expr(to, kind, value));
}

std::string literal(Kind kind, double value) {
    if (std::isinf(value)) {
        const char* sign = value < 0 ? "-" : "";
        return kind == Kind::kFloat ? std::format("({}__int_as_float(0x7f800000))", sign)
                                    : std::format("({}__longlong_as_double(0x7ff0000000000000ll))", sign);
    }
    std::string text = std::format("{:.17g}", value);
    if (text.find_first_of(".e") == std::string::npos) {
        text += ".0";
    }
    return kind == Kind::kFloat ? std::format("((float){})", text) : text;
}

/// Name of a math function for the compute kind ("expf" for floats, "exp" for doubles).
std::string math(Kind kind, std::string_view name) {
    return kind == Kind::kFloat ? std::format("{}f", name) : std::string(name);
}

struct Operand {
    std::string expr;
    Kind kind;
    DType dtype;
};

/// The expression of one operation, or why it is not supported.
Result<std::string> op_expr(std::string_view op, const Attributes& attrs, std::span<const Operand> in, DType out) {
    const Kind k = in.empty() ? *kind_of(out) : in[0].kind;
    const bool real = k == Kind::kFloat || k == Kind::kDouble;
    const std::string& a = in.empty() ? std::string() : in[0].expr;
    const std::string b = in.size() > 1 ? in[1].expr : std::string();
    const std::string c = in.size() > 2 ? in[2].expr : std::string();
    const std::string zero = literal(k == Kind::kDouble ? Kind::kDouble : Kind::kFloat, 0.0);
    const std::string one = literal(k == Kind::kDouble ? Kind::kDouble : Kind::kFloat, 1.0);
    const char* nan = k == Kind::kDouble ? "__longlong_as_double(0x7ff8000000000000ll)" : "__int_as_float(0x7fc00000)";
    auto float_only = [&](std::string expr) -> Result<std::string> {
        if (!real) {
            return errors::unsupported("'{}' of an integer type", op);
        }
        return expr;
    };
    auto wrap = [&](const char* symbol) {
        return std::format("(long long)((unsigned long long)({}) {} (unsigned long long)({}))", a, symbol, b);
    };

    // --- unary
    if (op == "neg") {
        return real ? std::format("(-{})", a)
                    : (k == Kind::kSigned ? std::format("(long long)(0ull - (unsigned long long)({}))", a)
                                          : (k == Kind::kUnsigned ? std::format("(0ull - {})", a) : a));
    }
    if (op == "abs") {
        if (real) {
            return std::format("{}({})", math(k, "fabs"), a);
        }
        return k == Kind::kSigned
                   ? std::format("(({0}) < 0 ? (long long)(0ull - (unsigned long long)({0})) : ({0}))", a)
                   : a;
    }
    if (op == "sign") {
        if (real) {
            return std::format("(({0}) != ({0}) ? ({0}) : (({0}) > {1} ? {2} : (({0}) < {1} ? -{2} : {1})))", a, zero,
                               one);
        }
        if (k == Kind::kSigned) {
            return std::format("(({0}) > 0 ? 1ll : (({0}) < 0 ? -1ll : 0ll))", a);
        }
        return k == Kind::kUnsigned ? std::format("(({0}) > 0 ? 1ull : 0ull)", a) : a;
    }
    if (op == "relu") {
        if (real) {
            return std::format("((({0}) > {1} || ({0}) != ({0})) ? ({0}) : {1})", a, zero);
        }
        return k == Kind::kSigned ? std::format("(({0}) > 0 ? ({0}) : 0ll)", a) : a;
    }
    if (op == "not") {
        return std::format("(({})(!({})))", compute_type(k), a);
    }
    if (op == "exp" || op == "log" || op == "sqrt" || op == "sin" || op == "cos" || op == "tanh" || op == "erf" ||
        op == "floor" || op == "ceil") {
        return float_only(std::format("{}({})", math(k, op), a));
    }
    if (op == "round") {
        return float_only(std::format("{}({})", math(k, "rint"), a));
    }
    if (op == "rsqrt") {
        return float_only(std::format("({} / {}({}))", one, math(k, "sqrt"), a));
    }
    if (op == "recip") {
        return float_only(std::format("({} / ({}))", one, a));
    }
    if (op == "sigmoid") {
        return float_only(std::format("({1} / ({1} + {2}(-({0}))))", a, one, math(k, "exp")));
    }
    if (op == "silu") {
        return float_only(std::format("(({0}) / ({1} + {2}(-({0}))))", a, one, math(k, "exp")));
    }
    if (op == "gelu") {
        if (attrs.get_string("approximate") == "tanh") {
            return float_only(std::format("({2} * ({0}) * ({1} + {3}({4} * (({0}) + {5} * ({0}) * ({0}) * ({0})))))", a,
                                          one, literal(k, 0.5), math(k, "tanh"), literal(k, 0.79788456080286535588),
                                          literal(k, 0.044715)));
        }
        return float_only(std::format("({2} * ({0}) * ({1} + {3}(({0}) * {4})))", a, one, literal(k, 0.5),
                                      math(k, "erf"), literal(k, 0.70710678118654752440)));
    }

    // --- binary
    if (op == "add" || op == "sub" || op == "mul") {
        const char* symbol = op == "add" ? "+" : (op == "sub" ? "-" : "*");
        if (real || k == Kind::kUnsigned) {
            return std::format("(({}) {} ({}))", a, symbol, b);
        }
        if (k == Kind::kSigned) {
            return wrap(symbol);
        }
        return op == "add" ? std::format("(({}) || ({}))", a, b)
                           : (op == "sub" ? std::format("(({}) != ({}))", a, b) : std::format("(({}) && ({}))", a, b));
    }
    if (op == "div") {
        if (real) {
            return std::format("(({}) / ({}))", a, b);
        }
        if (k == Kind::kSigned) {
            return std::format("div_s({}, {})", a, b);
        }
        return k == Kind::kUnsigned ? std::format("(({1}) == 0 ? 0ull : ({0}) / ({1}))", a, b) : a;
    }
    if (op == "pow") {
        if (real) {
            return std::format("{}({}, {})", math(k, "pow"), a, b);
        }
        if (k == Kind::kSigned) {
            return std::format("pow_s({}, {})", a, b);
        }
        return k == Kind::kUnsigned ? std::format("pow_u({}, {})", a, b) : std::format("(({}) || !({}))", a, b);
    }
    if (op == "max" || op == "min") {
        const std::string pick = op == "max" ? std::format("(({0}) < ({1}) ? ({1}) : ({0}))", a, b)
                                             : std::format("(({1}) < ({0}) ? ({1}) : ({0}))", a, b);
        if (real) {
            return std::format("((({0}) != ({0}) || ({1}) != ({1})) ? {2} : {3})", a, b, nan, pick);
        }
        return pick;
    }
    if (op == "mod") {
        const bool fmod = attrs.get_bool("fmod");
        switch (k) {
        case Kind::kFloat:
            return std::format("mod_f({}, {}, {})", a, b, fmod ? "true" : "false");
        case Kind::kDouble:
            return std::format("mod_d({}, {}, {})", a, b, fmod ? "true" : "false");
        case Kind::kSigned:
            return std::format("mod_s({}, {}, {})", a, b, fmod ? "true" : "false");
        case Kind::kUnsigned:
            return std::format("(({1}) == 0 ? 0ull : ({0}) % ({1}))", a, b);
        case Kind::kBool:
            return std::string("false");
        }
    }
    if (op == "eq" || op == "ne" || op == "lt" || op == "le" || op == "gt" || op == "ge") {
        const char* symbol = op == "eq"   ? "=="
                             : op == "ne" ? "!="
                             : op == "lt" ? "<"
                             : op == "le" ? "<="
                             : op == "gt" ? ">"
                                          : ">=";
        return std::format("(({}) {} ({}))", a, symbol, b);
    }
    if (op == "and" || op == "or" || op == "xor") {
        const char* symbol = op == "and" ? "&&" : (op == "or" ? "||" : "!=");
        return std::format("((bool)({}) {} (bool)({}))", a, symbol, b);
    }

    // --- others
    if (op == "where") {
        return std::format("(({}) ? ({}) : ({}))", a, b, c);
    }
    if (op == "cast") {
        return cast_expr(k, a, out);
    }
    if (op == "clamp") {
        const double low = attrs.get_float("min");
        const double high = attrs.get_float("max");
        if (real) {
            return std::format("clamp_r({}, {}, {})", a, literal(k, low), literal(k, high));
        }
        if (k == Kind::kBool) {
            return errors::unsupported("'clamp' of bool");
        }
        // Like the reference: out of range values become the bounds rounded inward, converted to the dtype.
        const std::string low_value =
            load_expr(in[0].dtype, store_expr(in[0].dtype, Kind::kDouble, literal(Kind::kDouble, std::ceil(low))));
        const std::string high_value =
            load_expr(in[0].dtype, store_expr(in[0].dtype, Kind::kDouble, literal(Kind::kDouble, std::floor(high))));
        return std::format("((double)({0}) < {1} ? {2} : ((double)({0}) > {3} ? {4} : ({0})))", a,
                           literal(Kind::kDouble, low), low_value, literal(Kind::kDouble, high), high_value);
    }
    return errors::unsupported("no CUDA code for '{}'", op);
}

constexpr const char* kPreamble = R"(
typedef unsigned short u16;
__forceinline__ float h2f(u16 h) { float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }
__forceinline__ u16 f2h(float f) { u16 h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h; }
__forceinline__ float bf2f(u16 b) { return __uint_as_float(((unsigned)b) << 16); }
__forceinline__ u16 f2bf(float f) {
    unsigned u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (u16)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (u16)(u >> 16);
}
template <class T> __forceinline__ T sat(double v, double lo, double hi, T tlo, T thi) {
    if (v != v) return (T)0;
    const double t = trunc(v);
    if (t <= lo) return tlo;
    if (t >= hi) return thi;
    return (T)t;
}
__forceinline__ long long div_s(long long a, long long b) {
    if (b == 0) return 0;
    if (a == (-9223372036854775807ll - 1) && b == -1) return a;
    return a / b;
}
__forceinline__ long long pow_s(long long base, long long e) {
    if (e < 0) return base == 1 ? 1 : (base == -1 ? ((e % 2 == 0) ? 1 : -1) : 0);
    unsigned long long r = 1, b = (unsigned long long)base;
    while (e > 0) { if (e & 1) r *= b; b *= b; e >>= 1; }
    return (long long)r;
}
__forceinline__ unsigned long long pow_u(unsigned long long b, unsigned long long e) {
    unsigned long long r = 1;
    while (e > 0) { if (e & 1) r *= b; b *= b; e >>= 1; }
    return r;
}
__forceinline__ float mod_f(float a, float b, bool c) {
    float r = fmodf(a, b);
    if (!c && r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}
__forceinline__ double mod_d(double a, double b, bool c) {
    double r = fmod(a, b);
    if (!c && r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}
__forceinline__ long long mod_s(long long a, long long b, bool c) {
    if (b == 0 || b == -1) return 0;
    long long r = a % b;
    if (!c && r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}
template <class T> __forceinline__ T clamp_r(T x, T lo, T hi) {
    if (x != x) return x;
    const T m = x < lo ? lo : x;
    return hi < m ? hi : m;
}
struct Args {
    long long n;
    int rank;
    long long dims[8];
    long long strides[16][8];
    const void* in[16];
    void* out;
};
)";

/// The kernel arguments, laid out like `Args` in the generated code.
struct Args {
    long long n = 0;
    int rank = 0;
    long long dims[kMaxRank] = {};
    long long strides[kMaxInputs][kMaxRank] = {};
    const void* in[kMaxInputs] = {};
    void* out = nullptr;
};

/// A program: inputs, then one value per operation.
struct SourceNode {
    std::string op;
    const Attributes* attrs = nullptr;
    std::vector<int> inputs;
    DType dtype = DType::kInvalid;
};
struct Program {
    std::vector<DType> input_dtypes;
    std::vector<SourceNode> nodes;
    int result = 0;
    DType out_dtype = DType::kInvalid;
    std::unique_ptr<ir::Graph> storage;  // the parsed fused program the attributes point into
};

Status add_graph(Program& program);

Result<Program> program_of(const ir::Node& node) {
    Program program;
    program.out_dtype = node.output()->dtype();
    if (node.op_name() != ops::kFusedElementwise) {
        SourceNode single{std::string(node.op_name()), &node.attrs(), {}, node.output()->dtype()};
        for (int i = 0; i < node.num_inputs(); ++i) {
            if (node.input(i) == nullptr) {
                return errors::unsupported("absent inputs");
            }
            program.input_dtypes.push_back(node.input(i)->dtype());
            single.inputs.push_back(i);
        }
        program.nodes.push_back(std::move(single));
        program.result = node.num_inputs();
        return program;
    }
    II_ASSIGN_OR_RETURN(program.storage, ops::parse_fused_program(node));
    II_TRY(add_graph(program));
    return program;
}

/// Adds the inputs and operations of `program.storage`.
Status add_graph(Program& program) {
    std::unordered_map<const ir::Value*, int> ids;
    for (const ir::Value* input : program.storage->inputs()) {
        ids.emplace(input, static_cast<int>(program.input_dtypes.size()));
        program.input_dtypes.push_back(input->dtype());
    }
    for (const auto& inner : program.storage->nodes()) {
        SourceNode item{std::string(inner->op_name()), &inner->attrs(), {}, inner->output()->dtype()};
        for (const ir::Value* value : inner->inputs()) {
            if (value == nullptr || !ids.contains(value)) {
                return errors::unsupported("fused program uses constants or absent inputs");
            }
            item.inputs.push_back(ids.at(value));
        }
        ids.emplace(inner->output(), static_cast<int>(program.input_dtypes.size() + program.nodes.size()));
        program.nodes.push_back(std::move(item));
    }
    program.result = ids.at(program.storage->outputs()[0].value);
    return ok_status();
}

/// The CUDA source of a program.
Result<std::string> generate(const Program& program) {
    const int inputs = static_cast<int>(program.input_dtypes.size());
    if (inputs > kMaxInputs) {
        return errors::unsupported("more than {} inputs", kMaxInputs);
    }
    std::vector<Operand> values;
    std::string body;
    for (int i = 0; i < inputs; ++i) {
        const DType dtype = program.input_dtypes[static_cast<std::size_t>(i)];
        const std::optional<Kind> kind = kind_of(dtype);
        if (!kind) {
            return errors::unsupported("dtype {}", to_string(dtype));
        }
        body += std::format("        const {} v{} = {};\n", compute_type(*kind), i,
                            load_expr(dtype, std::format("((const {}*)a.in[{}])[o{}]", storage_type(dtype), i, i)));
        values.push_back(Operand{std::format("v{}", i), *kind, dtype});
    }
    for (const SourceNode& node : program.nodes) {
        const std::optional<Kind> kind = kind_of(node.dtype);
        if (!kind) {
            return errors::unsupported("dtype {}", to_string(node.dtype));
        }
        std::vector<Operand> operands;
        for (const int input : node.inputs) {
            operands.push_back(values[static_cast<std::size_t>(input)]);
        }
        II_ASSIGN_OR_RETURN(const std::string expr, op_expr(node.op, *node.attrs, operands, node.dtype));
        const int id = static_cast<int>(values.size());
        body += std::format("        const {} v{} = {};\n", compute_type(*kind), id, expr);
        values.push_back(Operand{std::format("v{}", id), *kind, node.dtype});
    }
    const Operand& result = values[static_cast<std::size_t>(program.result)];
    if (!kind_of(program.out_dtype)) {
        return errors::unsupported("dtype {}", to_string(program.out_dtype));
    }
    body += std::format("        (({}*)a.out)[i] = {};\n", storage_type(program.out_dtype),
                        store_expr(program.out_dtype, result.kind, result.expr));

    std::string offsets;
    for (int i = 0; i < inputs; ++i) {
        offsets += std::format("        I o{0} = 0;\n", i);
    }
    std::string nd;
    std::string flat;
    for (int i = 0; i < inputs; ++i) {
        nd += std::format("                o{0} += idx * (I)a.strides[{0}][d];\n", i);
        flat += std::format("            o{0} = i * (I)a.strides[{0}][0];\n", i);
    }
    return std::format(R"({}
template <class I, bool kNd>
__forceinline__ void body(const Args& a) {{
    for (I i = (I)blockIdx.x * (I)blockDim.x + (I)threadIdx.x; i < (I)a.n; i += (I)gridDim.x * (I)blockDim.x) {{
{}        if (kNd) {{
            I rest = i;
            for (int d = a.rank - 1; d >= 0; --d) {{
                const I size = (I)a.dims[d];
                const I idx = rest % size;
                rest /= size;
{}            }}
        }} else {{
{}        }}
{}    }}
}}
extern "C" __global__ void ew_flat32(Args a) {{ body<int, false>(a); }}
extern "C" __global__ void ew_nd32(Args a) {{ body<int, true>(a); }}
extern "C" __global__ void ew_flat64(Args a) {{ body<long long, false>(a); }}
extern "C" __global__ void ew_nd64(Args a) {{ body<long long, true>(a); }}
)",
                       kPreamble, offsets, nd, flat, body);
}

// --- NVRTC compilation, shared by identical programs ----------------------------------------------------------------

Result<std::string> compile(const Libraries& l, const std::string& source, int compute_capability) {
    nvrtcProgram program = nullptr;
    if (const nvrtcResult r = l.nvrtc_create(&program, source.c_str(), "ii_elementwise.cu", 0, nullptr, nullptr);
        r != NVRTC_SUCCESS) {
        return errors::internal("nvrtcCreateProgram: {}", l.nvrtc_error_string(r));
    }
    struct Destroy {
        const Libraries& l;
        nvrtcProgram& program;
        ~Destroy() { l.nvrtc_destroy(&program); }
    } destroy{l, program};
    const std::string arch = std::format("-arch=sm_{}", compute_capability);
    const char* options[] = {arch.c_str(), "-default-device", "--std=c++17"};
    const nvrtcResult result = l.nvrtc_compile(program, 3, options);
    if (result != NVRTC_SUCCESS) {
        std::size_t size = 0;
        l.nvrtc_log_size(program, &size);
        std::string log(size, '\0');
        l.nvrtc_log(program, log.data());
        return errors::internal("NVRTC failed ({}): {}\n{}", l.nvrtc_error_string(result), log, source);
    }
    std::size_t size = 0;
    l.nvrtc_cubin_size(program, &size);
    std::string cubin(size, '\0');
    l.nvrtc_cubin(program, cubin.data());
    return cubin;
}

Result<const ElementwiseModule*> module_for(const CudaDevice& device, const std::string& source) {
    static std::mutex mutex;
    static auto* modules = new std::unordered_map<std::string, ElementwiseModule>();  // never freed (like CUDA)
    const std::string key = std::format("{}\n{}", device.compute_capability(), source);
    {
        const std::lock_guard lock(mutex);
        if (const auto it = modules->find(key); it != modules->end()) {
            return &it->second;
        }
    }
    II_ASSIGN_OR_RETURN(const std::string cubin, compile(device.libraries(), source, device.compute_capability()));
    cudaLibrary_t library = nullptr;
    II_CUDA_TRY(cudaLibraryLoadData(&library, cubin.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
    ElementwiseModule module;
    II_CUDA_TRY(cudaLibraryGetKernel(&module.flat32, library, "ew_flat32"));
    II_CUDA_TRY(cudaLibraryGetKernel(&module.nd32, library, "ew_nd32"));
    II_CUDA_TRY(cudaLibraryGetKernel(&module.flat64, library, "ew_flat64"));
    II_CUDA_TRY(cudaLibraryGetKernel(&module.nd64, library, "ew_nd64"));
    const std::lock_guard lock(mutex);
    return &modules->emplace(key, module).first->second;  // a concurrent compilation of the same program may win
}

class ElementwiseKernel final : public Kernel {
public:
    ElementwiseKernel(const CudaDevice& device, const ElementwiseModule& module, std::vector<DType> input_dtypes,
                      DType out)
        : device_(device), module_(module), input_dtypes_(std::move(input_dtypes)), out_(out) {}

    bool can_run_in_place(int input) const override {
        return element_size(input_dtypes_[static_cast<std::size_t>(input)]) == element_size(out_);
    }

    Status run(const KernelContext& context) const override {
        if (context.queue == nullptr) {
            return errors::internal("ii-cuda kernel run without a device queue");
        }
        II_TRY(device_.activate());
        return launch_elementwise(module_, context.inputs, context.output(0), stream_of(context));
    }

private:
    const CudaDevice& device_;
    const ElementwiseModule& module_;
    std::vector<DType> input_dtypes_;
    DType out_;
};

Status check_elementwise(const ir::Node& node) {
    if (node.num_inputs() > kMaxInputs || node.output()->rank() > kMaxRank) {
        return errors::unsupported("too many inputs or dimensions");
    }
    II_ASSIGN_OR_RETURN(const Program program, program_of(node));
    II_ASSIGN_OR_RETURN(const std::string source, generate(program));
    (void)source;
    return ok_status();
}

Result<std::unique_ptr<Kernel>> create_elementwise(const ir::Node& node, const CudaDevice& device) {
    II_ASSIGN_OR_RETURN(const Program program, program_of(node));
    II_ASSIGN_OR_RETURN(const std::string source, generate(program));
    II_ASSIGN_OR_RETURN(const ElementwiseModule* module, module_for(device, source));
    return std::unique_ptr<Kernel>(
        std::make_unique<ElementwiseKernel>(device, *module, program.input_dtypes, program.out_dtype));
}

}  // namespace

Status launch_elementwise(const ElementwiseModule& module, std::span<const TensorView> inputs, const TensorView& y,
                          cudaStream_t stream) {
    const std::int64_t n = y.numel();
    if (n == 0) {
        return ok_status();
    }
    const int rank = y.rank();
    const int count = static_cast<int>(inputs.size());
    if (count > kMaxInputs || rank > kMaxRank) {
        return errors::internal("elementwise kernel with {} inputs of rank {}", count, rank);
    }

    // Element strides of every input in the output's index space (0 along broadcast dimensions).
    long long strides[kMaxInputs][kMaxRank] = {};
    for (int i = 0; i < count; ++i) {
        const Shape& shape = inputs[static_cast<std::size_t>(i)].shape;
        long long stride = 1;
        for (int d = shape.rank() - 1; d >= 0; --d) {
            const int od = rank - shape.rank() + d;
            strides[i][od] = shape[d] == 1 ? 0 : stride;
            stride *= shape[d];
        }
    }
    // Merge dimensions that are contiguous for every operand; drop dimensions of size 1.
    Args args;
    args.n = n;
    for (int d = 0; d < rank; ++d) {
        if (y.shape[d] == 1) {
            continue;
        }
        const int last = args.rank - 1;
        bool merge = last >= 0;
        for (int i = 0; i < count && merge; ++i) {
            merge = args.strides[i][last] == strides[i][d] * y.shape[d];
        }
        if (merge) {
            args.dims[last] *= y.shape[d];
            for (int i = 0; i < count; ++i) {
                args.strides[i][last] = strides[i][d];
            }
            continue;
        }
        args.dims[args.rank] = y.shape[d];
        for (int i = 0; i < count; ++i) {
            args.strides[i][args.rank] = strides[i][d];
        }
        ++args.rank;
    }
    for (int i = 0; i < count; ++i) {
        args.in[i] = inputs[static_cast<std::size_t>(i)].data;
    }
    args.out = y.data;

    const bool flat = args.rank <= 1;
    const bool small = n < (std::int64_t{1} << 31);
    const cudaKernel_t kernel = flat ? (small ? module.flat32 : module.flat64) : (small ? module.nd32 : module.nd64);
    const std::int64_t blocks = std::min<std::int64_t>((n + 255) / 256, std::int64_t{65535} * 8);
    void* params[] = {&args};
    return launched(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(static_cast<unsigned>(blocks)),
                                     dim3(256), params, 0, stream),
                    "elementwise kernel");
}

Result<const ElementwiseModule*> compile_epilogue(const ir::Node& node, const CudaDevice* device) {
    Program program;
    program.out_dtype = node.output()->dtype();
    II_ASSIGN_OR_RETURN(program.storage, ops::parse_epilogue_program(node));
    II_TRY(add_graph(program));
    II_ASSIGN_OR_RETURN(const std::string source, generate(program));
    if (device == nullptr) {
        return nullptr;
    }
    return module_for(*device, source);
}

void register_elementwise_kernels(CudaBackend& backend) {
    for (const char* op : {"neg",  "abs",     "sign", "exp",  "log",   "sqrt",  "rsqrt", "recip", "sin",  "cos",
                           "tanh", "sigmoid", "erf",  "relu", "floor", "ceil",  "round", "not",   "add",  "sub",
                           "mul",  "div",     "pow",  "max",  "min",   "mod",   "eq",    "ne",    "lt",   "le",
                           "gt",   "ge",      "and",  "or",   "xor",   "where", "cast",  "clamp", "gelu", "silu"}) {
        backend.add_kernel(op, create_elementwise, check_elementwise);
    }
    backend.add_kernel(ops::kFusedElementwise, create_elementwise, check_elementwise);
}

}  // namespace ii::cuda
