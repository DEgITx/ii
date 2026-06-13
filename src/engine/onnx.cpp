// Загрузчик ONNX в граф движка `ii`.
//
// ONNX-модель — это сериализованный protobuf-message ModelProto. Чтобы
// движок оставался zero-dependency (без libprotobuf, без onnx — собирается
// на любой платформе), здесь свой минимальный ридер protobuf wire-формата
// и ровно те поля сообщений ONNX, что нужны для инференса:
//   ModelProto.graph -> GraphProto{node, initializer, input, output}
//   NodeProto{input, output, op_type, attribute}
//   TensorProto{dims, data_type, raw_data | *_data}  (веса -> float32)
//   ValueInfoProto -> формы входов/выходов
//
// Номера полей и enum'ы взяты из onnx.proto (стабильны между версиями).
// Раскладка тензоров в файле — row-major, как и в движке. Хост-порядок
// байт считаем little-endian (x86/ARM).

#include "engine/loader.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "half.h"  // inf::half_to_float — единая реализация на весь проект

namespace ii {

namespace {

// float16 -> float32 берём из общего half.h (раньше дублировалось здесь).
using inf::half_to_float;

// ---- минимальный protobuf-ридер -------------------------------------------
// Wire types: 0=varint, 1=fixed64, 2=length-delimited, 5=fixed32.
class Pb {
public:
    Pb(const std::uint8_t* p, const std::uint8_t* e) : p_(p), e_(e) {}

    bool eof() const { return p_ >= e_; }

    std::uint64_t varint() {
        std::uint64_t r = 0;
        int shift = 0;
        while (true) {
            if (p_ >= e_) throw std::runtime_error("protobuf: varint за границей");
            std::uint8_t b = *p_++;
            r |= (std::uint64_t)(b & 0x7F) << shift;
            if (!(b & 0x80)) break;
            shift += 7;
            if (shift > 63) throw std::runtime_error("protobuf: varint > 64 бит");
        }
        return r;
    }
    std::uint32_t fixed32() {
        if (p_ + 4 > e_) throw std::runtime_error("protobuf: fixed32 за границей");
        std::uint32_t v;
        std::memcpy(&v, p_, 4);
        p_ += 4;
        return v;
    }
    std::uint64_t fixed64() {
        if (p_ + 8 > e_) throw std::runtime_error("protobuf: fixed64 за границей");
        std::uint64_t v;
        std::memcpy(&v, p_, 8);
        p_ += 8;
        return v;
    }
    // Возвращает под-ридер на length-delimited регион (сообщение/строку/пакет).
    Pb sub() {
        std::uint64_t n = varint();
        if (p_ + n > e_) throw std::runtime_error("protobuf: длина за границей");
        Pb s(p_, p_ + n);
        p_ += n;
        return s;
    }
    std::string str() {
        std::uint64_t n = varint();
        if (p_ + n > e_) throw std::runtime_error("protobuf: строка за границей");
        std::string s((const char*)p_, (std::size_t)n);
        p_ += n;
        return s;
    }
    void skip(int wt) {
        switch (wt) {
            case 0: varint(); break;
            case 1: fixed64(); break;
            case 2: sub(); break;
            case 5: fixed32(); break;
            default: throw std::runtime_error("protobuf: неизвестный wire type");
        }
    }

private:
    const std::uint8_t* p_;
    const std::uint8_t* e_;
};

// Накопить repeated int64 (packed wt2 из varint'ов или одиночный wt0).
void read_ints(Pb& r, int wt, std::vector<std::int64_t>& out) {
    if (wt == 2) { Pb s = r.sub(); while (!s.eof()) out.push_back((std::int64_t)s.varint()); }
    else         { out.push_back((std::int64_t)r.varint()); }
}
// repeated float (packed wt2 из fixed32 или одиночный wt5).
void read_floats(Pb& r, int wt, std::vector<float>& out) {
    auto bf = [](std::uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; };
    if (wt == 2) { Pb s = r.sub(); while (!s.eof()) out.push_back(bf(s.fixed32())); }
    else         { out.push_back(bf(r.fixed32())); }
}

// ---- сырой TensorProto ----------------------------------------------------
struct RawTensor {
    std::string               name;
    std::vector<std::int64_t> dims;
    std::int32_t              data_type = 0;
    std::string               raw_data;
    std::vector<float>        float_data;
    std::vector<std::int64_t> int64_data;
    std::vector<std::int64_t> int32_data;  // храним расширенно
    std::vector<double>       double_data;
};

RawTensor parse_tensor(Pb r) {
    RawTensor t;
    while (!r.eof()) {
        std::uint64_t tag = r.varint();
        int field = (int)(tag >> 3), wt = (int)(tag & 7);
        switch (field) {
            case 1: read_ints(r, wt, t.dims); break;                  // dims
            case 2: t.data_type = (std::int32_t)r.varint(); break;    // data_type
            case 4: read_floats(r, wt, t.float_data); break;          // float_data
            case 5: read_ints(r, wt, t.int32_data); break;            // int32_data
            case 7: read_ints(r, wt, t.int64_data); break;            // int64_data
            case 8: t.name = r.str(); break;                          // name
            case 9: t.raw_data = r.str(); break;                      // raw_data
            case 10: {                                                // double_data
                if (wt == 2) { Pb s = r.sub();
                    while (!s.eof()) { std::uint64_t u = s.fixed64(); double d;
                        std::memcpy(&d, &u, 8); t.double_data.push_back(d); } }
                else { std::uint64_t u = r.fixed64(); double d;
                    std::memcpy(&d, &u, 8); t.double_data.push_back(d); }
                break;
            }
            default: r.skip(wt); break;
        }
    }
    return t;
}

// ONNX TensorProto.DataType.
enum {
    DT_FLOAT = 1, DT_UINT8 = 2, DT_INT8 = 3, DT_UINT16 = 4, DT_INT16 = 5,
    DT_INT32 = 6, DT_INT64 = 7, DT_BOOL = 9, DT_FLOAT16 = 10, DT_DOUBLE = 11,
    DT_UINT32 = 12, DT_UINT64 = 13
};

// RawTensor -> ii::Tensor (float32). Декодирует raw_data по dtype, иначе
// берёт типизированный *_data.
Tensor to_tensor(const RawTensor& rt) {
    Shape shape(rt.dims.begin(), rt.dims.end());
    std::int64_t n = numel(shape);
    Tensor out(shape);  // нули нужного размера

    if (!rt.raw_data.empty()) {
        const std::uint8_t* d = (const std::uint8_t*)rt.raw_data.data();
        std::size_t bytes = rt.raw_data.size();
        auto need = [&](std::size_t elem) {
            if (bytes < (std::size_t)n * elem)
                throw std::runtime_error("ONNX: raw_data короче, чем dims");
        };
        switch (rt.data_type) {
            case DT_FLOAT:  need(4); std::memcpy(out.data.data(), d, (std::size_t)n * 4); break;
            case DT_DOUBLE: need(8);
                for (std::int64_t i = 0; i < n; ++i) { double v; std::memcpy(&v, d + i * 8, 8);
                    out.data[i] = (float)v; } break;
            case DT_INT64:  need(8);
                for (std::int64_t i = 0; i < n; ++i) { std::int64_t v; std::memcpy(&v, d + i * 8, 8);
                    out.data[i] = (float)v; } break;
            case DT_UINT64: need(8);
                for (std::int64_t i = 0; i < n; ++i) { std::uint64_t v; std::memcpy(&v, d + i * 8, 8);
                    out.data[i] = (float)v; } break;
            case DT_INT32:  need(4);
                for (std::int64_t i = 0; i < n; ++i) { std::int32_t v; std::memcpy(&v, d + i * 4, 4);
                    out.data[i] = (float)v; } break;
            case DT_UINT32: need(4);
                for (std::int64_t i = 0; i < n; ++i) { std::uint32_t v; std::memcpy(&v, d + i * 4, 4);
                    out.data[i] = (float)v; } break;
            case DT_INT16:  need(2);
                for (std::int64_t i = 0; i < n; ++i) { std::int16_t v; std::memcpy(&v, d + i * 2, 2);
                    out.data[i] = (float)v; } break;
            case DT_UINT16: need(2);
                for (std::int64_t i = 0; i < n; ++i) { std::uint16_t v; std::memcpy(&v, d + i * 2, 2);
                    out.data[i] = (float)v; } break;
            case DT_FLOAT16: need(2);
                for (std::int64_t i = 0; i < n; ++i) { std::uint16_t v; std::memcpy(&v, d + i * 2, 2);
                    out.data[i] = half_to_float(v); } break;
            case DT_INT8:  need(1);
                for (std::int64_t i = 0; i < n; ++i) out.data[i] = (float)(std::int8_t)d[i];
                break;
            case DT_UINT8:
            case DT_BOOL:  need(1);
                for (std::int64_t i = 0; i < n; ++i) out.data[i] = (float)d[i];
                break;
            default:
                throw std::runtime_error("ONNX: неподдержанный data_type в raw_data: " +
                                         std::to_string(rt.data_type));
        }
        return out;
    }

    // Типизированные массивы (когда raw_data отсутствует).
    if (!rt.float_data.empty()) {
        for (std::int64_t i = 0; i < n && i < (std::int64_t)rt.float_data.size(); ++i)
            out.data[i] = rt.float_data[i];
    } else if (!rt.int64_data.empty()) {
        for (std::int64_t i = 0; i < n && i < (std::int64_t)rt.int64_data.size(); ++i)
            out.data[i] = (float)rt.int64_data[i];
    } else if (!rt.int32_data.empty()) {
        // int32_data в ONNX хранит также int8/uint8/int16/bool и (битами) float16.
        for (std::int64_t i = 0; i < n && i < (std::int64_t)rt.int32_data.size(); ++i) {
            if (rt.data_type == DT_FLOAT16)
                out.data[i] = half_to_float((std::uint16_t)rt.int32_data[i]);
            else
                out.data[i] = (float)rt.int32_data[i];
        }
    } else if (!rt.double_data.empty()) {
        for (std::int64_t i = 0; i < n && i < (std::int64_t)rt.double_data.size(); ++i)
            out.data[i] = (float)rt.double_data[i];
    }
    return out;
}

// ---- AttributeProto -------------------------------------------------------
struct RawAttr {
    std::string               name;
    float                     f = 0;
    std::int64_t              i = 0;
    std::string               s;
    std::vector<float>        floats;
    std::vector<std::int64_t> ints;
    RawTensor                 t;
    bool                      has_t = false;
};

RawAttr parse_attr(Pb r) {
    RawAttr a;
    while (!r.eof()) {
        std::uint64_t tag = r.varint();
        int field = (int)(tag >> 3), wt = (int)(tag & 7);
        switch (field) {
            case 1: a.name = r.str(); break;                 // name
            case 2: { std::uint32_t u = r.fixed32();         // f (float)
                      std::memcpy(&a.f, &u, 4); } break;
            case 3: a.i = (std::int64_t)r.varint(); break;   // i (int)
            case 4: a.s = r.str(); break;                    // s (bytes)
            case 5: a.t = parse_tensor(r.sub()); a.has_t = true; break;  // t (tensor)
            case 7: read_floats(r, wt, a.floats); break;     // floats
            case 8: read_ints(r, wt, a.ints); break;         // ints
            default: r.skip(wt); break;
        }
    }
    return a;
}

// Превратить RawAttr в ii::Attribute (что заполнено, то и кладём). Скаляры
// i/f кладём как одноэлементные списки — геттеры Node читают [0].
Attribute to_attr(const RawAttr& ra) {
    Attribute at;
    if (!ra.ints.empty())        at.ints = ra.ints;
    else                         at.ints = {ra.i};
    if (!ra.floats.empty())      at.floats = ra.floats;
    else                         at.floats = {ra.f};
    at.s = ra.s;
    return at;
}

// ---- NodeProto ------------------------------------------------------------
struct RawNode {
    std::string              op_type, name;
    std::vector<std::string> inputs, outputs;
    std::vector<RawAttr>     attrs;
};

RawNode parse_node(Pb r) {
    RawNode n;
    while (!r.eof()) {
        std::uint64_t tag = r.varint();
        int field = (int)(tag >> 3), wt = (int)(tag & 7);
        switch (field) {
            case 1: n.inputs.push_back(r.str()); break;      // input
            case 2: n.outputs.push_back(r.str()); break;     // output
            case 3: n.name = r.str(); break;                 // name
            case 4: n.op_type = r.str(); break;              // op_type
            case 5: n.attrs.push_back(parse_attr(r.sub())); break;  // attribute
            default: r.skip(wt); break;
        }
    }
    return n;
}

// ---- ValueInfoProto -> (имя, форма) ---------------------------------------
// Достаём форму из type.tensor_type.shape.dim[*].dim_value. Динамическая
// ось (dim_param без значения) кодируется как -1.
struct ValueInfo {
    std::string               name;
    std::vector<std::int64_t> shape;
};

std::vector<std::int64_t> parse_shape(Pb r) {  // TensorShapeProto
    std::vector<std::int64_t> dims;
    while (!r.eof()) {
        std::uint64_t tag = r.varint();
        int field = (int)(tag >> 3), wt = (int)(tag & 7);
        if (field == 1 && wt == 2) {            // dim (Dimension)
            Pb d = r.sub();
            std::int64_t val = -1;
            while (!d.eof()) {
                std::uint64_t dt = d.varint();
                int df = (int)(dt >> 3), dw = (int)(dt & 7);
                if (df == 1 && dw == 0) val = (std::int64_t)d.varint();  // dim_value
                else d.skip(dw);                                         // dim_param и пр.
            }
            dims.push_back(val);
        } else {
            r.skip(wt);
        }
    }
    return dims;
}

ValueInfo parse_value_info(Pb r) {
    ValueInfo vi;
    while (!r.eof()) {
        std::uint64_t tag = r.varint();
        int field = (int)(tag >> 3), wt = (int)(tag & 7);
        if (field == 1) {                       // name
            vi.name = r.str();
        } else if (field == 2 && wt == 2) {     // type (TypeProto)
            Pb tp = r.sub();
            while (!tp.eof()) {
                std::uint64_t tt = tp.varint();
                int tf = (int)(tt >> 3), tw = (int)(tt & 7);
                if (tf == 1 && tw == 2) {        // tensor_type (TypeProto.Tensor)
                    Pb tn = tp.sub();
                    while (!tn.eof()) {
                        std::uint64_t st = tn.varint();
                        int sf = (int)(st >> 3), sw = (int)(st & 7);
                        if (sf == 2 && sw == 2)  // shape (TensorShapeProto)
                            vi.shape = parse_shape(tn.sub());
                        else tn.skip(sw);
                    }
                } else {
                    tp.skip(tw);
                }
            }
        } else {
            r.skip(wt);
        }
    }
    return vi;
}

// ---- GraphProto -----------------------------------------------------------
void parse_graph(Pb r, Graph& g) {
    std::vector<RawNode>   nodes;
    std::vector<ValueInfo> inputs, outputs;

    while (!r.eof()) {
        std::uint64_t tag = r.varint();
        int field = (int)(tag >> 3), wt = (int)(tag & 7);
        switch (field) {
            case 1: nodes.push_back(parse_node(r.sub())); break;          // node
            case 5: {                                                     // initializer
                RawTensor rt = parse_tensor(r.sub());
                g.initializers[rt.name] = to_tensor(rt);
                break;
            }
            case 11: inputs.push_back(parse_value_info(r.sub())); break;  // input
            case 12: outputs.push_back(parse_value_info(r.sub())); break; // output
            default: r.skip(wt); break;
        }
    }

    // Входы графа = объявленные input'ы за вычетом тех, что являются
    // initializer'ами (старые opset'ы дублируют веса во input).
    std::unordered_set<std::string> init_names;
    for (const auto& kv : g.initializers) init_names.insert(kv.first);
    for (const ValueInfo& vi : inputs) {
        if (init_names.count(vi.name)) continue;
        g.input_names.push_back(vi.name);
        g.input_shapes[vi.name] = Shape(vi.shape.begin(), vi.shape.end());
    }
    for (const ValueInfo& vi : outputs) g.output_names.push_back(vi.name);

    // Ноды: Constant сворачиваем в initializer (его выход = константа),
    // остальные превращаем в ii::Node. Порядок нод в ONNX топологический.
    for (const RawNode& rn : nodes) {
        if (rn.op_type == "Constant" && !rn.outputs.empty()) {
            // ONNX Constant несёт значение ровно в одном из атрибутов: tensor
            // (value/sparse_value) или скаляр/список (value_float[s]/value_int[s]).
            // Поддерживаем числовые варианты — движок и так считает во float32.
            const std::string& out = rn.outputs[0];
            for (const RawAttr& a : rn.attrs) {
                if (a.name == "value" && a.has_t) {
                    g.initializers[out] = to_tensor(a.t);
                } else if (a.name == "value_float") {
                    g.initializers[out] = Tensor(Shape{}, std::vector<float>{a.f});
                } else if (a.name == "value_floats") {
                    g.initializers[out] =
                        Tensor(Shape{(std::int64_t)a.floats.size()}, a.floats);
                } else if (a.name == "value_int") {
                    g.initializers[out] =
                        Tensor(Shape{}, std::vector<float>{(float)a.i});
                } else if (a.name == "value_ints") {
                    std::vector<float> d(a.ints.begin(), a.ints.end());
                    g.initializers[out] =
                        Tensor(Shape{(std::int64_t)a.ints.size()}, std::move(d));
                }
                // value_string(s)/sparse_value не поддержаны (движок числовой).
            }
            continue;
        }
        Node n;
        n.op_type = rn.op_type;
        n.name = rn.name.empty() ? rn.op_type : rn.name;
        n.inputs = rn.inputs;
        n.outputs = rn.outputs;
        for (const RawAttr& a : rn.attrs) n.attrs[a.name] = to_attr(a);
        g.nodes.push_back(std::move(n));
    }
}

// ---- ModelProto -----------------------------------------------------------
void parse_model(Pb r, Graph& g) {
    bool found = false;
    while (!r.eof()) {
        std::uint64_t tag = r.varint();
        int field = (int)(tag >> 3), wt = (int)(tag & 7);
        if (field == 7 && wt == 2) {            // graph (GraphProto)
            parse_graph(r.sub(), g);
            found = true;
        } else {
            r.skip(wt);
        }
    }
    if (!found) throw std::runtime_error("ONNX: в ModelProto нет graph (поле 7)");
}

}  // namespace

bool parse_onnx(const void* data, std::size_t size, Graph& g, std::string& err) {
    g = Graph{};
    try {
        const std::uint8_t* p = (const std::uint8_t*)data;
        Pb r(p, p + size);
        parse_model(r, g);
    } catch (const std::exception& e) {
        err = std::string("ONNX: ") + e.what();
        return false;
    }
    if (g.nodes.empty()) {
        err = "ONNX: граф без операций";
        return false;
    }
    return true;
}

bool load_onnx(const std::string& path, Graph& g, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        err = "ONNX: не удалось открыть файл: " + path;
        return false;
    }
    std::vector<char> buf((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
    if (buf.empty()) {
        err = "ONNX: пустой файл: " + path;
        return false;
    }
    return parse_onnx(buf.data(), buf.size(), g, err);
}

}  // namespace ii
