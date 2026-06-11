// Реализация математических ядер движка `ii`. См. engine/ops.h по составу и
// соглашениям. Приоритет — корректность и читаемость: это эталон, по
// которому проверяются «настоящие» бэкенды, а не их замена по скорости.

#include "engine/ops.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ii {

namespace {

// Дополнить форму s слева единицами до длины n (выравнивание справа, как
// в broadcasting NumPy/ONNX).
Shape pad_left(const Shape& s, int n) {
    Shape out(n, 1);
    int ns = static_cast<int>(s.size());
    for (int i = 0; i < ns; ++i) out[n - ns + i] = s[i];
    return out;
}

// Шаги по форме s в пространстве из n осей (s выровнена справа) с обнулением
// шага там, где размер оси == 1 — это и есть «растяжение» при broadcasting.
std::vector<std::int64_t> bcast_strides(const Shape& s, int n) {
    Shape ps = pad_left(s, n);
    std::vector<std::int64_t> st(n, 0);
    std::int64_t acc = 1;
    for (int i = n - 1; i >= 0; --i) {
        st[i] = (ps[i] == 1) ? 0 : acc;
        acc *= ps[i];
    }
    return st;
}

template <class F>
Tensor ewise(const Tensor& a, const Tensor& b, F f) {
    Shape os = broadcast_shape(a.shape, b.shape);
    int n = static_cast<int>(os.size());
    auto sa = bcast_strides(a.shape, n);
    auto sb = bcast_strides(b.shape, n);
    Tensor out(os);
    std::int64_t total = out.numel();
    for (std::int64_t lin = 0; lin < total; ++lin) {
        std::int64_t rem = lin, ia = 0, ib = 0;
        for (int d = n - 1; d >= 0; --d) {
            std::int64_t c = rem % os[d];
            rem /= os[d];
            ia += c * sa[d];
            ib += c * sb[d];
        }
        out.data[static_cast<std::size_t>(lin)] =
            f(a.data[static_cast<std::size_t>(ia)],
              b.data[static_cast<std::size_t>(ib)]);
    }
    return out;
}

template <class F>
Tensor map1(const Tensor& x, F f) {
    Tensor out(x.shape);
    for (std::size_t i = 0; i < x.data.size(); ++i) out.data[i] = f(x.data[i]);
    return out;
}

}  // namespace

// ---- broadcasting форм -----------------------------------------------------
Shape broadcast_shape(const Shape& a, const Shape& b) {
    int n = std::max(a.size(), b.size());
    Shape pa = pad_left(a, n), pb = pad_left(b, n), out(n, 1);
    for (int i = 0; i < n; ++i) {
        std::int64_t da = pa[i], db = pb[i];
        if (da == db || db == 1)      out[i] = da;
        else if (da == 1)             out[i] = db;
        else throw std::runtime_error(
            "broadcast: несовместимые формы " + shape_to_string(a) +
            " и " + shape_to_string(b));
    }
    return out;
}

// ---- поэлементные бинарные --------------------------------------------------
Tensor add(const Tensor& a, const Tensor& b) {
    return ewise(a, b, [](float x, float y) { return x + y; });
}
Tensor sub(const Tensor& a, const Tensor& b) {
    return ewise(a, b, [](float x, float y) { return x - y; });
}
Tensor mul(const Tensor& a, const Tensor& b) {
    return ewise(a, b, [](float x, float y) { return x * y; });
}
Tensor div(const Tensor& a, const Tensor& b) {
    return ewise(a, b, [](float x, float y) { return x / y; });
}

// ---- активации --------------------------------------------------------------
Tensor relu(const Tensor& x) {
    return map1(x, [](float v) { return v > 0.0f ? v : 0.0f; });
}
Tensor sigmoid(const Tensor& x) {
    return map1(x, [](float v) { return 1.0f / (1.0f + std::exp(-v)); });
}
Tensor silu(const Tensor& x) {
    return map1(x, [](float v) { return v / (1.0f + std::exp(-v)); });
}
Tensor tanh_(const Tensor& x) {
    return map1(x, [](float v) { return std::tanh(v); });
}
Tensor leaky_relu(const Tensor& x, float alpha) {
    return map1(x, [alpha](float v) { return v >= 0.0f ? v : alpha * v; });
}
Tensor clip(const Tensor& x, float lo, float hi) {
    return map1(x, [lo, hi](float v) { return v < lo ? lo : (v > hi ? hi : v); });
}
Tensor gelu(const Tensor& x) {
    // Точный GELU: 0.5*x*(1+erf(x/sqrt(2))).
    const float inv_sqrt2 = 0.7071067811865476f;
    return map1(x, [inv_sqrt2](float v) {
        return 0.5f * v * (1.0f + std::erf(v * inv_sqrt2));
    });
}
Tensor add_scalar(const Tensor& x, float v) {
    return map1(x, [v](float a) { return a + v; });
}
Tensor mul_scalar(const Tensor& x, float v) {
    return map1(x, [v](float a) { return a * v; });
}

// ---- softmax / layernorm ----------------------------------------------------
Tensor softmax(const Tensor& x, std::int64_t axis) {
    int ax = norm_axis(axis, x.ndim());
    if (ax < 0 || ax >= x.ndim())
        throw std::runtime_error("softmax: ось вне диапазона");
    std::int64_t A = x.shape[ax];
    std::int64_t inner = 1;
    for (int i = ax + 1; i < x.ndim(); ++i) inner *= x.shape[i];
    std::int64_t outer = 1;
    for (int i = 0; i < ax; ++i) outer *= x.shape[i];

    Tensor out(x.shape);
    for (std::int64_t o = 0; o < outer; ++o) {
        for (std::int64_t in = 0; in < inner; ++in) {
            std::int64_t base = (o * A) * inner + in;
            float m = -INFINITY;
            for (std::int64_t k = 0; k < A; ++k)
                m = std::max(m, x.data[static_cast<std::size_t>(base + k * inner)]);
            float sum = 0.0f;
            for (std::int64_t k = 0; k < A; ++k) {
                float e = std::exp(x.data[static_cast<std::size_t>(base + k * inner)] - m);
                out.data[static_cast<std::size_t>(base + k * inner)] = e;
                sum += e;
            }
            for (std::int64_t k = 0; k < A; ++k)
                out.data[static_cast<std::size_t>(base + k * inner)] /= sum;
        }
    }
    return out;
}

Tensor layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias,
                  std::int64_t axis, float eps) {
    int ax = norm_axis(axis, x.ndim());
    std::int64_t norm = 1;
    for (int i = ax; i < x.ndim(); ++i) norm *= x.shape[i];
    std::int64_t outer = x.numel() / norm;
    if (weight.numel() != norm)
        throw std::runtime_error("layer_norm: размер weight != нормируемого хвоста");
    bool has_bias = !bias.empty();
    if (has_bias && bias.numel() != norm)
        throw std::runtime_error("layer_norm: размер bias != нормируемого хвоста");

    Tensor out(x.shape);
    for (std::int64_t o = 0; o < outer; ++o) {
        std::int64_t base = o * norm;
        float mean = 0.0f;
        for (std::int64_t i = 0; i < norm; ++i)
            mean += x.data[static_cast<std::size_t>(base + i)];
        mean /= static_cast<float>(norm);
        float var = 0.0f;
        for (std::int64_t i = 0; i < norm; ++i) {
            float d = x.data[static_cast<std::size_t>(base + i)] - mean;
            var += d * d;
        }
        var /= static_cast<float>(norm);
        float inv = 1.0f / std::sqrt(var + eps);
        for (std::int64_t i = 0; i < norm; ++i) {
            float v = (x.data[static_cast<std::size_t>(base + i)] - mean) * inv;
            v *= weight.data[static_cast<std::size_t>(i)];
            if (has_bias) v += bias.data[static_cast<std::size_t>(i)];
            out.data[static_cast<std::size_t>(base + i)] = v;
        }
    }
    return out;
}

// ---- матумножение -----------------------------------------------------------
namespace {

// Одно 2-D умножение (M,K)x(K,N)->(M,N), аккумуляция в out по плоским
// смещениям. Никаких блокировок — эталон, важна понятность.
void matmul_2d(const float* A, const float* B, float* C,
               std::int64_t M, std::int64_t K, std::int64_t N) {
    for (std::int64_t i = 0; i < M; ++i) {
        for (std::int64_t k = 0; k < K; ++k) {
            float a = A[i * K + k];
            const float* brow = B + k * N;
            float* crow = C + i * N;
            for (std::int64_t j = 0; j < N; ++j) crow[j] += a * brow[j];
        }
    }
}

}  // namespace

Tensor matmul(const Tensor& a, const Tensor& b) {
    // 1-D промоутинг по семантике NumPy/ONNX.
    bool a1d = a.ndim() == 1, b1d = b.ndim() == 1;
    Shape as = a.shape, bs = b.shape;
    if (a1d) as.insert(as.begin(), 1);   // (K) -> (1,K)
    if (b1d) bs.push_back(1);            // (K) -> (K,1)
    if (as.size() < 2 || bs.size() < 2)
        throw std::runtime_error("matmul: нужны тензоры >= 1-D");

    std::int64_t M = as[as.size() - 2], K = as[as.size() - 1];
    std::int64_t K2 = bs[bs.size() - 2], N = bs[bs.size() - 1];
    if (K != K2)
        throw std::runtime_error("matmul: несовпадение внутренней размерности " +
                                 shape_to_string(a.shape) + " x " +
                                 shape_to_string(b.shape));

    Shape abatch(as.begin(), as.end() - 2), bbatch(bs.begin(), bs.end() - 2);
    Shape batch = broadcast_shape(abatch, bbatch);
    int nb = static_cast<int>(batch.size());
    auto sa = bcast_strides(abatch, nb);   // в единицах «матриц», не элементов
    auto sb = bcast_strides(bbatch, nb);
    std::int64_t nbatch = numel(batch);

    Shape os = batch;
    os.push_back(M);
    os.push_back(N);
    Tensor out(os);  // нули — matmul_2d аккумулирует

    for (std::int64_t bi = 0; bi < nbatch; ++bi) {
        std::int64_t rem = bi, ia = 0, ib = 0;
        for (int d = nb - 1; d >= 0; --d) {
            std::int64_t c = rem % batch[d];
            rem /= batch[d];
            ia += c * sa[d];
            ib += c * sb[d];
        }
        const float* Ap = a.data.data() + ia * M * K;
        const float* Bp = b.data.data() + ib * K * N;
        float*       Cp = out.data.data() + bi * M * N;
        matmul_2d(Ap, Bp, Cp, M, K, N);
    }

    // Снять оси, добавленные при промоутинге 1-D входов.
    if (b1d) os.erase(os.end() - 1);                 // убрать N
    if (a1d) os.erase(os.end() - (b1d ? 1 : 2));     // убрать M
    out.shape = os;
    return out;
}

Tensor gemm(const Tensor& a, const Tensor& b, const Tensor& c,
            float alpha, float beta, bool trans_a, bool trans_b) {
    if (a.ndim() != 2 || b.ndim() != 2)
        throw std::runtime_error("gemm: A и B должны быть 2-D");
    std::int64_t M = trans_a ? a.shape[1] : a.shape[0];
    std::int64_t K = trans_a ? a.shape[0] : a.shape[1];
    std::int64_t Kb = trans_b ? b.shape[1] : b.shape[0];
    std::int64_t N = trans_b ? b.shape[0] : b.shape[1];
    if (K != Kb) throw std::runtime_error("gemm: несовпадение K");

    Tensor out(Shape{M, N}, 0.0f);
    for (std::int64_t i = 0; i < M; ++i) {
        for (std::int64_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (std::int64_t k = 0; k < K; ++k) {
                float av = trans_a ? a.data[static_cast<std::size_t>(k * a.shape[1] + i)]
                                   : a.data[static_cast<std::size_t>(i * a.shape[1] + k)];
                float bv = trans_b ? b.data[static_cast<std::size_t>(j * b.shape[1] + k)]
                                   : b.data[static_cast<std::size_t>(k * b.shape[1] + j)];
                acc += av * bv;
            }
            out.data[static_cast<std::size_t>(i * N + j)] = alpha * acc;
        }
    }
    if (!c.empty()) {
        Tensor bc = mul_scalar(c, beta);
        out = add(out, bc);  // broadcast C к (M,N)
    }
    return out;
}

// ---- свёртка / пулинг -------------------------------------------------------
Tensor conv2d(const Tensor& x, const Tensor& w, const Tensor& bias,
              const Conv2DParams& p) {
    if (x.ndim() != 4 || w.ndim() != 4)
        throw std::runtime_error("conv2d: ожидаются 4-D x (NCHW) и w (M,C/g,kH,kW)");
    std::int64_t Nn = x.shape[0], C = x.shape[1], H = x.shape[2], W = x.shape[3];
    std::int64_t M = w.shape[0], Cg = w.shape[1], kH = w.shape[2], kW = w.shape[3];
    std::int64_t g = p.groups;
    if (C != Cg * g)
        throw std::runtime_error("conv2d: каналы входа != (C/groups)*groups");
    if (M % g != 0)
        throw std::runtime_error("conv2d: число фильтров не делится на groups");
    if (!bias.empty() && bias.numel() != M)
        throw std::runtime_error("conv2d: размер bias != числу фильтров");

    std::int64_t sH = p.stride[0], sW = p.stride[1];
    std::int64_t pT = p.pad[0], pL = p.pad[1], pB = p.pad[2], pR = p.pad[3];
    std::int64_t dH = p.dilation[0], dW = p.dilation[1];
    std::int64_t OH = (H + pT + pB - (dH * (kH - 1) + 1)) / sH + 1;
    std::int64_t OW = (W + pL + pR - (dW * (kW - 1) + 1)) / sW + 1;
    if (OH <= 0 || OW <= 0)
        throw std::runtime_error("conv2d: непозитивный выходной размер");

    std::int64_t Mg = M / g;  // фильтров на группу
    Tensor out(Shape{Nn, M, OH, OW}, 0.0f);

    for (std::int64_t n = 0; n < Nn; ++n) {
        for (std::int64_t gi = 0; gi < g; ++gi) {
            for (std::int64_t mo = 0; mo < Mg; ++mo) {
                std::int64_t m = gi * Mg + mo;
                float b = bias.empty() ? 0.0f
                                       : bias.data[static_cast<std::size_t>(m)];
                for (std::int64_t oy = 0; oy < OH; ++oy) {
                    for (std::int64_t ox = 0; ox < OW; ++ox) {
                        float acc = b;
                        for (std::int64_t ci = 0; ci < Cg; ++ci) {
                            std::int64_t c = gi * Cg + ci;
                            for (std::int64_t ky = 0; ky < kH; ++ky) {
                                std::int64_t iy = oy * sH + ky * dH - pT;
                                if (iy < 0 || iy >= H) continue;
                                for (std::int64_t kx = 0; kx < kW; ++kx) {
                                    std::int64_t ix = ox * sW + kx * dW - pL;
                                    if (ix < 0 || ix >= W) continue;
                                    float xv = x.data[static_cast<std::size_t>(
                                        ((n * C + c) * H + iy) * W + ix)];
                                    float wv = w.data[static_cast<std::size_t>(
                                        ((m * Cg + ci) * kH + ky) * kW + kx)];
                                    acc += xv * wv;
                                }
                            }
                        }
                        out.data[static_cast<std::size_t>(
                            ((n * M + m) * OH + oy) * OW + ox)] = acc;
                    }
                }
            }
        }
    }
    return out;
}

namespace {

// Общая обёртка пулинга: pick — редуктор (max/avg) по элементам окна.
template <class Init, class Acc, class Fin>
Tensor pool2d(const Tensor& x, const PoolParams& p, Init init, Acc acc, Fin fin) {
    if (x.ndim() != 4) throw std::runtime_error("pool2d: ожидается 4-D NCHW");
    std::int64_t Nn = x.shape[0], C = x.shape[1], H = x.shape[2], W = x.shape[3];
    std::int64_t kH = p.kernel[0], kW = p.kernel[1];
    std::int64_t sH = p.stride[0], sW = p.stride[1];
    std::int64_t pT = p.pad[0], pL = p.pad[1], pB = p.pad[2], pR = p.pad[3];
    std::int64_t OH = (H + pT + pB - kH) / sH + 1;
    std::int64_t OW = (W + pL + pR - kW) / sW + 1;
    if (OH <= 0 || OW <= 0) throw std::runtime_error("pool2d: непозитивный выход");

    Tensor out(Shape{Nn, C, OH, OW});
    for (std::int64_t n = 0; n < Nn; ++n)
      for (std::int64_t c = 0; c < C; ++c)
        for (std::int64_t oy = 0; oy < OH; ++oy)
          for (std::int64_t ox = 0; ox < OW; ++ox) {
            float a = init();
            std::int64_t cnt = 0;
            for (std::int64_t ky = 0; ky < kH; ++ky) {
                std::int64_t iy = oy * sH + ky - pT;
                for (std::int64_t kx = 0; kx < kW; ++kx) {
                    std::int64_t ix = ox * sW + kx - pL;
                    bool inside = (iy >= 0 && iy < H && ix >= 0 && ix < W);
                    if (inside) {
                        a = acc(a, x.data[static_cast<std::size_t>(
                                       ((n * C + c) * H + iy) * W + ix)]);
                        ++cnt;
                    } else if (p.count_include_pad) {
                        a = acc(a, 0.0f);
                        ++cnt;
                    }
                }
            }
            out.data[static_cast<std::size_t>(((n * C + c) * OH + oy) * OW + ox)] =
                fin(a, cnt);
          }
    return out;
}

}  // namespace

Tensor maxpool2d(const Tensor& x, const PoolParams& p) {
    return pool2d(x, p,
                  []() { return -INFINITY; },
                  [](float a, float v) { return std::max(a, v); },
                  [](float a, std::int64_t) { return a; });
}

Tensor avgpool2d(const Tensor& x, const PoolParams& p) {
    return pool2d(x, p,
                  []() { return 0.0f; },
                  [](float a, float v) { return a + v; },
                  [](float a, std::int64_t cnt) {
                      return cnt > 0 ? a / static_cast<float>(cnt) : 0.0f;
                  });
}

Tensor global_average_pool(const Tensor& x) {
    if (x.ndim() != 4) throw std::runtime_error("global_average_pool: 4-D NCHW");
    std::int64_t Nn = x.shape[0], C = x.shape[1], H = x.shape[2], W = x.shape[3];
    Tensor out(Shape{Nn, C, 1, 1});
    std::int64_t hw = H * W;
    for (std::int64_t n = 0; n < Nn; ++n)
        for (std::int64_t c = 0; c < C; ++c) {
            float s = 0.0f;
            std::int64_t base = (n * C + c) * hw;
            for (std::int64_t i = 0; i < hw; ++i)
                s += x.data[static_cast<std::size_t>(base + i)];
            out.data[static_cast<std::size_t>(n * C + c)] =
                s / static_cast<float>(hw);
        }
    return out;
}

// ---- перестановки формы -----------------------------------------------------
Tensor concat(const std::vector<const Tensor*>& xs, std::int64_t axis) {
    if (xs.empty()) throw std::runtime_error("concat: пустой список входов");
    int nd = xs[0]->ndim();
    int ax = norm_axis(axis, nd);
    Shape os = xs[0]->shape;
    std::int64_t sum = 0;
    for (const Tensor* t : xs) {
        if (t->ndim() != nd)
            throw std::runtime_error("concat: разный ndim входов");
        for (int d = 0; d < nd; ++d)
            if (d != ax && t->shape[d] != os[d])
                throw std::runtime_error("concat: формы расходятся не по оси склейки");
        sum += t->shape[ax];
    }
    os[ax] = sum;
    Tensor out(os);

    // outer = произведение осей до ax; copy_block — элементы одной «строки»
    // одного входа = (его размер по ax) * (произведение осей после ax).
    std::int64_t inner = 1;
    for (int d = ax + 1; d < nd; ++d) inner *= os[d];
    std::int64_t outer = 1;
    for (int d = 0; d < ax; ++d) outer *= os[d];

    for (std::int64_t o = 0; o < outer; ++o) {
        std::int64_t off = 0;
        for (const Tensor* t : xs) {
            std::int64_t blk = t->shape[ax] * inner;
            std::int64_t src = o * blk;
            std::int64_t dst = o * (sum * inner) + off;
            std::copy_n(t->data.begin() + src, blk, out.data.begin() + dst);
            off += blk;
        }
    }
    return out;
}

Tensor reshape(const Tensor& x, const Shape& new_shape) {
    Shape out = new_shape;
    std::int64_t known = 1;
    int infer = -1;
    for (std::size_t i = 0; i < out.size(); ++i) {
        if (out[i] == -1) {
            if (infer >= 0) throw std::runtime_error("reshape: больше одного -1");
            infer = static_cast<int>(i);
        } else if (out[i] == 0) {
            // ONNX allowzero=0: скопировать размерность входа на этой позиции.
            if (i >= x.shape.size())
                throw std::runtime_error("reshape: 0 вне ranga входа");
            out[i] = x.shape[i];
            known *= out[i];
        } else {
            known *= out[i];
        }
    }
    if (infer >= 0) {
        if (known == 0) throw std::runtime_error("reshape: деление на 0 при выводе оси");
        out[static_cast<std::size_t>(infer)] = x.numel() / known;
    }
    if (numel(out) != x.numel())
        throw std::runtime_error("reshape: число элементов не сохраняется " +
                                 shape_to_string(x.shape) + " -> " +
                                 shape_to_string(new_shape));
    Tensor r(out, x.data);  // raw-копия буфера, раскладка не меняется
    return r;
}

Tensor transpose(const Tensor& x, const std::vector<std::int64_t>& perm) {
    int nd = x.ndim();
    std::vector<std::int64_t> p = perm;
    if (p.empty()) {  // по умолчанию — реверс осей
        p.resize(nd);
        for (int i = 0; i < nd; ++i) p[i] = nd - 1 - i;
    }
    if (static_cast<int>(p.size()) != nd)
        throw std::runtime_error("transpose: длина perm != ndim");
    Shape os(nd);
    for (int i = 0; i < nd; ++i) os[i] = x.shape[static_cast<std::size_t>(p[i])];

    auto in_strides = row_major_strides(x.shape);
    auto out_strides = row_major_strides(os);
    Tensor out(os);
    std::int64_t total = out.numel();
    for (std::int64_t lin = 0; lin < total; ++lin) {
        // координаты в выходе -> исходное смещение через perm
        std::int64_t rem = lin, src = 0;
        for (int d = 0; d < nd; ++d) {
            std::int64_t c = (rem / out_strides[d]) % os[d];
            src += c * in_strides[static_cast<std::size_t>(p[d])];
        }
        out.data[static_cast<std::size_t>(lin)] =
            x.data[static_cast<std::size_t>(src)];
    }
    return out;
}

Tensor upsample_nearest(const Tensor& x, std::int64_t sh, std::int64_t sw) {
    if (x.ndim() != 4) throw std::runtime_error("upsample_nearest: 4-D NCHW");
    std::int64_t Nn = x.shape[0], C = x.shape[1], H = x.shape[2], W = x.shape[3];
    std::int64_t OH = H * sh, OW = W * sw;
    Tensor out(Shape{Nn, C, OH, OW});
    for (std::int64_t n = 0; n < Nn; ++n)
      for (std::int64_t c = 0; c < C; ++c)
        for (std::int64_t oy = 0; oy < OH; ++oy)
          for (std::int64_t ox = 0; ox < OW; ++ox) {
            std::int64_t iy = oy / sh, ix = ox / sw;
            out.data[static_cast<std::size_t>(((n * C + c) * OH + oy) * OW + ox)] =
                x.data[static_cast<std::size_t>(((n * C + c) * H + iy) * W + ix)];
          }
    return out;
}

// ---- доп. поэлементные ------------------------------------------------------
Tensor exp_(const Tensor& x)       { return map1(x, [](float v){ return std::exp(v); }); }
Tensor sqrt_(const Tensor& x)      { return map1(x, [](float v){ return std::sqrt(v); }); }
Tensor abs_(const Tensor& x)       { return map1(x, [](float v){ return std::fabs(v); }); }
Tensor neg(const Tensor& x)        { return map1(x, [](float v){ return -v; }); }
Tensor reciprocal(const Tensor& x) { return map1(x, [](float v){ return 1.0f / v; }); }
Tensor pow_(const Tensor& a, const Tensor& b) {
    return ewise(a, b, [](float x, float y){ return std::pow(x, y); });
}
Tensor min_(const Tensor& a, const Tensor& b) {
    return ewise(a, b, [](float x, float y){ return std::min(x, y); });
}
Tensor max_(const Tensor& a, const Tensor& b) {
    return ewise(a, b, [](float x, float y){ return std::max(x, y); });
}

// ---- resize (nearest, per-axis scales) -------------------------------------
Tensor resize_nearest(const Tensor& x, const std::vector<float>& scales) {
    int nd = x.ndim();
    if ((int)scales.size() != nd)
        throw std::runtime_error("resize_nearest: len(scales) != ndim");
    Shape os(nd);
    for (int i = 0; i < nd; ++i) {
        os[i] = (std::int64_t)std::floor(x.shape[i] * scales[i]);
        if (os[i] < 1) os[i] = 1;
    }
    auto in_str = row_major_strides(x.shape);
    auto out_str = row_major_strides(os);
    Tensor out(os);
    std::int64_t total = out.numel();
    for (std::int64_t lin = 0; lin < total; ++lin) {
        std::int64_t rem = lin, src = 0;
        for (int d = 0; d < nd; ++d) {
            std::int64_t oc = (rem / out_str[d]) % os[d];
            std::int64_t ic = (std::int64_t)std::floor(oc / scales[d]);
            if (ic < 0) ic = 0;
            if (ic >= x.shape[d]) ic = x.shape[d] - 1;
            src += ic * in_str[d];
        }
        out.data[static_cast<std::size_t>(lin)] = x.data[static_cast<std::size_t>(src)];
    }
    return out;
}

// ---- split -----------------------------------------------------------------
std::vector<Tensor> split(const Tensor& x, std::int64_t axis,
                          const std::vector<std::int64_t>& sizes) {
    int ax = norm_axis(axis, x.ndim());
    std::int64_t sum = 0;
    for (auto s : sizes) sum += s;
    if (sum != x.shape[ax])
        throw std::runtime_error("split: сумма частей != размеру оси");
    std::int64_t inner = 1;
    for (int d = ax + 1; d < x.ndim(); ++d) inner *= x.shape[d];
    std::int64_t outer = 1;
    for (int d = 0; d < ax; ++d) outer *= x.shape[d];
    std::int64_t axdim = x.shape[ax];

    std::vector<Tensor> outs;
    std::int64_t off = 0;
    for (std::int64_t part : sizes) {
        Shape os = x.shape;
        os[ax] = part;
        Tensor o(os);
        for (std::int64_t ot = 0; ot < outer; ++ot) {
            const float* src = x.data.data() + (ot * axdim + off) * inner;
            float* dst = o.data.data() + ot * part * inner;
            std::copy_n(src, part * inner, dst);
        }
        outs.push_back(std::move(o));
        off += part;
    }
    return outs;
}

// ---- slice -----------------------------------------------------------------
Tensor slice(const Tensor& x,
             const std::vector<std::int64_t>& starts,
             const std::vector<std::int64_t>& ends,
             const std::vector<std::int64_t>& axes,
             const std::vector<std::int64_t>& steps) {
    int nd = x.ndim();
    std::vector<std::int64_t> st(nd, 0), cnt(nd), stp(nd, 1);
    for (int i = 0; i < nd; ++i) cnt[i] = x.shape[i];  // по умолчанию весь диапазон

    std::size_t m = starts.size();
    for (std::size_t k = 0; k < m; ++k) {
        int ax = axes.empty() ? (int)k : norm_axis(axes[k], nd);
        std::int64_t dim = x.shape[ax];
        std::int64_t step = steps.empty() ? 1 : steps[k];
        if (step == 0) throw std::runtime_error("slice: нулевой шаг");
        std::int64_t s = starts[k], e = ends[k];
        if (s < 0) s += dim;
        if (e < 0) e += dim;
        std::int64_t count;
        if (step > 0) {
            s = std::min(std::max(s, (std::int64_t)0), dim);
            e = std::min(std::max(e, (std::int64_t)0), dim);
            count = (e > s) ? (e - s + step - 1) / step : 0;
        } else {
            s = std::min(std::max(s, (std::int64_t)0), dim - 1);
            e = std::min(std::max(e, (std::int64_t)-1), dim - 1);
            count = (s > e) ? (s - e + (-step) - 1) / (-step) : 0;
        }
        st[ax] = s;
        stp[ax] = step;
        cnt[ax] = count;
    }

    Shape os(nd);
    for (int i = 0; i < nd; ++i) os[i] = cnt[i];
    auto in_str = row_major_strides(x.shape);
    auto out_str = row_major_strides(os);
    Tensor out(os);
    std::int64_t total = out.numel();
    for (std::int64_t lin = 0; lin < total; ++lin) {
        std::int64_t rem = lin, src = 0;
        for (int d = 0; d < nd; ++d) {
            std::int64_t oc = (rem / out_str[d]) % (os[d] == 0 ? 1 : os[d]);
            src += (st[d] + oc * stp[d]) * in_str[d];
        }
        out.data[static_cast<std::size_t>(lin)] = x.data[static_cast<std::size_t>(src)];
    }
    return out;
}

// ---- gather ----------------------------------------------------------------
Tensor gather(const Tensor& x, const Tensor& indices, std::int64_t axis) {
    int ax = norm_axis(axis, x.ndim());
    std::int64_t axdim = x.shape[ax];
    std::int64_t inner = 1;
    for (int d = ax + 1; d < x.ndim(); ++d) inner *= x.shape[d];
    std::int64_t outer = 1;
    for (int d = 0; d < ax; ++d) outer *= x.shape[d];
    std::int64_t idxn = indices.numel();

    Shape os;
    for (int d = 0; d < ax; ++d) os.push_back(x.shape[d]);
    for (auto d : indices.shape) os.push_back(d);
    for (int d = ax + 1; d < x.ndim(); ++d) os.push_back(x.shape[d]);
    Tensor out(os);

    for (std::int64_t o = 0; o < outer; ++o)
        for (std::int64_t k = 0; k < idxn; ++k) {
            std::int64_t iv = (std::int64_t)std::llround(indices.data[(std::size_t)k]);
            if (iv < 0) iv += axdim;
            if (iv < 0 || iv >= axdim)
                throw std::runtime_error("gather: индекс вне диапазона");
            const float* src = x.data.data() + (o * axdim + iv) * inner;
            float* dst = out.data.data() + (o * idxn + k) * inner;
            std::copy_n(src, inner, dst);
        }
    return out;
}

// ---- unsqueeze / squeeze ---------------------------------------------------
Tensor unsqueeze(const Tensor& x, std::vector<std::int64_t> axes) {
    int out_nd = x.ndim() + (int)axes.size();
    for (auto& a : axes) if (a < 0) a += out_nd;
    std::vector<char> is_new(out_nd, 0);
    for (auto a : axes) {
        if (a < 0 || a >= out_nd) throw std::runtime_error("unsqueeze: ось вне диапазона");
        is_new[(std::size_t)a] = 1;
    }
    Shape os(out_nd);
    int si = 0;
    for (int i = 0; i < out_nd; ++i) os[i] = is_new[i] ? 1 : x.shape[si++];
    return Tensor(os, x.data);
}

Tensor squeeze(const Tensor& x, std::vector<std::int64_t> axes) {
    int nd = x.ndim();
    std::vector<char> drop(nd, 0);
    if (axes.empty()) {
        for (int i = 0; i < nd; ++i) if (x.shape[i] == 1) drop[i] = 1;
    } else {
        for (auto a : axes) { int ax = norm_axis(a, nd); drop[(std::size_t)ax] = 1; }
    }
    Shape os;
    for (int i = 0; i < nd; ++i) if (!drop[i]) os.push_back(x.shape[i]);
    return Tensor(os, x.data);
}

Tensor shape_of(const Tensor& x) {
    Shape os{(std::int64_t)x.ndim()};
    Tensor out(os);
    for (int i = 0; i < x.ndim(); ++i) out.data[(std::size_t)i] = (float)x.shape[i];
    return out;
}

// ---- редукции --------------------------------------------------------------
Tensor reduce(const Tensor& x, std::vector<std::int64_t> axes,
              bool keepdims, int kind) {
    int nd = x.ndim();
    std::vector<char> red(nd, 0);
    if (axes.empty()) { for (int i = 0; i < nd; ++i) red[i] = 1; }
    else { for (auto a : axes) red[(std::size_t)norm_axis(a, nd)] = 1; }

    // Компактная форма выхода (без редуцируемых осей) + её шаги.
    Shape compact;
    std::vector<int> pos(nd, -1);
    for (int i = 0; i < nd; ++i) if (!red[i]) { pos[i] = (int)compact.size(); compact.push_back(x.shape[i]); }
    auto out_str = row_major_strides(compact);

    std::int64_t out_n = numel(compact);
    float init = (kind == 2) ? -INFINITY : (kind == 3 ? INFINITY : 0.0f);
    std::vector<float> acc((std::size_t)out_n, init);

    auto in_str = row_major_strides(x.shape);
    std::int64_t total = x.numel();
    std::int64_t red_count = total / (out_n == 0 ? 1 : out_n);
    for (std::int64_t lin = 0; lin < total; ++lin) {
        std::int64_t rem = lin, off = 0;
        for (int d = 0; d < nd; ++d) {
            std::int64_t c = (rem / in_str[d]) % x.shape[d];
            if (!red[d]) off += c * out_str[pos[d]];
        }
        float v = x.data[static_cast<std::size_t>(lin)];
        float& a = acc[static_cast<std::size_t>(off)];
        switch (kind) {
            case 0: case 1: a += v; break;
            case 2: a = std::max(a, v); break;
            case 3: a = std::min(a, v); break;
        }
    }
    if (kind == 1) for (auto& a : acc) a /= (float)red_count;  // mean

    Shape os;
    if (keepdims) { os.resize(nd); for (int i = 0; i < nd; ++i) os[i] = red[i] ? 1 : x.shape[i]; }
    else          os = compact;
    return Tensor(os, std::move(acc));
}

}  // namespace ii
