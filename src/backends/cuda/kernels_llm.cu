// SPDX-License-Identifier: MIT
// ii-cuda kernels of language models: rope, slice_update (KV cache writes) and attention.
#include "backends/cuda/kernels_common.cuh"

namespace ii::cuda {

namespace {

// --- rope -----------------------------------------------------------------------------------------------------------

template <class T>
__global__ void rope_kernel(T* __restrict__ y, const T* __restrict__ x, const T* __restrict__ cos,
                            const T* __restrict__ sin, RopeParams p, std::int64_t rows) {
    const std::int64_t size = p.dims[p.rank - 1];
    const std::int64_t n = rows * size;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t row = i / size;
        const std::int64_t f = i % size;
        if (f >= 2 * p.half) {
            y[i] = x[i];
            continue;
        }
        std::int64_t rest = row;
        std::int64_t table = 0;
        for (int d = p.rank - 2; d >= 0; --d) {
            table += (rest % p.dims[d]) * p.table_strides[d];
            rest /= p.dims[d];
        }
        // Element f is member `second` of pair `pair`.
        const bool second = p.interleaved ? (f & 1) != 0 : f >= p.half;
        const std::int64_t pair = p.interleaved ? f / 2 : (second ? f - p.half : f);
        const std::int64_t partner = p.interleaved ? (second ? f - 1 : f + 1) : (second ? f - p.half : f + p.half);
        const float c = Traits<T>::load(cos[table + pair]);
        const float s = Traits<T>::load(sin[table + pair]);
        const T* in = x + row * size;
        const float self = Traits<T>::load(in[f]);
        const float other = Traits<T>::load(in[partner]);
        // y_a = a cos - b sin, y_b = b cos + a sin
        y[i] = Traits<T>::store(second ? self * c + other * s : self * c - other * s);
    }
}

// --- slice_update ---------------------------------------------------------------------------------------------------

template <class W>
__global__ void slice_update_kernel(std::uint8_t* y, const std::uint8_t* update, SliceUpdateParams p,
                                    std::int64_t words_per_row, std::int64_t n) {
    const std::int64_t start =
        p.start64 ? *static_cast<const std::int64_t*>(p.start) : *static_cast<const std::int32_t*>(p.start);
    if (start < 0 || start > p.axis_size - p.length) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            report_error(p.error, DeviceErrorCode::kSliceUpdateStart, start, p.axis_size, p.length);
        }
        return;
    }
    const W* src = reinterpret_cast<const W*>(update);
    W* dst = reinterpret_cast<W*>(y);
    const std::int64_t chunk = p.length * words_per_row;  // words of one outer slice of update
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t o = i / chunk;
        const std::int64_t rest = i % chunk;
        dst[(o * p.axis_size + start) * words_per_row + rest] = src[i];
    }
}

// --- attention ------------------------------------------------------------------------------------------------------

constexpr int kAttentionWarps = 8;
constexpr int kMaxChunks = kAttentionMaxHead / 32;
constexpr int kMaxSplits = 256;
/// Queries from which a prompt runs on tensor cores (flash_attention).
constexpr std::int64_t kFlashMinRows = 16;

/// How a launch covers the problem: a block handles `heads_per_block` heads of one KV head group and `rows` query
/// positions, for the keys of one split.
struct AttentionLaunch {
    int heads_per_block = 1;
    int head_chunks = 1;  ///< blocks per group of query heads
    int rows = 1;         ///< query positions per block
    int tile = 32;        ///< keys per tile
    int splits = 1;
    std::int64_t split_size = 0;  ///< keys per split (a multiple of the tile)
    std::int64_t blocks = 0;      ///< without the split
    std::size_t shared = 0;
};

AttentionLaunch plan(const AttentionParams& p, int sm_count) {
    AttentionLaunch l;
    const std::int64_t group = p.heads / p.kv_heads;
    l.heads_per_block = static_cast<int>(group < kAttentionWarps ? group : kAttentionWarps);
    l.head_chunks = static_cast<int>((group + l.heads_per_block - 1) / l.heads_per_block);
    l.rows = p.sq > 1 ? kAttentionWarps / l.heads_per_block : 1;
    l.tile = p.d <= 128 && p.dv <= 128 ? 32 : 16;
    const std::int64_t row_blocks = (p.sq + l.rows - 1) / l.rows;
    l.blocks = p.batch * p.kv_heads * row_blocks * l.head_chunks;
    const std::int64_t tiles = (p.skv + l.tile - 1) / l.tile;
    std::int64_t splits = 1;
    if (l.blocks < 2 * sm_count && tiles > 1) {
        // Few blocks (decoding): split the keys so that every multiprocessor gets work.
        splits = (2 * sm_count + l.blocks - 1) / l.blocks;
        splits = splits < tiles ? splits : tiles;
        splits = splits < kMaxSplits ? splits : kMaxSplits;
    }
    const std::int64_t tiles_per_split = (tiles + splits - 1) / splits;
    l.split_size = tiles_per_split * l.tile;
    l.splits = static_cast<int>(tiles_per_split == 0 ? 1 : (tiles + tiles_per_split - 1) / tiles_per_split);
    const int warps = l.heads_per_block * l.rows;
    const std::int64_t stride = p.d % 4 == 0 && p.dv % 4 == 0 ? p.d + 4 : p.d + 1;
    l.shared = sizeof(float) * static_cast<std::size_t>(l.tile * stride + l.tile * p.dv + warps * p.d + warps * l.tile);
    return l;
}

/// Keys [begin, end) that query position i attends among L keys in use.
struct KeyRange {
    std::int64_t begin;
    std::int64_t end;
};
__device__ __forceinline__ KeyRange key_range(const AttentionParams& p, std::int64_t i, std::int64_t used) {
    const std::int64_t position = i + used - p.sq;
    std::int64_t end = used;
    if (p.causal) {
        end = position + 1 < 0 ? 0 : (position + 1 > used ? used : position + 1);
    }
    std::int64_t begin = 0;
    if (p.window > 0) {
        const std::int64_t first = position - p.window + 1;
        begin = first < 0 ? 0 : (first > used ? used : first);
    }
    return {begin, end};
}

template <class TQ, class TKV>
__global__ void __launch_bounds__(kAttentionWarps * 32)
    attention_kernel(TQ* __restrict__ out, const TQ* __restrict__ q, const TKV* __restrict__ k,
                     const TKV* __restrict__ v, const void* __restrict__ mask, AttentionParams p, AttentionLaunch l,
                     float* __restrict__ partial) {
    extern __shared__ float shared[];
    const int warps = l.heads_per_block * l.rows;
    const int tile = l.tile;
    const std::int64_t d = p.d;
    const std::int64_t dv = p.dv;
    // K rows are padded (lanes read different keys): by 4 floats when 16-byte loads fit, else by 1.
    const bool vector = d % 4 == 0 && dv % 4 == 0;
    const std::int64_t stride = vector ? d + 4 : d + 1;
    float* ks = shared;              // [tile, stride]
    float* vs = ks + tile * stride;  // [tile, dv]
    float* qs = vs + tile * dv;      // [warps, d]
    float* ps = qs + warps * d;      // [warps, tile]
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    std::int64_t block = blockIdx.x;
    const int chunk = static_cast<int>(block % l.head_chunks);
    block /= l.head_chunks;
    const std::int64_t row_blocks = (p.sq + l.rows - 1) / l.rows;
    const std::int64_t row_block = block % row_blocks;
    block /= row_blocks;
    const std::int64_t hk = block % p.kv_heads;
    const std::int64_t b = block / p.kv_heads;
    const std::int64_t group = p.heads / p.kv_heads;
    const std::int64_t g = static_cast<std::int64_t>(chunk) * l.heads_per_block + warp % l.heads_per_block;
    const std::int64_t h = hk * group + g;
    const std::int64_t first_row = row_block * l.rows;
    const std::int64_t i = first_row + warp / l.heads_per_block;
    const bool active = warp < warps && g < group && i < p.sq;
    const std::int64_t row = ((b * p.heads + h) * p.sq + i);
    if (active) {
        for (std::int64_t e = lane; e < d; e += 32) {
            qs[warp * d + e] = Traits<TQ>::load(q[row * d + e]);
        }
    }

    std::int64_t used = p.skv;
    if (p.kv_len != nullptr) {
        used = p.kv_len64 ? *static_cast<const std::int64_t*>(p.kv_len) : *static_cast<const std::int32_t*>(p.kv_len);
        if (used < 0 || used > p.skv) {
            if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
                report_error(p.error, DeviceErrorCode::kKvLength, used, p.skv);
            }
            used = used < 0 ? 0 : p.skv;
        }
    }
    // Keys of the block: the union of its rows' ranges (they grow with i), within its split.
    const std::int64_t last_row = (first_row + l.rows < p.sq ? first_row + l.rows : p.sq) - 1;
    std::int64_t begin = key_range(p, first_row, used).begin;
    std::int64_t end = key_range(p, last_row, used).end;
    const std::int64_t split_begin = static_cast<std::int64_t>(blockIdx.y) * l.split_size;
    begin = begin > split_begin ? begin : split_begin;
    end = end < split_begin + l.split_size ? end : split_begin + l.split_size;
    const KeyRange mine = active ? key_range(p, i, used) : KeyRange{0, 0};
    const TKV* kh = k + (b * p.kv_heads + hk) * p.skv * d;
    const TKV* vh = v + (b * p.kv_heads + hk) * p.skv * dv;
    float m = -infinity();
    float sum = 0.0f;
    float acc[kMaxChunks];
#pragma unroll
    for (int c = 0; c < kMaxChunks; ++c) {
        acc[c] = 0.0f;
    }
    for (std::int64_t t0 = begin; t0 < end; t0 += tile) {
        __syncthreads();  // the previous tile is consumed
        const int count = static_cast<int>(end - t0 < tile ? end - t0 : tile);
        for (std::int64_t e = threadIdx.x; e < static_cast<std::int64_t>(count) * d; e += blockDim.x) {
            const std::int64_t j = e / d;
            ks[j * stride + e % d] = Traits<TKV>::load(kh[(t0 + j) * d + e % d]);
        }
        for (std::int64_t e = threadIdx.x; e < static_cast<std::int64_t>(count) * dv; e += blockDim.x) {
            vs[e] = Traits<TKV>::load(vh[t0 * dv + e]);
        }
        __syncthreads();
        if (!active) {
            continue;
        }
        const std::int64_t key = t0 + lane;
        float s = -infinity();
        if (lane < count && key >= mine.begin && key < mine.end) {
            const float* qr = qs + warp * d;
            const float* kr = ks + lane * stride;
            float dot = 0.0f;
            if (vector) {
                // Independent sums: no chain of d dependent multiply-adds.
                float d0 = 0.0f;
                float d1 = 0.0f;
                float d2 = 0.0f;
                float d3 = 0.0f;
                for (std::int64_t e = 0; e < d; e += 4) {
                    const float4 a = *reinterpret_cast<const float4*>(qr + e);
                    const float4 c = *reinterpret_cast<const float4*>(kr + e);
                    d0 = fmaf(a.x, c.x, d0);
                    d1 = fmaf(a.y, c.y, d1);
                    d2 = fmaf(a.z, c.z, d2);
                    d3 = fmaf(a.w, c.w, d3);
                }
                dot = (d0 + d1) + (d2 + d3);
            } else {
                for (std::int64_t e = 0; e < d; ++e) {
                    dot = fmaf(qr[e], kr[e], dot);
                }
            }
            s = dot * p.scale;
            if (p.softcap > 0.0f) {
                s = p.softcap * tanhf(s / p.softcap);
            }
            if (p.mask_kind != 0) {
                const std::int64_t offset =
                    b * p.mask_strides[0] + h * p.mask_strides[1] + i * p.mask_strides[2] + key * p.mask_strides[3];
                if (p.mask_kind == 1) {
                    s = static_cast<const bool*>(mask)[offset] ? s : -infinity();
                } else {
                    s += Traits<TQ>::load(static_cast<const TQ*>(mask)[offset]);
                }
            }
        }
        const float tile_max = warp_reduce(s, [](float a, float c) { return fmaxf(a, c); });
        const float m_new = fmaxf(m, tile_max);
        if (m_new == -infinity()) {
            continue;  // nothing attended yet
        }
        const float correction = expf(m - m_new);
        const float pj = s == -infinity() ? 0.0f : expf(s - m_new);
        sum = sum * correction + warp_reduce(pj, [](float a, float c) { return a + c; });
        if (lane < tile) {
            ps[warp * tile + lane] = pj;
        }
        __syncwarp();
#pragma unroll
        for (int c = 0; c < kMaxChunks; ++c) {
            acc[c] *= correction;
        }
        // Only the `count` keys of the tile have V rows loaded.
        const float* pw = ps + warp * tile;
        int j = 0;
        for (; j + 4 <= count; j += 4) {
            const float p0 = pw[j];
            const float p1 = pw[j + 1];
            const float p2 = pw[j + 2];
            const float p3 = pw[j + 3];
#pragma unroll
            for (int c = 0; c < kMaxChunks; ++c) {
                const std::int64_t e = c * 32 + lane;
                if (e < dv) {
                    const float* vj = vs + j * dv + e;
                    acc[c] = fmaf(p3, vj[3 * dv], fmaf(p2, vj[2 * dv], fmaf(p1, vj[dv], fmaf(p0, vj[0], acc[c]))));
                }
            }
        }
        for (; j < count; ++j) {
            const float pj1 = pw[j];
#pragma unroll
            for (int c = 0; c < kMaxChunks; ++c) {
                const std::int64_t e = c * 32 + lane;
                if (e < dv) {
                    acc[c] = fmaf(pj1, vs[j * dv + e], acc[c]);
                }
            }
        }
        m = m_new;
        __syncwarp();
    }
    if (!active) {
        return;
    }
    if (gridDim.y == 1) {
        const float inverse = 1.0f / sum;  // fully masked rows: 0 * inf = NaN
#pragma unroll
        for (int c = 0; c < kMaxChunks; ++c) {
            const std::int64_t e = c * 32 + lane;
            if (e < dv) {
                out[row * dv + e] = Traits<TQ>::store(sum == 0.0f ? quiet_nan() : acc[c] * inverse);
            }
        }
        return;
    }
    float* part = partial + (row * gridDim.y + blockIdx.y) * (dv + 2);
    if (lane == 0) {
        part[0] = m;
        part[1] = sum;
    }
#pragma unroll
    for (int c = 0; c < kMaxChunks; ++c) {
        const std::int64_t e = c * 32 + lane;
        if (e < dv) {
            part[2 + e] = acc[c];
        }
    }
}

/// Merges the partial results of the splits: one block per query row. Splits past the keys in use hold nothing and
/// are skipped.
template <class TQ>
__global__ void attention_combine_kernel(TQ* __restrict__ out, const float* __restrict__ partial, AttentionParams p,
                                         int splits, std::int64_t split_size) {
    __shared__ float weight[kMaxSplits];
    const std::int64_t row = blockIdx.x;
    const std::int64_t dv = p.dv;
    int active = splits;
    if (p.kv_len != nullptr) {
        std::int64_t used =
            p.kv_len64 ? *static_cast<const std::int64_t*>(p.kv_len) : *static_cast<const std::int32_t*>(p.kv_len);
        used = used < 0 ? 0 : (used > p.skv ? p.skv : used);
        const std::int64_t needed = (used + split_size - 1) / split_size;
        active = static_cast<int>(needed < splits ? needed : splits);
    }
    const float* parts = partial + row * splits * (dv + 2);
    float m = -infinity();
    for (int s = static_cast<int>(threadIdx.x); s < active; s += static_cast<int>(blockDim.x)) {
        m = fmaxf(m, parts[s * (dv + 2)]);
    }
    m = block_reduce(m, [](float a, float b) { return fmaxf(a, b); }, -infinity());
    float total = 0.0f;
    for (int s = static_cast<int>(threadIdx.x); s < active; s += static_cast<int>(blockDim.x)) {
        const float ms = parts[s * (dv + 2)];
        const float w = ms == -infinity() ? 0.0f : expf(ms - m);
        weight[s] = w;
        total += w * parts[s * (dv + 2) + 1];
    }
    total = block_reduce(total, [](float a, float b) { return a + b; }, 0.0f);  // also publishes `weight`
    for (std::int64_t e = threadIdx.x; e < dv; e += blockDim.x) {
        float value = 0.0f;
#pragma unroll 4
        for (int s = 0; s < active; ++s) {
            value = fmaf(weight[s], parts[s * (dv + 2) + 2 + e], value);
        }
        out[row * dv + e] = Traits<TQ>::store(total == 0.0f ? quiet_nan() : value / total);
    }
}

template <class TQ, class TKV>
cudaError_t launch_attention(void* out, const void* q, const void* k, const void* v, const void* mask,
                             const AttentionParams& p, const AttentionLaunch& l, void* workspace, cudaStream_t stream) {
    auto* kernel = attention_kernel<TQ, TKV>;
    if (l.shared > 48 * 1024) {
        if (const cudaError_t error =
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(l.shared));
            error != cudaSuccess) {
            return error;
        }
    }
    const dim3 grid(static_cast<unsigned>(l.blocks), static_cast<unsigned>(l.splits));
    kernel<<<grid, kAttentionWarps * 32, l.shared, stream>>>(static_cast<TQ*>(out), static_cast<const TQ*>(q),
                                                             static_cast<const TKV*>(k), static_cast<const TKV*>(v),
                                                             mask, p, l, static_cast<float*>(workspace));
    if (const cudaError_t error = cudaGetLastError(); error != cudaSuccess || l.splits == 1) {
        return error;
    }
    const std::int64_t rows = p.batch * p.heads * p.sq;
    attention_combine_kernel<TQ><<<static_cast<unsigned>(rows), 64, 0, stream>>>(
        static_cast<TQ*>(out), static_cast<const float*>(workspace), p, l.splits, l.split_size);
    return cudaGetLastError();
}

}  // namespace

cudaError_t rope(void* y, const void* x, const void* cos, const void* sin, Scalar type, const RopeParams& params,
                 cudaStream_t stream) {
    std::int64_t numel = 1;
    for (int d = 0; d < params.rank; ++d) {
        numel *= params.dims[d];
    }
    if (numel == 0) {
        return cudaSuccess;
    }
    const std::int64_t rows = numel / params.dims[params.rank - 1];
    return dispatch_float(type, [&](auto tag) {
        using T = decltype(tag);
        rope_kernel<T><<<blocks_for(numel), kThreads, 0, stream>>>(static_cast<T*>(y), static_cast<const T*>(x),
                                                                   static_cast<const T*>(cos),
                                                                   static_cast<const T*>(sin), params, rows);
        return cudaGetLastError();
    });
}

cudaError_t slice_update(void* y, const void* update, const SliceUpdateParams& params, cudaStream_t stream) {
    const std::int64_t bytes = params.outer * params.length * params.inner_bytes;
    auto launch = [&](auto word) {
        using W = decltype(word);
        const std::int64_t words_per_row = params.inner_bytes / static_cast<std::int64_t>(sizeof(W));
        const std::int64_t n = bytes / static_cast<std::int64_t>(sizeof(W));
        // At least one thread checks the start (and reports a bad one) even when nothing is written.
        slice_update_kernel<W><<<blocks_for(n), kThreads, 0, stream>>>(
            static_cast<std::uint8_t*>(y), static_cast<const std::uint8_t*>(update), params, words_per_row, n);
        return cudaGetLastError();
    };
    if (params.inner_bytes % 16 == 0) {
        return launch(uint4{});
    }
    if (params.inner_bytes % 4 == 0) {
        return launch(std::uint32_t{});
    }
    if (params.inner_bytes % 2 == 0) {
        return launch(std::uint16_t{});
    }
    return launch(std::uint8_t{});
}

std::size_t attention_workspace(const AttentionParams& params, int sm_count) {
    const AttentionLaunch l = plan(params, sm_count);
    if (l.splits == 1) {
        return 0;
    }
    return sizeof(float) *
           static_cast<std::size_t>(params.batch * params.heads * params.sq * l.splits * (params.dv + 2));
}

cudaError_t attention(void* out, const void* q, const void* k, const void* v, const void* mask, Scalar q_type,
                      Scalar kv_type, const AttentionParams& params, int sm_count, void* workspace,
                      cudaStream_t stream) {
    if (params.d > kAttentionMaxHead || params.dv > kAttentionMaxHead || params.kv_heads <= 0 ||
        params.heads % params.kv_heads != 0) {
        return cudaErrorInvalidValue;
    }
    if (params.batch * params.heads * params.sq == 0 || params.dv == 0) {
        return cudaSuccess;
    }
    if (params.sq >= kFlashMinRows && flash_attention_supported(params, q_type, kv_type, params.arch)) {
        return flash_attention(out, q, k, v, mask, q_type, kv_type, params, stream);
    }
    const AttentionLaunch l = plan(params, sm_count);
    if (q_type == kv_type) {
        return dispatch_float(q_type, [&](auto tag) {
            using T = decltype(tag);
            return launch_attention<T, T>(out, q, k, v, mask, params, l, workspace, stream);
        });
    }
    if (q_type != Scalar::kF32) {
        return cudaErrorInvalidValue;
    }
    return dispatch_float(kv_type, [&](auto tag) {
        using T = decltype(tag);
        return launch_attention<float, T>(out, q, k, v, mask, params, l, workspace, stream);
    });
}

}  // namespace ii::cuda
