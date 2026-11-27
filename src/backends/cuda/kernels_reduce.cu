// SPDX-License-Identifier: MIT
// ii-cuda reductions: reduce_sum / mean / max / min / prod over any axes, argmax / argmin.
#include "backends/cuda/kernels_common.cuh"

#include <climits>

namespace ii::cuda {

namespace {

template <class T>
__device__ __forceinline__ typename Traits<T>::Acc lowest_value() {
    if constexpr (std::is_same_v<T, std::int32_t>) {
        return INT_MIN;
    } else if constexpr (std::is_same_v<T, std::int64_t>) {
        return LLONG_MIN;
    } else {
        return -__int_as_float(0x7f800000);
    }
}

template <class T>
__device__ __forceinline__ typename Traits<T>::Acc highest_value() {
    if constexpr (std::is_same_v<T, std::int32_t>) {
        return INT_MAX;
    } else if constexpr (std::is_same_v<T, std::int64_t>) {
        return LLONG_MAX;
    } else {
        return __int_as_float(0x7f800000);
    }
}

/// The reduction: identity, combination and the final value.
template <class T, ReduceOp kOp>
struct Reducer {
    using Acc = typename Traits<T>::Acc;
    static constexpr bool kFloat = std::is_same_v<Acc, float>;

    static __device__ __forceinline__ Acc identity() {
        if constexpr (kOp == ReduceOp::kProd) {
            return Acc{1};
        } else if constexpr (kOp == ReduceOp::kMax) {
            return lowest_value<T>();
        } else if constexpr (kOp == ReduceOp::kMin) {
            return highest_value<T>();
        } else {
            return Acc{0};
        }
    }

    static __device__ __forceinline__ Acc combine(Acc a, Acc b) {
        if constexpr (kOp == ReduceOp::kSum || kOp == ReduceOp::kMean) {
            if constexpr (kFloat) {
                return a + b;
            } else {
                return static_cast<Acc>(static_cast<unsigned long long>(a) + static_cast<unsigned long long>(b));
            }
        } else if constexpr (kOp == ReduceOp::kProd) {
            if constexpr (kFloat) {
                return a * b;
            } else {
                return static_cast<Acc>(static_cast<unsigned long long>(a) * static_cast<unsigned long long>(b));
            }
        } else {
            if (is_nan(a) || is_nan(b)) {
                return a + b;  // NaN
            }
            if constexpr (kOp == ReduceOp::kMax) {
                return a < b ? b : a;
            } else {
                return b < a ? b : a;
            }
        }
    }

    static __device__ __forceinline__ T finish(Acc acc, std::int64_t count) {
        if constexpr (kOp == ReduceOp::kMean) {
            if constexpr (kFloat) {
                return Traits<T>::store(count == 0 ? quiet_nan() : acc / static_cast<float>(count));
            } else {
                return Traits<T>::store(count == 0 ? 0 : acc / count);
            }
        } else {
            return Traits<T>::store(acc);
        }
    }
};

struct Offsets {
    __device__ static std::int64_t of(std::int64_t index, int rank, const std::int64_t* dims,
                                      const std::int64_t* strides) {
        std::int64_t offset = 0;
        for (int k = rank - 1; k >= 0; --k) {
            offset += (index % dims[k]) * strides[k];
            index /= dims[k];
        }
        return offset;
    }
};

template <class T, ReduceOp kOp, bool kContiguous>
__device__ __forceinline__ typename Traits<T>::Acc partial(const T* in, std::int64_t base, const ReduceParams& p,
                                                           std::int64_t first, std::int64_t step) {
    using R = Reducer<T, kOp>;
    typename Traits<T>::Acc acc = R::identity();
    for (std::int64_t r = first; r < p.count; r += step) {
        const std::int64_t offset = kContiguous ? r : Offsets::of(r, p.reduced_rank, p.reduced_dims, p.reduced_strides);
        acc = R::combine(acc, Traits<T>::load(in[base + offset]));
    }
    return acc;
}

/// One thread per output (short reductions).
template <class T, ReduceOp kOp, bool kContiguous>
__global__ void reduce_thread_kernel(T* out, const T* in, ReduceParams p) {
    for (std::int64_t o = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; o < p.outputs;
         o += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t base = Offsets::of(o, p.kept_rank, p.kept_dims, p.kept_strides);
        out[o] = Reducer<T, kOp>::finish(partial<T, kOp, kContiguous>(in, base, p, 0, 1), p.count);
    }
}

/// One warp per output.
template <class T, ReduceOp kOp, bool kContiguous>
__global__ void reduce_warp_kernel(T* out, const T* in, ReduceParams p) {
    using R = Reducer<T, kOp>;
    const int lane = threadIdx.x % 32;
    const std::int64_t warps = static_cast<std::int64_t>(gridDim.x) * (blockDim.x / 32);
    for (std::int64_t o = static_cast<std::int64_t>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32; o < p.outputs;
         o += warps) {
        const std::int64_t base = Offsets::of(o, p.kept_rank, p.kept_dims, p.kept_strides);
        auto acc = partial<T, kOp, kContiguous>(in, base, p, lane, 32);
        acc = warp_reduce(acc, [](auto a, auto b) { return R::combine(a, b); });
        if (lane == 0) {
            out[o] = R::finish(acc, p.count);
        }
    }
}

/// One block per output (long reductions).
template <class T, ReduceOp kOp, bool kContiguous>
__global__ void reduce_block_kernel(T* out, const T* in, ReduceParams p) {
    using R = Reducer<T, kOp>;
    for (std::int64_t o = blockIdx.x; o < p.outputs; o += gridDim.x) {
        const std::int64_t base = Offsets::of(o, p.kept_rank, p.kept_dims, p.kept_strides);
        auto acc = partial<T, kOp, kContiguous>(in, base, p, threadIdx.x, blockDim.x);
        acc = block_reduce(acc, [](auto a, auto b) { return R::combine(a, b); }, R::identity());
        if (threadIdx.x == 0) {
            out[o] = R::finish(acc, p.count);
        }
    }
}

/// Merges adjacent dimensions that are contiguous with each other.
void merge(int& rank, std::int64_t* dims, std::int64_t* strides) {
    int out = 0;
    for (int k = 0; k < rank; ++k) {
        if (dims[k] == 1) {
            continue;
        }
        if (out > 0 && strides[out - 1] == strides[k] * dims[k]) {
            dims[out - 1] *= dims[k];
            strides[out - 1] = strides[k];
            continue;
        }
        dims[out] = dims[k];
        strides[out] = strides[k];
        ++out;
    }
    rank = out;
}

template <class T, ReduceOp kOp, bool kContiguous>
cudaError_t launch_reduce(T* out, const T* in, const ReduceParams& p, cudaStream_t stream) {
    if (p.count <= 16) {
        reduce_thread_kernel<T, kOp, kContiguous><<<blocks_for(p.outputs), kThreads, 0, stream>>>(out, in, p);
    } else if (p.count < 4096 || p.outputs >= 4096) {
        reduce_warp_kernel<T, kOp, kContiguous><<<blocks_for(p.outputs * 32), kThreads, 0, stream>>>(out, in, p);
    } else {
        const std::int64_t blocks = p.outputs < 65535 ? p.outputs : 65535;
        reduce_block_kernel<T, kOp, kContiguous><<<static_cast<unsigned>(blocks), 512, 0, stream>>>(out, in, p);
    }
    return cudaGetLastError();
}

template <class T, ReduceOp kOp>
cudaError_t reduce_typed(void* out, const void* in, ReduceParams p, cudaStream_t stream) {
    const bool contiguous = p.reduced_rank == 0 || (p.reduced_rank == 1 && p.reduced_strides[0] == 1);
    if (contiguous) {
        return launch_reduce<T, kOp, true>(static_cast<T*>(out), static_cast<const T*>(in), p, stream);
    }
    return launch_reduce<T, kOp, false>(static_cast<T*>(out), static_cast<const T*>(in), p, stream);
}

// --- argmax / argmin --------------------------------------------------------------------------------------------

struct ArgBest {
    float value;  // floats; integer inputs use `ivalue`
    long long ivalue;
    long long index;
    bool nan;
};

template <class T>
__device__ __forceinline__ ArgBest arg_item(const T* in, std::int64_t offset, std::int64_t index) {
    ArgBest item{};
    const auto v = Traits<T>::load(in[offset]);
    if constexpr (std::is_same_v<typename Traits<T>::Acc, float>) {
        item.value = v;
        item.nan = v != v;
    } else {
        item.ivalue = v;
    }
    item.index = index;
    return item;
}

/// The better of two candidates; `a` has the smaller index when both are valid.
template <class T>
__device__ __forceinline__ ArgBest arg_better(const ArgBest& a, const ArgBest& b, bool max, bool last) {
    if (a.index < 0) {
        return b;
    }
    if (b.index < 0) {
        return a;
    }
    const ArgBest& first = a.index < b.index ? a : b;
    const ArgBest& second = a.index < b.index ? b : a;
    if (first.nan) {
        return first;
    }
    if (second.nan) {
        return second;
    }
    bool second_better;
    if constexpr (std::is_same_v<typename Traits<T>::Acc, float>) {
        second_better = max ? second.value > first.value : second.value < first.value;
        second_better = second_better || (last && second.value == first.value);
    } else {
        second_better = max ? second.ivalue > first.ivalue : second.ivalue < first.ivalue;
        second_better = second_better || (last && second.ivalue == first.ivalue);
    }
    return second_better ? second : first;
}

template <class T>
__global__ void arg_column_kernel(std::int64_t* out, const T* in, ArgParams p) {
    const std::int64_t n = p.outer * p.inner;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t o = i / p.inner;
        const std::int64_t j = i % p.inner;
        ArgBest best{};
        best.index = -1;
        for (std::int64_t k = 0; k < p.size; ++k) {
            best = arg_better<T>(best, arg_item(in, (o * p.size + k) * p.inner + j, k), p.max, p.select_last);
            if (best.nan) {
                break;
            }
        }
        out[i] = best.index < 0 ? 0 : best.index;
    }
}

template <class T>
__device__ ArgBest shuffle(const ArgBest& item, int offset) {
    ArgBest other;
    other.value = __shfl_xor_sync(0xffffffffu, item.value, offset);
    other.ivalue = __shfl_xor_sync(0xffffffffu, item.ivalue, offset);
    other.index = __shfl_xor_sync(0xffffffffu, item.index, offset);
    other.nan = __shfl_xor_sync(0xffffffffu, static_cast<int>(item.nan), offset) != 0;
    return other;
}

/// One block per row (inner == 1).
template <class T>
__global__ void arg_row_kernel(std::int64_t* out, const T* in, ArgParams p) {
    __shared__ ArgBest partial_best[32];
    for (std::int64_t o = blockIdx.x; o < p.outer; o += gridDim.x) {
        ArgBest best{};
        best.index = -1;
        for (std::int64_t k = threadIdx.x; k < p.size; k += blockDim.x) {
            best = arg_better<T>(best, arg_item(in, o * p.size + k, k), p.max, p.select_last);
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            best = arg_better<T>(best, shuffle<T>(best, offset), p.max, p.select_last);
        }
        if (threadIdx.x % 32 == 0) {
            partial_best[threadIdx.x / 32] = best;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            ArgBest result = partial_best[0];
            for (unsigned w = 1; w < blockDim.x / 32; ++w) {
                result = arg_better<T>(result, partial_best[w], p.max, p.select_last);
            }
            out[o] = result.index < 0 ? 0 : result.index;
        }
        __syncthreads();
    }
}

}  // namespace

cudaError_t reduce(void* out, const void* in, Scalar type, ReduceOp op, const ReduceParams& params,
                   cudaStream_t stream) {
    ReduceParams p = params;
    merge(p.kept_rank, p.kept_dims, p.kept_strides);
    merge(p.reduced_rank, p.reduced_dims, p.reduced_strides);
    if (p.outputs == 0) {
        return cudaSuccess;
    }
    return dispatch_all(type, [&](auto tag) {
        using T = decltype(tag);
        switch (op) {
        case ReduceOp::kSum:
            return reduce_typed<T, ReduceOp::kSum>(out, in, p, stream);
        case ReduceOp::kMean:
            return reduce_typed<T, ReduceOp::kMean>(out, in, p, stream);
        case ReduceOp::kMax:
            return reduce_typed<T, ReduceOp::kMax>(out, in, p, stream);
        case ReduceOp::kMin:
            return reduce_typed<T, ReduceOp::kMin>(out, in, p, stream);
        case ReduceOp::kProd:
            return reduce_typed<T, ReduceOp::kProd>(out, in, p, stream);
        }
        return cudaErrorInvalidValue;
    });
}

cudaError_t arg_reduce(std::int64_t* out, const void* in, Scalar type, const ArgParams& params, cudaStream_t stream) {
    if (params.outer * params.inner == 0) {
        return cudaSuccess;
    }
    return dispatch_all(type, [&](auto tag) {
        using T = decltype(tag);
        if (params.inner == 1 && params.size >= 256) {
            const std::int64_t blocks = params.outer < 65535 ? params.outer : 65535;
            arg_row_kernel<T>
                <<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(out, static_cast<const T*>(in), params);
        } else {
            arg_column_kernel<T><<<blocks_for(params.outer * params.inner), kThreads, 0, stream>>>(
                out, static_cast<const T*>(in), params);
        }
        return cudaGetLastError();
    });
}

}  // namespace ii::cuda
