// SPDX-License-Identifier: MIT
// ii-cuda data movement kernels: strided copies, gather, pad.
#include "backends/cuda/kernels_common.cuh"

#include <climits>

namespace ii::cuda {

namespace {

/// Element storage of a given size.
template <int kSize>
struct Element;
template <>
struct Element<1> {
    using type = std::uint8_t;
};
template <>
struct Element<2> {
    using type = std::uint16_t;
};
template <>
struct Element<4> {
    using type = std::uint32_t;
};
template <>
struct Element<8> {
    using type = std::uint64_t;
};

template <class Fn>
cudaError_t dispatch_size(int element_size, Fn&& fn) {
    switch (element_size) {
    case 1:
        return fn(Element<1>{});
    case 2:
        return fn(Element<2>{});
    case 4:
        return fn(Element<4>{});
    case 8:
        return fn(Element<8>{});
    default:
        return cudaErrorInvalidValue;
    }
}

template <class T, class I>
__global__ void copy_strided_kernel(T* dst, const T* src, CopyParams p, I n) {
    for (I i = static_cast<I>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<I>(gridDim.x) * blockDim.x) {
        I rest = i;
        std::int64_t s = 0;
        std::int64_t d = 0;
        for (int k = p.rank - 1; k >= 0; --k) {
            const I size = static_cast<I>(p.dims[k]);
            const I index = rest % size;
            rest /= size;
            s += index * p.src_strides[k];
            d += index * p.dst_strides[k];
        }
        dst[d] = src[s];
    }
}

/// Merges dimensions that are contiguous in both views and drops dimensions of size 1.
CopyParams simplify(const CopyParams& in) {
    CopyParams out;
    for (int k = 0; k < in.rank; ++k) {
        if (in.dims[k] == 1) {
            continue;
        }
        if (out.rank > 0) {
            const int last = out.rank - 1;
            if (out.src_strides[last] == in.src_strides[k] * in.dims[k] &&
                out.dst_strides[last] == in.dst_strides[k] * in.dims[k]) {
                out.dims[last] *= in.dims[k];
                out.src_strides[last] = in.src_strides[k];
                out.dst_strides[last] = in.dst_strides[k];
                continue;
            }
        }
        out.dims[out.rank] = in.dims[k];
        out.src_strides[out.rank] = in.src_strides[k];
        out.dst_strides[out.rank] = in.dst_strides[k];
        ++out.rank;
    }
    return out;
}

template <class T>
__global__ void gather_kernel(T* out, const T* data, const void* indices, GatherParams p, std::int64_t words_per_slice,
                              std::int64_t n) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t word = i % words_per_slice;
        const std::int64_t slice = i / words_per_slice;  // o * count + j
        const std::int64_t j = slice % p.count;
        const std::int64_t o = slice / p.count;
        std::int64_t index =
            p.index64 ? static_cast<const std::int64_t*>(indices)[j] : static_cast<const std::int32_t*>(indices)[j];
        const std::int64_t given = index;
        if (index < 0) {
            index += p.axis_size;
        }
        const bool inside = index >= 0 && index < p.axis_size;
        if (!inside && word == 0 && o == 0 &&
            atomicCAS(reinterpret_cast<unsigned long long*>(&p.error->code), 0ull,
                      static_cast<unsigned long long>(DeviceErrorCode::kGatherIndex)) == 0ull) {
            p.error->value = given;
            p.error->limit = p.axis_size;
        }
        out[i] = inside ? data[(o * p.axis_size + index) * words_per_slice + word] : T{};
    }
}

template <class T, class I>
__global__ void pad_kernel(T* out, const T* in, PadParams p, I n) {
    const T fill = static_cast<T>(p.fill);
    for (I i = static_cast<I>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<I>(gridDim.x) * blockDim.x) {
        I rest = i;
        I offset = 0;
        I stride = 1;
        bool inside = true;
        for (int k = p.rank - 1; k >= 0; --k) {
            const I size = static_cast<I>(p.out_dims[k]);
            const I in_size = static_cast<I>(p.in_dims[k]);
            I pos = rest % size - static_cast<I>(p.pad_begin[k]);
            rest /= size;
            if (pos < 0 || pos >= in_size) {
                switch (p.mode) {
                case PadMode::kConstant:
                    inside = false;
                    break;
                case PadMode::kEdge:
                    pos = pos < 0 ? 0 : in_size - 1;
                    break;
                case PadMode::kReflect: {
                    const I period = 2 * (in_size - 1);
                    pos = period == 0 ? 0 : ((pos % period) + period) % period;
                    if (pos >= in_size) {
                        pos = period - pos;
                    }
                    break;
                }
                case PadMode::kWrap:
                    pos = ((pos % in_size) + in_size) % in_size;
                    break;
                }
            }
            offset += pos * stride;
            stride *= in_size;
        }
        out[i] = inside ? in[offset] : fill;
    }
}

template <class T>
__global__ void fill_kernel(T* out, T value, std::int64_t n) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        out[i] = value;
    }
}

/// Converts with the `cast` rules: floats round to nearest; integers truncate and saturate, NaN -> 0.
template <class T>
__device__ __forceinline__ T convert(double v) {
    if constexpr (std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::int64_t>) {
        if (v != v) {
            return 0;
        }
        const double t = trunc(v);
        const double low = std::is_same_v<T, std::int32_t> ? -2147483648.0 : -9223372036854775808.0;
        const double high = std::is_same_v<T, std::int32_t> ? 2147483647.0 : 9223372036854775808.0;
        if (t <= low) {
            return std::is_same_v<T, std::int32_t> ? T(INT_MIN) : T(LLONG_MIN);
        }
        if (t >= high) {
            return std::is_same_v<T, std::int32_t> ? T(INT_MAX) : T(LLONG_MAX);
        }
        return static_cast<T>(t);
    } else {
        return Traits<T>::store(static_cast<float>(v));
    }
}

template <class T>
__global__ void iota_kernel(T* out, double start, double step, std::int64_t size, std::int64_t inner, std::int64_t n) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t position = size == 0 ? 0 : (i / inner) % size;
        out[i] = convert<T>(start + step * static_cast<double>(position));
    }
}

}  // namespace

cudaError_t fill(void* out, int element_size, std::uint64_t fill, std::int64_t count, cudaStream_t stream) {
    if (count == 0) {
        return cudaSuccess;
    }
    return dispatch_size(element_size, [&](auto tag) {
        using T = typename decltype(tag)::type;
        fill_kernel<T><<<blocks_for(count), kThreads, 0, stream>>>(static_cast<T*>(out), static_cast<T>(fill), count);
        return cudaGetLastError();
    });
}

cudaError_t iota(void* out, Scalar type, double start, double step, std::int64_t count, std::int64_t size,
                 std::int64_t inner, cudaStream_t stream) {
    if (count == 0) {
        return cudaSuccess;
    }
    return dispatch_all(type, [&](auto tag) {
        using T = decltype(tag);
        iota_kernel<T>
            <<<blocks_for(count), kThreads, 0, stream>>>(static_cast<T*>(out), start, step, size, inner, count);
        return cudaGetLastError();
    });
}

cudaError_t copy_strided(void* dst, const void* src, int element_size, const CopyParams& params, cudaStream_t stream) {
    const CopyParams p = simplify(params);
    std::int64_t n = 1;
    for (int k = 0; k < p.rank; ++k) {
        n *= p.dims[k];
    }
    if (n == 0) {
        return cudaSuccess;
    }
    if (p.rank <= 1) {
        const bool contiguous = p.rank == 0 || (p.src_strides[0] == 1 && p.dst_strides[0] == 1);
        if (contiguous) {
            return cudaMemcpyAsync(dst, src, static_cast<std::size_t>(n) * static_cast<std::size_t>(element_size),
                                   cudaMemcpyDeviceToDevice, stream);
        }
    }
    return dispatch_size(element_size, [&](auto tag) {
        using T = typename decltype(tag)::type;
        if (n < (std::int64_t{1} << 31)) {
            copy_strided_kernel<T, std::int32_t><<<blocks_for(n), kThreads, 0, stream>>>(
                static_cast<T*>(dst), static_cast<const T*>(src), p, static_cast<std::int32_t>(n));
        } else {
            copy_strided_kernel<T, std::int64_t>
                <<<blocks_for(n), kThreads, 0, stream>>>(static_cast<T*>(dst), static_cast<const T*>(src), p, n);
        }
        return cudaGetLastError();
    });
}

cudaError_t gather(void* out, const void* data, const void* indices, const GatherParams& params, cudaStream_t stream) {
    const std::int64_t slices = params.outer * params.count;
    if (slices == 0 || params.slice_bytes == 0) {
        return cudaSuccess;
    }
    const int word =
        params.slice_bytes % 8 == 0 ? 8 : (params.slice_bytes % 4 == 0 ? 4 : (params.slice_bytes % 2 == 0 ? 2 : 1));
    const std::int64_t words = params.slice_bytes / word;
    const std::int64_t n = slices * words;
    return dispatch_size(word, [&](auto tag) {
        using T = typename decltype(tag)::type;
        gather_kernel<T><<<blocks_for(n), kThreads, 0, stream>>>(static_cast<T*>(out), static_cast<const T*>(data),
                                                                 indices, params, words, n);
        return cudaGetLastError();
    });
}

cudaError_t pad(void* out, const void* in, int element_size, const PadParams& params, cudaStream_t stream) {
    std::int64_t n = 1;
    std::int64_t in_n = 1;
    for (int k = 0; k < params.rank; ++k) {
        n *= params.out_dims[k];
        in_n *= params.in_dims[k];
    }
    if (n == 0) {
        return cudaSuccess;
    }
    return dispatch_size(element_size, [&](auto tag) {
        using T = typename decltype(tag)::type;
        if (n < (std::int64_t{1} << 31) && in_n < (std::int64_t{1} << 31)) {
            pad_kernel<T, std::int32_t><<<blocks_for(n), kThreads, 0, stream>>>(
                static_cast<T*>(out), static_cast<const T*>(in), params, static_cast<std::int32_t>(n));
        } else {
            pad_kernel<T, std::int64_t>
                <<<blocks_for(n), kThreads, 0, stream>>>(static_cast<T*>(out), static_cast<const T*>(in), params, n);
        }
        return cudaGetLastError();
    });
}

}  // namespace ii::cuda
