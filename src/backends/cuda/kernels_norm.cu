// SPDX-License-Identifier: MIT
// ii-cuda softmax, log_softmax, layer_norm and rms_norm.
#include "backends/cuda/kernels_common.cuh"

namespace ii::cuda {

namespace {

__device__ __forceinline__ float max_nan(float a, float b) {
    return (a != a || b != b) ? quiet_nan() : (a < b ? b : a);
}

/// One block per row (inner == 1).
template <class T>
__global__ void softmax_row_kernel(T* out, const T* in, std::int64_t rows, std::int64_t size, bool log) {
    const float negative_infinity = -__int_as_float(0x7f800000);
    for (std::int64_t r = blockIdx.x; r < rows; r += gridDim.x) {
        const T* x = in + r * size;
        T* y = out + r * size;
        float max = negative_infinity;
        for (std::int64_t k = threadIdx.x; k < size; k += blockDim.x) {
            max = max_nan(max, Traits<T>::load(x[k]));
        }
        max = block_reduce(max, max_nan, negative_infinity);
        float sum = 0;
        for (std::int64_t k = threadIdx.x; k < size; k += blockDim.x) {
            sum += expf(Traits<T>::load(x[k]) - max);
        }
        sum = block_reduce(sum, [](float a, float b) { return a + b; }, 0.0f);
        const float log_sum = logf(sum);
        const float inverse = 1.0f / sum;
        for (std::int64_t k = threadIdx.x; k < size; k += blockDim.x) {
            const float shifted = Traits<T>::load(x[k]) - max;
            y[k] = Traits<T>::store(log ? shifted - log_sum : expf(shifted) * inverse);
        }
    }
}

/// One thread per column (inner > 1).
template <class T>
__global__ void softmax_column_kernel(T* out, const T* in, std::int64_t outer, std::int64_t size, std::int64_t inner,
                                      bool log) {
    const std::int64_t n = outer * inner;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t base = (i / inner) * size * inner + i % inner;
        float max = -__int_as_float(0x7f800000);
        for (std::int64_t k = 0; k < size; ++k) {
            max = max_nan(max, Traits<T>::load(in[base + k * inner]));
        }
        float sum = 0;
        for (std::int64_t k = 0; k < size; ++k) {
            sum += expf(Traits<T>::load(in[base + k * inner]) - max);
        }
        const float log_sum = logf(sum);
        for (std::int64_t k = 0; k < size; ++k) {
            const float shifted = Traits<T>::load(in[base + k * inner]) - max;
            out[base + k * inner] = Traits<T>::store(log ? shifted - log_sum : expf(shifted) / sum);
        }
    }
}

template <class T>
__global__ void normalize_kernel(T* out, const T* in, const T* scale, const T* bias, std::int64_t rows,
                                 std::int64_t size, float epsilon, bool rms) {
    const auto add = [](float a, float b) { return a + b; };
    for (std::int64_t r = blockIdx.x; r < rows; r += gridDim.x) {
        const T* x = in + r * size;
        T* y = out + r * size;
        float mean = 0;
        if (!rms) {
            float sum = 0;
            for (std::int64_t k = threadIdx.x; k < size; k += blockDim.x) {
                sum += Traits<T>::load(x[k]);
            }
            mean = block_reduce(sum, add, 0.0f) / static_cast<float>(size);
        }
        float squares = 0;
        for (std::int64_t k = threadIdx.x; k < size; k += blockDim.x) {
            const float d = Traits<T>::load(x[k]) - mean;
            squares += d * d;
        }
        const float variance = block_reduce(squares, add, 0.0f) / static_cast<float>(size);
        const float inverse = rsqrtf(variance + epsilon);
        for (std::int64_t k = threadIdx.x; k < size; k += blockDim.x) {
            float v = (Traits<T>::load(x[k]) - mean) * inverse;
            if (scale != nullptr) {
                v *= Traits<T>::load(scale[k]);
            }
            if (bias != nullptr) {
                v += Traits<T>::load(bias[k]);
            }
            y[k] = Traits<T>::store(v);
        }
    }
}

/// Threads per block for rows of `size` elements.
int row_threads(std::int64_t size) {
    if (size >= 2048) {
        return 512;
    }
    if (size >= 512) {
        return 256;
    }
    if (size >= 128) {
        return 128;
    }
    return 32;
}

unsigned row_blocks(std::int64_t rows) {
    return static_cast<unsigned>(rows < (1 << 20) ? rows : (1 << 20));
}

}  // namespace

cudaError_t softmax(void* out, const void* in, Scalar type, std::int64_t outer, std::int64_t size, std::int64_t inner,
                    bool log, cudaStream_t stream) {
    if (outer * size * inner == 0) {
        return cudaSuccess;
    }
    return dispatch_float(type, [&](auto tag) {
        using T = decltype(tag);
        if (inner == 1) {
            softmax_row_kernel<T><<<row_blocks(outer), row_threads(size), 0, stream>>>(
                static_cast<T*>(out), static_cast<const T*>(in), outer, size, log);
        } else {
            softmax_column_kernel<T><<<blocks_for(outer * inner), kThreads, 0, stream>>>(
                static_cast<T*>(out), static_cast<const T*>(in), outer, size, inner, log);
        }
        return cudaGetLastError();
    });
}

cudaError_t normalize(void* out, const void* in, const void* scale, const void* bias, Scalar type, std::int64_t rows,
                      std::int64_t size, float epsilon, bool rms, cudaStream_t stream) {
    if (rows * size == 0) {
        return cudaSuccess;
    }
    return dispatch_float(type, [&](auto tag) {
        using T = decltype(tag);
        normalize_kernel<T><<<row_blocks(rows), row_threads(size), 0, stream>>>(
            static_cast<T*>(out), static_cast<const T*>(in), static_cast<const T*>(scale), static_cast<const T*>(bias),
            rows, size, epsilon, rms);
        return cudaGetLastError();
    });
}

}  // namespace ii::cuda
