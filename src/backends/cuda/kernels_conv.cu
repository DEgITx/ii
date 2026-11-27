// SPDX-License-Identifier: MIT
// ii-cuda convolution helpers (im2col, direct grouped convolution, bias), pooling and resize.
#include "backends/cuda/kernels_common.cuh"

namespace ii::cuda {

namespace {

template <class T>
__global__ void im2col_kernel(T* col, const T* x, ConvParams p, std::int64_t first, std::int64_t n) {
    const std::int64_t positions = p.out_h * p.out_w;
    const std::int64_t window = p.kernel_h * p.kernel_w;
    const std::int64_t rows = p.channels * window;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t position = i % positions;
        const std::int64_t row = (i / positions) % rows;
        const std::int64_t image = first + i / (positions * rows);
        const std::int64_t c = row / window;
        const std::int64_t ky = (row % window) / p.kernel_w;
        const std::int64_t kx = row % p.kernel_w;
        const std::int64_t iy = (position / p.out_w) * p.stride_h - p.pad_h + ky * p.dilation_h;
        const std::int64_t ix = (position % p.out_w) * p.stride_w - p.pad_w + kx * p.dilation_w;
        const bool inside = iy >= 0 && iy < p.in_h && ix >= 0 && ix < p.in_w;
        col[i] = inside ? x[((image * p.channels + c) * p.in_h + iy) * p.in_w + ix] : T{};
    }
}

template <class T>
__global__ void conv_direct_kernel(T* y, const T* x, const T* w, const T* bias, ConvParams p, std::int64_t n) {
    const std::int64_t group_in = p.channels / p.groups;
    const std::int64_t group_out = p.out_channels / p.groups;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t ox = i % p.out_w;
        const std::int64_t oy = (i / p.out_w) % p.out_h;
        const std::int64_t oc = (i / (p.out_w * p.out_h)) % p.out_channels;
        const std::int64_t image = i / (p.out_w * p.out_h * p.out_channels);
        const std::int64_t g = oc / group_out;
        float acc = bias != nullptr ? Traits<T>::load(bias[oc]) : 0.0f;
        for (std::int64_t c = 0; c < group_in; ++c) {
            const T* plane = x + (image * p.channels + g * group_in + c) * p.in_h * p.in_w;
            const T* kernel = w + (oc * group_in + c) * p.kernel_h * p.kernel_w;
            for (std::int64_t ky = 0; ky < p.kernel_h; ++ky) {
                const std::int64_t iy = oy * p.stride_h - p.pad_h + ky * p.dilation_h;
                if (iy < 0 || iy >= p.in_h) {
                    continue;
                }
                for (std::int64_t kx = 0; kx < p.kernel_w; ++kx) {
                    const std::int64_t ix = ox * p.stride_w - p.pad_w + kx * p.dilation_w;
                    if (ix >= 0 && ix < p.in_w) {
                        acc += Traits<T>::load(plane[iy * p.in_w + ix]) * Traits<T>::load(kernel[ky * p.kernel_w + kx]);
                    }
                }
            }
        }
        y[i] = Traits<T>::store(acc);
    }
}

template <class T>
__global__ void widen_kernel(float* out, const T* in, std::int64_t n) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        out[i] = Traits<T>::load(in[i]);
    }
}

template <class T>
__global__ void add_bias_kernel(T* y, const T* bias, std::int64_t channels, std::int64_t inner, std::int64_t n) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t c = (i / inner) % channels;
        y[i] = Traits<T>::store(Traits<T>::load(y[i]) + Traits<T>::load(bias[c]));
    }
}

template <class T>
__global__ void pool_kernel(T* y, const T* x, PoolParams p, std::int64_t n) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t ox = i % p.out_w;
        const std::int64_t oy = (i / p.out_w) % p.out_h;
        const std::int64_t plane = i / (p.out_w * p.out_h);
        const T* in = x + plane * p.in_h * p.in_w;
        float best = 0;
        bool have = false;
        bool nan = false;
        float sum = 0;
        std::int64_t valid = 0;
        std::int64_t padded = 0;
        for (std::int64_t ky = 0; ky < p.kernel_h; ++ky) {
            const std::int64_t iy = oy * p.stride_h - p.pad_h + ky * p.dilation_h;
            const bool y_padding_area = iy >= -p.pad_h && iy < p.in_h + p.pad_h_end;
            for (std::int64_t kx = 0; kx < p.kernel_w; ++kx) {
                const std::int64_t ix = ox * p.stride_w - p.pad_w + kx * p.dilation_w;
                if (y_padding_area && ix >= -p.pad_w && ix < p.in_w + p.pad_w_end) {
                    ++padded;
                }
                if (iy < 0 || iy >= p.in_h || ix < 0 || ix >= p.in_w) {
                    continue;
                }
                const float v = Traits<T>::load(in[iy * p.in_w + ix]);
                ++valid;
                if (p.max) {
                    nan = nan || v != v;
                    if (!have || v > best) {
                        best = v;
                        have = true;
                    }
                } else {
                    sum += v;
                }
            }
        }
        float result;
        if (p.max) {
            result = nan ? quiet_nan() : (have ? best : -__int_as_float(0x7f800000));
        } else {
            const std::int64_t divisor = p.count_include_pad ? padded : valid;
            result = divisor == 0 ? quiet_nan() : sum / static_cast<float>(divisor);
        }
        y[i] = Traits<T>::store(result);
    }
}

template <class T>
__global__ void resize_kernel(T* y, const T* x, ResizeParams p, std::int64_t n) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        std::int64_t position[kMaxRank];
        std::int64_t rest = i;
        for (int d = p.rank - 1; d >= 0; --d) {
            position[d] = rest % p.out_dims[d];
            rest /= p.out_dims[d];
        }
        int tap[kMaxRank] = {};
        float sum = 0;
        while (true) {
            float weight = 1;
            std::int64_t offset = 0;
            for (int d = 0; d < p.rank; ++d) {
                const std::int64_t k = position[d] * p.taps[d] + tap[d];
                weight *= p.weight[d][k];
                offset += p.index[d][k] * p.in_strides[d];
            }
            if (weight != 0) {
                sum += weight * Traits<T>::load(x[offset]);
            }
            int d = p.rank - 1;
            while (d >= 0 && ++tap[d] == p.taps[d]) {
                tap[d] = 0;
                --d;
            }
            if (d < 0) {
                break;
            }
        }
        y[i] = Traits<T>::store(sum);
    }
}

}  // namespace

cudaError_t im2col(void* col, const void* x, int element_size, const ConvParams& params, std::int64_t first,
                   std::int64_t images, cudaStream_t stream) {
    const std::int64_t n = images * params.channels * params.kernel_h * params.kernel_w * params.out_h * params.out_w;
    if (n == 0) {
        return cudaSuccess;
    }
    if (element_size == 4) {
        im2col_kernel<std::uint32_t><<<blocks_for(n), kThreads, 0, stream>>>(
            static_cast<std::uint32_t*>(col), static_cast<const std::uint32_t*>(x), params, first, n);
    } else if (element_size == 2) {
        im2col_kernel<std::uint16_t><<<blocks_for(n), kThreads, 0, stream>>>(
            static_cast<std::uint16_t*>(col), static_cast<const std::uint16_t*>(x), params, first, n);
    } else {
        return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

cudaError_t conv_direct(void* y, const void* x, const void* w, const void* bias, Scalar type, const ConvParams& params,
                        cudaStream_t stream) {
    const std::int64_t n = params.batch * params.out_channels * params.out_h * params.out_w;
    if (n == 0) {
        return cudaSuccess;
    }
    return dispatch_float(type, [&](auto tag) {
        using T = decltype(tag);
        conv_direct_kernel<T><<<blocks_for(n), kThreads, 0, stream>>>(static_cast<T*>(y), static_cast<const T*>(x),
                                                                      static_cast<const T*>(w),
                                                                      static_cast<const T*>(bias), params, n);
        return cudaGetLastError();
    });
}

cudaError_t widen(float* out, const void* in, Scalar type, std::int64_t count, cudaStream_t stream) {
    if (count == 0) {
        return cudaSuccess;
    }
    return dispatch_float(type, [&](auto tag) {
        using T = decltype(tag);
        widen_kernel<T><<<blocks_for(count), kThreads, 0, stream>>>(out, static_cast<const T*>(in), count);
        return cudaGetLastError();
    });
}

cudaError_t add_bias(void* y, const void* bias, Scalar type, std::int64_t outer, std::int64_t channels,
                     std::int64_t inner, cudaStream_t stream) {
    const std::int64_t n = outer * channels * inner;
    if (n == 0) {
        return cudaSuccess;
    }
    return dispatch_float(type, [&](auto tag) {
        using T = decltype(tag);
        add_bias_kernel<T><<<blocks_for(n), kThreads, 0, stream>>>(static_cast<T*>(y), static_cast<const T*>(bias),
                                                                   channels, inner, n);
        return cudaGetLastError();
    });
}

cudaError_t pool(void* y, const void* x, Scalar type, const PoolParams& params, cudaStream_t stream) {
    const std::int64_t n = params.planes * params.out_h * params.out_w;
    if (n == 0) {
        return cudaSuccess;
    }
    return dispatch_float(type, [&](auto tag) {
        using T = decltype(tag);
        pool_kernel<T><<<blocks_for(n), kThreads, 0, stream>>>(static_cast<T*>(y), static_cast<const T*>(x), params, n);
        return cudaGetLastError();
    });
}

cudaError_t resize(void* y, const void* x, Scalar type, const ResizeParams& params, cudaStream_t stream) {
    std::int64_t n = 1;
    for (int d = 0; d < params.rank; ++d) {
        n *= params.out_dims[d];
    }
    if (n == 0) {
        return cudaSuccess;
    }
    return dispatch_float(type, [&](auto tag) {
        using T = decltype(tag);
        resize_kernel<T>
            <<<blocks_for(n), kThreads, 0, stream>>>(static_cast<T*>(y), static_cast<const T*>(x), params, n);
        return cudaGetLastError();
    });
}

}  // namespace ii::cuda
