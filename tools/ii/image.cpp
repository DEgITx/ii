// SPDX-License-Identifier: MIT
#include "image.h"

#include <ii/half.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#pragma GCC diagnostic ignored "-Wformat"
#elif defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_WINDOWS_UTF8
#define STBIW_WINDOWS_UTF8
#include <stb_image.h>
#include <stb_image_write.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace ii::cli {

namespace {

std::string utf8(const std::filesystem::path& path) {
    const std::u8string text = path.u8string();
    return std::string(text.begin(), text.end());
}

std::string lower_extension(const std::string& name) {
    const std::size_t dot = name.find_last_of('.');
    std::string ext = dot == std::string::npos ? std::string() : name.substr(dot + 1);
    for (char& c : ext) {
        c = static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
    }
    return ext;
}

}  // namespace

bool is_image_file(const std::string& name) {
    const std::string ext = lower_extension(name);
    return ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp" || ext == "tga" || ext == "gif" ||
           ext == "psd" || ext == "hdr" || ext == "pnm" || ext == "ppm" || ext == "pgm";
}

Result<Image> load_image(const std::filesystem::path& path) {
    const std::string name = utf8(path);
    int width = 0;
    int height = 0;
    int channels = 0;
    std::unique_ptr<stbi_uc, void (*)(void*)> data(stbi_load(name.c_str(), &width, &height, &channels, 3),
                                                   stbi_image_free);
    if (!data) {
        return errors::data_corrupted("cannot decode image '{}': {}", name, stbi_failure_reason());
    }
    Image image;
    image.width = width;
    image.height = height;
    image.pixels.assign(data.get(),
                        data.get() + static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3);
    return image;
}

Status save_image(const std::filesystem::path& path, const Image& image) {
    const std::string name = utf8(path);
    const std::string ext = lower_extension(name);
    int ok = 0;
    if (ext == "jpg" || ext == "jpeg") {
        ok = stbi_write_jpg(name.c_str(), image.width, image.height, 3, image.pixels.data(), 90);
    } else if (ext == "png") {
        ok = stbi_write_png(name.c_str(), image.width, image.height, 3, image.pixels.data(), image.width * 3);
    } else {
        return errors::invalid_argument("cannot write '{}': use a .png or .jpg file name", name);
    }
    if (ok == 0) {
        return errors::io_error("cannot write image '{}'", name);
    }
    return ok_status();
}

Image resize_image(const Image& image, int width, int height, bool letterbox, ImagePlacement* placement) {
    Image out;
    out.width = width;
    out.height = height;
    out.pixels.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3, 114);
    double scale_x = static_cast<double>(width) / image.width;
    double scale_y = static_cast<double>(height) / image.height;
    int inner_w = width;
    int inner_h = height;
    int pad_x = 0;
    int pad_y = 0;
    if (letterbox) {
        const double s = std::min(scale_x, scale_y);
        scale_x = scale_y = s;
        inner_w = std::max(1, static_cast<int>(std::lround(image.width * s)));
        inner_h = std::max(1, static_cast<int>(std::lround(image.height * s)));
        pad_x = (width - inner_w) / 2;
        pad_y = (height - inner_h) / 2;
    }
    if (placement != nullptr) {
        placement->scale = scale_x;
        placement->pad_x = pad_x;
        placement->pad_y = pad_y;
    }
    // Bilinear with half-pixel centers.
    for (int y = 0; y < inner_h; ++y) {
        const double sy = std::clamp((y + 0.5) * image.height / inner_h - 0.5, 0.0, image.height - 1.0);
        const int y0 = static_cast<int>(sy);
        const int y1 = std::min(y0 + 1, image.height - 1);
        const double fy = sy - y0;
        for (int x = 0; x < inner_w; ++x) {
            const double sx = std::clamp((x + 0.5) * image.width / inner_w - 0.5, 0.0, image.width - 1.0);
            const int x0 = static_cast<int>(sx);
            const int x1 = std::min(x0 + 1, image.width - 1);
            const double fx = sx - x0;
            for (int c = 0; c < 3; ++c) {
                const auto at = [&](int yy, int xx) {
                    return static_cast<double>(
                        image.pixels[(static_cast<std::size_t>(yy) * static_cast<std::size_t>(image.width) +
                                      static_cast<std::size_t>(xx)) *
                                         3 +
                                     static_cast<std::size_t>(c)]);
                };
                const double v = (at(y0, x0) * (1 - fx) + at(y0, x1) * fx) * (1 - fy) +
                                 (at(y1, x0) * (1 - fx) + at(y1, x1) * fx) * fy;
                out.pixels[(static_cast<std::size_t>(y + pad_y) * static_cast<std::size_t>(width) +
                            static_cast<std::size_t>(x + pad_x)) *
                               3 +
                           static_cast<std::size_t>(c)] =
                    static_cast<std::uint8_t>(std::clamp(std::lround(v), 0L, 255L));
            }
        }
    }
    return out;
}

Result<Tensor> image_to_tensor(const Image& image, const TensorInfo& input, const ImageOptions& options,
                               ImagePlacement* placement) {
    const int rank = static_cast<int>(input.shape.size());
    if (rank != 3 && rank != 4) {
        return errors::invalid_argument("input '{}' has shape {}; an image needs a 3-D or 4-D input", input.name,
                                        to_string(input.shape));
    }
    const int offset = rank - 3;  // batch dimension present
    const auto constant = [&](int d) -> std::optional<std::int64_t> {
        return input.shape[static_cast<std::size_t>(d)].constant();
    };
    // Layout: channels first ([N,] C, H, W) when that dimension is 1 or 3, otherwise channels last.
    const std::optional<std::int64_t> first = constant(offset);
    const bool channels_first = first && (*first == 1 || *first == 3);
    const int c_dim = channels_first ? offset : rank - 1;
    const int h_dim = channels_first ? offset + 1 : offset;
    const int w_dim = h_dim + 1;
    const std::int64_t channels = constant(c_dim).value_or(3);
    if (channels != 1 && channels != 3) {
        return errors::invalid_argument("input '{}' has {} channels; images have 1 or 3", input.name, channels);
    }
    int height = options.height != 0 ? options.height : static_cast<int>(constant(h_dim).value_or(image.height));
    int width = options.width != 0 ? options.width : static_cast<int>(constant(w_dim).value_or(image.width));
    const Image resized = width == image.width && height == image.height && !options.letterbox
                              ? image
                              : resize_image(image, width, height, options.letterbox, placement);
    Shape shape;
    if (offset == 1) {
        shape.push_back(1);
    }
    if (channels_first) {
        shape.push_back(channels);
        shape.push_back(height);
        shape.push_back(width);
    } else {
        shape.push_back(height);
        shape.push_back(width);
        shape.push_back(channels);
    }
    II_ASSIGN_OR_RETURN(Tensor tensor, Tensor::empty(input.dtype, shape));
    const std::size_t plane = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    for (std::size_t p = 0; p < plane; ++p) {
        for (std::int64_t c = 0; c < channels; ++c) {
            double value = 0;
            if (channels == 1) {
                const std::uint8_t* px = &resized.pixels[p * 3];
                value = 0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2];
            } else {
                const std::int64_t source = options.bgr ? 2 - c : c;
                value = resized.pixels[p * 3 + static_cast<std::size_t>(source)];
            }
            value *= options.scale;
            const auto ci = static_cast<std::size_t>(c);
            if (ci < options.mean.size()) {
                value -= options.mean[ci];
            }
            if (ci < options.std.size() && options.std[ci] != 0) {
                value /= options.std[ci];
            }
            const std::size_t index =
                channels_first ? static_cast<std::size_t>(c) * plane + p : p * static_cast<std::size_t>(channels) + ci;
            switch (input.dtype) {
            case DType::kF32:
                tensor.data_as<float>()[index] = static_cast<float>(value);
                break;
            case DType::kF16:
                tensor.data_as<f16>()[index] = f16(static_cast<float>(value));
                break;
            case DType::kU8:
                tensor.data_as<std::uint8_t>()[index] = static_cast<std::uint8_t>(std::clamp(value, 0.0, 255.0));
                break;
            default:
                return errors::unsupported("image input '{}' of type {}", input.name, to_string(input.dtype));
            }
        }
    }
    return tensor;
}

}  // namespace ii::cli
