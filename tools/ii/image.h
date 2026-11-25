// SPDX-License-Identifier: MIT
/// @file
/// Images as model inputs: decoding (stb_image: JPEG, PNG, BMP, TGA, GIF, PSD, HDR, PNM), resizing and conversion to
/// a tensor in the layout of a model input.
#pragma once

#include <ii/model.h>
#include <ii/status.h>
#include <ii/tensor.h>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ii::cli {

/// An 8-bit RGB image.
struct Image {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;  ///< height * width * 3, row-major
};

Result<Image> load_image(const std::filesystem::path& path);
Status save_image(const std::filesystem::path& path, const Image& image);  ///< PNG or JPEG by extension

/// True for file names with an image extension.
bool is_image_file(const std::string& name);

struct ImageOptions {
    int height = 0;  ///< target size; 0: from the model input, else the image size
    int width = 0;
    bool letterbox = false;    ///< keep the aspect ratio and pad (with 114, like YOLO) instead of stretching
    double scale = 1.0 / 255;  ///< pixel values are multiplied by this ...
    std::vector<double> mean;  ///< ... then (value - mean[c]) / std[c] per channel
    std::vector<double> std;
    bool bgr = false;  ///< channel order BGR instead of RGB
};

/// How an image was mapped into the input (for converting results back to image coordinates).
struct ImagePlacement {
    double scale = 1;  ///< input pixels per image pixel
    double pad_x = 0;
    double pad_y = 0;
};

/// Resizes `image` to `width` x `height` (bilinear, or letterboxed).
Image resize_image(const Image& image, int width, int height, bool letterbox, ImagePlacement* placement = nullptr);

/// Converts an image into a tensor for `input` (NCHW or NHWC with 1 or 3 channels, f32 / f16 / u8 / i8).
Result<Tensor> image_to_tensor(const Image& image, const TensorInfo& input, const ImageOptions& options,
                               ImagePlacement* placement = nullptr);

}  // namespace ii::cli
