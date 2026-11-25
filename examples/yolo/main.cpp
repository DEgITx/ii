// SPDX-License-Identifier: MIT
// Object detection with a YOLOv8 / YOLO11 model exported to ONNX (`yolo export model=yolov8n.pt format=onnx`):
// load an image, letterbox it to the model input, run the model, decode boxes, apply non-maximum suppression and
// draw the result.
//
//   ii_example_yolo yolov8n.onnx bus.jpg [result.jpg] [--conf 0.25] [--iou 0.45]
#include <ii/ii.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

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

namespace {

constexpr std::array<const char*, 80> kCocoClasses = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

struct Image {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> rgb;  // height * width * 3
};

struct Detection {
    float x0, y0, x1, y1;  // in image pixels
    float score;
    int label;
};

bool load_image(const std::string& path, Image& image) {
    int channels = 0;
    unsigned char* data = stbi_load(path.c_str(), &image.width, &image.height, &channels, 3);
    if (data == nullptr) {
        return false;
    }
    image.rgb.assign(data, data + static_cast<std::size_t>(image.width) * image.height * 3);
    stbi_image_free(data);
    return true;
}

/// Letterboxes the image into a size x size NCHW f32 tensor (gray padding, values in [0, 1]); returns the scale and
/// padding to map boxes back to the image.
ii::Tensor letterbox(const Image& image, int size, float& scale, float& pad_x, float& pad_y) {
    scale = std::min(static_cast<float>(size) / image.width, static_cast<float>(size) / image.height);
    const int w = std::max(1, static_cast<int>(std::lround(image.width * scale)));
    const int h = std::max(1, static_cast<int>(std::lround(image.height * scale)));
    pad_x = static_cast<float>((size - w) / 2);
    pad_y = static_cast<float>((size - h) / 2);
    ii::Tensor tensor = ii::Tensor::empty(ii::DType::kF32, ii::Shape{1, 3, size, size}).value();
    float* out = tensor.data_as<float>();
    const std::size_t plane = static_cast<std::size_t>(size) * size;
    std::fill(out, out + 3 * plane, 114.0f / 255.0f);
    for (int y = 0; y < h; ++y) {
        // Bilinear sampling with half-pixel centers.
        const float sy = std::clamp((y + 0.5f) / scale - 0.5f, 0.0f, image.height - 1.0f);
        const int y0 = static_cast<int>(sy);
        const int y1 = std::min(y0 + 1, image.height - 1);
        const float fy = sy - y0;
        for (int x = 0; x < w; ++x) {
            const float sx = std::clamp((x + 0.5f) / scale - 0.5f, 0.0f, image.width - 1.0f);
            const int x0 = static_cast<int>(sx);
            const int x1 = std::min(x0 + 1, image.width - 1);
            const float fx = sx - x0;
            for (int c = 0; c < 3; ++c) {
                const auto at = [&](int yy, int xx) {
                    return static_cast<float>(image.rgb[(static_cast<std::size_t>(yy) * image.width + xx) * 3 + c]);
                };
                const float v = (at(y0, x0) * (1 - fx) + at(y0, x1) * fx) * (1 - fy) +
                                (at(y1, x0) * (1 - fx) + at(y1, x1) * fx) * fy;
                out[c * plane + (static_cast<std::size_t>(y) + static_cast<std::size_t>(pad_y)) * size +
                    static_cast<std::size_t>(x) + static_cast<std::size_t>(pad_x)] = v / 255.0f;
            }
        }
    }
    return tensor;
}

float iou(const Detection& a, const Detection& b) {
    const float w = std::max(0.0f, std::min(a.x1, b.x1) - std::max(a.x0, b.x0));
    const float h = std::max(0.0f, std::min(a.y1, b.y1) - std::max(a.y0, b.y0));
    const float inter = w * h;
    const float uni = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter;
    return uni > 0 ? inter / uni : 0.0f;
}

/// Decodes the YOLOv8 head output [1, 4 + classes, anchors] (cx, cy, w, h in input pixels, then class scores) and
/// applies per-class non-maximum suppression.
std::vector<Detection> decode(const ii::Tensor& output, float conf, float iou_threshold, float scale, float pad_x,
                              float pad_y, const Image& image) {
    const std::int64_t attributes = output.shape()[1];
    const std::int64_t anchors = output.shape()[2];
    const std::int64_t classes = attributes - 4;
    const float* p = output.data_as<float>();
    std::vector<Detection> candidates;
    for (std::int64_t a = 0; a < anchors; ++a) {
        int best = 0;
        float best_score = 0;
        for (std::int64_t c = 0; c < classes; ++c) {
            const float s = p[(4 + c) * anchors + a];
            if (s > best_score) {
                best_score = s;
                best = static_cast<int>(c);
            }
        }
        if (best_score < conf) {
            continue;
        }
        const float cx = p[0 * anchors + a];
        const float cy = p[1 * anchors + a];
        const float w = p[2 * anchors + a];
        const float h = p[3 * anchors + a];
        const auto to_image_x = [&](float v) { return std::clamp((v - pad_x) / scale, 0.0f, float(image.width)); };
        const auto to_image_y = [&](float v) { return std::clamp((v - pad_y) / scale, 0.0f, float(image.height)); };
        candidates.push_back({to_image_x(cx - w / 2), to_image_y(cy - h / 2), to_image_x(cx + w / 2),
                              to_image_y(cy + h / 2), best_score, best});
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });
    std::vector<Detection> kept;
    for (const Detection& d : candidates) {
        const bool suppressed = std::any_of(kept.begin(), kept.end(), [&](const Detection& k) {
            return k.label == d.label && iou(k, d) > iou_threshold;
        });
        if (!suppressed) {
            kept.push_back(d);
        }
    }
    return kept;
}

void draw(Image& image, const Detection& d) {
    static const unsigned char kPalette[][3] = {{255, 56, 56},  {255, 157, 151}, {255, 112, 31}, {255, 178, 29},
                                                {207, 210, 49}, {72, 249, 10},   {146, 204, 23}, {61, 219, 134},
                                                {26, 147, 52},  {0, 212, 187},   {44, 153, 168}, {0, 194, 255}};
    const unsigned char* color = kPalette[d.label % 12];
    const int thickness = std::max(2, std::min(image.width, image.height) / 300);
    const int x0 = static_cast<int>(d.x0);
    const int y0 = static_cast<int>(d.y0);
    const int x1 = std::min(static_cast<int>(d.x1), image.width - 1);
    const int y1 = std::min(static_cast<int>(d.y1), image.height - 1);
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            const bool edge = y - y0 < thickness || y1 - y < thickness || x - x0 < thickness || x1 - x < thickness;
            if (edge) {
                unsigned char* px = &image.rgb[(static_cast<std::size_t>(y) * image.width + x) * 3];
                px[0] = color[0];
                px[1] = color[1];
                px[2] = color[2];
            }
        }
    }
}

int fail(const ii::Status& status) {
    std::fprintf(stderr, "error: %s\n", status.to_string().c_str());
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> positional;
    float conf = 0.25f;
    float iou_threshold = 0.45f;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--conf" || arg == "--iou") && i + 1 < argc) {
            (arg == "--conf" ? conf : iou_threshold) = std::strtof(argv[++i], nullptr);
        } else {
            positional.push_back(arg);
        }
    }
    if (positional.size() < 2) {
        std::fprintf(stderr, "usage: ii_example_yolo <model.onnx> <image> [result.jpg] [--conf 0.25] [--iou 0.45]\n");
        return 2;
    }

    ii::Result<ii::Context> context = ii::Context::create();
    if (!context.ok()) {
        return fail(context.status());
    }
    ii::Result<ii::Model> model =
        ii::Model::load(*context, std::filesystem::path(std::u8string(positional[0].begin(), positional[0].end())));
    if (!model.ok()) {
        return fail(model.status());
    }
    ii::Result<ii::Program> program = model->compile();
    if (!program.ok()) {
        return fail(program.status());
    }
    ii::Result<ii::Session> session = program->new_session();
    if (!session.ok()) {
        return fail(session.status());
    }
    const ii::TensorInfo input = program->inputs().at(0);
    const std::optional<std::int64_t> size = input.shape.size() == 4 ? input.shape[3].constant() : std::nullopt;

    Image image;
    if (!load_image(positional[1], image)) {
        std::fprintf(stderr, "error: cannot read image '%s': %s\n", positional[1].c_str(), stbi_failure_reason());
        return 1;
    }
    float scale = 1;
    float pad_x = 0;
    float pad_y = 0;
    const ii::Tensor tensor = letterbox(image, static_cast<int>(size.value_or(640)), scale, pad_x, pad_y);
    if (ii::Status status = session->set_input(input.name, tensor); !status.ok()) {
        return fail(status);
    }
    if (ii::Status status = session->run(); !status.ok()) {
        return fail(status);
    }
    ii::Result<ii::Tensor> output = session->output(program->outputs().at(0).name);
    if (!output.ok()) {
        return fail(output.status());
    }
    if (output->rank() != 3 || output->shape()[1] <= 4) {
        std::fprintf(stderr, "error: expected a YOLOv8 head output [1, 4 + classes, anchors], got %s\n",
                     output->shape().to_string().c_str());
        return 1;
    }
    const std::vector<Detection> detections = decode(*output, conf, iou_threshold, scale, pad_x, pad_y, image);

    std::printf("%zu objects\n", detections.size());
    for (const Detection& d : detections) {
        const char* name =
            d.label < static_cast<int>(kCocoClasses.size()) ? kCocoClasses[static_cast<std::size_t>(d.label)] : "?";
        std::printf("  %-14s %.2f  [%.0f, %.0f, %.0f, %.0f]\n", name, d.score, d.x0, d.y0, d.x1, d.y1);
        draw(image, d);
    }
    if (positional.size() >= 3) {
        const std::string& out = positional[2];
        const bool png = out.size() > 4 && out.substr(out.size() - 4) == ".png";
        const int ok =
            png ? stbi_write_png(out.c_str(), image.width, image.height, 3, image.rgb.data(), image.width * 3)
                : stbi_write_jpg(out.c_str(), image.width, image.height, 3, image.rgb.data(), 90);
        if (ok == 0) {
            std::fprintf(stderr, "error: cannot write '%s'\n", out.c_str());
            return 1;
        }
        std::printf("result written to %s\n", out.c_str());
    }
    return 0;
}
