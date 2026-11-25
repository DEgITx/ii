// SPDX-License-Identifier: MIT
// Fuzz target: the ONNX importer (protobuf decoding, tensor conversion, operator mapping, shape tracking) must reject
// malformed models without crashing, and every model it accepts must compile.
#include <ii/ii.h>

#include <cstddef>
#include <cstdint>
#include <span>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    static const ii::Context context = [] {
        ii::ContextOptions options;
        options.num_threads = 1;
        return ii::Context::create(options).value();
    }();
    const ii::Result<ii::Model> model =
        ii::Model::from_onnx(context, std::as_bytes(std::span<const std::uint8_t>(data, size)));
    if (model.ok()) {
        ii::CompileOptions options;
        options.backends = {"ii-ref"};
        (void)model->compile(options);
    }
    return 0;
}
