// SPDX-License-Identifier: MIT
// Fuzz target: the GGUF and safetensors readers must reject malformed input without crashing; a GGUF file that parses
// is also read as a language model (hyper-parameters, tokenizer, graph construction), which must fail cleanly.
#include <ii/gguf.h>

#include "formats/safetensors.h"
#include "llm/graph.h"
#include "llm/hparams.h"
#include "llm/tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <span>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    const std::span<const std::byte> bytes = std::as_bytes(std::span<const std::uint8_t>(data, size));
    (void)ii::formats::SafetensorsFile::parse(bytes, nullptr);
    const ii::Result<ii::gguf::File> file = ii::gguf::File::parse(bytes);
    if (!file.ok()) {
        return 0;
    }
    for (const ii::gguf::TensorInfo& info : file->tensors()) {
        (void)file->tensor(info);
    }
    for (const auto& [key, value] : file->metadata()) {
        (void)value.to_string();
    }
    (void)ii::llm::detail::tokenizer_from_gguf(*file);
    const ii::Result<ii::llm::detail::HParams> hparams = ii::llm::detail::hparams_from_gguf(*file);
    if (hparams.ok() && hparams->layers <= 4 && hparams->embedding <= 4096) {
        static const ii::Context context = ii::Context::create().value();
        const auto weights = ii::llm::detail::gguf_weights(*file);
        ii::llm::detail::GraphOptions options;
        options.context = 16;
        (void)ii::llm::detail::build_model(context, *hparams, *weights, options, nullptr);
    }
    return 0;
}
