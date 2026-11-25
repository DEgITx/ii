// SPDX-License-Identifier: MIT
// Fuzz target: the JSON parser and the base64 decoder must reject malformed input without crashing.
#include "core/json.h"

#include <cstdint>
#include <string_view>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    const std::string_view text(reinterpret_cast<const char*>(data), size);
    const ii::Result<ii::Json> document = ii::Json::parse(text);
    if (document.ok()) {
        (void)document->size();
    }
    (void)ii::base64_decode(text);
    return 0;
}
