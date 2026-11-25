// SPDX-License-Identifier: MIT
// Fuzz target: the IR text parser. Any input must either fail with a Status or produce a valid graph whose printed
// form parses back to the same text.
#include "ir/text.h"

#include <cstdint>
#include <cstdlib>
#include <string_view>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    const std::string_view text(reinterpret_cast<const char*>(data), size);
    ii::Result<std::unique_ptr<ii::ir::Graph>> graph = ii::ir::parse_graph(text);
    if (!graph.ok()) {
        return 0;
    }
    const ii::ir::PrintOptions everything{-1};
    const std::string printed = (*graph)->to_text(everything);
    ii::Result<std::unique_ptr<ii::ir::Graph>> again = ii::ir::parse_graph(printed);
    if (!again.ok() || (*again)->to_text(everything) != printed) {
        std::abort();  // the printer and the parser disagree
    }
    return 0;
}
