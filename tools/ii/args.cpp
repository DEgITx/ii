// SPDX-License-Identifier: MIT
#include "args.h"

#include <charconv>
#include <cstdlib>

namespace ii::cli {

Result<Args> Args::parse(const std::vector<std::string>& args, const std::set<std::string, std::less<>>& flags,
                         const std::set<std::string, std::less<>>& options) {
    Args result;
    bool only_positionals = false;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (only_positionals || !arg.starts_with("--")) {
            result.positionals_.push_back(arg);
            continue;
        }
        if (arg == "--") {
            only_positionals = true;
            continue;
        }
        std::string name = arg.substr(2);
        std::optional<std::string> value;
        if (const auto equals = name.find('='); equals != std::string::npos) {
            value = name.substr(equals + 1);
            name.resize(equals);
        }
        if (flags.contains(name)) {
            if (value) {
                return errors::invalid_argument("option --{} does not take a value", name);
            }
            result.flags_.insert(name);
        } else if (options.contains(name)) {
            if (!value) {
                if (i + 1 >= args.size()) {
                    return errors::invalid_argument("option --{} requires a value", name);
                }
                value = args[++i];
            }
            result.options_[name] = *value;
        } else {
            return errors::invalid_argument("unknown option --{}", name);
        }
    }
    return result;
}

std::optional<std::string> Args::option(std::string_view name) const {
    if (const auto it = options_.find(name); it != options_.end()) {
        return it->second;
    }
    return std::nullopt;
}

Result<int> Args::int_option(std::string_view name, int fallback) const {
    const auto text = option(name);
    if (!text) {
        return fallback;
    }
    int value = 0;
    const auto [end, error] = std::from_chars(text->data(), text->data() + text->size(), value);
    if (error != std::errc() || end != text->data() + text->size()) {
        return errors::invalid_argument("option --{} expects an integer, got '{}'", name, *text);
    }
    return value;
}

Result<double> Args::double_option(std::string_view name, double fallback) const {
    const auto text = option(name);
    if (!text) {
        return fallback;
    }
    // std::from_chars for double is not available everywhere yet (older libc++), so use strtod.
    char* end = nullptr;
    const double value = std::strtod(text->c_str(), &end);
    if (end != text->c_str() + text->size() || text->empty()) {
        return errors::invalid_argument("option --{} expects a number, got '{}'", name, *text);
    }
    return value;
}

}  // namespace ii::cli
