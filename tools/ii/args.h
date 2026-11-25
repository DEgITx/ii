// SPDX-License-Identifier: MIT
#pragma once

#include <ii/status.h>

#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace ii::cli {

/// Parsed command-line arguments of one subcommand: `--flag`, `--option value`, `--option=value` and positionals.
class Args {
public:
    /// Parses `args`. `flags` are options without a value; every other `--name` takes a value.
    static Result<Args> parse(const std::vector<std::string>& args, const std::set<std::string, std::less<>>& flags,
                              const std::set<std::string, std::less<>>& options);

    bool flag(std::string_view name) const { return flags_.contains(std::string(name)); }
    std::optional<std::string> option(std::string_view name) const;
    Result<int> int_option(std::string_view name, int fallback) const;
    Result<double> double_option(std::string_view name, double fallback) const;
    const std::vector<std::string>& positionals() const { return positionals_; }

private:
    std::set<std::string, std::less<>> flags_;
    std::map<std::string, std::string, std::less<>> options_;
    std::vector<std::string> positionals_;
};

}  // namespace ii::cli
