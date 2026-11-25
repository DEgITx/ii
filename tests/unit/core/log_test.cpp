// SPDX-License-Identifier: MIT
#include <ii/log.h>

#include <gtest/gtest.h>

#include <format>
#include <string>
#include <utility>
#include <vector>

namespace ii_test {
/// Counts how often it is formatted.
struct FormatCounter {
    int* count;
};
}  // namespace ii_test

template <>
struct std::formatter<ii_test::FormatCounter> : std::formatter<int> {
    auto format(const ii_test::FormatCounter& counter, std::format_context& context) const {
        ++*counter.count;
        return std::formatter<int>::format(*counter.count, context);
    }
};

namespace ii {
namespace {

struct Captured {
    std::vector<std::pair<LogLevel, std::string>> messages;

    LogSink sink() {
        return [this](LogLevel level, std::string_view message) { messages.emplace_back(level, message); };
    }
};

TEST(Logger, FiltersByLevel) {
    Captured captured;
    const Logger logger(LogLevel::kWarning, captured.sink());
    logger.debug("hidden {}", 1);
    logger.info("hidden");
    logger.warning("shown {}", 2);
    logger.error("also shown");
    ASSERT_EQ(captured.messages.size(), 2u);
    EXPECT_EQ(captured.messages[0].first, LogLevel::kWarning);
    EXPECT_EQ(captured.messages[0].second, "shown 2");
    EXPECT_EQ(captured.messages[1].first, LogLevel::kError);
}

TEST(Logger, OffDisablesEverything) {
    Captured captured;
    const Logger logger(LogLevel::kOff, captured.sink());
    logger.error("nothing");
    logger.log(LogLevel::kOff, "nothing either");
    EXPECT_TRUE(captured.messages.empty());
}

TEST(Logger, DisabledLevelsAreNotFormatted) {
    int formatted = 0;
    Captured captured;
    const Logger logger(LogLevel::kError, captured.sink());
    logger.info("{}", ii_test::FormatCounter{&formatted});
    EXPECT_EQ(formatted, 0);
    logger.error("{}", ii_test::FormatCounter{&formatted});
    EXPECT_EQ(formatted, 1);
    ASSERT_EQ(captured.messages.size(), 1u);
    EXPECT_EQ(captured.messages[0].second, "1");
}

TEST(Logger, EmptySinkIsDisabled) {
    const Logger logger(LogLevel::kTrace, nullptr);
    EXPECT_FALSE(logger.enabled(LogLevel::kError));
    logger.error("does not crash");
}

TEST(Logger, LevelNames) {
    EXPECT_EQ(to_string(LogLevel::kTrace), "trace");
    EXPECT_EQ(to_string(LogLevel::kWarning), "warning");
    EXPECT_EQ(to_string(LogLevel::kOff), "off");
}

}  // namespace
}  // namespace ii
