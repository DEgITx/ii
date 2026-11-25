// SPDX-License-Identifier: MIT
/// @file
/// Error handling: `Status`, `Result<T>` and the `II_TRY` / `II_ASSIGN_OR_RETURN` propagation macros.
///
/// ii never throws exceptions across its API. Fallible functions return `Status` (no value) or `Result<T>`
/// (a value or an error). Both are `[[nodiscard]]`.
///
/// @code
/// ii::Result<Model> load(const std::filesystem::path& path) {
///     II_ASSIGN_OR_RETURN(auto file, MappedFile::open(path));
///     if (file.size() < kHeaderSize) {
///         return ii::errors::data_corrupted("'{}' is too small to be a model", path.string());
///     }
///     II_TRY(validate(file));
///     return Model(std::move(file));
/// }
/// @endcode
#pragma once

#include <ii/export.h>

#include <format>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace ii {

/// Error category of a `Status`.
enum class StatusCode : int {
    kOk = 0,           ///< Success.
    kCancelled,        ///< The operation was cancelled by the caller.
    kInvalidArgument,  ///< The caller passed an invalid argument.
    kNotFound,         ///< A requested entity (file, symbol, tensor, ...) does not exist.
    kAlreadyExists,    ///< An entity that should be created already exists.
    kOutOfRange,       ///< A value or index is outside of the valid range.
    kOutOfMemory,      ///< A memory allocation failed.
    kUnsupported,      ///< The operation is not supported (by the platform, backend, model, ...).
    kUnavailable,      ///< A resource is temporarily or permanently unavailable (device, plugin runtime, ...).
    kIoError,          ///< An operating-system level I/O error.
    kDataCorrupted,    ///< Input data (a model file, for example) is malformed.
    kInternal,         ///< An internal invariant was violated; indicates a bug in ii.
};

/// Returns the name of a status code, e.g. "NotFound".
II_API std::string_view to_string(StatusCode code) noexcept;

/// The outcome of an operation: success, or an error code with a human-readable message.
///
/// A successful `Status` holds no allocation; copying and moving it is cheap.
class [[nodiscard]] Status {
public:
    /// Creates a successful status.
    Status() noexcept = default;

    /// Creates a status with the given code and message. `StatusCode::kOk` produces a successful status.
    Status(StatusCode code, std::string message) {
        if (code != StatusCode::kOk) {
            rep_ = std::make_unique<Rep>(Rep{code, std::move(message)});
        }
    }

    Status(const Status& other) : rep_(other.rep_ ? std::make_unique<Rep>(*other.rep_) : nullptr) {}
    Status& operator=(const Status& other) {
        if (this != &other) {
            rep_ = other.rep_ ? std::make_unique<Rep>(*other.rep_) : nullptr;
        }
        return *this;
    }
    Status(Status&&) noexcept = default;
    Status& operator=(Status&&) noexcept = default;
    ~Status() = default;

    /// True on success.
    bool ok() const noexcept { return rep_ == nullptr; }

    StatusCode code() const noexcept { return rep_ ? rep_->code : StatusCode::kOk; }

    /// The error message; empty on success.
    std::string_view message() const noexcept { return rep_ ? std::string_view(rep_->message) : std::string_view(); }

    /// "OK" on success, otherwise "<Code>: <message>".
    std::string to_string() const {
        if (ok()) {
            return "OK";
        }
        std::string result(ii::to_string(rep_->code));
        if (!rep_->message.empty()) {
            result += ": ";
            result += rep_->message;
        }
        return result;
    }

    /// Prepends context to the message of an error ("<context>: <message>"). Does nothing on success.
    Status& add_context(std::string_view context) {
        if (rep_) {
            std::string message(context);
            if (!rep_->message.empty()) {
                message += ": ";
                message += rep_->message;
            }
            rep_->message = std::move(message);
        }
        return *this;
    }

private:
    struct Rep {
        StatusCode code;
        std::string message;
    };
    std::unique_ptr<Rep> rep_;
};

/// Returns a successful status.
inline Status ok_status() noexcept {
    return {};
}

/// Creates an error status with a `std::format`-style message.
template <class... Args>
Status make_status(StatusCode code, std::format_string<Args...> format, Args&&... args) {
    return Status(code, std::format(format, std::forward<Args>(args)...));
}

/// Shorthands for creating error statuses: `return ii::errors::not_found("no tensor '{}'", name);`
namespace errors {

#define II_DEFINE_ERROR_FACTORY(function, code)                                                                        \
    template <class... Args>                                                                                           \
    Status function(std::format_string<Args...> format, Args&&... args) {                                              \
        return Status(StatusCode::code, std::format(format, std::forward<Args>(args)...));                             \
    }

II_DEFINE_ERROR_FACTORY(cancelled, kCancelled)
II_DEFINE_ERROR_FACTORY(invalid_argument, kInvalidArgument)
II_DEFINE_ERROR_FACTORY(not_found, kNotFound)
II_DEFINE_ERROR_FACTORY(already_exists, kAlreadyExists)
II_DEFINE_ERROR_FACTORY(out_of_range, kOutOfRange)
II_DEFINE_ERROR_FACTORY(out_of_memory, kOutOfMemory)
II_DEFINE_ERROR_FACTORY(unsupported, kUnsupported)
II_DEFINE_ERROR_FACTORY(unavailable, kUnavailable)
II_DEFINE_ERROR_FACTORY(io_error, kIoError)
II_DEFINE_ERROR_FACTORY(data_corrupted, kDataCorrupted)
II_DEFINE_ERROR_FACTORY(internal, kInternal)

#undef II_DEFINE_ERROR_FACTORY

}  // namespace errors

namespace detail {
/// Terminates the process after `Result::value()` was called on an error.
[[noreturn]] II_API void result_value_failed(const Status& status) noexcept;
}  // namespace detail

/// Either a value of type `T` or an error `Status`.
///
/// Accessing the value of an error result with `value()` terminates the process with the error message;
/// `operator*` and `operator->` do not check.
template <class T>
class [[nodiscard]] Result {
    static_assert(!std::is_reference_v<T>, "Result<T&> is not supported; use a pointer or std::reference_wrapper");
    static_assert(!std::is_same_v<std::remove_cv_t<T>, Status>, "Result<Status> is not meaningful; use Status");

public:
    using value_type = T;

    /// Constructs a successful result from a value.
    template <class U = T>
        requires(std::is_constructible_v<T, U &&> && !std::is_same_v<std::remove_cvref_t<U>, Result> &&
                 !std::is_same_v<std::remove_cvref_t<U>, Status> &&
                 !std::is_same_v<std::remove_cvref_t<U>, std::in_place_t>)
    Result(U&& value) : value_(std::in_place, std::forward<U>(value)) {}

    /// Constructs the value in place.
    template <class... Args>
    explicit Result(std::in_place_t, Args&&... args) : value_(std::in_place, std::forward<Args>(args)...) {}

    /// Constructs an error result. Passing a successful status is a programming error and produces an internal error.
    Result(Status status) : status_(std::move(status)) {
        if (status_.ok()) {
            status_ = Status(StatusCode::kInternal, "Result constructed from a successful Status without a value");
        }
    }

    /// True when the result holds a value.
    bool ok() const noexcept { return value_.has_value(); }

    /// The error; a successful status when the result holds a value.
    const Status& status() const& noexcept { return status_; }
    Status status() && { return std::move(status_); }

    // Accessors below are checked by check() or documented as unchecked; the optional is never read when empty.
    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    T& value() & {
        check();
        return *value_;
    }
    const T& value() const& {
        check();
        return *value_;
    }
    T&& value() && {
        check();
        return std::move(*value_);
    }

    /// Returns the value, or `fallback` on error.
    template <class U>
    T value_or(U&& fallback) const& {
        return value_ ? *value_ : static_cast<T>(std::forward<U>(fallback));
    }
    template <class U>
    T value_or(U&& fallback) && {
        return value_ ? std::move(*value_) : static_cast<T>(std::forward<U>(fallback));
    }

    T& operator*() & noexcept { return *value_; }
    const T& operator*() const& noexcept { return *value_; }
    T&& operator*() && noexcept { return std::move(*value_); }
    T* operator->() noexcept { return &*value_; }
    const T* operator->() const noexcept { return &*value_; }
    // NOLINTEND(bugprone-unchecked-optional-access)

private:
    void check() const {
        if (!value_) [[unlikely]] {
            detail::result_value_failed(status_);
        }
    }

    std::optional<T> value_;
    Status status_;
};

namespace detail {
inline Status status_of(const Status& status) {
    return status;
}
template <class T>
const Status& status_of(const Result<T>& result) noexcept {
    return result.status();
}
}  // namespace detail

}  // namespace ii

#define II_DETAIL_CONCAT_IMPL(a, b) a##b
#define II_DETAIL_CONCAT(a, b) II_DETAIL_CONCAT_IMPL(a, b)

/// Evaluates an expression returning `Status` or `Result<T>`; on error returns the error from the enclosing function.
#define II_TRY(expr) II_DETAIL_TRY_IMPL(II_DETAIL_CONCAT(ii_try_result_, __COUNTER__), expr)

// The temporary has a unique name so that II_TRY can be nested (e.g. inside a lambda passed to a function whose
// result is itself checked with II_TRY) without shadowing warnings. `tmp` is an identifier, so it cannot be
// parenthesized.
// NOLINTNEXTLINE(bugprone-macro-parentheses)
#define II_DETAIL_TRY_IMPL(tmp, expr)                                                                                  \
    do {                                                                                                               \
        auto&& tmp = (expr);                                                                                           \
        if (!tmp.ok()) [[unlikely]] {                                                                                  \
            return ::ii::detail::status_of(tmp);                                                                       \
        }                                                                                                              \
    } while (false)

/// Evaluates an expression returning `Result<T>`; on error returns the error from the enclosing function, otherwise
/// assigns (or declares, e.g. `auto x`) `lhs` from the value.
#define II_ASSIGN_OR_RETURN(lhs, expr)                                                                                 \
    II_DETAIL_ASSIGN_OR_RETURN_IMPL(II_DETAIL_CONCAT(ii_assign_result_, __COUNTER__), lhs, expr)

// `tmp` is an identifier and `lhs` may be a declaration ("auto x"), so neither can be parenthesized.
// NOLINTBEGIN(bugprone-macro-parentheses)
#define II_DETAIL_ASSIGN_OR_RETURN_IMPL(tmp, lhs, expr)                                                                \
    auto tmp = (expr);                                                                                                 \
    if (!tmp.ok()) [[unlikely]] {                                                                                      \
        return std::move(tmp).status();                                                                                \
    }                                                                                                                  \
    lhs = std::move(tmp).value()
// NOLINTEND(bugprone-macro-parentheses)
