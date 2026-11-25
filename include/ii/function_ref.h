// SPDX-License-Identifier: MIT
/// @file
/// `FunctionRef`: a non-owning, non-allocating reference to a callable.
#pragma once

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace ii {

template <class Signature>
class FunctionRef;

/// A lightweight reference to a callable object, for passing callbacks that are invoked before the callee returns.
///
/// Unlike `std::function` it never allocates and costs one indirect call. The referenced callable must outlive the
/// `FunctionRef` — never store one beyond the call it was passed to.
template <class R, class... Args>
class FunctionRef<R(Args...)> {
public:
    template <class F>
        requires(!std::is_same_v<std::remove_cvref_t<F>, FunctionRef> && std::is_invocable_r_v<R, F&, Args...>)
    FunctionRef(F&& callable) noexcept {  // NOLINT(bugprone-forwarding-reference-overload)
        using Callable = std::remove_reference_t<F>;
        if constexpr (std::is_function_v<Callable>) {
            // Function pointers cannot be stored as object pointers.
            target_.function = reinterpret_cast<void (*)()>(&callable);
            invoke_ = [](Target target, Args... args) -> R {
                return std::invoke(reinterpret_cast<Callable*>(target.function), std::forward<Args>(args)...);
            };
        } else {
            target_.object = const_cast<void*>(static_cast<const volatile void*>(std::addressof(callable)));
            invoke_ = [](Target target, Args... args) -> R {
                return std::invoke(*static_cast<Callable*>(target.object), std::forward<Args>(args)...);
            };
        }
    }

    R operator()(Args... args) const { return invoke_(target_, std::forward<Args>(args)...); }

private:
    union Target {
        void* object;
        void (*function)();
    };

    Target target_{};
    R (*invoke_)(Target, Args...) = nullptr;
};

}  // namespace ii
