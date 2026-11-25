// SPDX-License-Identifier: MIT
// Token sampling: repetition penalty, top-k, top-p, min-p and temperature (the order of llama.cpp's default chain),
// then a draw from the renormalized distribution. Temperature <= 0 is greedy decoding.
#include <ii/llm.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace ii::llm {

namespace {

std::uint64_t splitmix64(std::uint64_t& x) {
    std::uint64_t z = (x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

}  // namespace

Sampler::Sampler(const SamplingOptions& options) : options_(options) {
    std::uint64_t seed = options.seed;
    if (seed == 0) {
        std::random_device device;
        seed = (static_cast<std::uint64_t>(device()) << 32) ^ device();
    }
    state_[0] = splitmix64(seed);
    state_[1] = splitmix64(seed);
}

Token Sampler::sample(std::span<const float> logits, std::span<const Token> history) {
    last_ = sample_impl(logits, history);
    return last_;
}

Token Sampler::sample_impl(std::span<const float> logits, std::span<const Token> history) {
    const auto vocab = static_cast<Token>(logits.size());
    if (vocab == 0) {
        return -1;
    }
    candidates_.resize(logits.size());
    for (Token i = 0; i < vocab; ++i) {
        candidates_[static_cast<std::size_t>(i)] = {logits[static_cast<std::size_t>(i)], i};
    }
    // Repetition penalty over the last repeat_last_n tokens (each token once).
    if (options_.repeat_penalty != 1.0f && options_.repeat_last_n != 0) {
        const std::size_t window = options_.repeat_last_n < 0
                                       ? history.size()
                                       : std::min(history.size(), static_cast<std::size_t>(options_.repeat_last_n));
        for (std::size_t i = history.size() - window; i < history.size(); ++i) {
            const Token token = history[i];
            if (token < 0 || token >= vocab) {
                continue;
            }
            float& logit = candidates_[static_cast<std::size_t>(token)].first;
            if (std::isnan(logit)) {
                continue;
            }
            logit = logit > 0 ? logit / options_.repeat_penalty : logit * options_.repeat_penalty;
            // Mark as penalized so a repeated token is not penalized twice.
            candidates_[static_cast<std::size_t>(token)].second = -token - 1;
        }
        for (auto& [logit, token] : candidates_) {
            if (token < 0) {
                token = -token - 1;
            }
        }
    }
    auto larger = [](const std::pair<float, Token>& a, const std::pair<float, Token>& b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    };
    if (options_.temperature <= 0.0f) {
        return std::min_element(candidates_.begin(), candidates_.end(), larger)->second;
    }
    std::size_t count = candidates_.size();
    if (options_.top_k > 0 && static_cast<std::size_t>(options_.top_k) < count) {
        count = static_cast<std::size_t>(options_.top_k);
        std::partial_sort(candidates_.begin(), candidates_.begin() + static_cast<std::ptrdiff_t>(count),
                          candidates_.end(), larger);
    } else {
        std::sort(candidates_.begin(), candidates_.end(), larger);
    }
    // Probabilities of the (sorted) candidates at temperature 1, for top-p and min-p.
    const float max_logit = candidates_[0].first;
    double sum = 0;
    for (std::size_t i = 0; i < count; ++i) {
        sum += std::exp(static_cast<double>(candidates_[i].first - max_logit));
    }
    if (options_.top_p < 1.0f) {
        double cumulative = 0;
        for (std::size_t i = 0; i < count; ++i) {
            cumulative += std::exp(static_cast<double>(candidates_[i].first - max_logit)) / sum;
            if (cumulative >= options_.top_p) {
                count = i + 1;
                break;
            }
        }
    }
    if (options_.min_p > 0.0f) {
        // p_i >= min_p * p_max  <=>  logit_i >= max_logit + log(min_p)
        const float threshold = max_logit + std::log(options_.min_p);
        std::size_t kept = 1;
        while (kept < count && candidates_[kept].first >= threshold) {
            ++kept;
        }
        count = kept;
    }
    // Temperature, then draw.
    double total = 0;
    for (std::size_t i = 0; i < count; ++i) {
        total += std::exp(static_cast<double>(candidates_[i].first - max_logit) / options_.temperature);
    }
    // xorshift128+
    std::uint64_t s1 = state_[0];
    const std::uint64_t s0 = state_[1];
    state_[0] = s0;
    s1 ^= s1 << 23;
    state_[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    const double u = static_cast<double>((state_[1] + s0) >> 11) * 0x1.0p-53;
    double target = u * total;
    for (std::size_t i = 0; i < count; ++i) {
        target -= std::exp(static_cast<double>(candidates_[i].first - max_logit) / options_.temperature);
        if (target < 0) {
            return candidates_[i].second;
        }
    }
    return candidates_[count - 1].second;
}

}  // namespace ii::llm
