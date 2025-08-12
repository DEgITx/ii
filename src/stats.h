// Метрики стабильности кадра для видео-пайплайна.
//
// FpsCounter:
//   * tick() вызывается ровно один раз на кадр (после show_rgb / invoke);
//   * считает скользящее окно последних N интервалов между tick();
//   * выдаёт FPS, среднее время кадра, min/max и jitter (СКО).
//
// Метрики деталей:
//   * dt — миллисекунды между двумя соседними tick();
//   * jitter (СКО) — численная мера «дёрганья» кадра. < 1 мс — гладко,
//     5+ мс — заметно.

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <numeric>
#include <string>

class FpsCounter {
public:
    explicit FpsCounter(std::size_t window = 120) : window_(window) {}

    // Вызывать один раз на кадр.
    void tick() {
        auto now = clock::now();
        if (have_prev_) {
            double dt = std::chrono::duration<double, std::milli>(now - prev_).count();
            dts_.push_back(dt);
            if (dts_.size() > window_) dts_.pop_front();
        }
        prev_ = now;
        have_prev_ = true;
    }

    bool empty() const { return dts_.empty(); }

    double avg_ms() const {
        if (dts_.empty()) return 0.0;
        return std::accumulate(dts_.begin(), dts_.end(), 0.0) / dts_.size();
    }
    double min_ms() const {
        return dts_.empty() ? 0.0 : *std::min_element(dts_.begin(), dts_.end());
    }
    double max_ms() const {
        return dts_.empty() ? 0.0 : *std::max_element(dts_.begin(), dts_.end());
    }
    // Стандартное отклонение интервалов кадра — мера jitter.
    double jitter_ms() const {
        if (dts_.size() < 2) return 0.0;
        double m = avg_ms();
        double s = 0.0;
        for (double d : dts_) s += (d - m) * (d - m);
        return std::sqrt(s / dts_.size());
    }
    double fps() const {
        double a = avg_ms();
        return a > 0.0 ? 1000.0 / a : 0.0;
    }

    // Возвращает true ровно один раз за каждый interval_ms — удобно для
    // периодического лога без захламления stdout.
    bool log_due(double interval_ms = 1000.0) {
        auto now = clock::now();
        if (!last_log_set_) {
            last_log_ = now;
            last_log_set_ = true;
            return false;
        }
        double since = std::chrono::duration<double, std::milli>(now - last_log_).count();
        if (since >= interval_ms) {
            last_log_ = now;
            return true;
        }
        return false;
    }

    // Однострочное представление для оверлея/лога.
    std::string format() const {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "FPS %5.1f  dt %5.2f ms  min %5.2f  max %5.2f  jitter %5.2f",
            fps(), avg_ms(), min_ms(), max_ms(), jitter_ms());
        return buf;
    }

private:
    using clock = std::chrono::steady_clock;
    using time_point = clock::time_point;

    std::size_t window_;
    std::deque<double> dts_;
    time_point prev_{};
    bool       have_prev_    = false;
    time_point last_log_{};
    bool       last_log_set_ = false;
};
