// Простой CSV-экспортёр для замеров раннера (бенчмарк / FPS / compare).
//
// Особенности:
//   * RAII: открывается в open(), закрывается в деструкторе;
//   * шапка с метаданными в виде комментариев `# key=value` —
//     pandas/numpy/gnuplot читают их через `comment='#'`;
//   * printf-style API row()/writef() — никакой динамической аллокации
//     на горячем пути (важно для бенчмарка на тысячах итераций);
//   * без зависимостей кроме <cstdio>; работает на любой целевой
//     платформе, где собирается основной раннер.
//
// Использование:
//   CsvExport e;
//   if (e.open("run.bench.csv")) {
//       e.meta("model",    "%s", path.c_str());
//       e.meta("delegate", "%s", "NPU");
//       e.meta("started",  "%s", iso_timestamp_now().c_str());
//       e.header("run,ms");
//       for (int i = 0; i < runs; ++i)
//           e.writef("%d,%.6f", i, latencies[i]);
//   }

#pragma once

#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <string>

class CsvExport {
public:
    CsvExport() = default;
    CsvExport(const CsvExport&) = delete;
    CsvExport& operator=(const CsvExport&) = delete;
    ~CsvExport() { close(); }

    // Открыть файл на запись (перезаписывает существующий). Возвращает
    // false и печатает причину в stderr, если открытие не удалось —
    // при этом is_open() == false и все последующие writef/meta/header
    // молча игнорируются (no-op), чтобы вызывающий код можно было
    // писать без проверок «если экспорт включён».
    bool open(const std::string& path) {
        close();
        f_ = std::fopen(path.c_str(), "wb");
        if (!f_) {
            std::fprintf(stderr,
                "CSV export: не удалось открыть %s на запись\n",
                path.c_str());
            return false;
        }
        path_ = path;
        return true;
    }

    bool is_open() const { return f_ != nullptr; }
    const std::string& path() const { return path_; }

    void close() {
        if (f_) {
            std::fclose(f_);
            f_ = nullptr;
        }
    }

    // Строка метаданных вида `# key=value`. value форматируется printf.
    // Должна вызываться до header().
    void meta(const char* key, const char* fmt, ...) {
        if (!f_) return;
        std::fprintf(f_, "# %s=", key);
        va_list ap;
        va_start(ap, fmt);
        std::vfprintf(f_, fmt, ap);
        va_end(ap);
        std::fputc('\n', f_);
    }

    // Произвольная строка комментария (без префикса key=).
    void comment(const char* fmt, ...) {
        if (!f_) return;
        std::fputs("# ", f_);
        va_list ap;
        va_start(ap, fmt);
        std::vfprintf(f_, fmt, ap);
        va_end(ap);
        std::fputc('\n', f_);
    }

    // Шапка таблицы — обычная CSV-строка с именами колонок.
    void header(const char* cols) {
        if (!f_) return;
        std::fprintf(f_, "%s\n", cols);
    }

    // Запись строки данных: printf-формат + перенос строки.
    void writef(const char* fmt, ...) {
        if (!f_) return;
        va_list ap;
        va_start(ap, fmt);
        std::vfprintf(f_, fmt, ap);
        va_end(ap);
        std::fputc('\n', f_);
    }

    void flush() { if (f_) std::fflush(f_); }

private:
    std::FILE*  f_ = nullptr;
    std::string path_;
};

// ISO-8601 UTC timestamp без миллисекунд — компактен, сортируется
// лексикографически. Кросс-платформенный (gmtime_s на Windows,
// gmtime_r на POSIX).
inline std::string iso_timestamp_now() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

// Экранирует строку для CSV: оборачивает в кавычки, если содержит
// запятую/кавычку/перенос строки; внутренние кавычки удваиваются.
// Используется для имён моделей/тензоров с произвольными символами.
inline std::string csv_escape(const std::string& s) {
    bool need = false;
    for (char c : s) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') {
            need = true; break;
        }
    }
    if (!need) return s;
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char c : s) {
        if (c == '"') out.push_back('"');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}
