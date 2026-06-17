// Лёгкий монитор использования CPU и памяти процесса (и системы).
//
// Назначение:
//   Опциональный замер «снаружи» инференса: сколько RAM держит процесс,
//   во сколько ядер он реально упирается, насколько занят хост в целом.
//   Полезно отделить нагрузку CPU-препроцессинга / постобработки YOLO /
//   делегата от чистого NPU (при --no-delegate сравнивать с делегатом).
//
// API:
//   SysMonitor mon;
//   if (mon.init()) {                  // baseline-семпл; false на не-Linux
//       ...                            // полезная работа
//       SysSample s = mon.sample();    // дельта от предыдущего sample()
//       printf("RSS %ld kB, CPU %.1f%%\n", s.rss_kb, s.cpu_proc_pct);
//   }
//
// Реализация:
//   * Linux: парсим /proc/self/{stat,status,statm} и /proc/stat. Никаких
//     зависимостей кроме libc. CPU-проценты считаются как дельта тиков
//     /proc/self/stat (utime+stime) поделённая на дельту суммарных тиков
//     /proc/stat (для системы) и переведённая в проценты одного ядра
//     для процесса (>100% = используем больше одного ядра, что нормально
//     для многопоточного интерпретатора TFLite).
//   * Опционально — загрузка аппаратных ускорителей (GPU / NPU) «помимо
//     CPU». Решение универсальное: монитор сканирует набор типовых
//     sysfs/debugfs-узлов утилизации и регистрирует те, что присутствуют
//     и читаются на данной платформе (см. detect_accelerators_). Формат
//     узла определяется автоматически по содержимому:
//       - счётчики busy/idle (накопительные, наносекунды) → util =
//         d_busy / (d_busy + d_idle);
//       - один накопительный busy-счётчик (нс) → util = d_busy / d_wall;
//       - готовый процент (число 0..100, опц. со знаком «%») → как есть.
//     Любой узел можно задать явно через переменные окружения, не трогая
//     код (полезно для нестандартных драйверов):
//       II_SYSMON_GPU=/path   — узел утилизации GPU;
//       II_SYSMON_NPU=/path   — узел утилизации NPU;
//       II_SYSMON_ACCEL=label:gpu:/path,label2:npu:/path2  — произвольный
//                               список (kind = gpu|npu|other).
//     Если ускоритель не найден — он просто не появляется в отчёте; на
//     платформах без /proc раздел ускорителей пуст, как и весь sysmon.
//   * Не-Linux (Windows/macOS-сборки на dev-хосте): init() возвращает
//     false, sample() выдаёт нулевой SysSample{ok=false}. Так раннер
//     компилируется без #ifdef в вызывающем коде, а на платформах без
//     /proc просто молча пропускает блок sysmon.
//
// Замечания:
//   * Первый sample() после init() может вернуть короткий wall_ms и,
//     соответственно, неинформативные проценты — лучше делать первый
//     «прогревочный» вызов перед началом замеров.
//   * Все значения памяти — в килобайтах (как в /proc/self/status, чтобы
//     не пересчитывать туда-сюда).

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

// Класс ускорителя — только для маршрутизации в фиксированные колонки
// отчёта (gpu / npu) и агрегаты. Other — задел на прочие блоки (DSP/ISP),
// в CSV сейчас не выносится, но виден в построчном логе.
enum class AccelKind { Gpu, Npu, Other };

inline const char* accel_kind_name(AccelKind k) {
    switch (k) {
        case AccelKind::Gpu: return "gpu";
        case AccelKind::Npu: return "npu";
        default:             return "accel";
    }
}

// Один замер утилизации ускорителя за интервал между sample().
// util_pct ∈ [0..100]; 100 = блок занят весь интервал.
struct AccelSample {
    std::string label;       // человекочитаемая метка узла («gpu», «npu», …)
    AccelKind   kind = AccelKind::Other;
    double      util_pct = 0.0;
};

struct SysSample {
    // Память процесса (kB). Источник: /proc/self/status.
    long rss_kb       = 0;   // VmRSS — реально в RAM
    long vsz_kb       = 0;   // VmSize — виртуальная (резерв)
    long peak_rss_kb  = 0;   // VmHWM — максимум за жизнь процесса
    long swap_kb      = 0;   // VmSwap — выгружено в swap
    int  threads      = 0;   // /proc/self/status:Threads

    // Память системы (kB). Источник: /proc/meminfo / sysinfo().
    long mem_total_kb = 0;
    long mem_avail_kb = 0;   // MemAvailable — лучшая оценка «свободно»

    // Загрузка CPU за интервал между двумя sample().
    // Процесс: 100.0 = одно полное ядро; на 4-ядерном хосте максимум 400.
    // Система: 100.0 = все ядра заняты целиком (нормирована на num_cpus).
    double cpu_proc_pct = 0.0;
    double sys_cpu_pct  = 0.0;

    // Стенное время в мс с предыдущего sample() (или с init() для
    // первого вызова). cpu_*_pct посчитаны именно на этом интервале.
    double wall_ms = 0.0;

    // Утилизация аппаратных ускорителей (GPU/NPU/…) за тот же интервал.
    // Список динамический: ровно те узлы, что монитор нашёл на платформе
    // (пусто, если ускорители не обнаружены или сборка не Linux).
    std::vector<AccelSample> accel;

    bool ok = false;         // false на не-Linux или при ошибке чтения /proc

    // Утилизация (%) ускорителя данного класса или <0, если такого зонда
    // на платформе нет. Берём первый узел нужного класса.
    double accel_util(AccelKind k) const {
        for (const auto& a : accel)
            if (a.kind == k) return a.util_pct;
        return -1.0;
    }
    double gpu_util() const { return accel_util(AccelKind::Gpu); }
    double npu_util() const { return accel_util(AccelKind::Npu); }
};

class SysMonitor {
public:
    // Подготавливает монитор и снимает baseline. Возвращает false на
    // платформах без /proc (Windows/macOS-сборки) или если /proc/self/stat
    // нечитаем. После false sample() безопасно вызывать — он просто
    // вернёт SysSample{ok=false}.
    bool init() {
#if defined(__linux__)
        num_cpus_ = (int)sysconf(_SC_NPROCESSORS_ONLN);
        if (num_cpus_ < 1) num_cpus_ = 1;
        clk_tck_ = (long)sysconf(_SC_CLK_TCK);
        if (clk_tck_ < 1) clk_tck_ = 100;

        if (!read_proc_ticks_(prev_proc_ticks_)) return false;
        if (!read_total_cpu_ticks_(prev_total_ticks_, prev_idle_ticks_))
            return false;

        prev_wall_ns_ = mono_ns_();
        detect_accelerators_();   // GPU/NPU-зонды — опционально, если есть
        ok_ = true;
        return true;
#else
        return false;
#endif
    }

    // Снять текущий замер. CPU-проценты считаются от предыдущего sample()
    // (или от init() для первого вызова). Память — мгновенный снимок.
    SysSample sample() {
        SysSample s;
#if defined(__linux__)
        if (!ok_) return s;

        uint64_t now_ns = mono_ns_();
        double dt_ns = (double)(now_ns - prev_wall_ns_);
        s.wall_ms = dt_ns / 1.0e6;

        // ---- CPU процесса (utime + stime тики /proc/self/stat) ----
        uint64_t proc_now = 0;
        if (read_proc_ticks_(proc_now)) {
            uint64_t d_proc = proc_now - prev_proc_ticks_;
            // 1 тик = 1/clk_tck секунды; проценты «одного ядра».
            double proc_sec = (double)d_proc / (double)clk_tck_;
            double wall_sec = dt_ns / 1.0e9;
            s.cpu_proc_pct = (wall_sec > 0.0)
                ? (proc_sec / wall_sec) * 100.0 : 0.0;
            prev_proc_ticks_ = proc_now;
        }

        // ---- CPU системы (агрегированные тики /proc/stat) ----
        uint64_t total_now = 0, idle_now = 0;
        if (read_total_cpu_ticks_(total_now, idle_now)) {
            uint64_t d_total = total_now - prev_total_ticks_;
            uint64_t d_idle  = idle_now  - prev_idle_ticks_;
            if (d_total > 0) {
                double busy = (double)(d_total - d_idle) / (double)d_total;
                s.sys_cpu_pct = busy * 100.0;
            }
            prev_total_ticks_ = total_now;
            prev_idle_ticks_  = idle_now;
        }

        // ---- Утилизация ускорителей (GPU/NPU) ----
        // dt_ns нужен для зондов-«одиночных счётчиков busy» (util = d_busy/d_wall).
        sample_accelerators_(s, dt_ns);

        prev_wall_ns_ = now_ns;

        // ---- Память процесса (kB) ----
        read_status_mem_(s);

        // ---- Память системы ----
        read_meminfo_(s.mem_total_kb, s.mem_avail_kb);

        s.ok = true;
#endif
        return s;
    }

    int  num_cpus() const { return num_cpus_; }
    bool initialized() const { return ok_; }

    // Обнаружен ли хотя бы один зонд ускорителя данного класса.
    bool has_accel(AccelKind k) const {
        for (const auto& p : accel_probes_)
            if (p.kind == k) return true;
        return false;
    }
    bool has_gpu() const { return has_accel(AccelKind::Gpu); }
    bool has_npu() const { return has_accel(AccelKind::Npu); }

    // «gpu(/sys/...), npu(/sys/...)» — для записи в meta-шапку CSV/лог.
    std::string accel_summary() const {
        std::string out;
        for (const auto& p : accel_probes_) {
            if (!out.empty()) out += ", ";
            out += p.label;
            out += "(";
            out += p.path;
            out += ")";
        }
        return out;
    }

private:
    int  num_cpus_ = 1;
    long clk_tck_  = 100;
    bool ok_       = false;

    // Способ интерпретации содержимого узла утилизации (определяется
    // автоматически при detect_accelerators_ по первому чтению файла).
    enum class AccelFmt {
        BusyIdle,   // счётчики busy_time/idle_time (нс) → d_busy/(d_busy+d_idle)
        BusyNs,     // один накопительный busy_time (нс) → d_busy/d_wall
        Gauge,      // готовый процент 0..100 (опц. со знаком «%»)
    };
    struct AccelProbe {
        std::string label;
        AccelKind   kind = AccelKind::Other;
        AccelFmt    fmt  = AccelFmt::Gauge;
        std::string path;
        uint64_t    prev_busy = 0;   // для BusyIdle/BusyNs
        uint64_t    prev_idle = 0;   // для BusyIdle
    };
    std::vector<AccelProbe> accel_probes_;

    uint64_t prev_proc_ticks_  = 0;
    uint64_t prev_total_ticks_ = 0;
    uint64_t prev_idle_ticks_  = 0;
    uint64_t prev_wall_ns_     = 0;

#if defined(__linux__)
    static uint64_t mono_ns_() {
        struct timespec ts{};
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    }

    // /proc/self/stat: utime (поле 14) + stime (поле 15), тики clock(2).
    // Парсим аккуратно: имя процесса в скобках может содержать пробелы и
    // скобки, поэтому ищем последний ')' и работаем дальше.
    static bool read_proc_ticks_(uint64_t& out) {
        std::FILE* f = std::fopen("/proc/self/stat", "r");
        if (!f) return false;
        char buf[1024];
        size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
        std::fclose(f);
        if (n == 0) return false;
        buf[n] = '\0';
        char* rp = std::strrchr(buf, ')');
        if (!rp) return false;
        // После ')' идут: ' S ppid pgrp ... ' — нужны поля 14 и 15
        // относительно начала строки. От ')' это поля 12 и 13 (0-based 11/12)
        // если считать от первого после ')'.
        // Безопаснее просто пропустить N пробелов: после ') ' поле 3.
        // utime — 14-е поле всей строки = 12-е после ')'.
        char* p = rp + 1;
        for (int field = 3; field <= 13; ++field) {
            while (*p == ' ') ++p;
            while (*p && *p != ' ') ++p;
        }
        // Сейчас p указывает на пробел перед utime (поле 14).
        while (*p == ' ') ++p;
        unsigned long long utime = std::strtoull(p, &p, 10);
        while (*p == ' ') ++p;
        unsigned long long stime = std::strtoull(p, &p, 10);
        out = (uint64_t)(utime + stime);
        return true;
    }

    // /proc/stat первая строка: "cpu user nice system idle iowait irq
    // softirq steal guest guest_nice". total = сумма всех; idle = idle+iowait.
    static bool read_total_cpu_ticks_(uint64_t& total, uint64_t& idle) {
        std::FILE* f = std::fopen("/proc/stat", "r");
        if (!f) return false;
        char buf[512];
        if (!std::fgets(buf, sizeof(buf), f)) {
            std::fclose(f);
            return false;
        }
        std::fclose(f);
        if (std::strncmp(buf, "cpu ", 4) != 0
            && std::strncmp(buf, "cpu\t", 4) != 0) return false;
        char* p = buf + 4;
        unsigned long long vals[10] = {0};
        int got = 0;
        while (got < 10) {
            while (*p == ' ' || *p == '\t') ++p;
            if (!*p || *p == '\n') break;
            vals[got++] = std::strtoull(p, &p, 10);
        }
        uint64_t t = 0;
        for (int i = 0; i < got; ++i) t += vals[i];
        total = t;
        // idle = vals[3]; iowait = vals[4]
        idle = vals[3] + (got > 4 ? vals[4] : 0);
        return true;
    }

    // /proc/self/status: VmRSS, VmSize, VmHWM, VmSwap, Threads.
    // Все Vm* в kB (так и пишем без перевода).
    static void read_status_mem_(SysSample& s) {
        std::FILE* f = std::fopen("/proc/self/status", "r");
        if (!f) return;
        char line[256];
        while (std::fgets(line, sizeof(line), f)) {
            if (std::strncmp(line, "VmRSS:",   6) == 0)
                s.rss_kb      = std::atol(line + 6);
            else if (std::strncmp(line, "VmSize:",  7) == 0)
                s.vsz_kb      = std::atol(line + 7);
            else if (std::strncmp(line, "VmHWM:",   6) == 0)
                s.peak_rss_kb = std::atol(line + 6);
            else if (std::strncmp(line, "VmSwap:",  7) == 0)
                s.swap_kb     = std::atol(line + 7);
            else if (std::strncmp(line, "Threads:", 8) == 0)
                s.threads     = std::atoi(line + 8);
        }
        std::fclose(f);
    }

    // /proc/meminfo: MemTotal, MemAvailable. MemAvailable — оценка ядра
    // «сколько можно выделить без свопа», лучше чем MemFree.
    static void read_meminfo_(long& total, long& avail) {
        std::FILE* f = std::fopen("/proc/meminfo", "r");
        if (!f) return;
        char line[256];
        while (std::fgets(line, sizeof(line), f)) {
            if (std::strncmp(line, "MemTotal:",     9) == 0)
                total = std::atol(line + 9);
            else if (std::strncmp(line, "MemAvailable:", 13) == 0)
                avail = std::atol(line + 13);
        }
        std::fclose(f);
    }

    // ===== Ускорители (GPU/NPU): чтение и интерпретация sysfs/debugfs =====

    // Прочитать узел целиком (узлы утилизации крошечные — сотни байт).
    static std::string read_file_(const std::string& path) {
        std::FILE* f = std::fopen(path.c_str(), "r");
        if (!f) return std::string();
        char buf[1024];
        size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
        std::fclose(f);
        return std::string(buf, n);
    }

    // Найти "<key>" в тексте и распарсить следующее за ним беззнаковое
    // число (пропуская ':', пробелы и табы). true — если ключ найден.
    static bool find_u64_after_(const std::string& s, const char* key,
                                uint64_t& out) {
        const char* p = std::strstr(s.c_str(), key);
        if (!p) return false;
        p += std::strlen(key);
        while (*p == ':' || *p == ' ' || *p == '\t' || *p == '=') ++p;
        char* end = nullptr;
        unsigned long long v = std::strtoull(p, &end, 10);
        if (end == p) return false;
        out = (uint64_t)v;
        return true;
    }

    // Первое число в строке как double (для Gauge: "57", "57%", "load 57 %").
    static bool parse_first_number_(const std::string& s, double& out) {
        const char* p = s.c_str();
        while (*p && !((*p >= '0' && *p <= '9') || *p == '.' ||
                       ((*p == '-' || *p == '+') && p[1] >= '0' && p[1] <= '9')))
            ++p;
        if (!*p) return false;
        char* end = nullptr;
        double v = std::strtod(p, &end);
        if (end == p) return false;
        out = v;
        return true;
    }

    static double clamp_pct_(double v) {
        if (v < 0.0)   return 0.0;
        if (v > 100.0) return 100.0;
        return v;
    }

    static AccelKind parse_kind_(const std::string& s) {
        if (s == "gpu" || s == "GPU") return AccelKind::Gpu;
        if (s == "npu" || s == "NPU") return AccelKind::Npu;
        return AccelKind::Other;
    }

    // Определить формат узла по содержимому и снять baseline-счётчики.
    // Возвращает false, если файл нечитаем или формат не распознан.
    bool sniff_and_baseline_(AccelProbe& p) {
        std::string c = read_file_(p.path);
        if (c.empty()) return false;
        uint64_t busy = 0, idle = 0;
        if (find_u64_after_(c, "busy_time", busy)) {
            p.prev_busy = busy;
            if (find_u64_after_(c, "idle_time", idle)) {
                p.prev_idle = idle;
                p.fmt = AccelFmt::BusyIdle;
            } else {
                p.fmt = AccelFmt::BusyNs;
            }
            return true;
        }
        double v = 0.0;
        if (parse_first_number_(c, v)) {
            p.fmt = AccelFmt::Gauge;   // готовый процент
            return true;
        }
        return false;
    }

    // Зарегистрировать зонд по пути (если узел существует и распознан).
    // dedupe_kind=true — не добавлять второй зонд того же класса.
    void register_probe_(const std::string& label, AccelKind kind,
                         const std::string& path, bool dedupe_kind) {
        if (path.empty()) return;
        if (dedupe_kind && has_accel(kind)) return;
        AccelProbe p;
        p.label = label.empty() ? accel_kind_name(kind) : label;
        p.kind  = kind;
        p.path  = path;
        if (!sniff_and_baseline_(p)) return;
        accel_probes_.push_back(std::move(p));
    }

    // Разобрать II_SYSMON_ACCEL = "label:kind:/path,label2:kind2:/path2".
    void parse_accel_env_list_(const char* env) {
        std::string s(env);
        size_t i = 0;
        while (i < s.size()) {
            size_t comma = s.find(',', i);
            std::string item = s.substr(i, comma == std::string::npos
                                              ? std::string::npos : comma - i);
            i = (comma == std::string::npos) ? s.size() : comma + 1;
            // item = label:kind:path  (path может содержать ':'? на Linux нет)
            size_t c1 = item.find(':');
            if (c1 == std::string::npos) continue;
            size_t c2 = item.find(':', c1 + 1);
            if (c2 == std::string::npos) continue;
            std::string label = item.substr(0, c1);
            std::string kind  = item.substr(c1 + 1, c2 - c1 - 1);
            std::string path  = item.substr(c2 + 1);
            register_probe_(label, parse_kind_(kind), path,
                            /*dedupe_kind=*/false);
        }
    }

    // Сформировать список зондов: сначала явные переопределения из env
    // (полная свобода для нестандартных драйверов), затем — типовые узлы
    // утилизации GPU/NPU на embedded/desktop-платформах (берём первый
    // существующий на каждый класс).
    void detect_accelerators_() {
        if (const char* e = std::getenv("II_SYSMON_ACCEL"))
            if (*e) parse_accel_env_list_(e);
        if (const char* e = std::getenv("II_SYSMON_GPU"))
            if (*e) register_probe_("gpu", AccelKind::Gpu, e, true);
        if (const char* e = std::getenv("II_SYSMON_NPU"))
            if (*e) register_probe_("npu", AccelKind::Npu, e, true);

        // Типовые узлы (универсально, без привязки к конкретному вендору).
        // debugfs-варианты требуют примонтированного debugfs и обычно root.
        static const struct { const char* path; AccelKind kind; }
        kBuiltin[] = {
            // GPU
            {"/sys/kernel/debug/mali0/dvfs_utilization",            AccelKind::Gpu},
            {"/sys/kernel/debug/mali/dvfs_utilization",             AccelKind::Gpu},
            {"/sys/class/drm/card0/device/gpu_busy_percent",        AccelKind::Gpu},
            {"/sys/class/devfreq/gpu/utilization",                  AccelKind::Gpu},
            // NPU (узлы сильно зависят от драйвера — при необходимости
            // задаётся через II_SYSMON_NPU).
            {"/sys/kernel/debug/rknpu/load",                        AccelKind::Npu},
            {"/sys/class/devfreq/npu/utilization",                  AccelKind::Npu},
        };
        for (const auto& b : kBuiltin)
            register_probe_(accel_kind_name(b.kind), b.kind, b.path,
                            /*dedupe_kind=*/true);
    }

    // Снять утилизацию по всем зондам и заполнить s.accel.
    void sample_accelerators_(SysSample& s, double dt_ns) {
        for (auto& p : accel_probes_) {
            std::string c = read_file_(p.path);
            if (c.empty()) continue;
            double util = 0.0;
            bool ok = false;
            switch (p.fmt) {
                case AccelFmt::BusyIdle: {
                    uint64_t busy = 0, idle = 0;
                    if (find_u64_after_(c, "busy_time", busy) &&
                        find_u64_after_(c, "idle_time", idle)) {
                        uint64_t db = busy - p.prev_busy;
                        uint64_t di = idle - p.prev_idle;
                        p.prev_busy = busy;
                        p.prev_idle = idle;
                        uint64_t den = db + di;
                        util = den > 0 ? (double)db / (double)den * 100.0 : 0.0;
                        ok = true;
                    }
                    break;
                }
                case AccelFmt::BusyNs: {
                    uint64_t busy = 0;
                    if (find_u64_after_(c, "busy_time", busy)) {
                        uint64_t db = busy - p.prev_busy;
                        p.prev_busy = busy;
                        util = dt_ns > 0.0
                            ? (double)db / dt_ns * 100.0 : 0.0;
                        ok = true;
                    }
                    break;
                }
                case AccelFmt::Gauge: {
                    double v = 0.0;
                    if (parse_first_number_(c, v)) { util = v; ok = true; }
                    break;
                }
            }
            if (!ok) continue;
            AccelSample as;
            as.label    = p.label;
            as.kind     = p.kind;
            as.util_pct = clamp_pct_(util);
            s.accel.push_back(std::move(as));
        }
    }
#endif  // __linux__
};

// Аккумулятор замеров: считает агрегаты (среднее/пик) по серии sample().
// Используется для печати summary в конце бенчмарка / видео-цикла и для
// записи строки в *.sysmon.summary.csv. Игнорирует семплы с ok=false.
struct SysAccum {
    int    n            = 0;
    double cpu_proc_sum = 0.0;
    double cpu_proc_max = 0.0;
    double sys_cpu_sum  = 0.0;
    double sys_cpu_max  = 0.0;
    long   rss_max      = 0;
    long   vsz_max      = 0;
    long   peak_rss     = 0;   // последнее значение VmHWM (монотонно растёт)
    int    threads_max  = 0;

    // Ускорители: считаем отдельным счётчиком семплов, т.к. зонд может
    // появляться не во всех замерах (узел временно недоступен).
    int    gpu_n        = 0;
    double gpu_sum      = 0.0;
    double gpu_max      = 0.0;
    int    npu_n        = 0;
    double npu_sum      = 0.0;
    double npu_max      = 0.0;

    void add(const SysSample& s) {
        if (!s.ok) return;
        ++n;
        cpu_proc_sum += s.cpu_proc_pct;
        sys_cpu_sum  += s.sys_cpu_pct;
        if (s.cpu_proc_pct > cpu_proc_max) cpu_proc_max = s.cpu_proc_pct;
        if (s.sys_cpu_pct  > sys_cpu_max)  sys_cpu_max  = s.sys_cpu_pct;
        if (s.rss_kb       > rss_max)      rss_max      = s.rss_kb;
        if (s.vsz_kb       > vsz_max)      vsz_max      = s.vsz_kb;
        if (s.peak_rss_kb  > peak_rss)     peak_rss     = s.peak_rss_kb;
        if (s.threads      > threads_max)  threads_max  = s.threads;
        for (const auto& a : s.accel) {
            if (a.kind == AccelKind::Gpu) {
                ++gpu_n; gpu_sum += a.util_pct;
                if (a.util_pct > gpu_max) gpu_max = a.util_pct;
            } else if (a.kind == AccelKind::Npu) {
                ++npu_n; npu_sum += a.util_pct;
                if (a.util_pct > npu_max) npu_max = a.util_pct;
            }
        }
    }

    double cpu_proc_avg() const { return n     > 0 ? cpu_proc_sum / n     : 0.0; }
    double sys_cpu_avg()  const { return n     > 0 ? sys_cpu_sum  / n     : 0.0; }
    double gpu_avg()      const { return gpu_n > 0 ? gpu_sum      / gpu_n : 0.0; }
    double npu_avg()      const { return npu_n > 0 ? npu_sum      / npu_n : 0.0; }
    bool   has_gpu()      const { return gpu_n > 0; }
    bool   has_npu()      const { return npu_n > 0; }
    bool   empty()        const { return n == 0; }
};

// Однострочное представление семпла для лога/оверлея.
inline std::string sysmon_format(const SysSample& s) {
    if (!s.ok) return "sysmon: n/a";
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "CPU proc %5.1f%%  sys %5.1f%%  RSS %ld kB  VmHWM %ld kB  thr %d",
        s.cpu_proc_pct, s.sys_cpu_pct,
        s.rss_kb, s.peak_rss_kb, s.threads);
    std::string out = buf;
    // Дописываем все найденные ускорители (gpu/npu/…) по их меткам.
    for (const auto& a : s.accel) {
        char ab[64];
        std::snprintf(ab, sizeof(ab), "  %s %5.1f%%", a.label.c_str(),
                      a.util_pct);
        out += ab;
    }
    return out;
}
