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

#if defined(__linux__)
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

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

    bool ok = false;         // false на не-Linux или при ошибке чтения /proc
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

private:
    int  num_cpus_ = 1;
    long clk_tck_  = 100;
    bool ok_       = false;

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
    }

    double cpu_proc_avg() const { return n > 0 ? cpu_proc_sum / n : 0.0; }
    double sys_cpu_avg()  const { return n > 0 ? sys_cpu_sum  / n : 0.0; }
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
    return buf;
}
