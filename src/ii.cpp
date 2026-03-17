// Универсальный раннер моделей на NPU (через внешний делегат
// TensorFlow Lite) — а также на любом другом бэкенде, реализующем
// inf::Engine (см. inference.h). Сам раннер не знает про TFLite:
// конкретный бэкенд выбирается через --backend и подключается на этапе
// сборки (USE_TFLITE / USE_TENSORRT / USE_DIRECTML).
//
// Цель: пропустить изображение через INT8-квантованную модель
// (например yolov8m_int8.tflite) и получить «сырые» выходы.
//
// Поддерживаются модели:
//   * один вход; для путей с картинкой (загрузка image, --display, --yolo,
//     image-режим --compare) ожидается NHWC [1,H,W,3] (uint8/int8/float32);
//     в режиме --random-input shape произвольный (например [1,9,9,1] для
//     табличных/регрессионных сетей);
//   * любое число выходов произвольных типов;
//   * квантование scale/zero_point читается из самой модели.
//
// YOLO-постобработка (декодирование боксов, NMS) подключается сверху
// через флаг --yolo (см. yolo.h). Сам движок остаётся универсальным —
// без флага модель прогоняется как «чёрный ящик», на экран идёт сырое
// изображение.
//
// Запуск (на устройстве):
//   ./ii yolov8m_int8.tflite image.jpg
//   ./ii model.tflite image.jpg --no-delegate
//   ./ii model.tflite image.jpg --benchmark --runs 100
//   ./ii model.tflite image.jpg --display              # окно Wayland
//   ./ii model.tflite image.jpg --display --show-input # показать препроц.
//   ./ii yolov8m_int8.tflite img.jpg --display --yolo  # боксы поверх
//   ./ii yolov8m_int8.tflite img.jpg --display --yolo --conf 0.4 --iou 0.5
//   ./ii model.tflite --random-input --benchmark --runs 100   # без картинки
//   ./ii model.tflite --random-input --compare-cpu --random-runs 50
//   ./ii yolov8m_int8.tflite --camera --display --yolo  # видео-поток
//   ./ii yolov8m_int8.tflite --camera /dev/video2 --camera-size 1280x720
//       --camera-fps 30 --display --yolo --stats        # с замером FPS
//
// Image-to-image модели (SR / enhance / denoise):
//   ./ii fsrcnn_qat.tflite img.jpg --display --show-output    # SR на экране
//   ./ii fsrcnn_qat.tflite img.jpg --save-output out.png      # SR в файл
//   ./ii enhance_int8.tflite img.jpg --display --show-output  # low-light на экране
//   ./ii enhance_int8.tflite --camera --display --show-output # enhance live
//   ./ii model.tflite img.jpg --show-output --save-output o.png --output-range signed

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "camera.h"
#include "cli.h"
#include "csv_export.h"
#include "display.h"
#include "frame_source.h"
#include "image_proc.h"
#include "inference.h"
#include "preprocess.h"
#include "stats.h"
#include "sysmon.h"
#include "tensor_utils.h"
#include "tile.h"
#include "yolo.h"
#include "yolo_render.h"

#include <atomic>
#include <csignal>
#ifndef _WIN32
#include <signal.h>   // pthread_sigmask
#endif

namespace {
std::atomic<bool> g_interrupted{false};
void on_sigint(int) { g_interrupted = true; }

// «Осторожный» Ctrl+C: блокируем SIGINT в текущем потоке на время
// инференса. Без этого сигнал прилетает посреди poll() в драйвере
// внешний NPU-делегат, тот возвращает EINTR и репортит «Inference poll timeout»
// (code=1534) — NPU остаётся в неконсистентном состоянии и часто
// требует ребута устройства. С блокировкой SIGINT просто ставится
// в очередь ядром и доставляется сразу после возврата из invoke();
// обработчик on_sigint всё так же выставит g_interrupted, и цикл
// корректно выйдет на следующей итерации.
struct SigintGuard {
#ifndef _WIN32
    sigset_t old{};
    bool active{false};
    SigintGuard() {
        sigset_t s;
        sigemptyset(&s);
        sigaddset(&s, SIGINT);
        active = (pthread_sigmask(SIG_BLOCK, &s, &old) == 0);
    }
    ~SigintGuard() {
        if (active) pthread_sigmask(SIG_SETMASK, &old, nullptr);
    }
#endif
};

inline bool safe_invoke(inf::Engine& e) {
    SigintGuard g;
    return e.invoke();
}
}  // namespace

using namespace iirun;

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) return 1;

    // Источников входа три: картинка, --random-input, --camera. Хотя бы
    // один обязателен. --camera (если задана) перебивает остальные:
    // получает контроль над основным циклом и игнорирует image/random.
    const bool has_camera = !args.camera_device.empty();
    const bool has_image  = !args.image.empty() && !has_camera;
    if (!has_image && !args.random_input && !has_camera) {
        std::fprintf(stderr,
            "Не указан источник входа. Передайте image, либо "
            "--random-input, либо --camera.\n");
        print_usage(argv[0]);
        return 1;
    }
    if (args.random_runs < 1) {
        std::fprintf(stderr, "--random-runs должен быть >= 1.\n");
        return 1;
    }
    // При чистом --random-input без картинки/камеры показать оригинал нечего,
    // но с --show-output есть что: декодированный выход модели от случайного
    // входа — полезно для sanity-чека SR/enhance на dev-хосте.
    if (args.random_input && args.display && !has_image && !has_camera
        && !args.show_output) {
        std::fprintf(stderr,
            "--display требует источник кадров (image или --camera) либо "
            "--show-output; при чистом --random-input нечего показывать.\n");
        return 1;
    }
    if (has_camera && !args.image.empty()) {
        std::fprintf(stderr,
            "Внимание: image=%s проигнорирован, активен --camera %s.\n",
            args.image.c_str(), args.camera_device.c_str());
    }
    // --show-output (декодирование выхода как картинки) и --yolo (декодирование
    // выхода как боксов) — взаимоисключающие интерпретации одного и того же
    // выхода. Лучше упасть рано с понятным сообщением.
    if (args.show_output && args.yolo) {
        std::fprintf(stderr,
            "--show-output и --yolo несовместимы: выход модели — либо "
            "image, либо детекции.\n");
        return 1;
    }
    imgproc::OutputRange output_range = imgproc::OutputRange::Unit;
    if (!imgproc::parse_output_range(args.output_range_str, output_range)) {
        std::fprintf(stderr,
            "Неизвестный --output-range: '%s'. Допустимые: unit, signed, byte.\n",
            args.output_range_str.c_str());
        return 1;
    }
    if (!args.save_output_path.empty() && (args.loop || has_camera)) {
        std::fprintf(stderr,
            "Внимание: --save-output игнорируется в режиме --loop/--camera "
            "(имеет смысл только для одиночного инференса).\n");
    }
    // Валидация --tile: режим осмыслен только для image-to-image моделей с
    // реальным источником кадров и без альтернативных интерпретаций выхода.
    if (args.tile_mode) {
        if (args.yolo) {
            std::fprintf(stderr,
                "--tile и --yolo несовместимы: выход модели либо image-to-image, "
                "либо детекции.\n");
            return 1;
        }
        if (args.compare_cpu || !args.compare_model.empty()) {
            std::fprintf(stderr,
                "--tile несовместим с --compare*: сравнение работает с одиночным "
                "invoke, а не tile-pass'ом. Прогоните --compare отдельно.\n");
            return 1;
        }
        if (args.random_input) {
            std::fprintf(stderr,
                "--tile несовместим с --random-input: нужен реальный кадр для "
                "разбиения на тайлы.\n");
            return 1;
        }
        if (!has_image && !has_camera) {
            std::fprintf(stderr,
                "--tile требует источник кадров: image или --camera.\n");
            return 1;
        }
        if (args.tile_overlap < 0) {
            std::fprintf(stderr,
                "--tile-overlap должен быть >= 0 (получено %d).\n",
                args.tile_overlap);
            return 1;
        }
        if (args.show_input) {
            // input_rgb в tile-режиме содержит только последний обработанный
            // тайл — толку показывать его на экране нет. Не блокируем,
            // только предупреждаем: пусть пользователь сам решит.
            std::fprintf(stderr,
                "Внимание: --show-input в режиме --tile показывает только "
                "последний тайл (бессмысленно). Используйте --show-output "
                "для просмотра canvas’а или дефолт для исходника.\n");
        }
    }

    auto eng = inf::make_engine(args.backend);
    if (!eng) {
        std::fprintf(stderr,
            "Не удалось создать бэкенд '%s'. Доступные: ",
            args.backend.c_str());
        bool first = true;
        for (const auto& b : inf::available_backends()) {
            std::fprintf(stderr, "%s%s", first ? "" : ", ", b.c_str());
            first = false;
        }
        std::fprintf(stderr, "\n");
        return 2;
    }
    inf::Engine::Options eopts;
    eopts.delegate_path = args.no_delegate ? std::string{} : args.delegate;
    eopts.num_threads   = args.threads;
    if (!eng->load(args.model, eopts)) return 2;

    // Тег для логов: где именно крутится инференс. С NPU/делегатом
    // подразумевается ускоритель; без делегата — CPU. Бэкенд тоже
    // печатается отдельно (см. backend_name) — пользователю важно знать,
    // запускается ли он на tflite/tensorrt/directml.
    const char* label_main = args.no_delegate ? "CPU" : "NPU";
    std::printf("Бэкенд инференса: %s\n", eng->backend_name());

    // ---- Мониторинг CPU/памяти ----
    // Инициализируем заранее, чтобы baseline-семпл захватил состояние ещё
    // до загрузки тензоров и первого invoke. На не-Linux init() вернёт
    // false; SysAccum останется пустым, а вызовы sample() — безвредными.
    SysMonitor sysmon;
    SysAccum   sysmon_acc;
    if (args.sysmon) {
        if (!sysmon.init()) {
            std::fprintf(stderr,
                "Внимание: --sysmon недоступен (не Linux или /proc нечитаем) — "
                "пропускаем замеры.\n");
        } else {
            std::printf("Мониторинг CPU/памяти включён "
                        "(ядер в системе: %d, интервал %d мс)\n",
                        sysmon.num_cpus(), args.sysmon_interval_ms);
        }
    }
    // Открываем оба CSV (per-sample и summary) лениво по факту первого
    // использования, чтобы не плодить пустые файлы для путей, в которых
    // sysmon не активирован (например, --camera при --no-export).
    CsvExport sysmon_csv;
    bool      sysmon_csv_opened = false;
    const double sysmon_t0 = now_ms();

    // ---- Сводка по тензорам ----
    // Бэкенд уже подготовил описание тензоров в load(); просто берём ссылки.
    // Копируем в локальные вектора: дальнейший код переиндексирует in_info[0]
    // и т.п., а ссылки бэкенда живут до конца жизни Engine, так что в
    // принципе можно было бы и константной ссылкой — но копия 50–200 байт
    // погоды не делает, а независимость от бэкенда удобнее.
    std::vector<TensorInfo> in_info  = eng->inputs();
    std::vector<TensorInfo> out_info = eng->outputs();

    std::printf("Входы:\n");
    for (auto& t : in_info)  print_tensor(" -", t);
    std::printf("Выходы:\n");
    for (auto& t : out_info) print_tensor(" -", t);

    if (in_info.size() != 1) {
        std::fprintf(stderr,
            "Эта реализация рассчитана на 1 вход, у модели %zu.\n",
            in_info.size());
        return 3;
    }

    // ---- Индекс выхода-изображения (для --show-output / --save-output) ----
    // Один раз после load: либо берём явно указанный --output-index, либо
    // ищем единственный image-shaped выход среди всех. Решение кешируется,
    // в видео-цикле не повторяется. Если не нашлось — это фатально для
    // соответствующих режимов.
    const bool decode_output_needed =
        args.show_output || !args.save_output_path.empty() || args.tile_mode;
    int image_output_idx = -1;
    if (decode_output_needed) {
        if (args.output_index >= 0) {
            if (args.output_index >= (int)out_info.size()) {
                std::fprintf(stderr,
                    "--output-index %d вне диапазона (выходов %zu).\n",
                    args.output_index, out_info.size());
                return 3;
            }
            image_output_idx = args.output_index;
            if (!imgproc::is_image_output(out_info[image_output_idx])) {
                std::fprintf(stderr,
                    "Выход[%d] '%s' shape=%s не похож на изображение "
                    "(ожидаю NHWC [1,H,W,1|3]).\n",
                    image_output_idx,
                    out_info[image_output_idx].name.c_str(),
                    shape_to_str(out_info[image_output_idx].shape).c_str());
                return 3;
            }
        } else {
            image_output_idx = imgproc::detect_image_output_index(out_info);
            if (image_output_idx < 0) {
                std::fprintf(stderr,
                    "Не нашёл единственного image-shaped выхода среди %zu. "
                    "Уточните --output-index <i>.\n",
                    out_info.size());
                return 3;
            }
        }
        std::printf(
            "Выход для отображения: [%d] %s shape=%s, диапазон=%s\n",
            image_output_idx,
            out_info[image_output_idx].name.c_str(),
            shape_to_str(out_info[image_output_idx].shape).c_str(),
            args.output_range_str.c_str());
    }

    // ---- Масштаб модели для tile-режима ---------------------------------
    // Размер выхода относительно входа должен быть целым: для SR это ×2/×3/
    // ×4, для denoise/enhance — ×1. Нецелочисленный масштаб сделал бы
    // позиционирование тайлов в canvas неоднозначным, поэтому валим сразу.
    // Значения по умолчанию (1,1) безопасны и не используются вне tile-режима.
    int scale_x = 1, scale_y = 1;
    if (args.tile_mode) {
        if (image_output_idx < 0) {
            std::fprintf(stderr,
                "--tile требует image-shaped выход (NHWC [1,H,W,1|3]).\n");
            return 3;
        }
        // Вход модели уже проверен ниже как [1,H,W,1|3]; здесь предполагаем
        // те же поля. Берём их явно из shape входа.
        const auto& is_ = in_info[0].shape;
        if (is_.size() != 4 || is_[0] != 1) {
            std::fprintf(stderr,
                "--tile поддерживает только NHWC [1,H,W,1|3] вход (получено %s).\n",
                shape_to_str(is_).c_str());
            return 3;
        }
        const int in_h_v = is_[1];
        const int in_w_v = is_[2];
        if (in_h_v <= 0 || in_w_v <= 0) {
            std::fprintf(stderr,
                "--tile: размеры входа [%d,%d] должны быть статическими и > 0 "
                "(динамические/неизвестные оси не поддержаны).\n",
                in_h_v, in_w_v);
            return 3;
        }
        const auto& os = out_info[image_output_idx].shape;
        if (os[1] <= 0 || os[2] <= 0
            || os[1] % in_h_v != 0 || os[2] % in_w_v != 0) {
            std::fprintf(stderr,
                "--tile: размер выхода [%d,%d] должен быть целочисленно "
                "кратен размеру входа [%d,%d].\n",
                os[1], os[2], in_h_v, in_w_v);
            return 3;
        }
        scale_y = os[1] / in_h_v;
        scale_x = os[2] / in_w_v;
        if (args.tile_overlap >= in_w_v || args.tile_overlap >= in_h_v) {
            std::fprintf(stderr,
                "--tile-overlap=%d должен быть строго меньше входа %dx%d "
                "(иначе шаг сетки <= 0).\n",
                args.tile_overlap, in_w_v, in_h_v);
            return 3;
        }
        std::printf(
            "Tile: %dx%d, scale=%dx (выход %dx%d), overlap=%d%s\n",
            in_w_v, in_h_v, scale_x, os[2], os[1], args.tile_overlap,
            args.tile_overlap > 0 ? " (linear feathering)" : "");
    }

    // ---- Экспорт замеров в CSV ----
    // Префикс из --export. Конкретные файлы открываются лениво в
    // соответствующих режимах (бенчмарк / видео-цикл / compare),
    // чтобы не плодить пустые файлы для неактивных путей. Шапка
    // самодокументируема: модель, делегат, threads, shape входа,
    // timestamp — этого хватает, чтобы потом склеивать прогоны
    // разных моделей в одну таблицу.
    auto open_export = [&](CsvExport& e, const char* suffix,
                           const char* kind) -> bool {
        if (args.export_prefix.empty()) return false;
        std::string path = args.export_prefix + "." + suffix + ".csv";
        if (!e.open(path)) return false;
        e.meta("kind",         "%s", kind);
        e.meta("started",      "%s", iso_timestamp_now().c_str());
        e.meta("model",        "%s", csv_escape(args.model).c_str());
        e.meta("delegate",     "%s", args.no_delegate ? "cpu" : "npu");
        if (!args.no_delegate)
            e.meta("delegate_path", "%s",
                   csv_escape(args.delegate).c_str());
        e.meta("threads",      "%d", args.threads);
        e.meta("input_shape",  "%s",
               shape_to_str(in_info[0].shape).c_str());
        e.meta("input_dtype",  "%s", dtype_name(in_info[0].dtype));
        std::printf("CSV экспорт (%s): %s\n", kind, e.path().c_str());
        return true;
    };

    // Открыть (один раз) <prefix>.sysmon.csv с шапкой колонок per-sample.
    // Возвращает true, если файл готов к writef. Если --export не задан
    // или sysmon не инициализирован, экспорт молча отключён.
    auto ensure_sysmon_csv = [&]() -> bool {
        if (!args.sysmon || !sysmon.initialized()) return false;
        if (sysmon_csv_opened) return sysmon_csv.is_open();
        sysmon_csv_opened = true;
        if (!open_export(sysmon_csv, "sysmon", "sysmon")) return false;
        sysmon_csv.meta("interval_ms", "%d", args.sysmon_interval_ms);
        sysmon_csv.meta("num_cpus",    "%d", sysmon.num_cpus());
        sysmon_csv.header(
            "t_ms,phase,cpu_proc_pct,sys_cpu_pct,"
            "rss_kb,vsz_kb,peak_rss_kb,swap_kb,threads,"
            "mem_total_kb,mem_avail_kb,wall_ms");
        return true;
    };

    // Финальная сводка sysmon: один последний семпл + строка-агрегат в
    // stdout и в <prefix>.sysmon.summary.csv. Идемпотентен — повторные
    // вызовы перезатрут .summary тем же содержимым; на практике каждая
    // ветка main вызывает её ровно один раз перед своим return.
    bool sysmon_finalized = false;
    auto sysmon_finalize = [&]() {
        if (sysmon_finalized) return;
        sysmon_finalized = true;
        if (!args.sysmon || !sysmon.initialized()) return;
        // последний семпл захватываем тут же, чтобы попал в агрегат
        SysSample s = sysmon.sample();
        if (s.ok) {
            sysmon_acc.add(s);
            std::printf("[sysmon final   ] %s\n", sysmon_format(s).c_str());
            if (sysmon_csv.is_open()) {
                sysmon_csv.writef(
                    "%.3f,%s,%.3f,%.3f,"
                    "%ld,%ld,%ld,%ld,%d,"
                    "%ld,%ld,%.3f",
                    now_ms() - sysmon_t0, "final",
                    s.cpu_proc_pct, s.sys_cpu_pct,
                    s.rss_kb, s.vsz_kb, s.peak_rss_kb, s.swap_kb, s.threads,
                    s.mem_total_kb, s.mem_avail_kb, s.wall_ms);
            }
        }
        if (sysmon_acc.empty()) return;
        std::printf(
            "[sysmon summary] samples=%d  CPU proc avg=%.1f%% "
            "max=%.1f%%  sys avg=%.1f%% max=%.1f%%  RSS max=%ld kB  "
            "VmHWM=%ld kB  threads max=%d\n",
            sysmon_acc.n,
            sysmon_acc.cpu_proc_avg(), sysmon_acc.cpu_proc_max,
            sysmon_acc.sys_cpu_avg(),  sysmon_acc.sys_cpu_max,
            sysmon_acc.rss_max, sysmon_acc.peak_rss,
            sysmon_acc.threads_max);
        if (!args.export_prefix.empty()) {
            CsvExport ss;
            if (open_export(ss, "sysmon.summary", "sysmon_summary")) {
                ss.meta("interval_ms", "%d", args.sysmon_interval_ms);
                ss.meta("num_cpus",    "%d", sysmon.num_cpus());
                ss.header(
                    "samples,cpu_proc_avg_pct,cpu_proc_max_pct,"
                    "sys_cpu_avg_pct,sys_cpu_max_pct,"
                    "rss_max_kb,vsz_max_kb,peak_rss_kb,threads_max");
                ss.writef("%d,%.3f,%.3f,%.3f,%.3f,%ld,%ld,%ld,%d",
                          sysmon_acc.n,
                          sysmon_acc.cpu_proc_avg(), sysmon_acc.cpu_proc_max,
                          sysmon_acc.sys_cpu_avg(),  sysmon_acc.sys_cpu_max,
                          sysmon_acc.rss_max, sysmon_acc.vsz_max,
                          sysmon_acc.peak_rss, sysmon_acc.threads_max);
            }
        }
        if (sysmon_csv.is_open()) sysmon_csv.flush();
    };

    // Один комплексный шаг: снять семпл, добавить в агрегат, напечатать
    // строку в stdout и (если включён --export) — в sysmon.csv.
    // phase — короткая метка фазы прогона (init/warmup/bench/video/final).
    auto sysmon_log = [&](const char* phase) {
        if (!args.sysmon || !sysmon.initialized()) return;
        SysSample s = sysmon.sample();
        if (!s.ok) return;
        sysmon_acc.add(s);
        std::printf("[sysmon %-8s] %s\n", phase, sysmon_format(s).c_str());
        std::fflush(stdout);
        if (ensure_sysmon_csv()) {
            sysmon_csv.writef(
                "%.3f,%s,%.3f,%.3f,"
                "%ld,%ld,%ld,%ld,%d,"
                "%ld,%ld,%.3f",
                now_ms() - sysmon_t0, phase,
                s.cpu_proc_pct, s.sys_cpu_pct,
                s.rss_kb, s.vsz_kb, s.peak_rss_kb, s.swap_kb, s.threads,
                s.mem_total_kb, s.mem_avail_kb, s.wall_ms);
        }
    };
    // NHWC [1,H,W,1|3] — типовой формат для путей с изображением:
    //   * C=3 — обычная RGB-модель;
    //   * C=1 — grayscale/Y-модель (FSRCNN-Y, denoise по яркости и т.п.);
    //     RGB-кадр конвертируем в luma BT.601 непосредственно перед
    //     fill_input, см. fill_model_from_rgb ниже.
    // Для чисто рандомного --compare-random разрешаем любой shape
    // (например [1,9,9,1]) — buffer трактуется плоско.
    const auto& s = in_info[0].shape;
    const int  in_c  = image_input_channels(s);
    const bool nhwc_img = in_c > 0;
    int in_h = 0, in_w = 0;
    if (nhwc_img) { in_h = s[1]; in_w = s[2]; }
    if (has_image && !nhwc_img) {
        std::fprintf(stderr,
            "Для путей с изображением поддерживается только NHWC [1,H,W,1|3]. "
            "Получено shape=%s.\n", shape_to_str(s).c_str());
        return 3;
    }

    // ---- Загрузка и preprocess изображения / генерация случайного входа ----
    // Возможные сценарии:
    //   * --camera                  — вход формируется в цикле захвата,
    //                                 здесь только проверяем параметры;
    //   * есть картинка             — letterbox + fill_input;
    //   * --random-input без картинки — генерируем случайный uint8-буфер
    //     по shape входа и заливаем его (один раз; benchmark/loop не
    //     перегенерируют — для замера скорости содержимое не важно).
    Image img;
    // input_rgb — letterbox-кадр для модели и для --show-input в окне
    // (всегда 3 канала, потому что Display::show_rgb принимает RGB).
    // input_gray используется только когда in_c == 1: RGB → Y перед
    // fill_input, чтобы байтовый размер буфера совпал с numel тензора.
    std::vector<uint8_t> input_rgb;
    std::vector<uint8_t> input_gray;
    void* in_t = eng->input_data(0);
    double t0 = 0.0, t1 = 0.0;

    // Один источник правды для «как залить RGB-кадр в модель» — вызывается
    // из image-пути, видео-цикла, камеры и одиночного random-пути (после
    // вспомогательной интерпретации random-буфера как RGB-кадра in_w x in_h,
    // см. ниже). Для C=1 делает RGB → luma; для C=3 — отдаёт буфер напрямую.
    auto fill_model_from_rgb = [&]() -> bool {
        if (in_c == 1) {
            rgb_to_gray(input_rgb.data(), (size_t)in_w * in_h, input_gray);
            return fill_input(input_gray, in_info[0], in_t);
        }
        return fill_input(input_rgb, in_info[0], in_t);
    };
    if (has_camera) {
        // Для камеры ждём первый кадр уже в основном цикле — здесь лишь
        // проверяем, что вход модели совместим (NHWC [1,H,W,1|3]).
        if (!nhwc_img) {
            std::fprintf(stderr,
                "Для --camera нужен вход NHWC [1,H,W,1|3], получено %s.\n",
                shape_to_str(s).c_str());
            return 4;
        }
        std::printf(
            "Камера: устройство=%s, запрошено %dx%d @ %d fps, "
            "вход модели %dx%dx%d\n",
            args.camera_device.c_str(),
            args.camera_w, args.camera_h, args.camera_fps,
            in_w, in_h, in_c);
    } else if (has_image) {
        if (!load_image(args.image, img)) return 4;
        if (args.tile_mode) {
            // В tile-режиме источник остаётся исходного разрешения; летающий
            // letterbox не нужен — каждый тайл нарезается из img.rgb прямо
            // в input_rgb внутри run_tile_pass.
            const auto layout = tile::plan_tiles(
                img.w, img.h, in_w, in_h, args.tile_overlap);
            std::printf(
                "Изображение: %s  %dx%d -> tile %dx%d×%d тайлов "
                "(scale=%dx, overlap=%d, выход %dx%d)%s\n",
                args.image.c_str(), img.w, img.h,
                in_w, in_h, layout.count(), scale_x, args.tile_overlap,
                img.w * scale_x, img.h * scale_y,
                in_c == 1 ? " [RGB→luma BT.601 каждый тайл]" : "");
        } else {
            std::printf(
                "Изображение: %s  %dx%d -> letterbox %dx%dx%d%s\n",
                args.image.c_str(), img.w, img.h, in_w, in_h, in_c,
                in_c == 1 ? " (RGB→luma BT.601 перед инференсом)" : "");
            letterbox(img, in_w, in_h, input_rgb);
            if (!fill_model_from_rgb()) return 5;
        }
    } else {
        // --random-input без картинки: формируем случайный буфер длиной
        // numel(shape). fill_input трактует его плоско, так что для любого
        // ранга/раскладки результат корректен (включая grayscale-входы).
        const size_t n_in = numel(in_info[0].shape);
        if (n_in == 0) {
            std::fprintf(stderr,
                "Динамический shape входа не поддерживается в --random-input.\n");
            return 4;
        }
        const uint32_t seed = args.random_seed
            ? args.random_seed
            : (uint32_t)std::random_device{}();
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> dist(0, 255);
        input_rgb.resize(n_in);
        for (auto& v : input_rgb) v = (uint8_t)dist(rng);
        if (!fill_input(input_rgb, in_info[0], in_t)) return 5;
        std::printf("Случайный вход: shape=%s, seed=%u\n",
                    shape_to_str(in_info[0].shape).c_str(), seed);
    }

    // Первый замер: фиксируем «старт» — память сразу после загрузки модели,
    // делегата и заливки входа. CPU-проценты будут посчитаны от init() до
    // этой точки, поэтому отражают активность во время AllocateTensors /
    // подготовки делегата.
    sysmon_log("init");

    // ---- YOLO: предварительная подготовка ----
    // Детектируем голову один раз (формат [1, C, A] vs [1, A, C]) до
    // первого invoke, чтобы в видео-цикле не делать это каждый кадр.
    YoloHead yolo_head;
    YoloPostOptions yolo_opts;
    if (args.yolo) {
        if (!has_image && !has_camera) {
            std::fprintf(stderr,
                "--yolo требует источник кадров (image или --camera).\n");
            return 6;
        }
        if (!detect_yolo_head(out_info, args.yolo_output, yolo_head)) return 6;
        yolo_opts.conf_thresh = args.conf;
        yolo_opts.iou_thresh  = args.iou;
        yolo_opts.max_dets    = args.max_dets;
        yolo_opts.normalized  = !args.yolo_pixel_coords;
        if (!args.classes_path.empty()) load_class_names(args.classes_path);
        std::printf("YOLO: выход[%d] layout=%s C=%d A=%d, conf=%.2f iou=%.2f%s\n",
                    yolo_head.output_index,
                    yolo_head.channels_first ? "[1,C,A]" : "[1,A,C]",
                    yolo_head.channels, yolo_head.num_anchors,
                    yolo_opts.conf_thresh, yolo_opts.iou_thresh,
                    yolo_opts.normalized ? " (норм. координаты)"
                                         : " (координаты в пикселях)");
    }

    // ---- Общие буферы постобработки YOLO ----
    // Один раз выделили — переиспользуется во всех путях (одиночный
    // инференс, видео-цикл, камера). Никаких per-frame аллокаций.
    std::vector<float>      yolo_dequant;
    std::vector<Detection>  dets;
    std::vector<DisplayBox> boxes;

    // ---- Общий буфер декодированного выхода-изображения ----
    // Используется одиночным инференсом, видео-циклом и камерой. После
    // первого invoke размер RGB-буфера фиксируется и в видео-цикле не
    // перевыделяется (decode_image_output делает .resize() того же размера).
    imgproc::OutputImage    out_img;
    imgproc::DecodeOptions  dec_opts;
    dec_opts.range = output_range;

    // Кэш LUT для INT8/UInt8 квантования выхода. Параметры (dtype, scale,
    // zp, range) не меняются за время прогона, поэтому LUT строится один
    // раз и переиспользуется во всех вызовах decode (в tile-режиме —
    // десятки/сотни раз за кадр). Без LUT каждый из ~150K пикселей
    // выхода 384×384 проходил через std::lround — ~30 тактов/пиксель;
    // с LUT — одна индексация на 1 такт.
    imgproc::DecodeCache    dec_cache;
    auto decode_current_output = [&]() -> bool {
        if (image_output_idx < 0) return false;
        const void* data = eng->output_data(image_output_idx);
        return imgproc::decode_image_output(out_info[image_output_idx],
                                            data, dec_opts, out_img,
                                            &dec_cache);
    };

    // ---- Tiling: sliding-window pass через всю исходную картинку --------
    // Один полный pass: спланировать сетку тайлов размером с вход модели,
    // для каждого тайла извлечь RGB → fill_input → invoke → декодировать →
    // вставить в canvas. На выходе out_img заполняется полноразмерным
    // RGB-кадром (исходное разрешение × scale модели), и весь дальнейший
    // display/save код подхватывает его без изменений.
    //
    // Оптимизация overwrite-пути: вместо «decode → промежуточный out_img →
    // paste в canvas» (две полные записи декодированного тайла в DRAM),
    // вызываем decode_image_output_to и пишем сразу в нужную позицию
    // canvas’а. На кадр это экономит ~канваc-в-байтах memcpy и столько
    // же memset (~16 МБ DRAM-трафика для 1996×1332 при 24 тайлах).
    // Blend-режим (overlap > 0) всё ещё идёт через out_img + paste,
    // потому что paste делает взвешенное накопление в float-acc.
    //
    // Возвращает false при фатальной ошибке (Invoke упал) — вызывающая
    // сторона должна прервать цикл. Декод одного тайла, не сработавший,
    // пропускается без обрыва — для видео-цикла это важно, иначе один
    // битый кадр уронит весь поток.
    tile::TileCanvas tile_canvas;
    auto run_tile_pass = [&](const uint8_t* src_rgb,
                             int src_w, int src_h) -> bool {
        if (image_output_idx < 0) return false;
        const auto layout = tile::plan_tiles(
            src_w, src_h, in_w, in_h, args.tile_overlap);
        const bool blend = (args.tile_overlap > 0);
        tile_canvas.reset(src_w * scale_x, src_h * scale_y,
                          args.tile_overlap * scale_x, blend);
        const std::size_t canvas_stride = (std::size_t)tile_canvas.width * 3;
        const auto& tile_info = out_info[image_output_idx];
        for (int j = 0; j < layout.ny(); ++j) {
            for (int i = 0; i < layout.nx(); ++i) {
                const int x0 = layout.x0[i];
                const int y0 = layout.y0[j];
                tile::extract_tile(src_rgb, src_w, src_h, x0, y0,
                                   in_w, in_h, input_rgb);
                if (!fill_model_from_rgb()) return false;
                if (!safe_invoke(*eng)) return false;
                const void* tdata = eng->output_data(image_output_idx);
                if (blend) {
                    // В blend-режиме без out_img не обойтись: paste
                    // выполняет взвешенное накопление в acc/weight.
                    if (!imgproc::decode_image_output(
                            tile_info, tdata, dec_opts, out_img,
                            &dec_cache)) {
                        continue;
                    }
                    tile_canvas.paste(out_img,
                                      x0 * scale_x, y0 * scale_y);
                } else {
                    // Overwrite-режим: пишем декодированный тайл сразу
                    // в нужную позицию canvas’а — никаких промежуточных
                    // буферов и memcpy. Тайл целиком помещается в canvas
                    // (plan_tiles снэпает последний к краю), поэтому
                    // дополнительного клиппинга не требуется.
                    if (!imgproc::decode_image_output_to(
                            tile_info, tdata, dec_opts,
                            tile_canvas.rgb.data(), canvas_stride,
                            x0 * scale_x, y0 * scale_y,
                            &dec_cache)) {
                        continue;
                    }
                }
            }
        }
        tile_canvas.finalize();
        // Перекидываем итог в out_img через swap (O(1), вместо copy ~8 МБ
        // для 1996×1332). Следующий вызов reset() переиспользует ту же
        // память — выделение не требуется. Размер фиксируется по первому
        // кадру (src_w/src_h в живом видео-цикле обычно постоянны).
        out_img.width        = tile_canvas.width;
        out_img.height       = tile_canvas.height;
        out_img.channels_src = 3;
        std::swap(out_img.rgb, tile_canvas.rgb);
        return true;
    };

    // Декодирование выхода YOLO для последнего инференса. Вызывается
    // из run_video_loop и из одиночного display-кадра. orig_w/orig_h —
    // размеры исходного кадра (для обратного letterbox-маппинга).
    auto run_yolo_postproc = [&](int orig_w, int orig_h) {
        if (!args.yolo) return;
        const void* t = eng->output_data(yolo_head.output_index);
        if (!dequantize_output(out_info[yolo_head.output_index], t,
                               yolo_dequant)) return;
        dets = decode_yolov8(yolo_dequant.data(),
                             yolo_head.channels, yolo_head.num_anchors,
                             yolo_head.channels_first,
                             in_w, in_h, yolo_opts);
        if (!args.show_input)
            scale_to_original(dets, in_w, in_h, orig_w, orig_h);
        boxes = detections_to_boxes(dets);
    };

    // ---- Унифицированный видео-цикл ----
    // Один цикл на все три сценария:
    //   * --camera [+ --display]       — src=CameraFrameSource;
    //   * image + --display + --loop   — src=StaticFrameSource;
    //   * --loop без --display и без камеры — src=nullptr (просто
    //     гоняем invoke, чтобы померить throughput NPU).
    // disp=nullptr, когда окно не нужно.
    //
    // Шаги одной итерации (всегда одинаковы, отсутствующие просто
    // пропускаются):
    //   1. poll() окна          — выйти, если пользователь закрыл;
    //   2. next() источника     — получить кадр и его размеры;
    //   3. (опц.) препроцесс    — letterbox + fill_input для живого
    //                              источника; для статического вход
    //                              уже заполнен снаружи;
    //   4. invoke               — инференс;
    //   5. fps.tick();
    //   6. (опц.) YOLO postproc — декодирование боксов в координатах
    //                              текущего кадра;
    //   7. (опц.) stats         — оверлей FPS на окно + периодический
    //                              лог в stdout;
    //   8. (опц.) show_rgb      — отрисовка кадра в окне (через
    //                              glTexSubImage2D, без realloc).
    auto run_video_loop = [&](FrameSource* src, Display* disp) -> int {
        std::signal(SIGINT, on_sigint);
        FpsCounter fps;
        std::printf("Видео-цикл запущен. Ctrl+C%s для выхода.\n",
                    disp ? " или закрытие окна" : "");

        // CSV-экспорт FPS-семплов. Открывается только при --export.
        // Пишем по одной строке за каждый интервал log-interval_ms,
        // независимо от --stats (stats влияет только на оверлей и
        // лог в stdout). Колонки — те же агрегаты, что и в format(),
        // плюс счётчик кадров для удобной перепроверки.
        CsvExport fps_csv;
        const bool dump_fps = open_export(fps_csv, "fps", "video_loop");
        if (dump_fps) {
            fps_csv.meta("log_interval_ms", "%d", args.log_interval_ms);
            fps_csv.meta("vsync",           "%s",
                         args.vsync ? "on" : "off");
            fps_csv.meta("source",          "%s",
                         src ? (has_camera ? "camera" : "static")
                             : "none");
            fps_csv.header("t_ms,frame,fps,dt_avg_ms,dt_min_ms,"
                           "dt_max_ms,jitter_ms");
        }
        const double loop_t0 = now_ms();
        std::size_t  frame_n = 0;
        // Отдельный таймер семплирования sysmon — независим от FPS-лога,
        // чтобы пользователь мог поставить, например, 5 с интервал sysmon
        // при ежесекундном FPS-логе. last_sysmon_t = -inf при старте,
        // чтобы первый семпл сработал почти сразу (после первого кадра).
        double last_sysmon_t = -1e18;

        // orig_w/orig_h — размеры «исходного» кадра. Если источника
        // нет, для YOLO принимаем их равными размерам входа модели —
        // scale_to_original в этом случае становится тождественным.
        int orig_w = in_w, orig_h = in_h;

        // Однократный лог «что и где показываем» — печатаем после
        // первого успешного prepass, когда orig_w/orig_h уже актуальны
        // и (для tile-режима) tile_canvas уже размечен под исходное
        // разрешение. Без этого пользователь видит, например, FPS 20.5
        // и не понимает, какое разрешение реально пошло в окно.
        bool first_frame_logged = false;

        while (!g_interrupted && (!disp || disp->poll())) {
            const uint8_t* frame = nullptr;
            if (src) {
                bool needs_preprocess = false;
                int  fw = 0, fh = 0;
                frame = src->next(fw, fh, needs_preprocess);
                if (!frame) {
                    // Источник не отдал кадр (таймаут камеры / skip) —
                    // даём циклу шанс заметить SIGINT и идём дальше.
                    if (g_interrupted) break;
                    continue;
                }
                orig_w = fw;
                orig_h = fh;
                if (args.tile_mode) {
                    // Tile-pass совмещает preprocess+fill+invoke+decode на
                    // уровне отдельных тайлов. Один FPS-«тик» цикла =
                    // один полный кадр, как пользователь и ожидает.
                    if (!run_tile_pass(frame, fw, fh)) break;
                } else if (needs_preprocess) {
                    letterbox(frame, fw, fh, in_w, in_h, input_rgb);
                    if (!fill_model_from_rgb()) break;
                }
            }

            if (!args.tile_mode) {
                if (!safe_invoke(*eng)) {
                    std::fprintf(stderr, "Invoke упал.\n");
                    break;
                }
            }
            fps.tick();
            ++frame_n;

            // Однократный лог: фактические размеры источника и (если
            // включён --display) что именно уйдёт в окно. Делаем это
            // после первого invoke / tile-pass — на этот момент orig_w/h
            // и (для tile-mode) tile_canvas размечены под актуальное
            // разрешение источника, а out_img/output-shape известны.
            if (!first_frame_logged) {
                first_frame_logged = true;
                if (src) {
                    std::printf("Источник кадров: %dx%d%s\n",
                                orig_w, orig_h,
                                has_camera ? " (камера)" : " (статический)");
                }
                if (args.tile_mode && src) {
                    const auto layout = tile::plan_tiles(
                        orig_w, orig_h, in_w, in_h, args.tile_overlap);
                    std::printf(
                        "Tile pass: %d тайлов/кадр (%dx%d сетка), "
                        "canvas %dx%d\n",
                        layout.count(), layout.nx(), layout.ny(),
                        orig_w * scale_x, orig_h * scale_y);
                }
                if (disp) {
                    // Та же логика приоритета, что и в show_rgb ниже.
                    const char* kind;
                    int sw = orig_w, sh = orig_h;
                    if (args.show_output && image_output_idx >= 0) {
                        kind = "выход модели (--show-output)";
                        if (args.tile_mode) {
                            sw = orig_w * scale_x;
                            sh = orig_h * scale_y;
                        } else {
                            const auto& os =
                                out_info[image_output_idx].shape;
                            if (os.size() == 4) {
                                sh = os[1];
                                sw = os[2];
                            }
                        }
                    } else if (args.show_input) {
                        kind = "letterbox-вход (--show-input)";
                        sw = in_w;
                        sh = in_h;
                    } else {
                        kind = "исходный кадр";
                    }
                    std::printf(
                        "Display показывает: %s %dx%d "
                        "(окно %dx%d, letterbox)\n",
                        kind, sw, sh, args.win_w, args.win_h);
                    if (args.tile_mode && !args.show_output) {
                        std::printf(
                            "Подсказка: NPU собирает canvas %dx%d, "
                            "но в окно идёт исходник — добавьте "
                            "--show-output, чтобы увидеть результат.\n",
                            orig_w * scale_x, orig_h * scale_y);
                    }
                    std::fflush(stdout);
                }
            }

            if (args.yolo) {
                run_yolo_postproc(orig_w, orig_h);
                if (disp) disp->set_boxes(boxes);
            }
            // Декодирование выхода-картинки делаем только когда оно реально
            // нужно для отображения — в чистом --loop без дисплея смысла
            // тратить CPU на dequant миллионов пикселей нет. В tile-режиме
            // out_img уже заполнен внутри run_tile_pass (canvas) — повторный
            // decode не нужен.
            if (args.show_output && disp && !args.tile_mode) {
                decode_current_output();
            }

            // log_due() надо звать всегда, когда есть хоть один потребитель
            // периодического тика (stdout-лог под --stats или CSV-экспорт),
            // иначе таймер просто не сработает.
            const bool want_periodic = args.stats || dump_fps;
            const bool periodic = want_periodic
                && fps.log_due(args.log_interval_ms);
            if (args.stats) {
                std::string overlay = fps.format();
                if (disp) disp->set_overlay_text(overlay.c_str());
                if (periodic) {
                    std::printf("[%s] %s\n", label_main, overlay.c_str());
                    std::fflush(stdout);
                }
            }
            if (dump_fps && periodic && !fps.empty()) {
                fps_csv.writef("%.3f,%zu,%.3f,%.4f,%.4f,%.4f,%.4f",
                               now_ms() - loop_t0, frame_n,
                               fps.fps(), fps.avg_ms(),
                               fps.min_ms(), fps.max_ms(),
                               fps.jitter_ms());
            }
            // Семплирование sysmon в видео-цикле: раз в sysmon_interval_ms.
            // Семплим даже без --stats — нагрузка на CPU/память актуальна
            // и для чистого --camera/--loop без оверлея.
            if (args.sysmon && sysmon.initialized()) {
                const double now = now_ms();
                if (now - last_sysmon_t >= (double)args.sysmon_interval_ms) {
                    sysmon_log("video");
                    last_sysmon_t = now;
                }
            }

            if (disp && frame) {
                // Приоритет: --show-output > --show-input > исходный кадр.
                // Для --show-output берём актуально-декодированный буфер;
                // если он пуст (ошибка decode на этом кадре) — фолбэк на
                // обычный путь, чтобы окно не «замерзало» из-за одного бэда.
                const uint8_t* show;
                int sw, sh;
                if (args.show_output && !out_img.rgb.empty()) {
                    show = out_img.rgb.data();
                    sw   = out_img.width;
                    sh   = out_img.height;
                } else if (args.show_input) {
                    show = input_rgb.data();
                    sw   = in_w;
                    sh   = in_h;
                } else {
                    show = frame;
                    sw   = orig_w;
                    sh   = orig_h;
                }
                if (!disp->show_rgb(show, sw, sh)) break;
            }
        }
        if (dump_fps) fps_csv.flush();

        // ---- FPS summary ----
        // Одна строка с агрегатами за весь видео-цикл:
        //   * frames / wall_s / avg_fps — lifetime (точно по счётчику
        //     кадров и стенным часам, не по окну);
        //   * dt_*  / jitter           — текущее состояние FpsCounter
        //     (скользящее окно 120 кадров) — характеризует стабильность
        //     к моменту остановки. Для коротких прогонов окно = весь
        //     прогон, разницы нет.
        // Файл пишется только если был хоть один кадр — иначе таблица
        // была бы из NaN’ов.
        if (args.export_prefix.size() && frame_n > 0) {
            CsvExport fs;
            if (open_export(fs, "fps.summary", "video_loop_summary")) {
                fs.meta("source", "%s",
                        src ? (has_camera ? "camera" : "static") : "none");
                fs.meta("vsync",  "%s", args.vsync ? "on" : "off");
                fs.header("frames,wall_s,avg_fps,"
                          "dt_avg_ms,dt_min_ms,dt_max_ms,jitter_ms");
                const double wall_s = (now_ms() - loop_t0) / 1000.0;
                const double avg_fps = wall_s > 0.0
                    ? (double)frame_n / wall_s : 0.0;
                fs.writef("%zu,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f",
                          frame_n, wall_s, avg_fps,
                          fps.avg_ms(), fps.min_ms(), fps.max_ms(),
                          fps.jitter_ms());
            }
        }
        std::printf("\nОстановлено.\n");
        return 0;
    };

    // ---- Камера ----
    // Открывает V4L2-камеру и (опционально) окно, после чего отдаёт
    // управление общему run_video_loop. Заменяет собой одиночный
    // инференс / --benchmark / --loop / --compare* при наличии --camera.
    if (has_camera) {
        if (args.benchmark || args.loop || args.compare_cpu
            || !args.compare_model.empty() || args.random_input) {
            std::fprintf(stderr,
                "Внимание: --benchmark/--loop/--compare*/--random-input "
                "игнорируются в режиме --camera (используется общий "
                "видео-цикл).\n");
        }

        auto cam = make_camera();
        if (!cam) {
            std::fprintf(stderr,
                "Поддержка камеры не собрана (USE_CAMERA=OFF) или "
                "недоступна на этой платформе.\n");
            return 7;
        }
        if (!cam->open(args.camera_device,
                       args.camera_w, args.camera_h, args.camera_fps)) {
            return 7;
        }

        // Раскладка тайлов теперь известна — камера сообщила фактические
        // width()/height() (драйвер мог выбрать ближайшее к запрошенному
        // --camera-size). Печатаем по тому же шаблону, что и для картинки,
        // чтобы пользователю не пришлось считать тайлы в уме.
        if (args.tile_mode) {
            const auto layout = tile::plan_tiles(
                cam->width(), cam->height(), in_w, in_h, args.tile_overlap);
            std::printf(
                "Камера: %dx%d -> tile %dx%d×%d тайлов "
                "(scale=%dx, overlap=%d, выход %dx%d)%s\n",
                cam->width(), cam->height(),
                in_w, in_h, layout.count(), scale_x, args.tile_overlap,
                cam->width() * scale_x, cam->height() * scale_y,
                in_c == 1 ? " [RGB→luma BT.601 каждый тайл]" : "");
        }

        std::unique_ptr<Display> disp;
        if (args.display) {
            disp = make_display();
            if (!disp) {
                std::fprintf(stderr,
                    "Поддержка дисплея не собрана (USE_DISPLAY=OFF).\n");
                return 7;
            }
            if (!disp->init(args.win_w, args.win_h, "ii", args.vsync))
                return 7;
            std::printf("Display: окно %dx%d, vsync=%s\n",
                        args.win_w, args.win_h,
                        args.vsync ? "on" : "off");
        }

        CameraFrameSource src(*cam, /*timeout_ms=*/1000);
        int rc = run_video_loop(&src, disp.get());
        sysmon_finalize();
        return rc;
    }

    // ---- Один прогон ----
    // Делаем как только в input_t уже что-то залито (картинка или random).
    // В чистом --compare-random режиме вход для сравнения зальётся внутри
    // run_compare, поэтому одиночный прогон тут пропускаем. С камерой
    // одиночного прогона нет вовсе — кадры идут в собственном цикле ниже.
    const bool do_single_invoke = !has_camera && (has_image
        || (args.random_input && !args.compare_cpu && args.compare_model.empty()));
    if (do_single_invoke) {
        t0 = now_ms();
        if (args.tile_mode) {
            // Один полный pass по тайлам: внутри run_tile_pass — N инфёров,
            // декод каждого тайла и сборка canvas в out_img.
            if (!run_tile_pass(img.rgb.data(), img.w, img.h)) {
                std::fprintf(stderr, "Tile pass упал.\n");
                return 6;
            }
        } else {
            if (!safe_invoke(*eng)) { std::fprintf(stderr, "Invoke упал.\n"); return 6; }
        }
        t1 = now_ms();
        std::printf("Инференс: %.3f мс%s\n", t1 - t0,
                    args.tile_mode ? " (полный tile-pass)" : "");

        if (!args.tile_mode) {
            // В tile-режиме «последний» инвок — это последний тайл, печатать
            // его голову неинформативно: реальный итог — canvas в out_img.
            for (size_t i = 0; i < out_info.size(); ++i) {
                const void* t = eng->output_data((int)i);
                print_output_head(out_info[i], t);
            }
        }
        sysmon_log("invoke");

        // Декод выхода-изображения для одиночного прогона: нужен
        // и для --show-output (используется ниже в display-блоке),
        // и для --save-output (сохранение в PNG прямо здесь — это
        // единственный момент, когда оно осмысленно). В tile-режиме
        // out_img уже заполнен полноразмерным canvas’ом — декодировать
        // повторно не нужно.
        if (decode_output_needed) {
            const bool have_output =
                args.tile_mode ? !out_img.rgb.empty() : decode_current_output();
            if (have_output) {
                std::printf(
                    "Декодирован выход: %dx%d, исх. каналов=%d "
                    "(всего %zu uint8 RGB байт)\n",
                    out_img.width, out_img.height, out_img.channels_src,
                    out_img.rgb.size());
                if (!args.save_output_path.empty()) {
                    if (imgproc::save_png(out_img, args.save_output_path)) {
                        std::printf("Сохранено: %s\n",
                                    args.save_output_path.c_str());
                    }
                }
            }
        }
    }

    // ---- Сравнение точности с эталоном ----
    // Сравниваем выходы текущей модели с выходами эталонной (на CPU):
    //   * --compare-cpu       — та же .tflite, но без делегата;
    //   * --compare <path>    — указанная .tflite (например, исходная
    //                          float32 версия той же сети).
    // Метрики (MAE / RMSE / max|diff|) считаются после деквантования в
    // float, поэлементно. Если у выходов разные размеры — сравнивается
    // общий префикс длиной min(size_a, size_b) и печатается пометка.
    //
    // Дополнительно, если основной выход квантован (scale > 0), печатаем
    // те же метрики «в квантах INT8» — Δ/scale. Это даёт интуитивную
    // шкалу: 1 квант = минимально различимое значение этой модели,
    // т.е. предел разрешения INT8. MAE < 1·q считается идеальным
    // совпадением (лучше уже не передать восемью битами).
    // CSV-экспорт результатов сравнения. Один файл на оба возможных
    // вызова (--compare и --compare-cpu) — различаем их по колонке `ref`.
    // Открываем заранее, чтобы run_compare мог свободно писать.
    //
    // Пара файлов:
    //   * compare.csv         — per-run × per-output diff (сырые данные);
    //   * compare.summary.csv — агрегаты по всем прогонам (одна строка
    //                            на пару (ref, output) с MAE/RMSE/max|diff|
    //                            и теми же метриками в «квантах» INT8).
    CsvExport compare_csv;
    CsvExport compare_summary_csv;
    const bool will_compare = !args.compare_model.empty() || args.compare_cpu;
    const bool dump_compare = will_compare
        && open_export(compare_csv, "compare", "compare");
    if (dump_compare) {
        compare_csv.meta("random_input", "%s",
                         args.random_input ? "1" : "0");
        compare_csv.meta("random_runs",  "%d", args.random_runs);
        compare_csv.header(
            "ref,run,output_idx,output_name,elems,"
            "mae,rmse,max_abs_diff,scale,mae_q,rmse_q,maxd_q,trimmed");
    }
    const bool dump_compare_sum = will_compare
        && open_export(compare_summary_csv, "compare.summary",
                       "compare_summary");
    if (dump_compare_sum) {
        compare_summary_csv.meta("random_input", "%s",
                                 args.random_input ? "1" : "0");
        compare_summary_csv.meta("random_runs",  "%d", args.random_runs);
        compare_summary_csv.header(
            "ref,output_idx,output_name,runs,elems,"
            "mae,rmse,max_abs_diff,scale,mae_q,rmse_q,maxd_q,trimmed");
    }

    auto run_compare = [&](const std::string& ref_model_path,
                           const char* ref_label) {
        std::printf("\n=== Сравнение %s vs %s (%s) ===\n",
                    label_main, ref_label, ref_model_path.c_str());
        // Эталонная модель всегда крутится на CPU того же бэкенда, что
        // и основная: для tflite это «та же арифметика, без делегата»,
        // что и есть «золотой» референс. В будущем, если бэкенд эталона
        // понадобится менять (например, сверять TensorRT vs TFLite-CPU),
        // здесь можно будет принимать опцию --compare-backend.
        auto ref = inf::make_engine(args.backend);
        if (!ref) {
            std::fprintf(stderr,
                "[%s] не удалось создать бэкенд '%s' для эталона.\n",
                ref_label, args.backend.c_str());
            return;
        }
        inf::Engine::Options ropts;
        ropts.num_threads = args.threads;
        if (!ref->load(ref_model_path, ropts)) return;

        std::vector<TensorInfo> ref_in  = ref->inputs();
        std::vector<TensorInfo> ref_out = ref->outputs();

        if (ref_in.size() != 1) {
            std::fprintf(stderr,
                "[%s] ожидаю единственный вход, у модели %zu; пропуск.\n",
                ref_label, ref_in.size());
            return;
        }
        const auto& rs = ref_in[0].shape;
        const int  ref_c        = image_input_channels(rs);
        const bool ref_nhwc_img = ref_c > 0;
        int ref_h = 0, ref_w = 0;
        if (ref_nhwc_img) { ref_h = rs[1]; ref_w = rs[2]; }
        // Жёстко требуем [1,H,W,1|3] только в image-режиме compare: там мы
        // letterbox’им реальную картинку в ref. В --compare-random никаких
        // картинок нет — сравниваем тензор-в-тензор любого ранга.
        if (has_image && !ref_nhwc_img) {
            std::fprintf(stderr,
                "[%s] для сравнения по изображению ожидаю вход NHWC [1,H,W,1|3]; "
                "получено shape=%s; пропуск.\n",
                ref_label, shape_to_str(rs).c_str());
            return;
        }
        void* main_in_t = eng->input_data(0);
        void* ref_in_t  = ref->input_data(0);

        const size_t n_out = std::min(out_info.size(), ref_out.size());
        if (out_info.size() != ref_out.size()) {
            std::fprintf(stderr,
                "[%s] число выходов различается (%zu vs %zu); "
                "сравниваю первые %zu по индексу.\n",
                ref_label, out_info.size(), ref_out.size(), n_out);
        }

        // Аккумуляторы метрик по каждому выходу, агрегированные по всем
        // прогонам. В режиме одиночного запуска цифры совпадают со старым
        // поведением; в режиме --random-input N просто усредняются по
        // большему числу элементов (все runs × elems_per_run).
        struct Acc { double sum = 0, sse = 0, maxd = 0; size_t n = 0;
                     bool   trimmed = false; };
        std::vector<Acc> acc(n_out);

        // Одна итерация сравнения: оба интерпретатора уже должны быть
        // заполнены входами. Делает invoke обоих и аккумулирует поэлементные
        // расхождения. Возвращает false при ошибке инференса (фатально).
        // run_idx используется только для CSV-экспорта (per-run строки),
        // на агрегатную статистику не влияет.
        auto compare_once = [&](int run_idx) -> bool {
            if (!safe_invoke(*eng)) {
                std::fprintf(stderr, "[main] Invoke упал.\n");
                return false;
            }
            if (!safe_invoke(*ref)) {
                std::fprintf(stderr, "[%s] Invoke упал.\n", ref_label);
                return false;
            }
            std::vector<float> a, b;
            for (size_t i = 0; i < n_out; ++i) {
                const void* ta = eng->output_data((int)i);
                const void* tb = ref->output_data((int)i);
                if (!dequantize_output(out_info[i], ta, a)) continue;
                if (!dequantize_output(ref_out[i], tb, b)) continue;
                const size_t m = std::min(a.size(), b.size());
                if (m == 0) continue;
                const bool trimmed_now = (a.size() != b.size());
                if (trimmed_now) acc[i].trimmed = true;
                // Локальные аккумуляторы для CSV-строки этой итерации.
                // Глобальный acc[i] обновляем тут же — двойной проход
                // по данным был бы лишним.
                double r_sum = 0.0, r_sse = 0.0, r_max = 0.0;
                for (size_t j = 0; j < m; ++j) {
                    double d = std::fabs((double)a[j] - (double)b[j]);
                    r_sum += d;
                    r_sse += d * d;
                    if (d > r_max) r_max = d;
                }
                acc[i].sum  += r_sum;
                acc[i].sse  += r_sse;
                if (r_max > acc[i].maxd) acc[i].maxd = r_max;
                acc[i].n    += m;

                if (dump_compare) {
                    const double mae  = r_sum / (double)m;
                    const double rmse = std::sqrt(r_sse / (double)m);
                    const float  s    = out_info[i].scale;
                    const double inv_s = (s > 0.0f)
                        ? 1.0 / (double)s : 0.0;
                    compare_csv.writef(
                        "%s,%d,%zu,%s,%zu,"
                        "%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%d",
                        ref_label, run_idx, i,
                        csv_escape(out_info[i].name).c_str(), m,
                        mae, rmse, r_max, (double)s,
                        mae  * inv_s,
                        rmse * inv_s,
                        r_max * inv_s,
                        trimmed_now ? 1 : 0);
                }
            }
            return true;
        };

        int runs_done = 0;
        if (args.random_input) {
            // ---- Режим случайных входов ----
            // Каждая итерация: один и тот же случайный буфер uint8 [0..255]
            // заливается в обе сети через fill_input (нормализация в [0,1]
            // и квантование берутся из самой модели — каждая сеть видит
            // свою «честную» подачу). Работает для произвольных шейпов:
            // картинок [1,H,W,3], таблиц [1,9,9,1], векторов [1,N] и т.д.
            const uint32_t seed = args.random_seed
                ? args.random_seed
                : (uint32_t)std::random_device{}();
            std::mt19937 rng(seed);
            std::uniform_int_distribution<int> dist(0, 255);

            const size_t main_n   = numel(in_info[0].shape);
            const size_t ref_n    = numel(rs);
            const bool same_shape = (in_info[0].shape == rs);
            const bool same_numel = (main_n == ref_n);
            // «letterbox под ref» поддерживаем только для RGB-картинок
            // (in_c == ref_c == 3). letterbox() работает на 3-канальном
            // буфере; для grayscale-vs-grayscale потребовался бы отдельный
            // 1-канальный letterbox, для grayscale-vs-RGB — ещё и конверсия
            // luma↔RGB. Случай редкий — фолбэкаем в «независимый рандом»
            // (помечается в логе как несопоставимый по diff’у).
            const bool both_img   = nhwc_img && ref_nhwc_img
                                    && in_c == 3 && ref_c == 3;

            const char* mode_note;
            if      (same_shape) mode_note = " (общий буфер)";
            else if (same_numel) mode_note = " (разный shape, тот же numel — общий буфер)";
            else if (both_img)   mode_note = " (letterbox под ref)";
            else                 mode_note = " (независимые случайные входы — diff не сопоставим)";

            std::printf("Случайные входы: runs=%d, seed=%u, main=%s, ref=%s%s\n",
                        args.random_runs, seed,
                        shape_to_str(in_info[0].shape).c_str(),
                        shape_to_str(rs).c_str(), mode_note);

            if (main_n == 0 || ref_n == 0) {
                std::fprintf(stderr,
                    "[%s] неподдерживаемый динамический shape входа; пропуск.\n",
                    ref_label);
                return;
            }

            std::vector<uint8_t> rnd_main(main_n);
            std::vector<uint8_t> rnd_ref;
            for (int r = 0; r < args.random_runs && !g_interrupted; ++r) {
                for (auto& v : rnd_main) v = (uint8_t)dist(rng);
                if (!fill_input(rnd_main, in_info[0], main_in_t)) return;
                if (same_shape || same_numel) {
                    // fill_input трактует входной буфер плоско — при равном
                    // числе элементов раскладка по осям не важна.
                    if (!fill_input(rnd_main, ref_in[0], ref_in_t)) return;
                } else if (both_img) {
                    Image tmp; tmp.rgb = rnd_main; tmp.w = in_w; tmp.h = in_h;
                    letterbox(tmp, ref_w, ref_h, rnd_ref);
                    if (!fill_input(rnd_ref, ref_in[0], ref_in_t)) return;
                } else {
                    // Шейпы и numel разные, картинками не являются — других
                    // разумных вариантов нет. Льём независимый рандом, метрики
                    // diff’а интерпретировать осторожно (пометили в логе).
                    rnd_ref.resize(ref_n);
                    for (auto& v : rnd_ref) v = (uint8_t)dist(rng);
                    if (!fill_input(rnd_ref, ref_in[0], ref_in_t)) return;
                }
                if (!compare_once(r)) return;
                ++runs_done;
            }
        } else {
            // ---- Режим одного изображения (старое поведение) ----
            if (!has_image) {
                std::fprintf(stderr,
                    "[%s] изображения нет; используйте --random-input.\n",
                    ref_label);
                return;
            }
            // main_in_t уже заполнен исходным letterbox’ом во внешнем коде —
            // не перезаливаем, чтобы избежать лишней работы. Заливаем только
            // ref (его размер и/или число каналов может отличаться: для
            // ref_c == 1 конвертируем RGB → luma BT.601, как в основном пути).
            std::vector<uint8_t> ref_rgb;
            letterbox(img, ref_w, ref_h, ref_rgb);
            if (ref_c == 1) {
                std::vector<uint8_t> ref_gray;
                rgb_to_gray(ref_rgb.data(), (size_t)ref_w * ref_h, ref_gray);
                if (!fill_input(ref_gray, ref_in[0], ref_in_t)) return;
            } else {
                if (!fill_input(ref_rgb, ref_in[0], ref_in_t)) return;
            }
            if (!compare_once(0)) return;
            ++runs_done;
        }

        if (runs_done == 0) {
            std::fprintf(stderr, "[%s] не выполнено ни одного прогона.\n",
                         ref_label);
            return;
        }
        if (runs_done > 1) {
            std::printf("Прогонов выполнено: %d (метрики усреднены по всем)\n",
                        runs_done);
        }

        double agg_mae_sum = 0.0, agg_max = 0.0;
        size_t agg_n = 0;
        for (size_t i = 0; i < n_out; ++i) {
            if (acc[i].n == 0) continue;
            const double mae  = acc[i].sum / (double)acc[i].n;
            const double rmse = std::sqrt(acc[i].sse / (double)acc[i].n);
            const char* note  = acc[i].trimmed ? "  (срез по min длине)" : "";
            std::printf("  output[%zu] %-32s elems=%zu  "
                        "MAE=%.6f  RMSE=%.6f  max|diff|=%.6f%s\n",
                        i, out_info[i].name.c_str(), acc[i].n,
                        mae, rmse, acc[i].maxd, note);
            const float s_main = out_info[i].scale;
            if (s_main > 0.0f) {
                std::printf("                                              "
                            "scale=%.6g  ->  MAE=%.2f*q  RMSE=%.2f*q  "
                            "max|diff|=%.2f*q\n",
                            s_main,
                            mae        / s_main,
                            rmse       / s_main,
                            acc[i].maxd/ s_main);
            }
            if (dump_compare_sum) {
                const double inv_s = (s_main > 0.0f)
                    ? 1.0 / (double)s_main : 0.0;
                compare_summary_csv.writef(
                    "%s,%zu,%s,%d,%zu,"
                    "%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%d",
                    ref_label, i,
                    csv_escape(out_info[i].name).c_str(),
                    runs_done, acc[i].n,
                    mae, rmse, acc[i].maxd, (double)s_main,
                    mae         * inv_s,
                    rmse        * inv_s,
                    acc[i].maxd * inv_s,
                    acc[i].trimmed ? 1 : 0);
            }
            agg_mae_sum += acc[i].sum;
            if (acc[i].maxd > agg_max) agg_max = acc[i].maxd;
            agg_n       += acc[i].n;
        }
        if (agg_n > 0) {
            std::printf("  -> суммарно: MAE=%.6f  max|diff|=%.6f  (%zu элементов)\n",
                        agg_mae_sum / (double)agg_n, agg_max, agg_n);
        }
        if (dump_compare)     compare_csv.flush();
        if (dump_compare_sum) compare_summary_csv.flush();
    };

    // Между «invoke» и «warmup» в режиме --compare* проходит долгий
    // отрезок времени — оба интерпретатора (целевой и эталонный)
    // прогоняются по N random-итераций. Без явных меток sysmon
    // получает «дырку»: длинный сегмент без сэмплов, который потом
    // выглядит на графике как плоская линия от init/invoke
    // до warmup и искажает картину «загрузки таргетной модели».
    // Поэтому ставим маркеры phase=compare до и после run_compare.
    // На сводных графиках эти точки потом фильтруются (это не
    // нагрузка целевой модели, а служебная стадия), но в raw CSV
    // они остаются для диагностики.
    const bool any_compare = !args.compare_model.empty()
        || (args.compare_cpu && !args.no_delegate);
    if (any_compare) sysmon_log("compare");
    if (!args.compare_model.empty()) {
        run_compare(args.compare_model, "REF");
    }
    if (args.compare_cpu) {
        if (args.no_delegate) {
            std::fprintf(stderr,
                "--compare-cpu проигнорирован: основной запуск уже на CPU.\n");
        } else {
            run_compare(args.model, "CPU");
        }
    }
    if (any_compare) sysmon_log("compare");

    // Если входа нет вообще — дальше идти нечего. В режиме --random-input
    // вход уже залит случайным буфером выше, так что --benchmark / --loop
    // (без --display) могут отработать. --display и YOLO-постпроцессинг
    // требуют реальной картинки и тут пропускаются.
    if (!has_image) {
        if (args.yolo) {
            std::fprintf(stderr,
                "--yolo требует входное изображение, пропуск постобработки.\n");
        }
        if (args.display) {
            // Сюда не должны попасть из-за проверки в начале main, но на
            // всякий случай.
            return 0;
        }
    }

    // ---- Бенчмарк ----
    // Без --export меряем только агрегат (один вызов now_ms() на N
    // итераций — минимум накладных). С --export таймим каждую
    // итерацию отдельно, чтобы получить распределение латентности
    // (для гистограмм / квантилей p50/p95/p99 / поиска выбросов).
    // Накладные ~50–100 нс на now_ms() ничтожны на фоне инференса.
    if (args.benchmark) {
        // В tile-режиме «один прогон» — это полный pass по всем тайлам
        // исходного кадра. Меряем именно его, чтобы пользователь видел
        // реальный FPS на исходном разрешении, а не на 96×96.
        // В обычном режиме — один eng->invoke() как раньше.
        auto bench_step = [&]() -> bool {
            if (args.tile_mode) {
                return run_tile_pass(img.rgb.data(), img.w, img.h);
            }
            return safe_invoke(*eng);
        };
        for (int i = 0; i < args.warmup; ++i) bench_step();
        // После прогрева — фиксируем CPU/память: на этом интервале
        // прогревочные invoke’ы уже отработали, кэши/JIT делегата
        // прогрелись, а сам бенчмарк ещё не начался.
        sysmon_log("warmup");

        CsvExport bench_csv;
        const bool dump_bench = open_export(bench_csv, "bench", "benchmark");
        if (dump_bench) {
            bench_csv.meta("warmup", "%d", args.warmup);
            bench_csv.meta("runs",   "%d", args.runs);
            if (args.tile_mode) {
                bench_csv.meta("tile_mode",    "%s", "1");
                bench_csv.meta("tile_overlap", "%d", args.tile_overlap);
                bench_csv.meta("tile_in",      "%dx%d", in_w, in_h);
                bench_csv.meta("source_in",    "%dx%d", img.w, img.h);
                bench_csv.meta("source_out",   "%dx%d",
                               img.w * scale_x, img.h * scale_y);
            }
            bench_csv.header("run,ms");
        }

        double sum = 0.0, mn = 1e18, mx = 0.0;
        // При активном экспорте дополнительно копим полный массив
        // латентностей — нужен для перцентилей в summary. 1000 double’ов
        // = 8 КБ, на пайплайн не влияет.
        std::vector<double> lat;
        // Периодический семпл sysmon во время бенчмарка: на длинных
        // прогонах (--runs 5000+ / десятки секунд) одной точкой не
        // увидишь ни троттлинга по нагреву, ни плавающего RSS, ни
        // постепенного роста системной нагрузки от соседей по хосту.
        // last_sysmon_t = -inf, чтобы первый интервал успел набраться.
        // Сами замеры идут с фазой "bench" — те же колонки, что у
        // финального семпла «после бенчмарка»; различить можно по t_ms.
        const bool sysmon_periodic =
            args.sysmon && sysmon.initialized()
            && args.sysmon_interval_ms > 0;
        double last_sysmon_t = -1e18;
        auto maybe_sysmon = [&]() {
            if (!sysmon_periodic) return;
            const double now = now_ms();
            if (now - last_sysmon_t >= (double)args.sysmon_interval_ms) {
                sysmon_log("bench");
                last_sysmon_t = now;
            }
        };

        if (dump_bench) {
            lat.reserve((size_t)args.runs);
            for (int i = 0; i < args.runs; ++i) {
                double t0b = now_ms();
                bench_step();
                double dt = now_ms() - t0b;
                sum += dt;
                if (dt < mn) mn = dt;
                if (dt > mx) mx = dt;
                lat.push_back(dt);
                bench_csv.writef("%d,%.6f", i, dt);
                maybe_sysmon();
            }
            bench_csv.flush();
            double avg = sum / args.runs;
            std::printf("Бенчмарк: %d %s, среднее %.3f мс "
                        "(min %.3f, max %.3f), %.1f %s/с\n",
                        args.runs,
                        args.tile_mode ? "полных кадров" : "итераций",
                        avg, mn, mx, 1000.0 / avg,
                        args.tile_mode ? "кадр" : "инф");
        } else {
            double s0 = now_ms();
            for (int i = 0; i < args.runs; ++i) {
                bench_step();
                maybe_sysmon();
            }
            double s1 = now_ms();
            double avg = (s1 - s0) / args.runs;
            std::printf("Бенчмарк: %d %s, среднее %.3f мс, %.1f %s/с\n",
                        args.runs,
                        args.tile_mode ? "полных кадров" : "итераций",
                        avg, 1000.0 / avg,
                        args.tile_mode ? "кадр" : "инф");
        }
        // Финальный семпл «сразу после бенчмарка» — даёт дельту от
        // последнего periodic-семпла (или от warmup, если periodic не
        // успел сработать на коротком прогоне). В CSV отличается от
        // periodic-семплов более коротким wall_ms.
        sysmon_log("bench");

        // ---- Bench summary ----
        // Одна строка с агрегатами. Файл само-достаточен: все
        // мета-поля (модель, делегат, threads, runs) уже есть в шапке
        // через open_export, плюс перцентили p50/p95/p99 — для оценки
        // хвоста распределения (важнее среднего, когда есть выбросы).
        // Несколько прогонов разных моделей собираются в одну таблицу
        // тривиально:  cat results/*.bench.summary.csv | grep -v '^#'
        if (dump_bench && !lat.empty()) {
            CsvExport bs;
            if (open_export(bs, "bench.summary", "benchmark_summary")) {
                bs.meta("warmup", "%d", args.warmup);
                bs.meta("runs",   "%d", args.runs);
                bs.header(
                    "runs,avg_ms,min_ms,max_ms,"
                    "p50_ms,p95_ms,p99_ms,std_ms,throughput_ips");
                std::sort(lat.begin(), lat.end());
                auto pct = [&](double q) {
                    size_t k = (size_t)std::lround(
                        q * (double)(lat.size() - 1));
                    if (k >= lat.size()) k = lat.size() - 1;
                    return lat[k];
                };
                const double avg = sum / (double)lat.size();
                double sse = 0.0;
                for (double v : lat) sse += (v - avg) * (v - avg);
                const double stddev = std::sqrt(sse / (double)lat.size());
                bs.writef("%zu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f",
                          lat.size(), avg, mn, mx,
                          pct(0.50), pct(0.95), pct(0.99),
                          stddev, 1000.0 / avg);
            }
        }
    }

    // ---- Видео-цикл без окна ----
    // Источника кадров нет — общий цикл просто крутит invoke + FPS.
    // Полезно, чтобы померить чистую пропускную способность NPU.
    // В tile-режиме источник обязателен (статический исходник),
    // чтобы run_tile_pass получал полноразмерный кадр для нарезки.
    if (args.loop && !args.display) {
        std::unique_ptr<StaticFrameSource> static_src;
        if (args.tile_mode && has_image) {
            static_src = std::make_unique<StaticFrameSource>(
                img.rgb.data(), img.w, img.h);
        }
        int rc = run_video_loop(static_src.get(), /*disp=*/nullptr);
        sysmon_finalize();
        return rc;
    }

    // ---- Графический вывод (Wayland/EGL) ----
    // Показываем один из трёх возможных буферов в зависимости от флагов:
    //   --show-output  → декодированный выход модели (SR/enhance/denoise);
    //   --show-input   → препроцессированный letterbox-вход (отладка);
    //   default        → оригинальная картинка.
    if (args.display) {
        auto disp = make_display();
        if (!disp) {
            std::fprintf(stderr,
                "Поддержка дисплея не собрана (USE_DISPLAY=OFF).\n");
            return 7;
        }
        if (!disp->init(args.win_w, args.win_h, "ii", args.vsync)) return 7;
        std::printf("Display: окно %dx%d, vsync=%s\n",
                    args.win_w, args.win_h,
                    args.vsync ? "on" : "off");

        const uint8_t* frame;
        int fw, fh;
        if (args.show_output && !out_img.rgb.empty()) {
            frame = out_img.rgb.data();
            fw    = out_img.width;
            fh    = out_img.height;
        } else if (args.show_input) {
            frame = input_rgb.data();
            fw    = in_w;
            fh    = in_h;
        } else {
            // Без картинки (--random-input) тут будет img.rgb=empty —
            // но мы сюда не попадём, проверка на старте отсекла такой
            // путь, требуя или image, или --camera, или --show-output.
            frame = img.rgb.data();
            fw    = img.w;
            fh    = img.h;
        }

        // ---- Цикл с окном ----
        // Источник статический: вход модели уже залит в image-блоке
        // выше, поэтому StaticFrameSource просит цикл ничего не
        // препроцессить — только invoke + show_rgb на каждой итерации.
        // В tile-режиме источник для run_tile_pass — это всегда исходное
        // изображение (а не letterbox/canvas), потому что тайлы нарезаются
        // из исходного разрешения. Frame, который отображается в окне,
        // выбирается отдельно внутри run_video_loop по тем же приоритетам
        // (--show-output > --show-input > исходный кадр).
        if (args.loop) {
            const uint8_t* src_buf = args.tile_mode ? img.rgb.data() : frame;
            const int      src_w   = args.tile_mode ? img.w : fw;
            const int      src_h   = args.tile_mode ? img.h : fh;
            StaticFrameSource src(src_buf, src_w, src_h);
            int rc = run_video_loop(&src, disp.get());
            sysmon_finalize();
            return rc;
        }

        // ---- Одиночный кадр ----
        if (args.stats) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "inference %.2f ms", t1 - t0);
            disp->set_overlay_text(buf);
        }
        if (args.yolo) {
            run_yolo_postproc(img.w, img.h);
            disp->set_boxes(boxes);
        }
        if (!disp->show_rgb(frame, fw, fh)) { sysmon_finalize(); return 0; }
        std::printf("Окно открыто. Закройте его, чтобы выйти.\n");
        // Один статический кадр: блокируемся на событиях Wayland до закрытия.
        while (disp->wait()) {}
    }

    sysmon_finalize();
    return 0;
}
