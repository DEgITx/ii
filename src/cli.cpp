// Реализация cli.h: print_usage + parse_args.

#include "cli.h"

#include <cstdio>
#include <cstdlib>

namespace ii {

void print_usage(const char* prog) {
    const char* def_delegate = ii::default_delegate_path();
    const char* def_delegate_show = (def_delegate && *def_delegate)
        ? def_delegate : "(нет на этой платформе)";
    // Собираем список собранных бэкендов для подсказки в --help.
    std::string backends_str;
    for (const auto& b : ii::available_backends()) {
        if (!backends_str.empty()) backends_str += ", ";
        backends_str += b;
    }
    if (backends_str.empty()) backends_str = "(нет)";
    std::printf(
        "Usage: %s <model.tflite> [image] [options]\n"
        "  image не обязателен в режиме --random-input.\n"
        "Options:\n"
        "  --backend <name>    бэкенд инференса (доступны: %s; def: авто)\n"
        "  --delegate <path>   путь к делегату/плагину (по умолчанию %s)\n"
        "  --no-delegate       запустить на CPU (без делегата)\n"
        "  --benchmark         прогрев + замер скорости\n"
        "  --runs <N>          число итераций бенчмарка (по умолчанию 100)\n"
        "  --warmup <N>        число итераций прогрева (по умолчанию 3)\n"
        "  --threads <N>       число CPU-потоков интерпретатора\n"
        "  --display           открыть окно (Wayland/EGL/GLES2)\n"
        "  --show-input        выводить препроцесированный (letterbox) вход\n"
        "  --win <WxH>         стартовый размер окна (по умолчанию 960x720)\n"
        "  --loop              непрерывный цикл инференс+отрисовка (Ctrl+C)\n"
        "  --stats             счётчик FPS/jitter (оверлей + лог в stdout)\n"
        "  --log-interval <ms> период лога FPS, мс (по умолчанию 1000)\n"
        "  --no-vsync          отключить vsync (макс. FPS, возможен tearing)\n"
        "  --yolo              декодировать выход как YOLOv8 и рисовать боксы\n"
        "  --conf <p>          порог уверенности (по умолчанию 0.25)\n"
        "  --iou <p>           порог IoU для NMS (по умолчанию 0.45)\n"
        "  --max-dets <N>      макс. число детекций (по умолчанию 300)\n"
        "  --yolo-pixel-coords координаты модели в пикселях, а не 0..1\n"
        "  --yolo-output <i>   индекс выхода с боксами (по умолчанию авто)\n"
        "  --classes <path>    файл имён классов (по строке; default = COCO80)\n"
        "  --compare-cpu       прогнать ту же модель на CPU и сравнить выходы\n"
        "  --compare <path>    прогнать эталонную .tflite на CPU и сравнить\n"
        "                      (например, исходный float32 vs INT8 на NPU)\n"
        "  --show-output       выводить декодированный выход модели как картинку\n"
        "                      (для image-to-image: FSRCNN, enhance, denoise).\n"
        "                      Несовместимо с --yolo. По умолчанию ищем единственный\n"
        "                      выход формы NHWC [1,H,W,1|3]; см. --output-index.\n"
        "  --output-index <i>  индекс выходного тензора-изображения (если у модели\n"
        "                      несколько image-shaped выходов; def — автодетект)\n"
        "  --output-range <r>  диапазон значений выхода: unit (=[0..1], default),\n"
        "                      signed (=[-1..1]), byte (=[0..255])\n"
        "  --save-output <p>   сохранить декодированный выход в PNG (только\n"
        "                      одиночный инференс — для --loop/--camera не имеет\n"
        "                      смысла плодить кадры на каждом такте)\n"
        "  --tile              tiling/sliding-window для image-to-image моделей\n"
        "                      с маленьким окном (FSRCNN 96x96 и т.п.).\n"
        "                      Кадр режется на тайлы размером со входом модели,\n"
        "                      каждый тайл прогоняется отдельно, результаты\n"
        "                      сшиваются в полноразмерный canvas. FPS считается\n"
        "                      на полный кадр (= 1 такт). Требует image-shaped\n"
        "                      выход с целочисленным масштабом; несовместимо с\n"
        "                      --yolo / --compare* / --random-input.\n"
        "  --tile-overlap <N>  перекрытие тайлов, пикселей в input-space\n"
        "                      (по умолчанию 0). >0 включает линейное\n"
        "                      feathering — швы становятся незаметны, ценой\n"
        "                      ~16 байт/пиксель canvas'а на float-буферы.\n"
        "  --tile-pipeline-depth <N>  глубина конвейера tile-режима (default 2):\n"
        "                      invoke тайла на NPU перекрывается с decode+\n"
        "                      composite предыдущего на CPU. 1 = выключить\n"
        "                      (последовательно, для A/B-замеров).\n"
        "  --random-input      использовать случайный входной буфер вместо\n"
        "                      изображения; работает с одиночным инференсом,\n"
        "                      --benchmark, --loop, --compare* (картинку можно\n"
        "                      не указывать). Алиас: --compare-random\n"
        "  --random-runs <N>   в --compare* число случайных прогонов для\n"
        "                      агрегации метрик (def 1). Алиас: --compare-runs\n"
        "  --random-seed <N>   seed RNG для воспроизводимости (def — случайный).\n"
        "                      Алиас: --compare-seed\n"
        "  --camera [dev]      захват с камеры (Linux: V4L2 /dev/videoN;\n"
        "                      Windows: Media Foundation, dev = индекс камеры).\n"
        "                      По умолчанию /dev/video0 (индекс 0).\n"
        "                      Включает собственный цикл инференса; image и\n"
        "                      --random-input игнорируются. Совместимо с\n"
        "                      --display, --yolo, --stats, --show-input.\n"
        "  --camera-size WxH   запрашиваемое разрешение камеры (def 640x480)\n"
        "  --camera-fps <N>    запрашиваемая частота кадров камеры (def 30)\n"
        "  --video <file>      видеофайл на входе (декодируется внешним\n"
        "                      ffmpeg в pipe; нужен ffmpeg/ffprobe в PATH).\n"
        "                      Запускает общий видео-цикл; image и\n"
        "                      --random-input игнорируются. Совместимо с\n"
        "                      --display, --yolo, --tile, --show-output, --stats.\n"
        "  --video-loop        зациклить воспроизведение файла по концу\n"
        "  --video-decoder <d> реализация декодера: auto (def) | pipeline\n"
        "                      (внешний ffmpeg) | libav (линковка libav*) |\n"
        "                      gstreamer (HW-VPU через decodebin на устройстве).\n"
        "                      Доступность зависит от сборки (USE_VIDEO_*).\n"
        "  --video-gl          (gstreamer) конверсия кадра через GL\n"
        "                      (glupload!glcolorconvert!gldownload). Нужно, когда\n"
        "                      VPU отдаёт только DMABuf/DMA_DRM; тянет EGL.\n"
        "  --ffmpeg <path>     путь к бинарю ffmpeg (только pipeline; def PATH)\n"
        "  --ffprobe <path>    путь к бинарю ffprobe (только pipeline; def PATH)\n"
        "  --export <prefix>   писать замеры в CSV: <prefix>.bench.csv (для\n"
        "                      --benchmark), <prefix>.fps.csv (для видео-\n"
        "                      цикла со --stats), <prefix>.compare.csv (для\n"
        "                      --compare/--compare-cpu). Пути с / создают\n"
        "                      нужные директории заранее: --export results/run01\n"
        "  --sysmon            мониторинг CPU/памяти процесса (RSS, VmHWM,\n"
        "                      потоки, %% ядра) и системного CPU. Сводка\n"
        "                      в stdout + при --export пишется sysmon.csv\n"
        "  --sysmon-interval <ms> период семплирования sysmon внутри длинного\n"
        "                      бенчмарка и видео-цикла (по умолчанию 1000)\n"
        "  --parallel <N>      параллельный бенчмарк: N независимых экземпляров\n"
        "                      одной модели в N потоках. Проверяет, сериализует\n"
        "                      ли NPU/делегат одновременные запросы и растёт ли\n"
        "                      суммарный throughput. Меряет single-instance\n"
        "                      baseline + N-way concurrency + scaling efficiency.\n"
        "                      Вход случайный; несовместимо с --display/--camera/\n"
        "                      --yolo/--tile/--compare*/--loop. При N>1 без явного\n"
        "                      --threads интер-оп параллелизм форсится в 1.\n"
        "  --parallel-models <a,b,c>  то же, но разнородный набор моделей\n"
        "                      (multi-tenant), по одному потоку на модель.\n"
        "                      Перебивает --parallel; позиционный model можно\n"
        "                      не указывать (берётся первая из списка).\n",
        prog, backends_str.c_str(), def_delegate_show);
}

bool parse_args(int argc, char** argv, Args& a) {
    if (argc < 2) { print_usage(argv[0]); return false; }
    // model — первый позиционный аргумент, НО только если он не начинается
    // с '-'. Иначе (например `ii --parallel-models a,b,c --random-input`)
    // модель не задана позиционно: её бэкфиллит main из --parallel-models.
    int start = 1;
    if (argv[1][0] != '-') {
        a.model = argv[1];
        start   = 2;
        // image — опциональный второй позиционный аргумент. Если argv[2]
        // начинается с '-' (или его нет) — картинку не передавали и весь
        // остаток разбираем как флаги (нужно, чтобы --random-input работал
        // без изображения вообще).
        if (argc >= 3 && argv[2][0] != '-') {
            a.image = argv[2];
            start   = 3;
        }
    }
    for (int i = start; i < argc; ++i) {
        std::string s = argv[i];
        if      (s == "--backend"     && i + 1 < argc) a.backend     = argv[++i];
        else if (s == "--delegate"    && i + 1 < argc) a.delegate    = argv[++i];
        else if (s == "--no-delegate")                 a.no_delegate = true;
        else if (s == "--benchmark")                   a.benchmark   = true;
        else if (s == "--runs"        && i + 1 < argc) a.runs        = std::atoi(argv[++i]);
        else if (s == "--warmup"      && i + 1 < argc) a.warmup      = std::atoi(argv[++i]);
        else if (s == "--threads"     && i + 1 < argc) a.threads     = std::atoi(argv[++i]);
        else if (s == "--display")                     a.display     = true;
        else if (s == "--show-input")                  a.show_input  = true;
        else if (s == "--loop")                        a.loop        = true;
        else if (s == "--stats")                       a.stats       = true;
        else if (s == "--log-interval"&& i + 1 < argc) a.log_interval_ms = std::atoi(argv[++i]);
        else if (s == "--no-vsync")                    a.vsync       = false;
        else if (s == "--yolo")                        a.yolo        = true;
        else if (s == "--conf"        && i + 1 < argc) a.conf        = (float)std::atof(argv[++i]);
        else if (s == "--iou"         && i + 1 < argc) a.iou         = (float)std::atof(argv[++i]);
        else if (s == "--max-dets"    && i + 1 < argc) a.max_dets    = std::atoi(argv[++i]);
        else if (s == "--yolo-pixel-coords")           a.yolo_pixel_coords = true;
        else if (s == "--yolo-output" && i + 1 < argc) a.yolo_output = std::atoi(argv[++i]);
        else if (s == "--classes"     && i + 1 < argc) a.classes_path = argv[++i];
        else if (s == "--compare-cpu")                 a.compare_cpu = true;
        else if (s == "--compare"     && i + 1 < argc) a.compare_model = argv[++i];
        else if (s == "--show-output")                 a.show_output  = true;
        else if (s == "--output-index"  && i + 1 < argc) a.output_index = std::atoi(argv[++i]);
        else if (s == "--output-range"  && i + 1 < argc) a.output_range_str = argv[++i];
        else if (s == "--save-output"   && i + 1 < argc) a.save_output_path = argv[++i];
        else if (s == "--tile")                          a.tile_mode    = true;
        else if (s == "--tile-overlap"  && i + 1 < argc) a.tile_overlap = std::atoi(argv[++i]);
        else if (s == "--tile-pipeline-depth" && i + 1 < argc) a.tile_pipeline_depth = std::atoi(argv[++i]);
        else if (s == "--random-input"
              || s == "--compare-random")              a.random_input = true;
        else if ((s == "--random-runs" || s == "--compare-runs")
                 && i + 1 < argc)                      a.random_runs  = std::atoi(argv[++i]);
        else if ((s == "--random-seed" || s == "--compare-seed")
                 && i + 1 < argc)                      a.random_seed  = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (s == "--camera") {
            // Опциональный позиционный параметр: путь к устройству.
            // Любая следующая опция (начинается с '-') означает, что
            // путь не задан и берётся дефолт /dev/video0.
            a.camera_device = "/dev/video0";
            if (i + 1 < argc && argv[i + 1][0] != '-')
                a.camera_device = argv[++i];
        }
        else if (s == "--camera-size" && i + 1 < argc) {
            std::string v = argv[++i];
            auto x = v.find('x');
            if (x == std::string::npos) {
                std::fprintf(stderr,
                    "--camera-size ожидает WxH, получено: %s\n", v.c_str());
                return false;
            }
            a.camera_w = std::atoi(v.substr(0, x).c_str());
            a.camera_h = std::atoi(v.substr(x + 1).c_str());
        }
        else if (s == "--camera-fps"  && i + 1 < argc) a.camera_fps  = std::atoi(argv[++i]);
        else if (s == "--video"        && i + 1 < argc) a.video_path    = argv[++i];
        else if (s == "--video-loop")                   a.video_loop    = true;
        else if (s == "--video-decoder" && i + 1 < argc) a.video_decoder = argv[++i];
        else if (s == "--video-gl")                     a.video_gl      = true;
        else if (s == "--ffmpeg"       && i + 1 < argc) a.ffmpeg_path   = argv[++i];
        else if (s == "--ffprobe"      && i + 1 < argc) a.ffprobe_path  = argv[++i];
        else if (s == "--export"      && i + 1 < argc) a.export_prefix = argv[++i];
        else if (s == "--sysmon")                      a.sysmon       = true;
        else if (s == "--sysmon-interval" && i + 1 < argc)
                                                      a.sysmon_interval_ms = std::atoi(argv[++i]);
        else if (s == "--parallel"    && i + 1 < argc) a.parallel    = std::atoi(argv[++i]);
        else if (s == "--parallel-models" && i + 1 < argc) a.parallel_models = argv[++i];
        else if (s == "--win"         && i + 1 < argc) {
            // Принимаем формат "WxH" (например 1280x720).
            std::string v = argv[++i];
            auto x = v.find('x');
            if (x == std::string::npos) {
                std::fprintf(stderr, "--win ожидает WxH, получено: %s\n", v.c_str());
                return false;
            }
            a.win_w = std::atoi(v.substr(0, x).c_str());
            a.win_h = std::atoi(v.substr(x + 1).c_str());
        }
        else if (s == "-h" || s == "--help") { print_usage(argv[0]); return false; }
        else {
            std::fprintf(stderr, "Неизвестный аргумент: %s\n", s.c_str());
            return false;
        }
    }
    return true;
}

} // namespace ii
