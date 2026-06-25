// Диспетчер реализаций видеодекодера (--video). Аналог inference.cpp::
// make_engine для бэкендов инференса: выбор реализации в рантайме по
// имени, доступность — по compile-time флагам VIDEO_HAS_* (их выставляет
// CMake под опции USE_VIDEO_PIPELINE / USE_VIDEO_LIBAV).
//
// Компилируется только когда USE_VIDEO=ON. При USE_VIDEO=OFF вместо него
// собирается video_stub.cpp (там make_video() возвращает nullptr).

#include "video.h"

#include <cstdio>

// Фабрики конкретных реализаций — определены в своих TU, объявлены здесь.
#if VIDEO_HAS_PIPELINE
std::unique_ptr<VideoSource> make_ffmpeg_pipeline_video();
#endif
#if VIDEO_HAS_LIBAV
std::unique_ptr<VideoSource> make_ffmpeg_libav_video();
#endif
#if VIDEO_HAS_GSTREAMER
std::unique_ptr<VideoSource> make_gstreamer_video();
#endif

std::vector<std::string> available_video_decoders() {
    std::vector<std::string> v;
    // Порядок = приоритет авто-выбора: libav (полноценный, точные параметры)
    // важнее gstreamer (нужен на устройстве ради аппаратного VPU, но включается
    // отдельно), тот в свою очередь важнее pipeline (внешний процесс).
#if VIDEO_HAS_LIBAV
    v.push_back("libav");
#endif
#if VIDEO_HAS_GSTREAMER
    v.push_back("gstreamer");
#endif
#if VIDEO_HAS_PIPELINE
    v.push_back("pipeline");
#endif
    return v;
}

std::unique_ptr<VideoSource> make_video(const std::string& decoder) {
    std::string d = decoder;
    if (d.empty() || d == "auto") {
        // Авто: предпочесть libav (без процесса, точные параметры), затем
        // gstreamer (аппаратный декод на устройстве), затем pipeline.
#if VIDEO_HAS_LIBAV
        d = "libav";
#elif VIDEO_HAS_GSTREAMER
        d = "gstreamer";
#elif VIDEO_HAS_PIPELINE
        d = "pipeline";
#else
        d = "";
#endif
    }

#if VIDEO_HAS_LIBAV
    if (d == "libav") return make_ffmpeg_libav_video();
#endif
#if VIDEO_HAS_GSTREAMER
    if (d == "gstreamer") return make_gstreamer_video();
#endif
#if VIDEO_HAS_PIPELINE
    if (d == "pipeline") return make_ffmpeg_pipeline_video();
#endif

    std::fprintf(stderr,
        "Видеодекодер '%s' недоступен в этой сборке. Собрано: ",
        decoder.empty() ? "auto" : decoder.c_str());
    const auto avail = available_video_decoders();
    if (avail.empty()) {
        std::fprintf(stderr, "ничего (USE_VIDEO=OFF).\n");
    } else {
        for (size_t i = 0; i < avail.size(); ++i)
            std::fprintf(stderr, "%s%s", i ? ", " : "", avail[i].c_str());
        std::fprintf(stderr, ".\n");
    }
    return nullptr;
}
