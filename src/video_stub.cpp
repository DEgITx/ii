// Заглушка модуля VideoSource при сборке с -DUSE_VIDEO=OFF. Реальные
// реализации — video_ffmpeg_pipeline.cpp (внешний ffmpeg + pipe) и
// video_ffmpeg_libav.cpp (линковка libav*), выбираемые диспетчером
// video.cpp. При выключенной поддержке всё это не компилируется, а
// make_video() всегда возвращает nullptr — клиентский код (ii.cpp)
// сообщит «поддержка видео не собрана».

#include "video.h"

std::unique_ptr<VideoSource> make_video(const std::string&, bool, bool) {
    return nullptr;
}

std::vector<std::string> available_video_decoders() { return {}; }
