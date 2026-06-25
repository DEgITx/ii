// Заглушка модуля VideoSource при сборке с -DUSE_VIDEO=OFF. Реальная
// реализация — video_ffmpeg_pipeline.cpp (внешний ffmpeg + pipe).
// make_video() возвращает nullptr — клиентский код (ii.cpp) увидит это
// и сообщит «поддержка видео не собрана».

#include "video.h"

std::unique_ptr<VideoSource> make_video() { return nullptr; }
