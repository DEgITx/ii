// Абстрактный источник кадров из ВИДЕОФАЙЛА (в отличие от camera.* —
// живого захвата с устройства).
//
// Первая и пока единственная реализация — video_ffmpeg_pipeline.cpp:
// внешний процесс ffmpeg декодирует файл и отдаёт сырые RGB-кадры в
// pipe, без линковки libav в бинарь (нужен лишь ffmpeg/ffprobe в
// рантайме). В будущем рядом могут встать другие реализации (libav
// напрямую, платформенные декодеры) — отсюда обобщённое имя интерфейса
// VideoSource, не привязанное к ffmpeg.
//
// Если USE_VIDEO=OFF — подключается video_stub.cpp и make_video()
// возвращает nullptr (клиентский код сообщит «поддержка видео не
// собрана»).
//
// Контракт grab() намеренно совпадает с Camera::grab(): возвращает
// указатель на ВНУТРЕННИЙ RGB888 HWC-буфер размером 3*width()*height(),
// валидный до следующего grab()/close(). За счёт этого VideoFrameSource
// в frame_source.h — такая же тонкая обёртка, как CameraFrameSource, и
// весь видео-цикл раннера переиспользуется без изменений. Единственное
// отличие от живой камеры — конечность потока: eof() сообщает, что файл
// дошёл до конца (в отличие от grab()==nullptr, который у камеры значит
// лишь «кадр ещё не готов»).

#pragma once

#include <cstdint>
#include <memory>
#include <string>

struct VideoSource {
    virtual ~VideoSource() = default;

    // Открыть видеофайл и запустить декодирование.
    //   path     — путь к файлу.
    //   ffmpeg   — путь/имя бинаря ffmpeg ("ffmpeg" если в PATH, либо
    //              абсолютный путь, напр. "/opt/bin/ffmpeg" на устройстве).
    //   ffprobe  — путь/имя ffprobe: им узнаём width/height/fps файла,
    //              чтобы знать размер одного rgb24-кадра в байтах.
    //   loop     — true: зациклить воспроизведение (eof() никогда не true).
    // Реальные параметры читайте через width()/height()/fps().
    virtual bool open(const std::string& path,
                      const std::string& ffmpeg,
                      const std::string& ffprobe,
                      bool loop) = 0;

    virtual int width()  const = 0;
    virtual int height() const = 0;
    virtual int fps()    const = 0;

    // Очередной кадр в RGB888 HWC. Возвращает указатель на внутренний
    // буфер (валиден до следующего grab()/close()) либо nullptr — конец
    // файла (тогда eof()==true) или ошибка чтения.
    virtual const uint8_t* grab() = 0;

    // true, если достигнут конец файла. Для --video-loop никогда не true.
    virtual bool eof() const = 0;

    // Остановить декодер, закрыть pipe. Идемпотентно; зовётся из деструктора.
    virtual void close() = 0;
};

// Возвращает реализацию VideoSource для текущей сборки или nullptr,
// если поддержка не собрана (USE_VIDEO=OFF).
std::unique_ptr<VideoSource> make_video();
