// Абстрактный источник кадров из ВИДЕОФАЙЛА (в отличие от camera.* —
// живого захвата с устройства).
//
// Три взаимозаменяемые реализации, как у бэкендов инференса (выбор в
// сборке через USE_VIDEO_*, диспетч в рантайме через make_video(name)):
//   * "pipeline" (video_ffmpeg_pipeline.cpp) — внешний процесс ffmpeg
//     отдаёт сырые RGB-кадры в pipe. Линк-зависимостей нет, нужен лишь
//     бинарь ffmpeg/ffprobe в рантайме. Максимально портируемо.
//   * "libav" (video_ffmpeg_libav.cpp) — линковка libavformat/libavcodec/
//     libswscale напрямую. Полноценный декодер: без накладных расходов на
//     процесс, точные fps/длительность, seek/loop средствами библиотеки.
//   * "gstreamer" (video_gstreamer.cpp) — конвейер filesrc ! decodebin !
//     videoconvert ! appsink. На встраиваемых платформах decodebin
//     подхватывает АППАРАТНЫЙ VPU-декодер — путь к hardware-decode.
// Отсюда обобщённое имя интерфейса VideoSource, не привязанное к способу
// декодирования.
//
// Если USE_VIDEO=OFF — подключается video_stub.cpp, make_video() всегда
// возвращает nullptr, available_video_decoders() пуст (клиентский код
// сообщит «поддержка видео не собрана»).
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
#include <vector>

struct VideoSource {
    virtual ~VideoSource() = default;

    // Открыть видеофайл и запустить декодирование.
    //   path     — путь к файлу.
    //   ffmpeg   — путь/имя бинаря ffmpeg ("ffmpeg" если в PATH, либо
    //              абсолютный путь, напр. "/opt/bin/ffmpeg" на устройстве).
    //              Используется только реализацией "pipeline"; "libav" его
    //              игнорирует (декодирует библиотекой).
    //   ffprobe  — путь/имя ffprobe: им узнаём width/height/fps файла
    //              (тоже только "pipeline"; "libav" берёт параметры из
    //              libavformat).
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

// Создать источник видео заданной реализацией:
//   decoder == "" / "auto"            — выбрать наилучшую из собранных
//       (приоритет libav -> gstreamer -> pipeline);
//   "pipeline" / "libav" / "gstreamer" — конкретная реализация.
//   gl — только для "gstreamer": собрать звено конверсии через GL
//       (glupload ! glcolorconvert ! gldownload) вместо videoconvert.
//       Нужно на SoC, где VPU отдаёт кадры только как DMABuf/DMA_DRM;
//       тянет EGL/Wayland-контекст. Прочие реализации игнорируют.
// Возвращает nullptr, если запрошенная реализация не собрана (или
// USE_VIDEO=OFF).
std::unique_ptr<VideoSource> make_video(const std::string& decoder = "",
                                        bool gl = false);

// Список собранных реализаций видеодекодера ("libav", "gstreamer",
// "pipeline"). Пуст при USE_VIDEO=OFF. Порядок = приоритет авто-выбора.
std::vector<std::string> available_video_decoders();
