// Абстрактный источник видеокадров для раннера.
//
// Реализация по умолчанию — V4L2 (camera_v4l2.cpp): UVC/CSI камеры на
// Linux. Если USE_CAMERA=OFF в CMake или сборка идёт
// под платформой без V4L2, подключается заглушка (camera_stub.cpp), и
// make_camera() возвращает nullptr — клиентский код просто сообщает,
// что захват не доступен.
//
// Дизайн заточен под видео-пайплайн с минимумом копий:
//   * open() выделяет mmap-буферы один раз;
//   * grab() делает DQBUF, конвертирует пиксели в RGB во ВНУТРЕННИЙ
//     буфер Camera и сразу возвращает буфер драйверу через QBUF;
//   * возвращаемый указатель валиден до следующего grab() / close().
// За счёт этого на каждый кадр идёт одна аллокация-free пара (внутри
// stb для MJPEG-веток) либо ноль (для YUYV — конвертация in-place в
// заранее выделенный буфер).

#pragma once

#include <cstdint>
#include <memory>
#include <string>

struct Camera {
    virtual ~Camera() = default;

    // Открыть устройство и запустить streaming.
    //   device  — путь, например "/dev/video0".
    //   w, h    — желаемое разрешение; драйвер может выбрать ближайшее.
    //             Передайте 0, чтобы оставить текущие настройки драйвера.
    //   fps     — желаемая частота кадров (best effort). 0 — не трогать.
    // Реальные параметры читайте через width()/height()/fps().
    virtual bool open(const std::string& device, int w, int h, int fps) = 0;

    virtual int width()  const = 0;
    virtual int height() const = 0;
    virtual int fps()    const = 0;

    // Захватить очередной кадр и сконвертировать в RGB HWC 8 bpp.
    // Возвращает указатель на ВНУТРЕННИЙ буфер размером 3*width()*height();
    // буфер валиден до следующего grab()/close().
    //
    //   timeout_ms > 0  — ждать максимум указанное время;
    //   timeout_ms == 0 — не ждать, вернуть nullptr если кадр ещё не готов;
    //   timeout_ms <  0 — блокироваться без таймаута.
    // nullptr — таймаут или ошибка (см. stderr).
    virtual const uint8_t* grab(int timeout_ms = 1000) = 0;

    // Остановить streaming, отдать буферы драйверу, закрыть fd.
    // Идемпотентно; вызывается из деструктора.
    virtual void close() = 0;
};

// Возвращает реализацию Camera для текущей платформы или nullptr,
// если поддержка не собрана (USE_CAMERA=OFF / не Linux).
std::unique_ptr<Camera> make_camera();
