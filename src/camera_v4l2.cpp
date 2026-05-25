// V4L2-источник видеокадров для раннера.
//
// Покрывает типовой сценарий встраиваемого Linux / SBC: USB UVC-камера
// или CSI-сенсор, экспортируемый драйвером как /dev/videoN. Используем
// самый простой
// и быстрый рабочий путь:
//
//   * mmap-буферы (V4L2_MEMORY_MMAP) — четыре штуки, чтобы драйвер
//     мог писать следующий кадр пока мы обрабатываем текущий, без
//     лишних копий между ядром и user space;
//   * predzapuska пробуем YUYV (4:2:2 packed) — нативный формат для
//     большинства UVC и CSI-сенсоров, конвертируется в RGB одним
//     проходом интегерной арифметикой BT.601;
//   * fallback — MJPEG (декодирование через stb_image, который уже
//     слинкован для путей с обычной картинкой).
//
// На каждый кадр: poll() -> DQBUF -> конвертация в наш RGB-буфер
// (выделенный один раз в open()) -> QBUF. Ни одной аллокации в
// стационарном режиме для YUYV; для MJPEG одна пара stbi_load/free
// (без неё JPEG распаковать нельзя).

#include "camera.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>

#include "stb_image.h"  // только объявления; реализация — в ii.cpp

namespace {

inline uint8_t clamp_u8(int v) {
    return v < 0 ? 0 : (v > 255 ? 255 : (uint8_t)v);
}

// YUYV 4:2:2 packed -> RGB888.
//
// YUYV хранит на пару пикселей 4 байта: Y0 U Y1 V (общий хром на двух
// соседних пикселях). Конвертация — стандартная BT.601 в Q8:
//   R = Y +              1.402 * (V-128)
//   G = Y - 0.344*(U-128) - 0.714 * (V-128)
//   B = Y + 1.772 * (U-128)
// Целочисленные коэффициенты подобраны как round(coeff * 256), сдвиг
// >>8 даёт ту же точность ±1 LSB при заметно более быстром коде, чем
// float-вариант. Один проход без побочных аллокаций.
void yuyv_to_rgb(const uint8_t* src, uint8_t* dst, int w, int h) {
    const int pairs = (w * h) / 2;  // 4 байта YUYV = 2 пикселя RGB
    for (int i = 0; i < pairs; ++i) {
        const int y0 =  src[0];
        const int u  =  src[1] - 128;
        const int y1 =  src[2];
        const int v  =  src[3] - 128;

        const int r_uv =                359 * v;
        const int g_uv = - 88 * u   -   183 * v;
        const int b_uv =  454 * u;

        dst[0] = clamp_u8(y0 + (r_uv >> 8));
        dst[1] = clamp_u8(y0 + (g_uv >> 8));
        dst[2] = clamp_u8(y0 + (b_uv >> 8));
        dst[3] = clamp_u8(y1 + (r_uv >> 8));
        dst[4] = clamp_u8(y1 + (g_uv >> 8));
        dst[5] = clamp_u8(y1 + (b_uv >> 8));

        src += 4;
        dst += 6;
    }
}

const char* fourcc_name(uint32_t f, char buf[5]) {
    buf[0] =  f        & 0xff;
    buf[1] = (f >>  8) & 0xff;
    buf[2] = (f >> 16) & 0xff;
    buf[3] = (f >> 24) & 0xff;
    buf[4] = 0;
    return buf;
}

struct MmapBuffer {
    void*  start  = nullptr;
    size_t length = 0;
};

class V4L2Camera : public Camera {
public:
    ~V4L2Camera() override { close(); }

    bool open(const std::string& device, int w, int h, int fps) override {
        device_ = device;

        // O_NONBLOCK: poll() и DQBUF не должны висеть в ядре, мы сами
        // управляем таймаутом через poll(). Гораздо проще корректно
        // обработать SIGINT.
        fd_ = ::open(device.c_str(), O_RDWR | O_NONBLOCK);
        if (fd_ < 0) {
            std::fprintf(stderr, "Camera: не открыть %s: %s\n",
                         device.c_str(), std::strerror(errno));
            return false;
        }

        v4l2_capability cap{};
        if (xioctl(VIDIOC_QUERYCAP, &cap) < 0) return fail("QUERYCAP");
        if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
            std::fprintf(stderr, "Camera: %s не умеет VIDEO_CAPTURE\n",
                         device.c_str());
            return false;
        }
        if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
            std::fprintf(stderr, "Camera: %s не поддерживает streaming\n",
                         device.c_str());
            return false;
        }

        // Пробуем форматы по приоритету: YUYV (быстрая конвертация) ->
        // MJPEG (распаковка через stb_image). Ничего не вышло — сдаёмся.
        if (!set_format(w, h, V4L2_PIX_FMT_YUYV)) {
            if (!set_format(w, h, V4L2_PIX_FMT_MJPEG)) {
                std::fprintf(stderr,
                    "Camera: %s не отдаёт ни YUYV, ни MJPEG.\n",
                    device.c_str());
                return false;
            }
        }

        // Best-effort установка FPS. Драйвер вправе проигнорировать /
        // округлить — реальное значение читаем через G_PARM.
        fps_ = 0;
        if (fps > 0) {
            v4l2_streamparm parm{};
            parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            parm.parm.capture.timeperframe.numerator   = 1;
            parm.parm.capture.timeperframe.denominator = (uint32_t)fps;
            xioctl(VIDIOC_S_PARM, &parm);
        }
        {
            v4l2_streamparm got{};
            got.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            if (xioctl(VIDIOC_G_PARM, &got) == 0
                && got.parm.capture.timeperframe.numerator > 0) {
                fps_ = (int)(got.parm.capture.timeperframe.denominator
                           / got.parm.capture.timeperframe.numerator);
            }
        }

        // Запрашиваем 4 mmap-буфера: один драйвер пишет, второй мы
        // обрабатываем, ещё два — запас от джиттера планировщика.
        v4l2_requestbuffers req{};
        req.count  = 4;
        req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        if (xioctl(VIDIOC_REQBUFS, &req) < 0) return fail("REQBUFS");
        if (req.count < 2) {
            std::fprintf(stderr,
                "Camera: драйвер выдал только %u буферов (нужно >=2).\n",
                req.count);
            return false;
        }

        buffers_.resize(req.count);
        for (uint32_t i = 0; i < req.count; ++i) {
            v4l2_buffer buf{};
            buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index  = i;
            if (xioctl(VIDIOC_QUERYBUF, &buf) < 0) return fail("QUERYBUF");
            buffers_[i].length = buf.length;
            buffers_[i].start  = mmap(nullptr, buf.length,
                                      PROT_READ | PROT_WRITE, MAP_SHARED,
                                      fd_, buf.m.offset);
            if (buffers_[i].start == MAP_FAILED) {
                buffers_[i].start = nullptr;
                std::fprintf(stderr, "Camera: mmap упал: %s\n",
                             std::strerror(errno));
                return false;
            }
        }

        // Все буферы — в очередь, и поехали.
        for (size_t i = 0; i < buffers_.size(); ++i) {
            v4l2_buffer buf{};
            buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index  = (uint32_t)i;
            if (xioctl(VIDIOC_QBUF, &buf) < 0) return fail("QBUF (init)");
        }
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (xioctl(VIDIOC_STREAMON, &type) < 0) return fail("STREAMON");
        streaming_ = true;

        rgb_.assign((size_t)width_ * height_ * 3, 0);

        char fcc[5];
        std::printf("Camera: %s %dx%d %s @ %d fps, %zu mmap-буферов\n",
                    device.c_str(), width_, height_,
                    fourcc_name(pix_fmt_, fcc), fps_, buffers_.size());
        return true;
    }

    int width()  const override { return width_; }
    int height() const override { return height_; }
    int fps()    const override { return fps_; }

    const uint8_t* grab(int timeout_ms) override {
        if (fd_ < 0) return nullptr;

        // Ждём, пока в очереди не появится готовый кадр. POLLERR
        // тоже сообщает драйвер на disconnect камеры. EINTR (например,
        // SIGINT) намеренно НЕ ретраим — возвращаем nullptr, чтобы
        // вызывающий цикл сразу проверил флаг прерывания.
        pollfd pfd{};
        pfd.fd     = fd_;
        pfd.events = POLLIN;
        int pr = ::poll(&pfd, 1, timeout_ms);
        if (pr <= 0) return nullptr;
        if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
            std::fprintf(stderr, "Camera: poll() сообщил об ошибке устройства.\n");
            return nullptr;
        }
        if (!(pfd.revents & POLLIN)) return nullptr;

        v4l2_buffer buf{};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (xioctl(VIDIOC_DQBUF, &buf) < 0) {
            if (errno == EAGAIN) return nullptr;
            std::fprintf(stderr, "Camera: DQBUF упал: %s\n",
                         std::strerror(errno));
            return nullptr;
        }

        bool ok = (buf.index < buffers_.size());
        if (ok) {
            const MmapBuffer& mb = buffers_[buf.index];
            const uint8_t* src   = (const uint8_t*)mb.start;
            const size_t   len   = buf.bytesused
                                 ? (size_t)buf.bytesused
                                 : mb.length;

            switch (pix_fmt_) {
                case V4L2_PIX_FMT_YUYV:
                    yuyv_to_rgb(src, rgb_.data(), width_, height_);
                    break;
                case V4L2_PIX_FMT_MJPEG: {
                    int jw = 0, jh = 0, jc = 0;
                    uint8_t* px = stbi_load_from_memory(
                        src, (int)len, &jw, &jh, &jc, 3);
                    if (!px) {
                        std::fprintf(stderr,
                            "Camera: декодирование MJPEG упало: %s\n",
                            stbi_failure_reason());
                        ok = false;
                    } else {
                        // Если драйвер вдруг отдал кадр другого размера,
                        // лучше отбросить кадр чем тихо порисовать мусор.
                        if (jw != width_ || jh != height_) {
                            std::fprintf(stderr,
                                "Camera: MJPEG-кадр %dx%d != заявленных %dx%d\n",
                                jw, jh, width_, height_);
                            ok = false;
                        } else {
                            std::memcpy(rgb_.data(), px,
                                        (size_t)width_ * height_ * 3);
                        }
                        stbi_image_free(px);
                    }
                    break;
                }
                default:
                    ok = false;
            }
        }

        // Буфер должен вернуться в очередь даже при ошибке конвертации,
        // иначе через несколько кадров поток встанет.
        if (xioctl(VIDIOC_QBUF, &buf) < 0) {
            std::fprintf(stderr, "Camera: QBUF (return) упал: %s\n",
                         std::strerror(errno));
        }
        return ok ? rgb_.data() : nullptr;
    }

    void close() override {
        if (fd_ < 0) return;
        if (streaming_) {
            v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            ioctl(fd_, VIDIOC_STREAMOFF, &type);
            streaming_ = false;
        }
        for (auto& b : buffers_) {
            if (b.start) munmap(b.start, b.length);
        }
        buffers_.clear();
        ::close(fd_);
        fd_ = -1;
    }

private:
    // ioctl, повторяющий вызов при EINTR — стандартная V4L2-обёртка.
    int xioctl(int req, void* arg) {
        int r;
        do { r = ioctl(fd_, req, arg); }
        while (r < 0 && errno == EINTR);
        return r;
    }

    bool fail(const char* op) {
        std::fprintf(stderr, "Camera: %s упал: %s\n", op, std::strerror(errno));
        return false;
    }

    bool set_format(int w, int h, uint32_t fourcc) {
        v4l2_format fmt{};
        fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width       = w > 0 ? (uint32_t)w : 0;
        fmt.fmt.pix.height      = h > 0 ? (uint32_t)h : 0;
        fmt.fmt.pix.pixelformat = fourcc;
        fmt.fmt.pix.field       = V4L2_FIELD_NONE;
        if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) return false;
        // Драйвер мог проигнорировать запрос и оставить свой формат —
        // это для нас провал, иначе кадр потом будет интерпретирован
        // не тем декодером.
        if (fmt.fmt.pix.pixelformat != fourcc) return false;
        width_   = (int)fmt.fmt.pix.width;
        height_  = (int)fmt.fmt.pix.height;
        pix_fmt_ = fmt.fmt.pix.pixelformat;
        return width_ > 0 && height_ > 0;
    }

    std::string device_;
    int      fd_       = -1;
    int      width_    = 0;
    int      height_   = 0;
    int      fps_      = 0;
    uint32_t pix_fmt_  = 0;
    bool     streaming_ = false;
    std::vector<MmapBuffer> buffers_;
    std::vector<uint8_t>    rgb_;   // конечный буфер кадра в RGB888
};

}  // namespace

std::unique_ptr<Camera> make_camera() {
    return std::make_unique<V4L2Camera>();
}
