// Media Foundation источник видеокадров для раннера (Windows).
//
// Аналог camera_v4l2.cpp, но для Windows. Покрывает типовой сценарий:
// USB UVC-камера, экспонируемая системой как видеоустройство захвата.
// Используем самый простой рабочий путь через IMFSourceReader:
//
//   * MFEnumDeviceSources перечисляет камеры; выбираем по индексу
//     (device-строка "/dev/video0" / "0" / "1" → хвостовое число, либо 0);
//   * source reader создаётся с MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING,
//     поэтому мы просто просим на ВЫХОДЕ RGB32, а конвертацию из родного
//     формата сенсора (YUY2 / NV12 / MJPG) делает встроенный Video
//     Processor MFT — не нужно тащить декодеры руками;
//   * на каждый кадр: ReadSample (синхронно) → ConvertToContiguousBuffer →
//     Lock → BGRA→RGB888 в наш буфер (выделенный один раз в open()) → Unlock.
//
// RGB32 в Media Foundation хранится как BGRA (B,G,R,X) по 4 байта на
// пиксель; ориентация задаётся знаком MF_MT_DEFAULT_STRIDE (отрицательный
// stride → строки снизу вверх). Обе ветки учтены в convert_bgra_to_rgb().
//
// Таймаут grab() для синхронного reader'а не поддерживается на уровне API
// (ReadSample блокируется до готовности кадра); на практике кадр приходит
// заметно быстрее секунды, а вызывающий цикл всё равно трактует nullptr
// как «пропусти итерацию», так что параметр timeout_ms игнорируется.

#include "camera.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>

namespace {

template <class T>
void safe_release(T*& p) {
    if (p) { p->Release(); p = nullptr; }
}

// "/dev/video0" / "video1" / "2" / "" → хвостовой индекс (по умолчанию 0).
int device_index(const std::string& dev) {
    int idx = 0, scale = 1;
    bool any = false;
    for (auto it = dev.rbegin(); it != dev.rend(); ++it) {
        if (*it < '0' || *it > '9') break;
        idx += (*it - '0') * scale;
        scale *= 10;
        any = true;
    }
    return any ? idx : 0;
}

class MFCamera : public Camera {
public:
    ~MFCamera() override { close(); }

    bool open(const std::string& device, int w, int h, int fps) override {
        const int want_idx = device_index(device);

        // COM + Media Foundation. CoInitializeEx может вернуть
        // RPC_E_CHANGED_MODE, если поток уже в другой apartment-модели —
        // это не фатально, просто не нам его деинициализировать.
        HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
        com_inited_ = SUCCEEDED(hr);
        hr = MFStartup(MF_VERSION, MFSTARTUP_LITE);
        if (FAILED(hr)) {
            std::fprintf(stderr, "Camera: MFStartup упал (0x%08lx)\n",
                         (unsigned long)hr);
            return false;
        }
        mf_inited_ = true;

        // Перечисляем видеоустройства захвата и активируем нужное.
        IMFMediaSource* source = activate_device(want_idx);
        if (!source) return false;

        IMFAttributes* ra = nullptr;
        MFCreateAttributes(&ra, 1);
        ra->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE);
        hr = MFCreateSourceReaderFromMediaSource(source, ra, &reader_);
        safe_release(ra);
        safe_release(source);
        if (FAILED(hr) || !reader_) {
            std::fprintf(stderr,
                "Camera: не создать SourceReader (0x%08lx)\n",
                (unsigned long)hr);
            return false;
        }

        // Best-effort: выбрать родной media type с запрошенным разрешением
        // (и заодно прихватить fps). Если не нашли — оставляем как есть,
        // конвертацию/масштаб сделает Video Processor.
        if (w > 0 && h > 0) select_native_size(w, h);
        if (fps > 0) (void)fps;  // запрос fps в MF делается через native type;
                                 // оставляем выбор драйверу для простоты.

        // Просим RGB32 на выходе — конвертацию вставит Video Processor MFT.
        IMFMediaType* out = nullptr;
        MFCreateMediaType(&out);
        out->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
        out->SetGUID(MF_MT_SUBTYPE,    MFVideoFormat_RGB32);
        hr = reader_->SetCurrentMediaType(
            MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, out);
        safe_release(out);
        if (FAILED(hr)) {
            std::fprintf(stderr,
                "Camera: камера не отдаёт RGB32 (0x%08lx)\n",
                (unsigned long)hr);
            return false;
        }

        // Фактические параметры берём из согласованного выходного типа.
        IMFMediaType* cur = nullptr;
        hr = reader_->GetCurrentMediaType(
            MF_SOURCE_READER_FIRST_VIDEO_STREAM, &cur);
        if (FAILED(hr) || !cur) {
            std::fprintf(stderr, "Camera: не прочитать выходной media type\n");
            return false;
        }
        UINT32 cw = 0, ch = 0;
        MFGetAttributeSize(cur, MF_MT_FRAME_SIZE, &cw, &ch);
        width_  = (int)cw;
        height_ = (int)ch;
        UINT32 stride = 0;
        if (SUCCEEDED(cur->GetUINT32(MF_MT_DEFAULT_STRIDE, &stride)))
            stride_ = (long)(int32_t)stride;   // знак важен (bottom-up)
        else
            stride_ = (long)width_ * 4;
        UINT32 num = 0, den = 0;
        if (SUCCEEDED(MFGetAttributeRatio(cur, MF_MT_FRAME_RATE, &num, &den))
            && den > 0)
            fps_ = (int)(num / den);
        safe_release(cur);

        if (width_ <= 0 || height_ <= 0) {
            std::fprintf(stderr, "Camera: некорректный размер кадра %dx%d\n",
                         width_, height_);
            return false;
        }

        rgb_.assign((size_t)width_ * height_ * 3, 0);
        std::printf("Camera: \"%s\" (#%d) %dx%d @ %d fps [Media Foundation, "
                    "RGB32→RGB888]\n",
                    name_.c_str(), want_idx, width_, height_, fps_);
        return true;
    }

    int width()  const override { return width_; }
    int height() const override { return height_; }
    int fps()    const override { return fps_; }

    const uint8_t* grab(int /*timeout_ms*/) override {
        if (!reader_) return nullptr;

        DWORD     stream_idx = 0, flags = 0;
        LONGLONG  ts = 0;
        IMFSample* sample = nullptr;
        HRESULT hr = reader_->ReadSample(
            MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0,
            &stream_idx, &flags, &ts, &sample);
        if (FAILED(hr)) {
            std::fprintf(stderr, "Camera: ReadSample упал (0x%08lx)\n",
                         (unsigned long)hr);
            return nullptr;
        }
        if (flags & MF_SOURCE_READERF_ENDOFSTREAM) {
            safe_release(sample);
            return nullptr;
        }
        // Драйвер мог пересогласовать формат на лету — обновим размеры.
        if (flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED)
            refresh_format();
        if (!sample) return nullptr;   // stream tick / нет кадра — пропускаем

        IMFMediaBuffer* buf = nullptr;
        hr = sample->ConvertToContiguousBuffer(&buf);
        bool ok = SUCCEEDED(hr) && buf;
        if (ok) {
            BYTE* data = nullptr;
            DWORD maxlen = 0, curlen = 0;
            if (SUCCEEDED(buf->Lock(&data, &maxlen, &curlen))) {
                convert_bgra_to_rgb(data);
                buf->Unlock();
            } else {
                ok = false;
            }
        }
        safe_release(buf);
        safe_release(sample);
        return ok ? rgb_.data() : nullptr;
    }

    void close() override {
        safe_release(reader_);
        if (mf_inited_)  { MFShutdown(); mf_inited_ = false; }
        if (com_inited_) { CoUninitialize(); com_inited_ = false; }
    }

private:
    // Активировать камеру по индексу; заполнить name_ дружелюбным именем.
    IMFMediaSource* activate_device(int want_idx) {
        IMFAttributes* attrs = nullptr;
        MFCreateAttributes(&attrs, 1);
        attrs->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                       MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);

        IMFActivate** devices = nullptr;
        UINT32 count = 0;
        HRESULT hr = MFEnumDeviceSources(attrs, &devices, &count);
        safe_release(attrs);
        if (FAILED(hr)) {
            std::fprintf(stderr, "Camera: MFEnumDeviceSources упал\n");
            return nullptr;
        }
        if (count == 0) {
            std::fprintf(stderr, "Camera: камеры не найдены\n");
            if (devices) CoTaskMemFree(devices);
            return nullptr;
        }
        if (want_idx < 0 || (UINT32)want_idx >= count) {
            std::fprintf(stderr,
                "Camera: индекс %d вне диапазона (найдено %u камер), "
                "беру #0\n", want_idx, count);
            want_idx = 0;
        }

        // Дружелюбное имя — для лога.
        WCHAR* wname = nullptr;
        UINT32 wlen = 0;
        if (SUCCEEDED(devices[want_idx]->GetAllocatedString(
                MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &wname, &wlen))) {
            char buf[256];
            int n = WideCharToMultiByte(CP_UTF8, 0, wname, (int)wlen,
                                        buf, sizeof(buf) - 1, nullptr, nullptr);
            if (n > 0) { buf[n] = 0; name_ = buf; }
            CoTaskMemFree(wname);
        }

        IMFMediaSource* source = nullptr;
        hr = devices[want_idx]->ActivateObject(
            IID_PPV_ARGS(&source));
        for (UINT32 i = 0; i < count; ++i) safe_release(devices[i]);
        CoTaskMemFree(devices);
        if (FAILED(hr) || !source) {
            std::fprintf(stderr, "Camera: ActivateObject упал (0x%08lx)\n",
                         (unsigned long)hr);
            return nullptr;
        }
        return source;
    }

    // Выбрать родной media type стрима с разрешением w×h, если он есть.
    void select_native_size(int w, int h) {
        for (DWORD i = 0;; ++i) {
            IMFMediaType* nt = nullptr;
            HRESULT hr = reader_->GetNativeMediaType(
                MF_SOURCE_READER_FIRST_VIDEO_STREAM, i, &nt);
            if (FAILED(hr)) break;   // больше типов нет
            UINT32 nw = 0, nh = 0;
            MFGetAttributeSize(nt, MF_MT_FRAME_SIZE, &nw, &nh);
            if ((int)nw == w && (int)nh == h) {
                reader_->SetCurrentMediaType(
                    MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, nt);
                safe_release(nt);
                return;
            }
            safe_release(nt);
        }
    }

    // Перечитать размер/stride после смены формата на лету.
    void refresh_format() {
        IMFMediaType* cur = nullptr;
        if (FAILED(reader_->GetCurrentMediaType(
                MF_SOURCE_READER_FIRST_VIDEO_STREAM, &cur)) || !cur)
            return;
        UINT32 cw = 0, ch = 0;
        MFGetAttributeSize(cur, MF_MT_FRAME_SIZE, &cw, &ch);
        UINT32 stride = 0;
        long new_stride = SUCCEEDED(cur->GetUINT32(MF_MT_DEFAULT_STRIDE, &stride))
                              ? (long)(int32_t)stride : (long)cw * 4;
        safe_release(cur);
        if ((int)cw != width_ || (int)ch != height_) {
            width_  = (int)cw;
            height_ = (int)ch;
            rgb_.assign((size_t)width_ * height_ * 3, 0);
        }
        stride_ = new_stride;
    }

    // BGRA (RGB32) → RGB888 с учётом знака stride (bottom-up при stride<0).
    void convert_bgra_to_rgb(const uint8_t* data) {
        const long abs_s = stride_ < 0 ? -stride_ : stride_;
        for (int y = 0; y < height_; ++y) {
            const uint8_t* row = stride_ < 0
                ? data + (size_t)(height_ - 1 - y) * abs_s
                : data + (size_t)y * abs_s;
            uint8_t* drow = rgb_.data() + (size_t)y * width_ * 3;
            for (int x = 0; x < width_; ++x) {
                drow[x * 3 + 0] = row[x * 4 + 2];  // R
                drow[x * 3 + 1] = row[x * 4 + 1];  // G
                drow[x * 3 + 2] = row[x * 4 + 0];  // B
            }
        }
    }

    IMFSourceReader* reader_ = nullptr;
    bool        com_inited_  = false;
    bool        mf_inited_   = false;
    int         width_       = 0;
    int         height_      = 0;
    int         fps_         = 0;
    long        stride_      = 0;     // байт на строку; знак = ориентация
    std::string name_        = "camera";
    std::vector<uint8_t> rgb_;        // конечный кадр в RGB888
};

}  // namespace

std::unique_ptr<Camera> make_camera() {
    return std::make_unique<MFCamera>();
}
