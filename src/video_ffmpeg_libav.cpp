// Источник кадров из видеофайла через БИБЛИОТЕКУ libav* (FFmpeg) —
// полноценный декодер без внешнего процесса.
//
// В отличие от video_ffmpeg_pipeline.cpp (внешний ffmpeg + pipe) здесь
// файл декодируется прямо в процессе раннера: libavformat демультиплексирует
// контейнер, libavcodec декодирует видеопоток, libswscale конвертирует
// кадр в RGB24. Плюсы: нет накладных расходов на процесс и парсинг stdout,
// точные width/height/fps/длительность из контейнера, аккуратный seek для
// --video-loop средствами библиотеки. Цена — линк-зависимость от FFmpeg
// dev-библиотек (USE_VIDEO_LIBAV + их поиск в CMake).
//
// Контракт grab() тот же, что у pipeline-реализации и у Camera::grab():
// один RGB888 HWC-кадр в нативном разрешении файла, указатель валиден до
// следующего grab()/close(). letterbox/RGB→luma делает общий C++-пайплайн.
//
// Модель декодирования libavcodec — send/receive: одному пакету может
// соответствовать 0..N кадров, поэтому grab() крутит цикл «достать кадр из
// декодера → не хватило, скормить ещё пакет», а на конце файла переходит
// в режим дренажа (send NULL), добирая буферизованные кадры.

#include "video.h"

#include <cstdio>
#include <vector>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

namespace {

class LibavVideo : public VideoSource {
public:
    ~LibavVideo() override { close(); }

    bool open(const std::string& path, const std::string& /*ffmpeg*/,
              const std::string& /*ffprobe*/, bool loop) override {
        loop_ = loop;

        if (avformat_open_input(&fmt_, path.c_str(), nullptr, nullptr) < 0) {
            std::fprintf(stderr, "Video(libav): не открыть %s\n", path.c_str());
            return false;
        }
        if (avformat_find_stream_info(fmt_, nullptr) < 0)
            return fail("avformat_find_stream_info");

        // Лучшая видеодорожка + её декодер.
        const AVCodec* dec = nullptr;
        stream_idx_ = av_find_best_stream(fmt_, AVMEDIA_TYPE_VIDEO,
                                          -1, -1, &dec, 0);
        if (stream_idx_ < 0 || !dec)
            return fail("видеодорожка не найдена");

        AVStream* st = fmt_->streams[stream_idx_];
        dec_ = avcodec_alloc_context3(dec);
        if (!dec_) return fail("avcodec_alloc_context3");
        if (avcodec_parameters_to_context(dec_, st->codecpar) < 0)
            return fail("avcodec_parameters_to_context");
        dec_->thread_count = 0;   // 0 = libav сам выберет число потоков
        if (avcodec_open2(dec_, dec, nullptr) < 0)
            return fail("avcodec_open2");

        width_  = dec_->width;
        height_ = dec_->height;
        if (width_ <= 0 || height_ <= 0)
            return fail("некорректный размер кадра");

        // fps: avg_frame_rate надёжнее для VFR; запасной — r_frame_rate.
        AVRational fr = st->avg_frame_rate;
        if (fr.num <= 0 || fr.den <= 0) fr = st->r_frame_rate;
        fps_ = (fr.num > 0 && fr.den > 0) ? (int)(av_q2d(fr) + 0.5) : 0;

        // Конвертер пиксельный формат декодера -> RGB24, размер не меняем.
        sws_ = sws_getContext(width_, height_, dec_->pix_fmt,
                              width_, height_, AV_PIX_FMT_RGB24,
                              SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!sws_) return fail("sws_getContext");

        frame_  = av_frame_alloc();
        packet_ = av_packet_alloc();
        if (!frame_ || !packet_) return fail("av_frame_alloc/av_packet_alloc");

        rgb_.assign((size_t)width_ * height_ * 3, 0);

        std::printf("Video: %s  %dx%d @ %d fps%s (libav, %s -> rgb24)\n",
                    path.c_str(), width_, height_, fps_,
                    loop_ ? ", loop" : "", dec->name);
        return true;
    }

    int width()  const override { return width_; }
    int height() const override { return height_; }
    int fps()    const override { return fps_; }

    const uint8_t* grab() override {
        if (!fmt_ || eof_) return nullptr;
        for (;;) {
            // 1) Попытаться достать готовый кадр из декодера.
            int r = avcodec_receive_frame(dec_, frame_);
            if (r == 0) return convert_current();
            if (r == AVERROR_EOF) { eof_ = true; return nullptr; }
            if (r != AVERROR(EAGAIN)) { log_err("avcodec_receive_frame", r);
                                        return nullptr; }

            // 2) Декодеру не хватает данных. Если уже дренируем — кадров
            //    больше не будет (хвост выберется на шаге 1, затем EOF).
            if (draining_) { eof_ = true; return nullptr; }

            // 3) Прочитать очередной пакет из контейнера.
            int rr = av_read_frame(fmt_, packet_);
            if (rr == AVERROR_EOF) {
                if (loop_ && seek_start()) continue;     // зацикливание
                avcodec_send_packet(dec_, nullptr);      // начать дренаж
                draining_ = true;
                continue;
            }
            if (rr < 0) { log_err("av_read_frame", rr); return nullptr; }

            // 4) Скормить пакет нашего видеопотока декодеру.
            if (packet_->stream_index == stream_idx_) {
                int sr = avcodec_send_packet(dec_, packet_);
                if (sr < 0 && sr != AVERROR(EAGAIN)) {
                    av_packet_unref(packet_);
                    log_err("avcodec_send_packet", sr);
                    return nullptr;
                }
            }
            av_packet_unref(packet_);
        }
    }

    bool eof() const override { return eof_; }

    void close() override {
        if (sws_)    { sws_freeContext(sws_); sws_ = nullptr; }
        if (frame_)  av_frame_free(&frame_);
        if (packet_) av_packet_free(&packet_);
        if (dec_)    avcodec_free_context(&dec_);
        if (fmt_)    avformat_close_input(&fmt_);
    }

private:
    // Сконвертировать текущий frame_ в rgb_ и подготовить кадр к переиспользованию.
    const uint8_t* convert_current() {
        uint8_t* dst[4]  = { rgb_.data(), nullptr, nullptr, nullptr };
        int      dls[4]  = { width_ * 3, 0, 0, 0 };
        sws_scale(sws_, frame_->data, frame_->linesize, 0, height_, dst, dls);
        av_frame_unref(frame_);
        return rgb_.data();
    }

    // Перемотать в начало для --video-loop и сбросить состояние декодера.
    bool seek_start() {
        if (av_seek_frame(fmt_, stream_idx_, 0, AVSEEK_FLAG_BACKWARD) < 0)
            return false;
        avcodec_flush_buffers(dec_);
        return true;
    }

    void log_err(const char* what, int err) {
        char buf[AV_ERROR_MAX_STRING_SIZE] = {0};
        av_strerror(err, buf, sizeof(buf));
        std::fprintf(stderr, "Video(libav): %s: %s\n", what, buf);
    }

    bool fail(const char* what) {
        std::fprintf(stderr, "Video(libav): %s\n", what);
        return false;
    }

    AVFormatContext* fmt_    = nullptr;
    AVCodecContext*  dec_    = nullptr;
    SwsContext*      sws_    = nullptr;
    AVFrame*         frame_  = nullptr;
    AVPacket*        packet_ = nullptr;
    int  stream_idx_ = -1;
    int  width_  = 0;
    int  height_ = 0;
    int  fps_    = 0;
    bool loop_     = false;
    bool eof_      = false;
    bool draining_ = false;     // дренаж декодера после конца файла
    std::vector<uint8_t> rgb_;  // буфер текущего кадра в RGB888
};

}  // namespace

std::unique_ptr<VideoSource> make_ffmpeg_libav_video() {
    return std::make_unique<LibavVideo>();
}
