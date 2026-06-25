// Источник кадров из видеофайла через GStreamer — путь к АППАРАТНОМУ
// декодированию на устройстве (i.MX95 и прочие SoC с VPU).
//
// Зачем третья реализация, когда уже есть pipeline (внешний ffmpeg) и
// libav: на BSP от NXP именно GStreamer — родной мультимедийный стек, и
// только через него `decodebin` автоматически подхватывает аппаратный
// видеодекодер (v4l2-VPU: v4l2h264dec и т.п.) с zero-copy через DMABUF.
// ffmpeg/libav на том же железе обычно уезжают в софтовый декод на ядрах
// A55 и упираются в CPU раньше, чем в NPU. Здесь же тяжёлый H.264/H.265
// 1080p/4K снимается аппаратно, оставляя CPU под препроцессинг и NPU под
// инференс. На dev-хосте GStreamer тоже работает (софтовый decodebin),
// но смысла поверх libav не добавляет — поэтому реализация OFF по
// умолчанию и включается под BSP-сборку (USE_VIDEO_GSTREAMER).
//
// Конвейер: filesrc ! decodebin ! videoconvert ! videoscale !
//           video/x-raw,format=RGB ! appsink
// decodebin сам выбирает наилучший доступный декодер (в т.ч. аппаратный),
// videoconvert/videoscale приводят любой формат декодера к плотному RGB,
// appsink в pull-режиме отдаёт кадры синхронно нашему grab().
//
// ВАЖНО: конвейер строится вручную (а не через gst_parse_launch), потому
// что decodebin отдаёт ДИНАМИЧЕСКИЕ pad'ы, причём у файла с аудиодорожкой
// их несколько (видео + аудио). Если аудио-pad оставить неподключённым,
// демуксер блокируется на его незаполняемом выходе и тянет за собой весь
// конвейер (preroll не завершается, grab() висит вечно). Поэтому в
// обработчике pad-added видео-pad линкуется в videoconvert, а всё
// остальное (аудио/субтитры/лишние видеодорожки) — в fakesink, который их
// молча сливает, чтобы демуксер не вставал.
//
// Контракт grab() тот же, что у pipeline-/libav-реализаций и Camera::grab():
// один RGB888 HWC-кадр в нативном разрешении файла, указатель валиден до
// следующего grab()/close(). letterbox/RGB→luma делает общий C++-пайплайн.
//
// sync=false на appsink: кадры отдаются по мере декодирования, без привязки
// к часам конвейера — как и pipeline/libav (темп задаёт раннер, а не файл).

#include "video.h"

#include <cstdio>
#include <cstring>
#include <vector>

extern "C" {
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>
}

namespace {

// gst_init достаточно вызвать один раз на процесс; ленивая инициализация
// при первом создании источника.
void ensure_gst_init() {
    static bool inited = [] { gst_init(nullptr, nullptr); return true; }();
    (void)inited;
}

class GstVideo : public VideoSource {
public:
    ~GstVideo() override { close(); }

    bool open(const std::string& path, const std::string& /*ffmpeg*/,
              const std::string& /*ffprobe*/, bool loop) override {
        loop_ = loop;
        ensure_gst_init();

        pipeline_ = gst_pipeline_new("ii-video");
        GstElement* src   = gst_element_factory_make("filesrc",      "src");
        GstElement* dec   = gst_element_factory_make("decodebin",    "dec");
        conv_             = gst_element_factory_make("videoconvert", "conv");
        GstElement* scale = gst_element_factory_make("videoscale",   "scale");
        GstElement* capsf = gst_element_factory_make("capsfilter",   "capsf");
        sink_             = gst_element_factory_make("appsink",      "iisink");
        if (!pipeline_ || !src || !dec || !conv_ || !scale || !capsf || !sink_)
            return fail("не создать элементы конвейера (нет плагинов GStreamer?)");

        g_object_set(src, "location", path.c_str(), nullptr);

        // appsink принимает только плотный RGB; videoconvert/videoscale
        // догоняют до него любой формат декодера. Resize кадра не просим —
        // letterbox в нативном разрешении делает раннер.
        GstCaps* rgb = gst_caps_new_simple("video/x-raw",
                                           "format", G_TYPE_STRING, "RGB", nullptr);
        g_object_set(capsf, "caps", rgb, nullptr);
        gst_caps_unref(rgb);
        // max-buffers=2, drop=FALSE, sync=FALSE: тянем кадры по мере деко-
        // дирования, без привязки к часам конвейера и без лишней буферизации.
        g_object_set(sink_, "max-buffers", (guint)2, "drop", FALSE,
                     "sync", FALSE, nullptr);

        // Кладём все элементы в pipeline (bin берёт на себя владение
        // floating-ссылками — отдельно освобождать не нужно, всё снимет
        // gst_object_unref(pipeline_)).
        gst_bin_add_many(GST_BIN(pipeline_), src, dec, conv_, scale, capsf,
                         sink_, nullptr);
        // Статическая часть линкуется сразу. Стык decodebin -> videoconvert
        // динамический: decodebin создаёт pad'ы по ходу разбора контейнера,
        // поэтому линкуем его в обработчике pad-added (см. on_pad_added).
        if (!gst_element_link(src, dec))
            return fail("не слинковать filesrc -> decodebin");
        if (!gst_element_link_many(conv_, scale, capsf, sink_, nullptr))
            return fail("не слинковать videoconvert -> appsink");
        g_signal_connect(dec, "pad-added", G_CALLBACK(&GstVideo::on_pad_added),
                         this);

        if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) ==
            GST_STATE_CHANGE_FAILURE) {
            drain_bus_errors();
            return fail("не запустить конвейер (PLAYING)");
        }

        // Первый кадр тянем уже здесь: из его caps берём width/height/fps
        // (нужны до первого grab()) и заодно убеждаемся, что декодер найден.
        // Сам кадр не теряем — откладываем как pending для первого grab().
        GstSample* s = gst_app_sink_pull_sample(GST_APP_SINK(sink_));
        if (!s) {
            drain_bus_errors();
            return fail("не получить первый кадр (битый файл / нет декодера?)");
        }
        bool ok = convert_sample(s);
        gst_sample_unref(s);
        if (!ok) return fail("не сконвертировать первый кадр в RGB");
        have_pending_ = true;

        std::printf("Video: %s  %dx%d @ %d fps%s (gstreamer, decodebin -> rgb24)\n",
                    path.c_str(), width_, height_, fps_,
                    loop_ ? ", loop" : "");
        return true;
    }

    int width()  const override { return width_; }
    int height() const override { return height_; }
    int fps()    const override { return fps_; }

    const uint8_t* grab() override {
        if (eof_) return nullptr;
        // Отложенный в open() первый кадр отдаём без нового pull.
        if (have_pending_) { have_pending_ = false; return rgb_.data(); }

        for (;;) {
            GstSample* s = gst_app_sink_pull_sample(GST_APP_SINK(sink_));
            if (s) {
                bool ok = convert_sample(s);
                gst_sample_unref(s);
                if (ok) return rgb_.data();
                eof_ = true;          // не смогли сконвертировать — стоп
                return nullptr;
            }
            // NULL: конец потока либо ошибка/останов.
            if (loop_ && gst_app_sink_is_eos(GST_APP_SINK(sink_)) &&
                seek_start())
                continue;             // зацикливание: перемотали, тянем дальше
            eof_ = true;
            return nullptr;
        }
    }

    bool eof() const override { return eof_; }

    void close() override {
        if (!pipeline_) return;
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);   // снимает и все дочерние элементы
        pipeline_ = nullptr;
        conv_ = sink_ = nullptr;       // принадлежали pipeline_, уже сняты
    }

private:
    // decodebin отдал новый декодированный pad. Видео — в videoconvert (один,
    // первый), всё прочее (аудио/субтитры/лишнее видео) — в fakesink, чтобы
    // демуксер не блокировался на неподключённом выходе.
    static void on_pad_added(GstElement* /*dec*/, GstPad* pad, gpointer user) {
        auto* self = static_cast<GstVideo*>(user);

        GstCaps* caps = gst_pad_get_current_caps(pad);
        if (!caps) caps = gst_pad_query_caps(pad, nullptr);
        const char* name = caps ? gst_structure_get_name(
                                      gst_caps_get_structure(caps, 0))
                                 : "";
        const bool is_video = name && g_str_has_prefix(name, "video/");
        if (caps) gst_caps_unref(caps);

        // Первую видеодорожку — в videoconvert, вторую (если есть) и не-видео
        // pad'ы — в fakesink.
        GstPad* sinkpad = is_video
            ? gst_element_get_static_pad(self->conv_, "sink") : nullptr;
        if (sinkpad && !gst_pad_is_linked(sinkpad))
            gst_pad_link(pad, sinkpad);
        else
            self->attach_fakesink(pad);
        if (sinkpad) gst_object_unref(sinkpad);
    }

    // Подключить pad к свежему fakesink (sync=false/async=false — просто
    // сливает буферы, не участвуя в preroll/часах), чтобы поток не вставал.
    void attach_fakesink(GstPad* pad) {
        GstElement* fs = gst_element_factory_make("fakesink", nullptr);
        if (!fs) return;
        g_object_set(fs, "sync", FALSE, "async", FALSE, nullptr);
        gst_bin_add(GST_BIN(pipeline_), fs);
        gst_element_sync_state_with_parent(fs);
        GstPad* sp = gst_element_get_static_pad(fs, "sink");
        gst_pad_link(pad, sp);
        gst_object_unref(sp);
    }

    // Скопировать кадр из GstSample в плотный RGB888-буфер rgb_.
    //
    // Размеры/fps и, главное, РЕАЛЬНЫЙ stride строки берём из GstVideoInfo:
    // gst_video_frame_map() читает выравнивание, заданное декодером
    // (на VPU оно бывает 16/32/64 байта, а не дефолтные 4), поэтому гадать
    // про padding не нужно — копируем построчно по фактическому src_stride.
    bool convert_sample(GstSample* s) {
        GstCaps* caps = gst_sample_get_caps(s);
        GstVideoInfo info;
        if (!caps || !gst_video_info_from_caps(&info, caps)) return false;

        width_  = GST_VIDEO_INFO_WIDTH(&info);
        height_ = GST_VIDEO_INFO_HEIGHT(&info);
        const int fn = GST_VIDEO_INFO_FPS_N(&info);
        const int fd = GST_VIDEO_INFO_FPS_D(&info);
        if (fd > 0) fps_ = (fn + fd / 2) / fd;   // округление к ближайшему
        if (width_ <= 0 || height_ <= 0) return false;

        GstBuffer* buf = gst_sample_get_buffer(s);
        if (!buf) return false;

        GstVideoFrame frame;
        if (!gst_video_frame_map(&frame, &info, buf, GST_MAP_READ)) return false;
        const uint8_t* src =
            static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 0));
        const size_t src_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 0);
        const size_t dst_stride = (size_t)width_ * 3;

        if (rgb_.size() != dst_stride * (size_t)height_)
            rgb_.resize(dst_stride * (size_t)height_);

        if (src_stride == dst_stride) {
            std::memcpy(rgb_.data(), src, dst_stride * (size_t)height_);
        } else {                                 // строки с padding'ом
            for (int y = 0; y < height_; ++y)
                std::memcpy(rgb_.data() + (size_t)y * dst_stride,
                            src + (size_t)y * src_stride, dst_stride);
        }
        gst_video_frame_unmap(&frame);
        return true;
    }

    // Перемотка в начало для --video-loop: flush-seek сбрасывает EOS appsink
    // и возобновляет поток кадров.
    bool seek_start() {
        return gst_element_seek_simple(
            pipeline_, GST_FORMAT_TIME,
            (GstSeekFlags)(GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_KEY_UNIT), 0);
    }

    // Высыпать накопившиеся ERROR-сообщения с шины в stderr (диагностика
    // неудачного open / отсутствующих плагинов декодера).
    void drain_bus_errors() {
        GstBus* bus = gst_element_get_bus(pipeline_);
        if (!bus) return;
        GstMessage* msg;
        while ((msg = gst_bus_pop_filtered(bus, GST_MESSAGE_ERROR)) != nullptr) {
            GError* e = nullptr; gchar* dbg = nullptr;
            gst_message_parse_error(msg, &e, &dbg);
            std::fprintf(stderr, "Video(gstreamer): %s\n",
                         e ? e->message : "ошибка конвейера");
            if (e) g_error_free(e);
            if (dbg) g_free(dbg);
            gst_message_unref(msg);
        }
        gst_object_unref(bus);
    }

    bool fail(const char* what) {
        std::fprintf(stderr, "Video(gstreamer): %s\n", what);
        return false;
    }

    GstElement* pipeline_ = nullptr;
    GstElement* conv_     = nullptr;  // videoconvert: цель линковки видео-pad'а
    GstElement* sink_     = nullptr;
    int  width_  = 0;
    int  height_ = 0;
    int  fps_    = 0;
    bool loop_         = false;
    bool eof_          = false;
    bool have_pending_ = false;  // первый кадр уже лежит в rgb_ (из open)
    std::vector<uint8_t> rgb_;   // буфер текущего кадра в RGB888
};

}  // namespace

std::unique_ptr<VideoSource> make_gstreamer_video() {
    return std::make_unique<GstVideo>();
}
