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
}

namespace {

// gst_init достаточно вызвать один раз на процесс; ленивая инициализация
// при первом создании источника.
void ensure_gst_init() {
    static bool inited = [] { gst_init(nullptr, nullptr); return true; }();
    (void)inited;
}

// Экранирование пути для строки gst_parse_launch: значение свойства
// location берём в кавычки, а спецсимволы (\ и ") внутри — гасим обратным
// слэшем. Заодно корректно проходят windows-пути C:\... и пробелы/кириллица.
std::string gst_escape(const std::string& s) {
    std::string r;
    r.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '\\' || c == '"') r += '\\';
        r += c;
    }
    return r;
}

class GstVideo : public VideoSource {
public:
    ~GstVideo() override { close(); }

    bool open(const std::string& path, const std::string& /*ffmpeg*/,
              const std::string& /*ffprobe*/, bool loop) override {
        loop_ = loop;
        ensure_gst_init();

        // decodebin автоплугинит аппаратный декодер, если он есть в системе;
        // videoconvert+videoscale гарантируют плотный RGB на appsink.
        const std::string desc =
            "filesrc location=\"" + gst_escape(path) + "\" ! "
            "decodebin ! videoconvert ! videoscale ! "
            "video/x-raw,format=RGB ! "
            "appsink name=iisink max-buffers=2 drop=false sync=false";

        GError* err = nullptr;
        pipeline_ = gst_parse_launch(desc.c_str(), &err);
        if (!pipeline_) {
            std::fprintf(stderr, "Video(gstreamer): gst_parse_launch: %s\n",
                         err ? err->message : "не построить конвейер");
            if (err) g_error_free(err);
            return false;
        }
        if (err) g_error_free(err);   // ненулевой err при ненулевом pipeline_ = предупреждение

        sink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "iisink");
        if (!sink_) return fail("appsink не найден в конвейере");

        if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) ==
            GST_STATE_CHANGE_FAILURE) {
            drain_bus_errors();
            return fail("не запустить конвейер (PLAYING)");
        }

        // Первый кадр забираем уже здесь: из его caps узнаём width/height/fps
        // (нужны до первого grab()), а сам кадр откладываем как pending, чтобы
        // не потерять его для раннера.
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
        if (pipeline_) {
            gst_element_set_state(pipeline_, GST_STATE_NULL);
            if (sink_) { gst_object_unref(sink_); sink_ = nullptr; }
            gst_object_unref(pipeline_);
            pipeline_ = nullptr;
        }
    }

private:
    // Прочитать параметры из caps и скопировать буфер кадра в rgb_ плотно.
    bool convert_sample(GstSample* s) {
        GstCaps* caps = gst_sample_get_caps(s);
        if (caps) {
            GstStructure* st = gst_caps_get_structure(caps, 0);
            int w = 0, h = 0;
            gst_structure_get_int(st, "width", &w);
            gst_structure_get_int(st, "height", &h);
            if (w > 0 && h > 0) { width_ = w; height_ = h; }
            int fn = 0, fd = 0;
            if (gst_structure_get_fraction(st, "framerate", &fn, &fd) && fd > 0)
                fps_ = (int)((double)fn / fd + 0.5);
        }
        if (width_ <= 0 || height_ <= 0) return false;

        GstBuffer* buf = gst_sample_get_buffer(s);
        if (!buf) return false;
        GstMapInfo map;
        if (!gst_buffer_map(buf, &map, GST_MAP_READ)) return false;

        const size_t dst_stride = (size_t)width_ * 3;
        // GStreamer по умолчанию выравнивает строки RGB на 4 байта, поэтому
        // src-строка может быть шире dst — копируем построчно, отбрасывая
        // padding. Если выравнивания нет (width*3 кратно 4) — один memcpy.
        const size_t src_stride = (dst_stride + 3) & ~(size_t)3;
        const size_t need = dst_stride * (size_t)height_;
        if (rgb_.size() != need) rgb_.resize(need);

        if (src_stride != dst_stride && map.size >= src_stride * (size_t)height_) {
            for (int y = 0; y < height_; ++y)
                std::memcpy(rgb_.data() + (size_t)y * dst_stride,
                            map.data + (size_t)y * src_stride, dst_stride);
        } else {
            const size_t n = map.size < need ? (size_t)map.size : need;
            std::memcpy(rgb_.data(), map.data, n);
        }
        gst_buffer_unmap(buf, &map);
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
