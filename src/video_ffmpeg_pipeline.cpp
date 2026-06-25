// Источник кадров из видеофайла через ВНЕШНИЙ процесс ffmpeg + pipe.
//
// Самый «лёгкий» способ подключить видеофайлы: не линковать libav, а
// запустить системный ffmpeg дочерним процессом и читать декодированные
// кадры из его stdout. Линк-зависимостей в сборке нет вовсе — нужен лишь
// бинарь ffmpeg (и ffprobe для определения размера) в рантайме. Тот же
// приём используется в прототипе infer.py.
//
// Поток данных:
//   1) ffprobe выдаёт width,height,r_frame_rate первой видеодорожки —
//      так мы узнаём, сколько байт занимает один rgb24-кадр (W*H*3);
//   2) ffmpeg -f rawvideo -pix_fmt rgb24 -  пишет сырые кадры в pipe в
//      НАТИВНОМ разрешении файла. letterbox / RGB→luma делает уже общий
//      C++-пайплайн раннера (как для камеры), поэтому здесь ничего не
//      масштабируем — отдаём кадр как есть, и весь видео-цикл (YOLO,
//      tile, show-output) работает без изменений;
//   3) grab() читает ровно W*H*3 байт; неполное чтение = конец файла.
//
// Кроссплатформенно за счёт popen/_popen — отдельных реализаций под
// Linux/Windows не требуется (в отличие от camera.*, где захват завязан
// на ОС-API). Будущая libav-реализация встанет рядом отдельным файлом.

#include "video.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// popen/pclose называются по-разному на Windows; режим "b" обязателен,
// иначе на Windows произойдёт CRLF-трансляция и сырые пиксели побьются.
#ifdef _WIN32
  #define II_POPEN  _popen
  #define II_PCLOSE _pclose
  static const char* kPopenMode = "rb";
#else
  #define II_POPEN  popen
  #define II_PCLOSE pclose
  static const char* kPopenMode = "r";
#endif

// Прочитать ровно n байт из pipe. fread на pipe вправе вернуть меньше
// запрошенного (граница кадра, планировщик), поэтому дочитываем в цикле.
// Возвращает фактически прочитанное число байт (< n означает EOF/ошибку).
std::size_t read_full(FILE* f, uint8_t* buf, std::size_t n) {
    std::size_t got = 0;
    while (got < n) {
        std::size_t r = std::fread(buf + got, 1, n - got, f);
        if (r == 0) break;          // EOF или ошибка чтения
        got += r;
    }
    return got;
}

class FfmpegPipeVideo : public VideoSource {
public:
    ~FfmpegPipeVideo() override { close(); }

    bool open(const std::string& path, const std::string& ffmpeg,
              const std::string& ffprobe, bool loop) override {
        loop_ = loop;

        if (!probe(path, ffprobe) || width_ <= 0 || height_ <= 0) {
            std::fprintf(stderr,
                "Video: не удалось определить параметры файла %s через '%s' "
                "(нет файла / ffprobe не в PATH?).\n",
                path.c_str(), ffprobe.c_str());
            return false;
        }
        frame_bytes_ = (std::size_t)width_ * height_ * 3;
        rgb_.assign(frame_bytes_, 0);

        // ffmpeg -> сырой rgb24 в stdout, нативное разрешение файла.
        // -stream_loop -1 (до -i) бесконечно повторяет ввод для --video-loop.
        std::string cmd = quote(ffmpeg);
        if (loop_) cmd += " -stream_loop -1";
        cmd += " -i " + quote(path)
             + " -f rawvideo -pix_fmt rgb24 -loglevel quiet -";
        pipe_ = II_POPEN(cmd.c_str(), kPopenMode);
        if (!pipe_) {
            std::fprintf(stderr, "Video: не запустить ffmpeg: %s\n",
                         cmd.c_str());
            return false;
        }

        std::printf("Video: %s  %dx%d @ %d fps%s (ffmpeg pipe, rgb24)\n",
                    path.c_str(), width_, height_, fps_,
                    loop_ ? ", loop" : "");
        return true;
    }

    int width()  const override { return width_; }
    int height() const override { return height_; }
    int fps()    const override { return fps_; }

    const uint8_t* grab() override {
        if (!pipe_ || eof_) return nullptr;
        const std::size_t got = read_full(pipe_, rgb_.data(), frame_bytes_);
        if (got != frame_bytes_) {
            // Неполный кадр — ffmpeg завершился, файл кончился.
            eof_ = true;
            return nullptr;
        }
        return rgb_.data();
    }

    bool eof() const override { return eof_; }

    void close() override {
        if (pipe_) {
            II_PCLOSE(pipe_);   // дождётся и завершит дочерний ffmpeg
            pipe_ = nullptr;
        }
    }

private:
    // Обернуть аргумент в кавычки — пути с пробелами / кириллицей.
    static std::string quote(const std::string& s) {
        return "\"" + s + "\"";
    }

    // ffprobe: width,height,r_frame_rate первой видеодорожки одной строкой.
    // Пример вывода: "1920,1080,30000/1001".
    bool probe(const std::string& path, const std::string& ffprobe) {
        const std::string cmd = quote(ffprobe)
            + " -v error -select_streams v:0"
              " -show_entries stream=width,height,r_frame_rate"
              " -of csv=p=0 " + quote(path);
        FILE* p = II_POPEN(cmd.c_str(), kPopenMode);
        if (!p) return false;
        char line[256] = {0};
        char* got = std::fgets(line, sizeof(line), p);
        II_PCLOSE(p);
        if (!got) return false;

        int w = 0, h = 0, num = 0, den = 1;
        // r_frame_rate приходит дробью ("30/1", "30000/1001"); если её нет,
        // sscanf вернёт 2 поля, fps останется 0 — это не ошибка.
        const int n = std::sscanf(line, "%d,%d,%d/%d", &w, &h, &num, &den);
        if (n < 2) return false;
        width_  = w;
        height_ = h;
        fps_    = (n >= 3 && num > 0 && den > 0)
                ? (int)((num + den / 2) / den)   // округление к ближайшему
                : 0;
        return w > 0 && h > 0;
    }

    FILE*       pipe_        = nullptr;
    bool        loop_        = false;
    bool        eof_         = false;
    int         width_       = 0;
    int         height_      = 0;
    int         fps_         = 0;
    std::size_t frame_bytes_ = 0;
    std::vector<uint8_t> rgb_;   // буфер текущего кадра в RGB888
};

}  // namespace

std::unique_ptr<VideoSource> make_video() {
    return std::make_unique<FfmpegPipeVideo>();
}
