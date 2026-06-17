// Wayland + EGL + GLES2 backend для отображения RGB-кадров.
//
// Логика:
//   * подключаемся к Wayland-композитору (Weston, Mutter, KWin и т.п.);
//   * создаём wl_surface + xdg_toplevel (xdg-shell, стандартное окно);
//   * поверх wl_egl_window поднимаем EGL-контекст + GLES2;
//   * рендерим один полноэкранный квад с RGB-текстурой;
//   * на каждый show_rgb() обновляем содержимое текстуры через
//     glTexSubImage2D (без перевыделения), eglSwapBuffers с vsync.
//
// Пропорции изображения сохраняются — viewport letterbox’ится в окно.

#include "display.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <poll.h>

#include <wayland-client.h>
#include <wayland-egl.h>

#include <EGL/egl.h>
#include <GLES2/gl2.h>

#include "xdg-shell-client-protocol.h"

// stb_easy_font — крошечный встроенный битмап-шрифт для отладочных
// оверлеев. Header-only, реализация подтягивается одним include’ом и
// должна жить ровно в одной TU — поэтому держим её здесь.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "stb_easy_font.h"
#pragma GCC diagnostic pop

namespace {

class WaylandDisplay : public Display {
public:
    ~WaylandDisplay() override { cleanup(); }

    bool init(int w, int h, const char* title, bool vsync) override;
    bool show_rgb(const uint8_t* rgb, int w, int h) override;
    bool poll() override;
    bool wait() override;
    void set_overlay_text(const char* text) override {
        overlay_text_ = (text && *text) ? text : std::string{};
    }
    void set_boxes(const std::vector<DisplayBox>& boxes) override {
        boxes_ = boxes;
    }

    // ---- GPU-сборка кадра из тайлов (см. display.h) ----
    bool supports_tile_frame() const override { return tile_supported_; }
    bool tile_frame_begin(const TileFrameDesc& desc) override;
    void tile_frame_add(const void* raw, int dst_x, int dst_y,
                        int ramp_x, int ramp_y) override;
    bool tile_frame_present() override;
    bool tile_frame_readback(std::vector<uint8_t>& rgb,
                             int& w, int& h) override;

    // ---- Wayland-callback’и (трамплины из C-API) ----
    // Публичные, потому что используются listener-структурами в namespace
    // scope и должны быть видны как обычные функции для C-кода Wayland.
    static void on_registry_global(void* data, wl_registry* r, uint32_t name,
                                   const char* iface, uint32_t version);
    static void on_registry_remove(void*, wl_registry*, uint32_t) {}
    static void on_wm_base_ping(void*, xdg_wm_base* b, uint32_t serial) {
        xdg_wm_base_pong(b, serial);
    }
    static void on_xdg_surface_configure(void* data, xdg_surface* s,
                                         uint32_t serial) {
        xdg_surface_ack_configure(s, serial);
        static_cast<WaylandDisplay*>(data)->configured_ = true;
    }
    static void on_toplevel_configure(void* data, xdg_toplevel*,
                                      int32_t w, int32_t h, wl_array*) {
        if (w <= 0 || h <= 0) return;
        auto* self = static_cast<WaylandDisplay*>(data);
        if (self->egl_window_ &&
            (w != self->win_w_ || h != self->win_h_)) {
            wl_egl_window_resize(self->egl_window_, w, h, 0, 0);
            self->win_w_ = w;
            self->win_h_ = h;
        }
    }
    static void on_toplevel_close(void* data, xdg_toplevel*) {
        static_cast<WaylandDisplay*>(data)->closed_ = true;
    }

private:
    bool init_wayland();
    bool init_egl();
    bool init_gl();
    bool init_text();
    bool init_compose();                       // GPU-сборка тайлов: шейдер+FBO
    bool ensure_compose_target(int w, int h);  // offscreen RGBA8 размером canvas
    GLuint ensure_tile_tex(int idx);           // кольцо текстур тайлов
    void render();
    void render_overlay();
    void render_boxes();
    void cleanup();
    // Вычисляет letterbox-viewport (vx, vy, vw, vh) под текущий размер
    // окна и текстуры. Результат используется и при отрисовке кадра
    // (glViewport), и при маппинге координат боксов из image space в окно.
    void compute_viewport(int& vx, int& vy, int& vw, int& vh) const;

    // ---- Wayland ----
    wl_display*    display_    = nullptr;
    wl_registry*   registry_   = nullptr;
    wl_compositor* compositor_ = nullptr;
    xdg_wm_base*   wm_base_    = nullptr;

    wl_surface*    surface_     = nullptr;
    xdg_surface*   xdg_surface_ = nullptr;
    xdg_toplevel*  toplevel_    = nullptr;
    wl_egl_window* egl_window_  = nullptr;

    // ---- EGL ----
    EGLDisplay egl_display_ = EGL_NO_DISPLAY;
    EGLContext egl_context_ = EGL_NO_CONTEXT;
    EGLSurface egl_surface_ = EGL_NO_SURFACE;

    // ---- GL ----
    GLuint program_ = 0;
    GLuint vbo_     = 0;
    GLuint texture_ = 0;
    GLint  a_pos_   = -1;
    GLint  a_uv_    = -1;
    GLint  u_tex_   = -1;

    int  win_w_      = 0;
    int  win_h_      = 0;
    int  tex_w_      = 0;
    int  tex_h_      = 0;
    bool configured_ = false;
    bool closed_     = false;
    bool vsync_      = true;

    // ---- Оверлей текста (FPS / debug) ----
    std::string overlay_text_;
    GLuint text_program_ = 0;
    GLuint text_vbo_     = 0;
    GLint  text_a_pos_   = -1;
    GLint  text_u_screen_= -1;
    GLint  text_u_color_ = -1;

    // ---- Боксы (например, YOLO-детекции) ----
    // Используют тот же шейдер, что и текст: вершины в пиксельных
    // координатах окна, цвет передаётся uniform’ом. Геометрия (контур
    // рамки + подложка под подпись + сам текст) пересоздаётся каждый
    // кадр — детекций мало (десятки), это копейки.
    std::vector<DisplayBox> boxes_;

    // ---- GPU-сборка тайлов (tile-режим, см. display.h) ----
    // compose-шейдер деквантует int8/uint8 тайлы и накладывает их (over-
    // composite с растушёвкой по ведущим краям) в offscreen RGBA8-текстуру
    // размером canvas; затем обычный render() рисует её как кадр (letterbox
    // + боксы + оверлей). Это снимает CPU-decode из горячего пути.
    bool   tile_supported_ = false;
    GLuint compose_prog_   = 0;
    GLuint fbo_            = 0;
    GLuint compose_tex_    = 0;     // offscreen RGBA8, размер canvas
    int    compose_w_      = 0;
    int    compose_h_      = 0;
    GLuint cur_frame_tex_  = 0;     // что рисует render(): 0 → texture_
    GLuint compose_vbo_    = 0;     // динамический квад (canvas xy + uv)
    GLint  c_a_canvas_ = -1, c_a_uv_ = -1;
    GLint  c_u_canvas_ = -1, c_u_tilesz_ = -1, c_u_tex_ = -1;
    GLint  c_u_scale_  = -1, c_u_zp_ = -1, c_u_channels_ = -1;
    GLint  c_u_dtype_  = -1, c_u_range_ = -1, c_u_ramp_ = -1;
    // Кольцо текстур тайлов: по одной на тайл в кадре, переиспользуются
    // между кадрами (запись-после-чтения на одной текстуре дала бы стол
    // на tile-based GPU — поэтому отдельная текстура на тайл).
    std::vector<GLuint> tile_texs_;
    int    tile_tex_w_ = 0, tile_tex_h_ = 0;
    GLenum tile_tex_fmt_ = 0;       // GL_LUMINANCE (C=1) | GL_RGB (C=3)
    int    tile_add_idx_ = 0;       // индекс текущего тайла в кадре
    TileFrameDesc tile_desc_{};
};

// Listener-структуры. Используем designated initializers (C++20),
// чтобы пропустить опциональные поля (configure_bounds, wm_capabilities,
// которые добавлены в новых версиях xdg-shell).
constexpr wl_registry_listener kRegistryListener = {
    .global        = WaylandDisplay::on_registry_global,
    .global_remove = WaylandDisplay::on_registry_remove,
};
constexpr xdg_wm_base_listener kWmBaseListener = {
    .ping = WaylandDisplay::on_wm_base_ping,
};
constexpr xdg_surface_listener kXdgSurfaceListener = {
    .configure = WaylandDisplay::on_xdg_surface_configure,
};
// Опциональные поля configure_bounds / wm_capabilities заполняем nullptr,
// чтобы убить -Wmissing-field-initializers и не зависеть от того, какой
// версии xdg-shell.h попался в системе (в старых их просто нет).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
constexpr xdg_toplevel_listener kToplevelListener = {
    .configure = WaylandDisplay::on_toplevel_configure,
    .close     = WaylandDisplay::on_toplevel_close,
};
#pragma GCC diagnostic pop

void WaylandDisplay::on_registry_global(void* data, wl_registry* r,
                                        uint32_t name, const char* iface,
                                        uint32_t version) {
    auto* self = static_cast<WaylandDisplay*>(data);
    if (std::strcmp(iface, wl_compositor_interface.name) == 0) {
        self->compositor_ = static_cast<wl_compositor*>(
            wl_registry_bind(r, name, &wl_compositor_interface,
                             std::min(version, 4u)));
    } else if (std::strcmp(iface, xdg_wm_base_interface.name) == 0) {
        self->wm_base_ = static_cast<xdg_wm_base*>(
            wl_registry_bind(r, name, &xdg_wm_base_interface, 1));
        xdg_wm_base_add_listener(self->wm_base_, &kWmBaseListener, self);
    }
}

bool WaylandDisplay::init_wayland() {
    display_ = wl_display_connect(nullptr);
    if (!display_) {
        std::fprintf(stderr,
            "Wayland: не удалось подключиться (WAYLAND_DISPLAY не задан?)\n");
        return false;
    }
    registry_ = wl_display_get_registry(display_);
    wl_registry_add_listener(registry_, &kRegistryListener, this);
    wl_display_roundtrip(display_);
    if (!compositor_ || !wm_base_) {
        std::fprintf(stderr,
            "Wayland: композитор не предоставляет wl_compositor / xdg_wm_base.\n");
        return false;
    }
    return true;
}

bool WaylandDisplay::init_egl() {
    egl_display_ = eglGetDisplay((EGLNativeDisplayType)display_);
    if (egl_display_ == EGL_NO_DISPLAY) {
        std::fprintf(stderr, "EGL: eglGetDisplay упал.\n");
        return false;
    }
    EGLint maj = 0, min = 0;
    if (!eglInitialize(egl_display_, &maj, &min)) {
        std::fprintf(stderr, "EGL: eglInitialize упал.\n");
        return false;
    }
    eglBindAPI(EGL_OPENGL_ES_API);

    const EGLint cfg_attr[] = {
        EGL_SURFACE_TYPE,    EGL_WINDOW_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_RED_SIZE,   8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE,  8,
        EGL_ALPHA_SIZE, 0,
        EGL_NONE,
    };
    EGLConfig cfg;
    EGLint n = 0;
    if (!eglChooseConfig(egl_display_, cfg_attr, &cfg, 1, &n) || n == 0) {
        std::fprintf(stderr, "EGL: не нашли подходящий config.\n");
        return false;
    }
    const EGLint ctx_attr[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
    egl_context_ = eglCreateContext(egl_display_, cfg, EGL_NO_CONTEXT, ctx_attr);
    if (egl_context_ == EGL_NO_CONTEXT) {
        std::fprintf(stderr, "EGL: eglCreateContext упал.\n");
        return false;
    }
    egl_window_ = wl_egl_window_create(surface_, win_w_, win_h_);
    if (!egl_window_) {
        std::fprintf(stderr, "EGL: wl_egl_window_create упал.\n");
        return false;
    }
    egl_surface_ = eglCreateWindowSurface(egl_display_, cfg,
                                          (EGLNativeWindowType)egl_window_,
                                          nullptr);
    if (egl_surface_ == EGL_NO_SURFACE) {
        std::fprintf(stderr, "EGL: eglCreateWindowSurface упал.\n");
        return false;
    }
    if (!eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_)) {
        std::fprintf(stderr, "EGL: eglMakeCurrent упал.\n");
        return false;
    }
    // 1 — ждём vblank (плавно, без tearing); 0 — отдаём максимум
    // (полезно для замеров и диагностики, не зависим от частоты дисплея).
    if (!eglSwapInterval(egl_display_, vsync_ ? 1 : 0)) {
        std::fprintf(stderr,
            "EGL: eglSwapInterval(%d) не поддержан драйвером — игнорирую.\n",
            vsync_ ? 1 : 0);
    }
    return true;
}

GLuint compile_shader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512] = {};
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        std::fprintf(stderr, "GLSL: compile error: %s\n", log);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

bool WaylandDisplay::init_gl() {
    static const char* kVS =
        "attribute vec2 a_pos;\n"
        "attribute vec2 a_uv;\n"
        "varying vec2 v_uv;\n"
        "void main(){ gl_Position = vec4(a_pos, 0.0, 1.0); v_uv = a_uv; }\n";
    static const char* kFS =
        "precision mediump float;\n"
        "varying vec2 v_uv;\n"
        "uniform sampler2D u_tex;\n"
        "void main(){ gl_FragColor = texture2D(u_tex, v_uv); }\n";

    GLuint vs = compile_shader(GL_VERTEX_SHADER,   kVS);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, kFS);
    if (!vs || !fs) return false;

    program_ = glCreateProgram();
    glAttachShader(program_, vs);
    glAttachShader(program_, fs);
    glLinkProgram(program_);
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint ok = 0;
    glGetProgramiv(program_, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512] = {};
        glGetProgramInfoLog(program_, sizeof(log), nullptr, log);
        std::fprintf(stderr, "GLSL: link error: %s\n", log);
        return false;
    }
    a_pos_ = glGetAttribLocation(program_,  "a_pos");
    a_uv_  = glGetAttribLocation(program_,  "a_uv");
    u_tex_ = glGetUniformLocation(program_, "u_tex");

    // Полноэкранный квад. UV перевёрнут по Y: текстуру с RGB HWC,
    // у которой строка 0 — верх изображения, OpenGL ожидает «снизу вверх»,
    // поэтому верхним вершинам даём v=0, нижним — v=1.
    static const float verts[] = {
        // pos        uv
        -1.f, -1.f,  0.f, 1.f,
         1.f, -1.f,  1.f, 1.f,
        -1.f,  1.f,  0.f, 0.f,
         1.f,  1.f,  1.f, 0.f,
    };
    glGenBuffers(1, &vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

    glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glClearColor(0.f, 0.f, 0.f, 1.f);
    return true;
}

bool WaylandDisplay::init_text() {
    // Шейдер для оверлея: вершины в пиксельных координатах
    // (0,0 в левом верхнем углу окна), цвет передаётся uniform’ом.
    // Используется и для текста, и для полупрозрачного фона под ним.
    static const char* kVS =
        "attribute vec2 a_pos;\n"
        "uniform vec2 u_screen;\n"
        "void main(){\n"
        "  vec2 ndc = (a_pos / u_screen) * 2.0 - 1.0;\n"
        "  ndc.y = -ndc.y;\n"
        "  gl_Position = vec4(ndc, 0.0, 1.0);\n"
        "}\n";
    static const char* kFS =
        "precision mediump float;\n"
        "uniform vec4 u_color;\n"
        "void main(){ gl_FragColor = u_color; }\n";

    GLuint vs = compile_shader(GL_VERTEX_SHADER,   kVS);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, kFS);
    if (!vs || !fs) return false;
    text_program_ = glCreateProgram();
    glAttachShader(text_program_, vs);
    glAttachShader(text_program_, fs);
    glLinkProgram(text_program_);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok = 0;
    glGetProgramiv(text_program_, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512] = {};
        glGetProgramInfoLog(text_program_, sizeof(log), nullptr, log);
        std::fprintf(stderr, "GLSL: text link error: %s\n", log);
        return false;
    }
    text_a_pos_    = glGetAttribLocation(text_program_,  "a_pos");
    text_u_screen_ = glGetUniformLocation(text_program_, "u_screen");
    text_u_color_  = glGetUniformLocation(text_program_, "u_color");
    glGenBuffers(1, &text_vbo_);
    return true;
}

void WaylandDisplay::render_overlay() {
    if (overlay_text_.empty()) return;

    // 1) Получаем геометрию текста и его пиксельные размеры.
    // Каждый символ stb_easy_font раскладывает на ~1-3 квада, каждый
    // квад — это 4 вершины по 16 байт (x,y,z + 4 байта цвета).
    static char raw[80 * 1024];
    char* text = const_cast<char*>(overlay_text_.c_str());
    int num_quads = stb_easy_font_print(0.0f, 0.0f, text, nullptr,
                                        raw, sizeof(raw));
    int tw = stb_easy_font_width(text);
    int th = stb_easy_font_height(text);

    constexpr float kPad = 4.0f;     // отступ внутри подложки
    constexpr float kMargin = 8.0f;  // отступ от края окна
    constexpr float kScale = 2.0f;   // увеличиваем шрифт (он крошечный)

    // 2) Конвертируем quads -> triangles в один float-буфер (x,y).
    // stb_easy_font выдаёт CCW-quad: v0,v1,v2,v3. Разбиваем на (0,1,2)+(0,2,3).
    struct EasyVert { float x, y, z; unsigned char c[4]; };
    auto* qv = reinterpret_cast<const EasyVert*>(raw);
    std::vector<float> tri;
    tri.reserve((std::size_t)num_quads * 6 * 2);
    auto push = [&](float x, float y) {
        tri.push_back(kMargin + kPad + x * kScale);
        tri.push_back(kMargin + kPad + y * kScale);
    };
    for (int q = 0; q < num_quads; ++q) {
        const EasyVert* v = &qv[q * 4];
        push(v[0].x, v[0].y); push(v[1].x, v[1].y); push(v[2].x, v[2].y);
        push(v[0].x, v[0].y); push(v[2].x, v[2].y); push(v[3].x, v[3].y);
    }

    // 3) Подложка — полупрозрачный чёрный прямоугольник под текстом.
    float bx = kMargin;
    float by = kMargin;
    float bw = tw * kScale + kPad * 2.0f;
    float bh = th * kScale + kPad * 2.0f;
    float bg[] = {
        bx,      by,
        bx + bw, by,
        bx,      by + bh,
        bx + bw, by,
        bx + bw, by + bh,
        bx,      by + bh,
    };

    // 4) Включаем blending, viewport на всё окно (оверлей рисуется в
    // экранных координатах, не в letterbox-viewport главного кадра).
    glViewport(0, 0, win_w_, win_h_);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(text_program_);
    glUniform2f(text_u_screen_, (float)win_w_, (float)win_h_);
    glBindBuffer(GL_ARRAY_BUFFER, text_vbo_);
    glEnableVertexAttribArray(text_a_pos_);
    glVertexAttribPointer(text_a_pos_, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    // Подложка.
    glUniform4f(text_u_color_, 0.0f, 0.0f, 0.0f, 0.55f);
    glBufferData(GL_ARRAY_BUFFER, sizeof(bg), bg, GL_DYNAMIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    // Текст ярко-жёлтым.
    glUniform4f(text_u_color_, 1.0f, 1.0f, 0.2f, 1.0f);
    glBufferData(GL_ARRAY_BUFFER,
                 (GLsizeiptr)(tri.size() * sizeof(float)),
                 tri.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(tri.size() / 2));

    glDisable(GL_BLEND);
}

bool WaylandDisplay::init(int w, int h, const char* title, bool vsync) {
    win_w_ = w;
    win_h_ = h;
    vsync_ = vsync;

    if (!init_wayland()) return false;

    surface_     = wl_compositor_create_surface(compositor_);
    xdg_surface_ = xdg_wm_base_get_xdg_surface(wm_base_, surface_);
    xdg_surface_add_listener(xdg_surface_, &kXdgSurfaceListener, this);
    toplevel_ = xdg_surface_get_toplevel(xdg_surface_);
    xdg_toplevel_add_listener(toplevel_, &kToplevelListener, this);
    if (title) xdg_toplevel_set_title(toplevel_, title);
    xdg_toplevel_set_app_id(toplevel_, "npu");

    wl_surface_commit(surface_);
    // Дожидаемся первого configure — без него EGL surface некуда вешать.
    while (!configured_ && wl_display_dispatch(display_) != -1) {}

    if (!init_egl())  return false;
    if (!init_gl())   return false;
    if (!init_text()) return false;
    // GPU-сборка тайлов — опциональна: если шейдер/FBO не поднялись,
    // tile_supported_ остаётся false и раннер использует CPU-путь.
    init_compose();

    // Пустой первый кадр, чтобы окно появилось до первого show_rgb().
    glViewport(0, 0, win_w_, win_h_);
    glClear(GL_COLOR_BUFFER_BIT);
    eglSwapBuffers(egl_display_, egl_surface_);
    return true;
}

bool WaylandDisplay::show_rgb(const uint8_t* rgb, int w, int h) {
    if (closed_) return false;

    glBindTexture(GL_TEXTURE_2D, texture_);
    if (w != tex_w_ || h != tex_h_) {
        // Размер изменился — единоразовая аллокация.
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, rgb);
        tex_w_ = w;
        tex_h_ = h;
    } else {
        // Тот же размер — заливаем поверх (на видео-стрим это hot path).
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h,
                        GL_RGB, GL_UNSIGNED_BYTE, rgb);
    }
    render();
    poll();  // обработать pending-события сразу после кадра
    return !closed_;
}

void WaylandDisplay::compute_viewport(int& vx, int& vy,
                                      int& vw, int& vh) const {
    vw = win_w_;
    vh = win_h_;
    if (tex_w_ > 0 && tex_h_ > 0) {
        float s = std::min((float)win_w_ / tex_w_, (float)win_h_ / tex_h_);
        vw = (int)(tex_w_ * s);
        vh = (int)(tex_h_ * s);
    }
    vx = (win_w_ - vw) / 2;
    vy = (win_h_ - vh) / 2;
}

void WaylandDisplay::render() {
    int vx, vy, vw, vh;
    compute_viewport(vx, vy, vw, vh);

    // Сначала чистим всё окно (чёрные «полосы» letterbox’а),
    // затем рисуем картинку в центральный viewport.
    glViewport(0, 0, win_w_, win_h_);
    glClear(GL_COLOR_BUFFER_BIT);
    glViewport(vx, vy, vw, vh);

    glUseProgram(program_);
    glActiveTexture(GL_TEXTURE0);
    // cur_frame_tex_ != 0 — рисуем собранную compose-цель (tile-режим),
    // иначе обычную залитую с CPU текстуру кадра.
    glBindTexture(GL_TEXTURE_2D, cur_frame_tex_ ? cur_frame_tex_ : texture_);
    glUniform1i(u_tex_, 0);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glEnableVertexAttribArray(a_pos_);
    glVertexAttribPointer(a_pos_, 2, GL_FLOAT, GL_FALSE,
                          4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(a_uv_);
    glVertexAttribPointer(a_uv_, 2, GL_FLOAT, GL_FALSE,
                          4 * sizeof(float),
                          (void*)(2 * sizeof(float)));
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    // Сначала боксы (рисуются в image-координатах, маппятся в letterbox-
    // viewport), затем — текстовый оверлей в полном экранном пространстве.
    render_boxes();
    render_overlay();

    eglSwapBuffers(egl_display_, egl_surface_);
}

void WaylandDisplay::render_boxes() {
    if (boxes_.empty() || tex_w_ <= 0 || tex_h_ <= 0) return;

    int vx, vy, vw, vh;
    compute_viewport(vx, vy, vw, vh);

    // Маппинг (image px) -> (window px). Y у нас сверху вниз, и шейдер
    // оверлея уже инвертирует Y — оставляем линейное соответствие.
    const float sx = (float)vw / tex_w_;
    const float sy = (float)vh / tex_h_;
    auto map_x = [&](float x) { return vx + x * sx; };
    auto map_y = [&](float y) { return vy + y * sy; };

    // Толщина рамки в пикселях окна — масштабируем от размера окна,
    // чтобы на 4K не превратилась в «волосок», а на маленьком — в кляксу.
    const float t = std::clamp(std::min(win_w_, win_h_) / 400.0f, 1.5f, 4.0f);

    glViewport(0, 0, win_w_, win_h_);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(text_program_);
    glUniform2f(text_u_screen_, (float)win_w_, (float)win_h_);
    glBindBuffer(GL_ARRAY_BUFFER, text_vbo_);
    glEnableVertexAttribArray(text_a_pos_);
    glVertexAttribPointer(text_a_pos_, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    // Геометрия одной рамки: 4 «полоски» (top/bottom/left/right),
    // каждая — 2 треугольника. Рисуем по одному боксу за раз, чтобы
    // менять цвет через uniform — для 10..50 детекций накладные расходы
    // мизерные, а код проще.
    auto rect = [](float x1, float y1, float x2, float y2,
                   std::vector<float>& out) {
        out.push_back(x1); out.push_back(y1);
        out.push_back(x2); out.push_back(y1);
        out.push_back(x1); out.push_back(y2);
        out.push_back(x2); out.push_back(y1);
        out.push_back(x2); out.push_back(y2);
        out.push_back(x1); out.push_back(y2);
    };

    std::vector<float> verts;
    verts.reserve(4 * 6 * 2);

    for (const DisplayBox& b : boxes_) {
        float x1 = map_x(b.x1), y1 = map_y(b.y1);
        float x2 = map_x(b.x2), y2 = map_y(b.y2);
        if (x2 < x1) std::swap(x1, x2);
        if (y2 < y1) std::swap(y1, y2);

        verts.clear();
        rect(x1,     y1,     x2,     y1 + t, verts);  // top
        rect(x1,     y2 - t, x2,     y2,     verts);  // bottom
        rect(x1,     y1,     x1 + t, y2,     verts);  // left
        rect(x2 - t, y1,     x2,     y2,     verts);  // right

        glUniform4f(text_u_color_,
                    b.r / 255.0f, b.g / 255.0f, b.b / 255.0f, 1.0f);
        glBufferData(GL_ARRAY_BUFFER,
                     (GLsizeiptr)(verts.size() * sizeof(float)),
                     verts.data(), GL_DYNAMIC_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(verts.size() / 2));

        // ---- Подпись ----
        if (b.label.empty() && b.score < 0.0f) continue;

        char label_buf[160];
        if (b.score >= 0.0f) {
            std::snprintf(label_buf, sizeof(label_buf), "%s %.0f%%",
                          b.label.c_str(), b.score * 100.0f);
        } else {
            std::snprintf(label_buf, sizeof(label_buf), "%s",
                          b.label.c_str());
        }

        static char raw[16 * 1024];
        int num_quads = stb_easy_font_print(0.0f, 0.0f, label_buf, nullptr,
                                            raw, sizeof(raw));
        int tw = stb_easy_font_width(label_buf);
        int th = stb_easy_font_height(label_buf);

        constexpr float kPad   = 3.0f;
        constexpr float kScale = 1.5f;
        float bw = tw * kScale + kPad * 2.0f;
        float bh = th * kScale + kPad * 2.0f;

        // Подпись прижимаем к верху рамки; если вылезает за верх окна —
        // переносим внутрь рамки (чтобы не обрезалась).
        float bx = x1;
        float by = y1 - bh;
        if (by < 0.0f) by = y1;

        // Подложка цвета рамки.
        float bg[] = {
            bx,      by,
            bx + bw, by,
            bx,      by + bh,
            bx + bw, by,
            bx + bw, by + bh,
            bx,      by + bh,
        };
        glUniform4f(text_u_color_,
                    b.r / 255.0f, b.g / 255.0f, b.b / 255.0f, 0.85f);
        glBufferData(GL_ARRAY_BUFFER, sizeof(bg), bg, GL_DYNAMIC_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // Сам текст — белым, чтобы читалось на цветной подложке.
        struct EasyVert { float x, y, z; unsigned char c[4]; };
        auto* qv = reinterpret_cast<const EasyVert*>(raw);
        std::vector<float> tri;
        tri.reserve((std::size_t)num_quads * 6 * 2);
        auto push = [&](float x, float y) {
            tri.push_back(bx + kPad + x * kScale);
            tri.push_back(by + kPad + y * kScale);
        };
        for (int q = 0; q < num_quads; ++q) {
            const EasyVert* v = &qv[q * 4];
            push(v[0].x, v[0].y); push(v[1].x, v[1].y); push(v[2].x, v[2].y);
            push(v[0].x, v[0].y); push(v[2].x, v[2].y); push(v[3].x, v[3].y);
        }
        glUniform4f(text_u_color_, 1.0f, 1.0f, 1.0f, 1.0f);
        glBufferData(GL_ARRAY_BUFFER,
                     (GLsizeiptr)(tri.size() * sizeof(float)),
                     tri.data(), GL_DYNAMIC_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(tri.size() / 2));
    }

    glDisable(GL_BLEND);
}

// ---- GPU-сборка кадра из тайлов -----------------------------------------

bool WaylandDisplay::init_compose() {
    // VS: позиция тайла задаётся в canvas-пикселях, переводим в NDC.
    // ВАЖНО (ориентация): без y-flip — canvas_y=0 (верх кадра) → ndc.y=-1
    // → низ FBO → texel row0 compose-текстуры. Так row0 = верх кадра, что
    // совпадает с конвенцией CPU-залитой texture_ (её v-flip’ает display-VS),
    // и glReadPixels отдаёт row0=верх без переворота.
    static const char* kVS =
        "attribute vec2 a_canvas;\n"
        "attribute vec2 a_uv;\n"
        "uniform vec2 u_canvas;\n"
        "uniform vec2 u_tilesz;\n"
        "varying vec2 v_uv;\n"
        "varying vec2 v_local;\n"
        "void main(){\n"
        "  vec2 ndc = (a_canvas / u_canvas) * 2.0 - 1.0;\n"
        "  gl_Position = vec4(ndc, 0.0, 1.0);\n"
        "  v_uv = a_uv;\n"
        "  v_local = a_uv * u_tilesz;\n"   // локальные px внутри тайла (для ramp)
        "}\n";
    // FS: highp обязателен — восстанавливаем точный байт из R8 UNORM
    // (b = round(s*255)) и деквантуем как decode_kernel в image_proc.cpp.
    static const char* kFS =
        "precision highp float;\n"
        "varying vec2 v_uv;\n"
        "varying vec2 v_local;\n"
        "uniform sampler2D u_tex;\n"
        "uniform float u_scale;\n"
        "uniform float u_zp;\n"
        "uniform float u_channels;\n"   // 1.0 (Y) | 3.0 (RGB)
        "uniform float u_dtype;\n"      // 0=int8 (bit-pattern), 1=uint8
        "uniform float u_range;\n"      // 0=unit 1=signed 2=byte
        "uniform vec2  u_ramp;\n"       // ширина растушёвки по левому/верхнему краю
        "float lead_alpha(float d, float r){\n"
        "  if (r <= 0.0) return 1.0;\n"
        "  if (d >= r) return 1.0;\n"
        "  return clamp((2.0*d + 1.0) / (2.0*r), 0.0, 1.0);\n"
        "}\n"
        "float deq(float s){\n"
        "  float b = floor(s * 255.0 + 0.5);\n"
        "  float q = b;\n"
        "  if (u_dtype < 0.5 && b > 127.5) q = b - 256.0;\n"   // int8
        "  float val = u_scale * (q - u_zp);\n"
        "  if (u_range > 1.5) val = val / 255.0;\n"            // byte
        "  else if (u_range > 0.5) val = val * 0.5 + 0.5;\n"   // signed
        "  return clamp(val, 0.0, 1.0);\n"
        "}\n"
        "void main(){\n"
        "  vec3 rgb;\n"
        "  if (u_channels < 1.5) {\n"
        "    float y = deq(texture2D(u_tex, v_uv).r);\n"
        "    rgb = vec3(y);\n"
        "  } else {\n"
        "    vec3 c = texture2D(u_tex, v_uv).rgb;\n"
        "    rgb = vec3(deq(c.r), deq(c.g), deq(c.b));\n"
        "  }\n"
        "  float a = lead_alpha(v_local.x, u_ramp.x)\n"
        "          * lead_alpha(v_local.y, u_ramp.y);\n"
        "  gl_FragColor = vec4(rgb, a);\n"
        "}\n";

    GLuint vs = compile_shader(GL_VERTEX_SHADER,   kVS);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, kFS);
    if (!vs || !fs) return false;
    compose_prog_ = glCreateProgram();
    glAttachShader(compose_prog_, vs);
    glAttachShader(compose_prog_, fs);
    glLinkProgram(compose_prog_);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok = 0;
    glGetProgramiv(compose_prog_, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512] = {};
        glGetProgramInfoLog(compose_prog_, sizeof(log), nullptr, log);
        std::fprintf(stderr, "GLSL: compose link error: %s\n", log);
        return false;
    }
    c_a_canvas_   = glGetAttribLocation(compose_prog_,  "a_canvas");
    c_a_uv_       = glGetAttribLocation(compose_prog_,  "a_uv");
    c_u_canvas_   = glGetUniformLocation(compose_prog_, "u_canvas");
    c_u_tilesz_   = glGetUniformLocation(compose_prog_, "u_tilesz");
    c_u_tex_      = glGetUniformLocation(compose_prog_, "u_tex");
    c_u_scale_    = glGetUniformLocation(compose_prog_, "u_scale");
    c_u_zp_       = glGetUniformLocation(compose_prog_, "u_zp");
    c_u_channels_ = glGetUniformLocation(compose_prog_, "u_channels");
    c_u_dtype_    = glGetUniformLocation(compose_prog_, "u_dtype");
    c_u_range_    = glGetUniformLocation(compose_prog_, "u_range");
    c_u_ramp_     = glGetUniformLocation(compose_prog_, "u_ramp");

    glGenBuffers(1, &compose_vbo_);
    glGenFramebuffers(1, &fbo_);
    tile_supported_ = true;
    return true;
}

bool WaylandDisplay::ensure_compose_target(int w, int h) {
    if (compose_tex_ && w == compose_w_ && h == compose_h_) return true;
    if (!compose_tex_) glGenTextures(1, &compose_tex_);
    glBindTexture(GL_TEXTURE_2D, compose_tex_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    compose_w_ = w;
    compose_h_ = h;

    // Проверяем полноту FBO один раз при (ре)аллокации. GL_RGBA/UNSIGNED_BYTE
    // как color-attachment поддержан не каждым ES2-драйвером (формально нужен
    // OES_rgb8_rgba8) — на неполном FBO гасим GPU-путь, раннер откатится на CPU.
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, compose_tex_, 0);
    GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    if (st != GL_FRAMEBUFFER_COMPLETE) {
        std::fprintf(stderr,
            "GL: FBO неполон (0x%x) — GPU-сборка тайлов отключена, CPU-путь.\n",
            (unsigned)st);
        tile_supported_ = false;
        return false;
    }
    return true;
}

GLuint WaylandDisplay::ensure_tile_tex(int idx) {
    while ((int)tile_texs_.size() <= idx) {
        GLuint t = 0;
        glGenTextures(1, &t);
        glBindTexture(GL_TEXTURE_2D, t);
        glTexImage2D(GL_TEXTURE_2D, 0, tile_tex_fmt_, tile_tex_w_, tile_tex_h_,
                     0, tile_tex_fmt_, GL_UNSIGNED_BYTE, nullptr);
        // NEAREST: квад 1:1 с текселями тайла (canvas-px == tile-px), плюс
        // точное восстановление байта во фрагменте требует точного семпла.
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        tile_texs_.push_back(t);
    }
    return tile_texs_[idx];
}

bool WaylandDisplay::tile_frame_begin(const TileFrameDesc& d) {
    if (!tile_supported_ || closed_) return false;
    tile_desc_ = d;

    const GLenum fmt = (d.channels == 1) ? GL_LUMINANCE : GL_RGB;
    if (fmt != tile_tex_fmt_ || d.tile_w != tile_tex_w_
        || d.tile_h != tile_tex_h_) {
        if (!tile_texs_.empty()) {
            glDeleteTextures((GLsizei)tile_texs_.size(), tile_texs_.data());
            tile_texs_.clear();
        }
        tile_tex_fmt_ = fmt;
        tile_tex_w_   = d.tile_w;
        tile_tex_h_   = d.tile_h;
    }
    if (!ensure_compose_target(d.canvas_w, d.canvas_h)) return false;
    tile_add_idx_ = 0;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, compose_tex_, 0);
    glViewport(0, 0, compose_w_, compose_h_);
    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(compose_prog_);
    glUniform1i(c_u_tex_, 0);
    glUniform2f(c_u_canvas_,  (float)compose_w_, (float)compose_h_);
    glUniform2f(c_u_tilesz_,  (float)d.tile_w,   (float)d.tile_h);
    glUniform1f(c_u_scale_,   d.scale);
    glUniform1f(c_u_zp_,      (float)d.zero_point);
    glUniform1f(c_u_channels_,(float)d.channels);
    glUniform1f(c_u_dtype_,   (float)d.dtype);
    glUniform1f(c_u_range_,   (float)d.range);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    return true;
}

void WaylandDisplay::tile_frame_add(const void* raw, int dst_x, int dst_y,
                                    int ramp_x, int ramp_y) {
    if (!tile_supported_ || !raw) return;
    const GLuint tex = ensure_tile_tex(tile_add_idx_++);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tile_tex_w_, tile_tex_h_,
                    tile_tex_fmt_, GL_UNSIGNED_BYTE, raw);

    // Квад в canvas-координатах + uv по тайлу (uv.y=0 — верх тайла).
    const float x0 = (float)dst_x;
    const float y0 = (float)dst_y;
    const float x1 = (float)(dst_x + tile_tex_w_);
    const float y1 = (float)(dst_y + tile_tex_h_);
    const float verts[] = {
        // a_canvas      a_uv
        x0, y0,   0.f, 0.f,   // TL
        x1, y0,   1.f, 0.f,   // TR
        x0, y1,   0.f, 1.f,   // BL
        x1, y1,   1.f, 1.f,   // BR
    };
    glUniform2f(c_u_ramp_, (float)ramp_x, (float)ramp_y);
    glBindBuffer(GL_ARRAY_BUFFER, compose_vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(c_a_canvas_);
    glVertexAttribPointer(c_a_canvas_, 2, GL_FLOAT, GL_FALSE,
                          4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(c_a_uv_);
    glVertexAttribPointer(c_a_uv_, 2, GL_FLOAT, GL_FALSE,
                          4 * sizeof(float), (void*)(2 * sizeof(float)));
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

bool WaylandDisplay::tile_frame_present() {
    if (closed_) return false;
    if (!tile_supported_ || !compose_tex_) return false;
    // Возврат к оконному фреймбуферу; собранную цель рисуем как обычный кадр.
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDisable(GL_BLEND);
    tex_w_ = compose_w_;
    tex_h_ = compose_h_;
    cur_frame_tex_ = compose_tex_;
    render();              // letterbox + боксы + оверлей + eglSwapBuffers
    cur_frame_tex_ = 0;
    poll();
    return !closed_;
}

bool WaylandDisplay::tile_frame_readback(std::vector<uint8_t>& rgb,
                                         int& w, int& h) {
    if (!tile_supported_ || !compose_tex_) return false;
    w = compose_w_;
    h = compose_h_;
    std::vector<uint8_t> rgba((std::size_t)w * h * 4);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, compose_tex_, 0);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    // glReadPixels(y=0) = низ FBO = texel row0 = верх кадра (см. ориентацию
    // в init_compose) — переворот не нужен. RGBA → RGB.
    rgb.resize((std::size_t)w * h * 3);
    for (std::size_t i = 0, n = (std::size_t)w * h; i < n; ++i) {
        rgb[i * 3 + 0] = rgba[i * 4 + 0];
        rgb[i * 3 + 1] = rgba[i * 4 + 1];
        rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }
    return true;
}

bool WaylandDisplay::poll() {
    if (closed_) return false;

    // Неблокирующий drain по канону libwayland
    while (wl_display_prepare_read(display_) != 0) {
        if (wl_display_dispatch_pending(display_) < 0) {
            closed_ = true;
            return false;
        }
    }

    // Сокет полон — даём композитору один короткий шанс вычитать буфер
    // и идём дальше: poll() вызывается из render loop, блокироваться нельзя.
    constexpr int kFlushRetries = 2;
    for (int i = 0; i < kFlushRetries; ++i) {
        int r = wl_display_flush(display_);
        if (r >= 0) break;
        if (errno == EINTR) { continue; }
        if (errno != EAGAIN) {
            wl_display_cancel_read(display_);
            closed_ = true;
            return false;
        }
        pollfd po{ wl_display_get_fd(display_), POLLOUT, 0 };
        int pr = ::poll(&po, 1, 2);
        if (pr < 0 && errno != EINTR) {
            wl_display_cancel_read(display_);
            closed_ = true;
            return false;
        }
        if (pr > 0 && (po.revents & (POLLHUP | POLLERR))) {
            wl_display_cancel_read(display_);
            closed_ = true;
            return false;
        }
    }
    // Если не дофлашили за отведённые попытки — не блокируем кадр,
    // дочитаем то, что уже пришло, и попробуем снова в следующий poll().

    pollfd pi{ wl_display_get_fd(display_), POLLIN, 0 };
    int pr = ::poll(&pi, 1, 0);
    if (pr > 0 && (pi.revents & (POLLHUP | POLLERR))) {
        wl_display_cancel_read(display_);
        closed_ = true;
        return false;
    }
    if (pr > 0 && (pi.revents & POLLIN)) {
        if (wl_display_read_events(display_) < 0) {
            closed_ = true;
            return false;
        }
    } else {
        wl_display_cancel_read(display_);
        if (pr < 0 && errno != EINTR) {
            closed_ = true;
            return false;
        }
    }

    if (wl_display_dispatch_pending(display_) < 0) {
        closed_ = true;
        return false;
    }
    return !closed_;
}

bool WaylandDisplay::wait() {
    if (closed_) return false;
    if (wl_display_dispatch(display_) < 0) closed_ = true;
    return !closed_;
}

void WaylandDisplay::cleanup() {
    if (egl_display_ != EGL_NO_DISPLAY) {
        eglMakeCurrent(egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE,
                       EGL_NO_CONTEXT);
        if (program_)      glDeleteProgram(program_);
        if (text_program_) glDeleteProgram(text_program_);
        if (compose_prog_) glDeleteProgram(compose_prog_);
        if (vbo_)          glDeleteBuffers(1, &vbo_);
        if (text_vbo_)     glDeleteBuffers(1, &text_vbo_);
        if (compose_vbo_)  glDeleteBuffers(1, &compose_vbo_);
        if (texture_)      glDeleteTextures(1, &texture_);
        if (compose_tex_)  glDeleteTextures(1, &compose_tex_);
        if (!tile_texs_.empty())
            glDeleteTextures((GLsizei)tile_texs_.size(), tile_texs_.data());
        if (fbo_)          glDeleteFramebuffers(1, &fbo_);
        if (egl_surface_ != EGL_NO_SURFACE)
            eglDestroySurface(egl_display_, egl_surface_);
        if (egl_context_ != EGL_NO_CONTEXT)
            eglDestroyContext(egl_display_, egl_context_);
        eglTerminate(egl_display_);
    }
    if (egl_window_)  wl_egl_window_destroy(egl_window_);
    if (toplevel_)    xdg_toplevel_destroy(toplevel_);
    if (xdg_surface_) xdg_surface_destroy(xdg_surface_);
    if (surface_)     wl_surface_destroy(surface_);
    if (wm_base_)     xdg_wm_base_destroy(wm_base_);
    if (compositor_)  wl_compositor_destroy(compositor_);
    if (registry_)    wl_registry_destroy(registry_);
    if (display_)     wl_display_disconnect(display_);
}

}  // namespace

std::unique_ptr<Display> make_display() {
    return std::make_unique<WaylandDisplay>();
}
