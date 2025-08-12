// Wayland + EGL + GLES2 backend для отображения RGB-кадров.
//
// Логика:
//   * подключаемся к Wayland-композитору (Wayland-композитор);
//   * создаём wl_surface + xdg_toplevel (xdg-shell, стандартное окно);
//   * поверх wl_egl_window поднимаем EGL-контекст + GLES2;
//   * рендерим один полноэкранный квад с RGB-текстурой;
//   * на каждый show_rgb() обновляем содержимое текстуры через
//     glTexSubImage2D (без перевыделения), eglSwapBuffers с vsync.
//
// Пропорции изображения сохраняются — viewport letterbox’ится в окно.

#include "display.h"

#include <algorithm>
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
    glBindTexture(GL_TEXTURE_2D, texture_);
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

bool WaylandDisplay::poll() {
    if (closed_) return false;
    // Корректный неблокирующий drain: сбрасываем исходящий буфер и
    // читаем входящие события, только если они уже лежат в сокете.
    wl_display_flush(display_);
    pollfd pfd{ wl_display_get_fd(display_), POLLIN, 0 };
    if (::poll(&pfd, 1, 0) > 0 && (pfd.revents & POLLIN)) {
        if (wl_display_dispatch(display_) < 0) closed_ = true;
    } else {
        wl_display_dispatch_pending(display_);
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
        if (vbo_)          glDeleteBuffers(1, &vbo_);
        if (text_vbo_)     glDeleteBuffers(1, &text_vbo_);
        if (texture_)      glDeleteTextures(1, &texture_);
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
