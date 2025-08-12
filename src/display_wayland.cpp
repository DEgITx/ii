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
#include <poll.h>

#include <wayland-client.h>
#include <wayland-egl.h>

#include <EGL/egl.h>
#include <GLES2/gl2.h>

#include "xdg-shell-client-protocol.h"

namespace {

class WaylandDisplay : public Display {
public:
    ~WaylandDisplay() override { cleanup(); }

    bool init(int w, int h, const char* title) override;
    bool show_rgb(const uint8_t* rgb, int w, int h) override;
    bool poll() override;
    bool wait() override;

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
    void render();
    void cleanup();

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
    eglSwapInterval(egl_display_, 1);  // vsync
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

bool WaylandDisplay::init(int w, int h, const char* title) {
    win_w_ = w;
    win_h_ = h;

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

    if (!init_egl()) return false;
    if (!init_gl())  return false;

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

void WaylandDisplay::render() {
    // Letterbox-viewport: сохраняем пропорции изображения внутри окна.
    int vw = win_w_;
    int vh = win_h_;
    if (tex_w_ > 0 && tex_h_ > 0) {
        float s = std::min((float)win_w_ / tex_w_, (float)win_h_ / tex_h_);
        vw = (int)(tex_w_ * s);
        vh = (int)(tex_h_ * s);
    }
    int vx = (win_w_ - vw) / 2;
    int vy = (win_h_ - vh) / 2;

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

    eglSwapBuffers(egl_display_, egl_surface_);
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
        if (program_) glDeleteProgram(program_);
        if (vbo_)     glDeleteBuffers(1, &vbo_);
        if (texture_) glDeleteTextures(1, &texture_);
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
