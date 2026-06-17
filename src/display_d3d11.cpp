// Win32 + Direct3D 11 backend для отображения RGB-кадров (Windows-аналог
// display_wayland.cpp).
//
// Логика — один в один с Wayland-вариантом, только нативный API другой:
//   * создаём обычное окно Win32 (CreateWindowEx + класс с WndProc);
//   * поднимаем D3D11-устройство + DXGI swapchain на это окно;
//   * держим одну динамическую текстуру под кадр и заливаем её на каждый
//     show_rgb() через Map/Unmap (без перевыделения, если размер тот же);
//   * рисуем один полноэкранный квад с этой текстурой;
//   * пропорции сохраняем — viewport letterbox’ится в окно.
//
// Текстовый оверлей (FPS/debug) и боксы детекций рисуются тем же приёмом,
// что и в Wayland-бэкенде: геометрию символов даёт stb_easy_font, а заливка
// идёт сплошным цветом через отдельный шейдер (цвет — в constant buffer).
//
// DXGI не знает формата RGB8 (только RGBA8 и шире), поэтому при заливке
// текстуры мы на лету разворачиваем RGB HWC -> RGBA прямо в замапленные
// строки (с учётом RowPitch).

#include "display.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <d3dcompiler.h>

// stb_easy_font — крошечный встроенный битмап-шрифт для оверлеев.
// Header-only: реализация подтягивается одним include’ом и должна жить
// ровно в одной TU — на Windows эта TU и есть единственная (Wayland-файл
// на Windows не компилируется). Глушим предупреждение о неиспользуемых
// static-функциях, которые объявляет stb (MSVC C4505 / GCC -Wunused-function;
// сборка под Windows возможна и MSVC, и MinGW).
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4505)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "stb_easy_font.h"
#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace {

constexpr wchar_t kWndClassName[] = L"iiD3D11Window";

template <class T>
void safe_release(T*& p) {
    if (p) { p->Release(); p = nullptr; }
}

class D3D11Display : public Display {
public:
    ~D3D11Display() override { cleanup(); }

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

private:
    // Трамплин из C-API окна в метод экземпляра. Указатель this кладём в
    // GWLP_USERDATA на WM_NCCREATE и достаём оттуда во всех последующих
    // сообщениях.
    static LRESULT CALLBACK wnd_proc(HWND, UINT, WPARAM, LPARAM);
    LRESULT handle_msg(HWND, UINT, WPARAM, LPARAM);

    bool init_window(int w, int h, const char* title);
    bool init_device();
    bool create_rtv();
    bool init_pipeline();
    bool init_compose();                 // GPU-сборка тайлов: шейдер + стейты
    bool ensure_rt(int w, int h);        // offscreen RGBA8 цель размером canvas
    bool ensure_tile_tex(int idx);       // кольцо текстур тайлов
    bool ensure_texture(int w, int h);
    bool ensure_overlay_vb(size_t bytes);
    void apply_pending_resize();
    void set_viewport(int x, int y, int w, int h);
    void render();
    void render_overlay();
    void render_boxes();
    void draw_solid(const std::vector<float>& verts,
                    float r, float g, float b, float a);
    void compute_viewport(int& vx, int& vy, int& vw, int& vh) const;
    void cleanup();

    ID3DBlob* compile(const char* src, const char* entry, const char* target);

    // ---- Win32 ----
    HWND      hwnd_      = nullptr;
    ATOM      wnd_class_ = 0;
    HINSTANCE hinst_     = nullptr;

    // ---- D3D11 / DXGI ----
    ID3D11Device*           device_ = nullptr;
    ID3D11DeviceContext*    ctx_    = nullptr;
    IDXGISwapChain*         swap_   = nullptr;
    ID3D11RenderTargetView* rtv_    = nullptr;

    // ---- Пайплайн кадра ----
    ID3D11VertexShader*       tex_vs_     = nullptr;
    ID3D11PixelShader*        tex_ps_     = nullptr;
    ID3D11InputLayout*        tex_layout_ = nullptr;
    ID3D11Buffer*             quad_vb_    = nullptr;
    ID3D11Texture2D*          texture_    = nullptr;
    ID3D11ShaderResourceView* srv_        = nullptr;
    ID3D11SamplerState*       sampler_    = nullptr;

    // ---- Пайплайн оверлея/боксов (сплошной цвет) ----
    ID3D11VertexShader* ov_vs_     = nullptr;
    ID3D11PixelShader*  ov_ps_     = nullptr;
    ID3D11InputLayout*  ov_layout_ = nullptr;
    ID3D11Buffer*       ov_cb_     = nullptr;
    ID3D11Buffer*       ov_vb_     = nullptr;
    size_t              ov_vb_cap_ = 0;

    // ---- Состояния ----
    ID3D11BlendState*      blend_  = nullptr;  // альфа-блендинг (текст/боксы)
    ID3D11RasterizerState* raster_ = nullptr;  // solid, cull none

    int  win_w_         = 0;
    int  win_h_         = 0;
    int  tex_w_         = 0;
    int  tex_h_         = 0;
    bool closed_        = false;
    bool vsync_         = true;
    bool resize_pending_ = false;
    int  pending_w_      = 0;
    int  pending_h_      = 0;

    // ---- Оверлей текста / боксы ----
    std::string             overlay_text_;
    std::vector<DisplayBox> boxes_;

    // ---- GPU-сборка тайлов (tile-режим, см. display.h) ----
    // compose-PS деквантует int8/uint8 тайлы и накладывает их (over-composite
    // с растушёвкой по ведущим краям) в offscreen RGBA8 RT размером canvas;
    // затем обычный render() рисует его как кадр. Снимает CPU-decode из
    // горячего пути tile-режима.
    bool tile_supported_ = false;
    ID3D11VertexShader*       cmp_vs_      = nullptr;
    ID3D11PixelShader*        cmp_ps_      = nullptr;
    ID3D11InputLayout*        cmp_layout_  = nullptr;
    ID3D11Buffer*             cmp_vb_      = nullptr;  // динамический квад
    ID3D11Buffer*             cmp_cb_      = nullptr;  // ComposeCB
    ID3D11SamplerState*       cmp_sampler_ = nullptr;  // point clamp

    // Offscreen-цель (RT + SRV) и staging для readback.
    ID3D11Texture2D*          rt_tex_ = nullptr;
    ID3D11RenderTargetView*   rt_rtv_ = nullptr;
    ID3D11ShaderResourceView* rt_srv_ = nullptr;
    int rt_w_ = 0, rt_h_ = 0;
    ID3D11Texture2D*          rb_tex_ = nullptr;       // staging, CPU read
    int rb_w_ = 0, rb_h_ = 0;

    // Кольцо текстур тайлов (по одной на тайл в кадре, переиспользуются).
    std::vector<ID3D11Texture2D*>          tile_texs_;
    std::vector<ID3D11ShaderResourceView*> tile_srvs_;
    int         tile_tex_w_   = 0, tile_tex_h_ = 0;
    int         tile_channels_ = 0;                    // 1 | 3
    DXGI_FORMAT tile_fmt_     = DXGI_FORMAT_R8_UNORM;
    int         tile_add_idx_ = 0;

    ID3D11ShaderResourceView* cur_frame_srv_ = nullptr; // что рисует render()
    TileFrameDesc tile_desc_{};
};

// Layout constant buffer’а оверлея: vec2 экрана + vec2 паддинга + vec4 цвета.
// HLSL требует 16-байтового выравнивания cbuffer’а — 32 байта подходят.
struct OverlayCB {
    float screen[2];
    float pad[2];
    float color[4];
};

// Constant buffer compose-шейдера. 48 байт, выровнено по 16.
struct ComposeCB {
    float canvas[2];     // размер canvas (output-space)
    float tilesz[2];     // размер тайла px
    float scale;         // квант-scale
    float zp;            // zero-point
    float channels;      // 1.0 | 3.0
    float dtype;         // 0=int8 1=uint8
    float range;         // 0=unit 1=signed 2=byte
    float ramp[2];       // растушёвка по левому/верхнему краю
    float _pad;
};

LRESULT CALLBACK D3D11Display::wnd_proc(HWND hwnd, UINT msg,
                                        WPARAM wp, LPARAM lp) {
    if (msg == WM_NCCREATE) {
        auto* cs = reinterpret_cast<CREATESTRUCT*>(lp);
        SetWindowLongPtr(hwnd, GWLP_USERDATA,
                         reinterpret_cast<LONG_PTR>(cs->lpCreateParams));
    }
    auto* self = reinterpret_cast<D3D11Display*>(
        GetWindowLongPtr(hwnd, GWLP_USERDATA));
    if (self) return self->handle_msg(hwnd, msg, wp, lp);
    return DefWindowProcW(hwnd, msg, wp, lp);
}

LRESULT D3D11Display::handle_msg(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_SIZE:
        if (wp != SIZE_MINIMIZED) {
            int w = LOWORD(lp), h = HIWORD(lp);
            if (w > 0 && h > 0 && (w != win_w_ || h != win_h_)) {
                pending_w_ = w;
                pending_h_ = h;
                resize_pending_ = true;
            }
        }
        return 0;
    case WM_CLOSE:
        closed_ = true;
        DestroyWindow(hwnd);
        return 0;
    case WM_DESTROY:
        closed_ = true;
        PostQuitMessage(0);
        return 0;
    default:
        return DefWindowProcW(hwnd, msg, wp, lp);
    }
}

bool D3D11Display::init_window(int w, int h, const char* title) {
    hinst_ = GetModuleHandleW(nullptr);

    // Работаем явно через -W API (Unicode), чтобы не зависеть от того,
    // определён ли макрос UNICODE в сборке хоста.
    WNDCLASSEXW wc = {};
    wc.cbSize        = sizeof(wc);
    wc.style         = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
    wc.lpfnWndProc   = &D3D11Display::wnd_proc;
    wc.hInstance     = hinst_;
    // IDC_ARROW — числовой атом ресурса (одинаков для A/W); приводим к
    // широкому типу, т.к. UNICODE в сборке хоста может быть не определён.
    wc.hCursor       = LoadCursorW(nullptr, reinterpret_cast<LPCWSTR>(IDC_ARROW));
    wc.lpszClassName = kWndClassName;
    wnd_class_ = RegisterClassExW(&wc);
    if (!wnd_class_) {
        std::fprintf(stderr, "D3D11: RegisterClassEx упал.\n");
        return false;
    }

    // Подгоняем размер окна так, чтобы клиентская область была ровно w×h.
    RECT r = { 0, 0, w, h };
    DWORD style = WS_OVERLAPPEDWINDOW;
    AdjustWindowRect(&r, style, FALSE);

    // Заголовок: ANSI -> широкая строка (окно создаём через -W API).
    wchar_t wtitle[256];
    const char* t = (title && *title) ? title : "ii";
    int n = MultiByteToWideChar(CP_UTF8, 0, t, -1, wtitle,
                                (int)(sizeof(wtitle) / sizeof(wtitle[0])));
    if (n <= 0) wtitle[0] = 0;

    hwnd_ = CreateWindowExW(
        0, kWndClassName, wtitle, style,
        CW_USEDEFAULT, CW_USEDEFAULT,
        r.right - r.left, r.bottom - r.top,
        nullptr, nullptr, hinst_, this);
    if (!hwnd_) {
        std::fprintf(stderr, "D3D11: CreateWindowEx упал.\n");
        return false;
    }

    ShowWindow(hwnd_, SW_SHOW);
    UpdateWindow(hwnd_);
    return true;
}

bool D3D11Display::init_device() {
    DXGI_SWAP_CHAIN_DESC sd = {};
    sd.BufferCount        = 2;
    sd.BufferDesc.Width   = win_w_;
    sd.BufferDesc.Height  = win_h_;
    sd.BufferDesc.Format  = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferUsage        = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow       = hwnd_;
    sd.SampleDesc.Count   = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed           = TRUE;
    sd.SwapEffect         = DXGI_SWAP_EFFECT_DISCARD;

    UINT flags = 0;
#ifndef NDEBUG
    // Debug-слой удобен при отладке, но есть не на всех машинах — если
    // создание с ним упадёт, ниже повторим без флага.
    flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    const D3D_FEATURE_LEVEL want[] = {
        D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
    };
    D3D_FEATURE_LEVEL got;

    HRESULT hr = D3D11CreateDeviceAndSwapChain(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags,
        want, (UINT)(sizeof(want) / sizeof(want[0])),
        D3D11_SDK_VERSION, &sd, &swap_, &device_, &got, &ctx_);

    if (FAILED(hr) && (flags & D3D11_CREATE_DEVICE_DEBUG)) {
        // Нет D3D11 SDK Layers — пробуем без debug-слоя.
        flags &= ~(UINT)D3D11_CREATE_DEVICE_DEBUG;
        hr = D3D11CreateDeviceAndSwapChain(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags,
            want, (UINT)(sizeof(want) / sizeof(want[0])),
            D3D11_SDK_VERSION, &sd, &swap_, &device_, &got, &ctx_);
    }
    if (FAILED(hr)) {
        std::fprintf(stderr,
            "D3D11: D3D11CreateDeviceAndSwapChain упал (hr=0x%08lX).\n",
            (unsigned long)hr);
        return false;
    }
    return create_rtv();
}

bool D3D11Display::create_rtv() {
    ID3D11Texture2D* back = nullptr;
    HRESULT hr = swap_->GetBuffer(0, __uuidof(ID3D11Texture2D),
                                  reinterpret_cast<void**>(&back));
    if (FAILED(hr) || !back) {
        std::fprintf(stderr, "D3D11: GetBuffer(backbuffer) упал.\n");
        return false;
    }
    hr = device_->CreateRenderTargetView(back, nullptr, &rtv_);
    back->Release();
    if (FAILED(hr)) {
        std::fprintf(stderr, "D3D11: CreateRenderTargetView упал.\n");
        return false;
    }
    return true;
}

ID3DBlob* D3D11Display::compile(const char* src, const char* entry,
                                const char* target) {
    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifndef NDEBUG
    flags |= D3DCOMPILE_DEBUG;
#endif
    ID3DBlob* code = nullptr;
    ID3DBlob* err  = nullptr;
    HRESULT hr = D3DCompile(src, std::strlen(src), nullptr, nullptr, nullptr,
                            entry, target, flags, 0, &code, &err);
    if (FAILED(hr)) {
        std::fprintf(stderr, "HLSL: compile error (%s): %s\n", entry,
                     err ? static_cast<const char*>(err->GetBufferPointer())
                         : "?");
        safe_release(err);
        return nullptr;
    }
    safe_release(err);
    return code;
}

bool D3D11Display::init_pipeline() {
    // ---- Шейдеры кадра: текстурированный полноэкранный квад ----
    // В D3D строка 0 текстуры соответствует v=0, а NDC y=+1 — это верх
    // экрана, поэтому переворот UV (как в GLES-варианте) НЕ нужен:
    // верхним вершинам даём v=0.
    static const char* kTexHLSL =
        "struct VSIn  { float2 pos : POSITION; float2 uv : TEXCOORD0; };\n"
        "struct VSOut { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; };\n"
        "VSOut vs_main(VSIn i){ VSOut o; o.pos=float4(i.pos,0,1); o.uv=i.uv; return o; }\n"
        "Texture2D   tex : register(t0);\n"
        "SamplerState smp : register(s0);\n"
        "float4 ps_main(VSOut i) : SV_TARGET { return tex.Sample(smp, i.uv); }\n";

    ID3DBlob* vsb = compile(kTexHLSL, "vs_main", "vs_4_0");
    ID3DBlob* psb = compile(kTexHLSL, "ps_main", "ps_4_0");
    if (!vsb || !psb) { safe_release(vsb); safe_release(psb); return false; }

    if (FAILED(device_->CreateVertexShader(vsb->GetBufferPointer(),
            vsb->GetBufferSize(), nullptr, &tex_vs_)) ||
        FAILED(device_->CreatePixelShader(psb->GetBufferPointer(),
            psb->GetBufferSize(), nullptr, &tex_ps_))) {
        safe_release(vsb); safe_release(psb);
        std::fprintf(stderr, "D3D11: создание шейдеров кадра упало.\n");
        return false;
    }

    const D3D11_INPUT_ELEMENT_DESC tex_elems[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0,
          D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 8,
          D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    HRESULT hr = device_->CreateInputLayout(
        tex_elems, 2, vsb->GetBufferPointer(), vsb->GetBufferSize(),
        &tex_layout_);
    safe_release(vsb);
    safe_release(psb);
    if (FAILED(hr)) {
        std::fprintf(stderr, "D3D11: CreateInputLayout(tex) упал.\n");
        return false;
    }

    // Полноэкранный квад (pos.xy + uv). Рисуется TRIANGLESTRIP’ом:
    // TL, TR, BL, BR.
    struct QuadV { float x, y, u, v; };
    static const QuadV quad[4] = {
        { -1.f,  1.f, 0.f, 0.f },  // top-left
        {  1.f,  1.f, 1.f, 0.f },  // top-right
        { -1.f, -1.f, 0.f, 1.f },  // bottom-left
        {  1.f, -1.f, 1.f, 1.f },  // bottom-right
    };
    D3D11_BUFFER_DESC bd = {};
    bd.ByteWidth = sizeof(quad);
    bd.Usage     = D3D11_USAGE_IMMUTABLE;
    bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    D3D11_SUBRESOURCE_DATA init = {};
    init.pSysMem = quad;
    if (FAILED(device_->CreateBuffer(&bd, &init, &quad_vb_))) {
        std::fprintf(stderr, "D3D11: создание quad VB упало.\n");
        return false;
    }

    // Семплер: билинейный, clamp по краям (как GL_LINEAR + CLAMP_TO_EDGE).
    D3D11_SAMPLER_DESC sda = {};
    sda.Filter   = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sda.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    sda.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    sda.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sda.MaxLOD   = D3D11_FLOAT32_MAX;
    if (FAILED(device_->CreateSamplerState(&sda, &sampler_))) {
        std::fprintf(stderr, "D3D11: CreateSamplerState упал.\n");
        return false;
    }

    // ---- Шейдеры оверлея/боксов: сплошной цвет из constant buffer ----
    static const char* kOvHLSL =
        "cbuffer CB : register(b0) { float2 u_screen; float2 _pad; float4 u_color; };\n"
        "float4 vs_main(float2 pos : POSITION) : SV_POSITION {\n"
        "  float2 ndc = (pos / u_screen) * 2.0 - 1.0;\n"
        "  ndc.y = -ndc.y;\n"
        "  return float4(ndc, 0, 1);\n"
        "}\n"
        "float4 ps_main() : SV_TARGET { return u_color; }\n";

    ID3DBlob* ovs = compile(kOvHLSL, "vs_main", "vs_4_0");
    ID3DBlob* ops = compile(kOvHLSL, "ps_main", "ps_4_0");
    if (!ovs || !ops) { safe_release(ovs); safe_release(ops); return false; }

    if (FAILED(device_->CreateVertexShader(ovs->GetBufferPointer(),
            ovs->GetBufferSize(), nullptr, &ov_vs_)) ||
        FAILED(device_->CreatePixelShader(ops->GetBufferPointer(),
            ops->GetBufferSize(), nullptr, &ov_ps_))) {
        safe_release(ovs); safe_release(ops);
        std::fprintf(stderr, "D3D11: создание шейдеров оверлея упало.\n");
        return false;
    }

    const D3D11_INPUT_ELEMENT_DESC ov_elems[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0,
          D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    hr = device_->CreateInputLayout(ov_elems, 1, ovs->GetBufferPointer(),
                                    ovs->GetBufferSize(), &ov_layout_);
    safe_release(ovs);
    safe_release(ops);
    if (FAILED(hr)) {
        std::fprintf(stderr, "D3D11: CreateInputLayout(overlay) упал.\n");
        return false;
    }

    D3D11_BUFFER_DESC cbd = {};
    cbd.ByteWidth      = sizeof(OverlayCB);
    cbd.Usage          = D3D11_USAGE_DEFAULT;
    cbd.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
    if (FAILED(device_->CreateBuffer(&cbd, nullptr, &ov_cb_))) {
        std::fprintf(stderr, "D3D11: создание overlay CB упало.\n");
        return false;
    }

    // Альфа-блендинг для текста/подложек/боксов.
    D3D11_BLEND_DESC bld = {};
    bld.RenderTarget[0].BlendEnable    = TRUE;
    bld.RenderTarget[0].SrcBlend       = D3D11_BLEND_SRC_ALPHA;
    bld.RenderTarget[0].DestBlend      = D3D11_BLEND_INV_SRC_ALPHA;
    bld.RenderTarget[0].BlendOp        = D3D11_BLEND_OP_ADD;
    bld.RenderTarget[0].SrcBlendAlpha  = D3D11_BLEND_ONE;
    bld.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    bld.RenderTarget[0].BlendOpAlpha   = D3D11_BLEND_OP_ADD;
    bld.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    if (FAILED(device_->CreateBlendState(&bld, &blend_))) {
        std::fprintf(stderr, "D3D11: CreateBlendState упал.\n");
        return false;
    }

    // Растеризатор: сплошная заливка, без отсечения граней (квад рисуем
    // без оглядки на winding) и без scissor.
    D3D11_RASTERIZER_DESC rd = {};
    rd.FillMode = D3D11_FILL_SOLID;
    rd.CullMode = D3D11_CULL_NONE;
    if (FAILED(device_->CreateRasterizerState(&rd, &raster_))) {
        std::fprintf(stderr, "D3D11: CreateRasterizerState упал.\n");
        return false;
    }

    return true;
}

bool D3D11Display::ensure_texture(int w, int h) {
    if (texture_ && w == tex_w_ && h == tex_h_) return true;

    safe_release(srv_);
    safe_release(texture_);

    D3D11_TEXTURE2D_DESC td = {};
    td.Width            = w;
    td.Height           = h;
    td.MipLevels        = 1;
    td.ArraySize        = 1;
    td.Format           = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc.Count = 1;
    td.Usage            = D3D11_USAGE_DYNAMIC;
    td.BindFlags        = D3D11_BIND_SHADER_RESOURCE;
    td.CPUAccessFlags   = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(device_->CreateTexture2D(&td, nullptr, &texture_))) {
        std::fprintf(stderr, "D3D11: CreateTexture2D(%dx%d) упал.\n", w, h);
        return false;
    }
    if (FAILED(device_->CreateShaderResourceView(texture_, nullptr, &srv_))) {
        std::fprintf(stderr, "D3D11: CreateShaderResourceView упал.\n");
        return false;
    }
    tex_w_ = w;
    tex_h_ = h;
    return true;
}

bool D3D11Display::ensure_overlay_vb(size_t bytes) {
    if (ov_vb_ && bytes <= ov_vb_cap_) return true;
    safe_release(ov_vb_);
    // Растём с запасом, чтобы не пересоздавать буфер на каждый кадр.
    size_t cap = ov_vb_cap_ ? ov_vb_cap_ : 4096;
    while (cap < bytes) cap *= 2;
    D3D11_BUFFER_DESC bd = {};
    bd.ByteWidth      = (UINT)cap;
    bd.Usage          = D3D11_USAGE_DYNAMIC;
    bd.BindFlags      = D3D11_BIND_VERTEX_BUFFER;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(device_->CreateBuffer(&bd, nullptr, &ov_vb_))) {
        std::fprintf(stderr, "D3D11: создание overlay VB упало.\n");
        ov_vb_cap_ = 0;
        return false;
    }
    ov_vb_cap_ = cap;
    return true;
}

void D3D11Display::apply_pending_resize() {
    if (!resize_pending_) return;
    resize_pending_ = false;
    win_w_ = pending_w_;
    win_h_ = pending_h_;

    // Перед ResizeBuffers нужно отпустить все ссылки на back buffer.
    safe_release(rtv_);
    HRESULT hr = swap_->ResizeBuffers(0, win_w_, win_h_,
                                      DXGI_FORMAT_UNKNOWN, 0);
    if (FAILED(hr)) {
        std::fprintf(stderr, "D3D11: ResizeBuffers упал (hr=0x%08lX).\n",
                     (unsigned long)hr);
        closed_ = true;
        return;
    }
    if (!create_rtv()) closed_ = true;
}

void D3D11Display::set_viewport(int x, int y, int w, int h) {
    D3D11_VIEWPORT vp = {};
    vp.TopLeftX = (FLOAT)x;
    vp.TopLeftY = (FLOAT)y;
    vp.Width    = (FLOAT)w;
    vp.Height   = (FLOAT)h;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    ctx_->RSSetViewports(1, &vp);
}

bool D3D11Display::init(int w, int h, const char* title, bool vsync) {
    win_w_ = w;
    win_h_ = h;
    vsync_ = vsync;

    if (!init_window(w, h, title)) return false;

    // Реальная клиентская область может слегка отличаться от запрошенной
    // (DPI/тема) — берём фактический размер под swapchain.
    RECT rc = {};
    GetClientRect(hwnd_, &rc);
    win_w_ = std::max<int>(1, rc.right - rc.left);
    win_h_ = std::max<int>(1, rc.bottom - rc.top);

    if (!init_device())   return false;
    if (!init_pipeline()) return false;
    // GPU-сборка тайлов — опциональна: при неудаче tile_supported_ = false
    // и раннер использует CPU-путь.
    init_compose();

    // Пустой первый кадр, чтобы окно не висело чёрным до первого show_rgb.
    ctx_->OMSetRenderTargets(1, &rtv_, nullptr);
    const float black[4] = { 0.f, 0.f, 0.f, 1.f };
    ctx_->ClearRenderTargetView(rtv_, black);
    swap_->Present(vsync_ ? 1 : 0, 0);
    return true;
}

bool D3D11Display::show_rgb(const uint8_t* rgb, int w, int h) {
    if (closed_) return false;
    if (!ensure_texture(w, h)) return false;

    // Заливаем текстуру: RGB HWC -> RGBA построчно (RowPitch может быть
    // больше w*4 из-за выравнивания драйвера).
    D3D11_MAPPED_SUBRESOURCE m = {};
    if (FAILED(ctx_->Map(texture_, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
        std::fprintf(stderr, "D3D11: Map(texture) упал.\n");
        return false;
    }
    auto* dst = static_cast<uint8_t*>(m.pData);
    for (int y = 0; y < h; ++y) {
        const uint8_t* src = rgb + (size_t)y * w * 3;
        uint8_t*       row = dst + (size_t)y * m.RowPitch;
        for (int x = 0; x < w; ++x) {
            row[x * 4 + 0] = src[x * 3 + 0];
            row[x * 4 + 1] = src[x * 3 + 1];
            row[x * 4 + 2] = src[x * 3 + 2];
            row[x * 4 + 3] = 255;
        }
    }
    ctx_->Unmap(texture_, 0);

    render();
    poll();
    return !closed_;
}

void D3D11Display::compute_viewport(int& vx, int& vy,
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

void D3D11Display::render() {
    apply_pending_resize();
    if (closed_ || !rtv_) return;

    ctx_->OMSetRenderTargets(1, &rtv_, nullptr);
    const float black[4] = { 0.f, 0.f, 0.f, 1.f };
    ctx_->ClearRenderTargetView(rtv_, black);  // полосы letterbox’а
    ctx_->RSSetState(raster_);

    // ---- Кадр: текстура в letterbox-viewport ----
    int vx, vy, vw, vh;
    compute_viewport(vx, vy, vw, vh);
    set_viewport(vx, vy, vw, vh);

    UINT stride = 4 * sizeof(float), offset = 0;
    ctx_->IASetInputLayout(tex_layout_);
    ctx_->IASetVertexBuffers(0, 1, &quad_vb_, &stride, &offset);
    ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    ctx_->VSSetShader(tex_vs_, nullptr, 0);
    ctx_->PSSetShader(tex_ps_, nullptr, 0);
    // cur_frame_srv_ != null — рисуем собранную compose-цель (tile-режим),
    // иначе обычную залитую с CPU текстуру кадра.
    ID3D11ShaderResourceView* frame_srv = cur_frame_srv_ ? cur_frame_srv_ : srv_;
    ctx_->PSSetShaderResources(0, 1, &frame_srv);
    ctx_->PSSetSamplers(0, 1, &sampler_);
    ctx_->OMSetBlendState(nullptr, nullptr, 0xffffffff);  // непрозрачно
    ctx_->Draw(4, 0);

    // ---- Боксы (в image-координатах -> окно), затем текстовый оверлей ----
    render_boxes();
    render_overlay();

    swap_->Present(vsync_ ? 1 : 0, 0);
}

void D3D11Display::draw_solid(const std::vector<float>& verts,
                              float r, float g, float b, float a) {
    if (verts.empty()) return;
    size_t bytes = verts.size() * sizeof(float);
    if (!ensure_overlay_vb(bytes)) return;

    D3D11_MAPPED_SUBRESOURCE m = {};
    if (FAILED(ctx_->Map(ov_vb_, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) return;
    std::memcpy(m.pData, verts.data(), bytes);
    ctx_->Unmap(ov_vb_, 0);

    OverlayCB cb = {};
    cb.screen[0] = (float)win_w_;
    cb.screen[1] = (float)win_h_;
    cb.color[0] = r; cb.color[1] = g; cb.color[2] = b; cb.color[3] = a;
    ctx_->UpdateSubresource(ov_cb_, 0, nullptr, &cb, 0, 0);

    UINT stride = 2 * sizeof(float), offset = 0;
    ctx_->IASetInputLayout(ov_layout_);
    ctx_->IASetVertexBuffers(0, 1, &ov_vb_, &stride, &offset);
    ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx_->VSSetShader(ov_vs_, nullptr, 0);
    ctx_->PSSetShader(ov_ps_, nullptr, 0);
    ctx_->VSSetConstantBuffers(0, 1, &ov_cb_);
    ctx_->PSSetConstantBuffers(0, 1, &ov_cb_);
    ctx_->OMSetBlendState(blend_, nullptr, 0xffffffff);
    ctx_->Draw((UINT)(verts.size() / 2), 0);
}

void D3D11Display::render_overlay() {
    if (overlay_text_.empty()) return;

    // Геометрия текста через stb_easy_font (как в Wayland-бэкенде).
    static char raw[80 * 1024];
    char* text = const_cast<char*>(overlay_text_.c_str());
    int num_quads = stb_easy_font_print(0.0f, 0.0f, text, nullptr,
                                        raw, sizeof(raw));
    int tw = stb_easy_font_width(text);
    int th = stb_easy_font_height(text);

    constexpr float kPad = 4.0f;
    constexpr float kMargin = 8.0f;
    constexpr float kScale = 2.0f;

    // Оверлей рисуется в экранных (пиксельных) координатах — viewport на
    // всё окно.
    set_viewport(0, 0, win_w_, win_h_);

    // Подложка — полупрозрачный чёрный прямоугольник под текстом.
    float bw = tw * kScale + kPad * 2.0f;
    float bh = th * kScale + kPad * 2.0f;
    std::vector<float> bg = {
        kMargin,      kMargin,
        kMargin + bw, kMargin,
        kMargin,      kMargin + bh,
        kMargin + bw, kMargin,
        kMargin + bw, kMargin + bh,
        kMargin,      kMargin + bh,
    };
    draw_solid(bg, 0.0f, 0.0f, 0.0f, 0.55f);

    // Сам текст: quads -> triangles, ярко-жёлтым.
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
    draw_solid(tri, 1.0f, 1.0f, 0.2f, 1.0f);
}

void D3D11Display::render_boxes() {
    if (boxes_.empty() || tex_w_ <= 0 || tex_h_ <= 0) return;

    int vx, vy, vw, vh;
    compute_viewport(vx, vy, vw, vh);

    const float sx = (float)vw / tex_w_;
    const float sy = (float)vh / tex_h_;
    auto map_x = [&](float x) { return vx + x * sx; };
    auto map_y = [&](float y) { return vy + y * sy; };

    // Толщина рамки масштабируется от размера окна (на 4K не «волосок»,
    // на маленьком окне не «клякса»).
    const float t = std::clamp(std::min(win_w_, win_h_) / 400.0f, 1.5f, 4.0f);

    set_viewport(0, 0, win_w_, win_h_);

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
        draw_solid(verts, b.r / 255.0f, b.g / 255.0f, b.b / 255.0f, 1.0f);

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
        // переносим внутрь рамки.
        float bx = x1;
        float by = y1 - bh;
        if (by < 0.0f) by = y1;

        // Подложка цвета рамки.
        std::vector<float> bg = {
            bx,      by,
            bx + bw, by,
            bx,      by + bh,
            bx + bw, by,
            bx + bw, by + bh,
            bx,      by + bh,
        };
        draw_solid(bg, b.r / 255.0f, b.g / 255.0f, b.b / 255.0f, 0.85f);

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
        draw_solid(tri, 1.0f, 1.0f, 1.0f, 1.0f);
    }
}

// ---- GPU-сборка кадра из тайлов -----------------------------------------

bool D3D11Display::init_compose() {
    // VS: позиция тайла в canvas-пикселях → NDC. ndc.y = -((y/h)*2-1):
    // canvas_y=0 (верх кадра) → ndc.y=+1 → верх RT (row0, D3D top-left).
    // Так RT row0 = верх кадра — совпадает с CPU-текстурой и readback’ом.
    // FS: восстанавливаем точный байт из R8_UNORM (полный fp32 в HLSL) и
    // деквантуем как decode_kernel в image_proc.cpp.
    static const char* kHLSL =
        "cbuffer CB : register(b0) {\n"
        "  float2 u_canvas; float2 u_tilesz;\n"
        "  float u_scale; float u_zp; float u_channels; float u_dtype;\n"
        "  float u_range; float2 u_ramp; float _pad;\n"
        "};\n"
        "struct VSIn  { float2 canvas : POSITION; float2 uv : TEXCOORD0; };\n"
        "struct VSOut { float4 pos : SV_POSITION; float2 uv : TEXCOORD0;\n"
        "               float2 local : TEXCOORD1; };\n"
        "VSOut vs_main(VSIn i){\n"
        "  VSOut o;\n"
        "  float2 ndc = (i.canvas / u_canvas) * 2.0 - 1.0;\n"
        "  ndc.y = -ndc.y;\n"
        "  o.pos = float4(ndc, 0, 1);\n"
        "  o.uv = i.uv;\n"
        "  o.local = i.uv * u_tilesz;\n"
        "  return o;\n"
        "}\n"
        "Texture2D    tex : register(t0);\n"
        "SamplerState smp : register(s0);\n"
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
        "float4 ps_main(VSOut i) : SV_TARGET {\n"
        "  float3 rgb;\n"
        "  if (u_channels < 1.5) {\n"
        "    float y = deq(tex.Sample(smp, i.uv).r);\n"
        "    rgb = float3(y, y, y);\n"
        "  } else {\n"
        "    float3 c = tex.Sample(smp, i.uv).rgb;\n"
        "    rgb = float3(deq(c.r), deq(c.g), deq(c.b));\n"
        "  }\n"
        "  float a = lead_alpha(i.local.x, u_ramp.x)\n"
        "          * lead_alpha(i.local.y, u_ramp.y);\n"
        "  return float4(rgb, a);\n"
        "}\n";

    ID3DBlob* vsb = compile(kHLSL, "vs_main", "vs_4_0");
    ID3DBlob* psb = compile(kHLSL, "ps_main", "ps_4_0");
    if (!vsb || !psb) { safe_release(vsb); safe_release(psb); return false; }
    if (FAILED(device_->CreateVertexShader(vsb->GetBufferPointer(),
            vsb->GetBufferSize(), nullptr, &cmp_vs_)) ||
        FAILED(device_->CreatePixelShader(psb->GetBufferPointer(),
            psb->GetBufferSize(), nullptr, &cmp_ps_))) {
        safe_release(vsb); safe_release(psb);
        std::fprintf(stderr, "D3D11: создание compose-шейдеров упало.\n");
        return false;
    }
    const D3D11_INPUT_ELEMENT_DESC elems[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0,
          D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 8,
          D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    HRESULT hr = device_->CreateInputLayout(elems, 2, vsb->GetBufferPointer(),
                                            vsb->GetBufferSize(), &cmp_layout_);
    safe_release(vsb);
    safe_release(psb);
    if (FAILED(hr)) {
        std::fprintf(stderr, "D3D11: CreateInputLayout(compose) упал.\n");
        return false;
    }

    // Динамический VB под квад (4 вершины × float4).
    D3D11_BUFFER_DESC vbd = {};
    vbd.ByteWidth      = 4 * 4 * sizeof(float);
    vbd.Usage          = D3D11_USAGE_DYNAMIC;
    vbd.BindFlags      = D3D11_BIND_VERTEX_BUFFER;
    vbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(device_->CreateBuffer(&vbd, nullptr, &cmp_vb_))) {
        std::fprintf(stderr, "D3D11: создание compose VB упало.\n");
        return false;
    }
    D3D11_BUFFER_DESC cbd = {};
    cbd.ByteWidth = sizeof(ComposeCB);
    cbd.Usage     = D3D11_USAGE_DEFAULT;
    cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    if (FAILED(device_->CreateBuffer(&cbd, nullptr, &cmp_cb_))) {
        std::fprintf(stderr, "D3D11: создание compose CB упало.\n");
        return false;
    }
    // Point clamp: квад 1:1 с текселями тайла, точный семпл для восстановления.
    D3D11_SAMPLER_DESC sda = {};
    sda.Filter   = D3D11_FILTER_MIN_MAG_MIP_POINT;
    sda.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    sda.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    sda.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sda.MaxLOD   = D3D11_FLOAT32_MAX;
    if (FAILED(device_->CreateSamplerState(&sda, &cmp_sampler_))) {
        std::fprintf(stderr, "D3D11: CreateSamplerState(compose) упал.\n");
        return false;
    }
    tile_supported_ = true;
    return true;
}

bool D3D11Display::ensure_rt(int w, int h) {
    if (rt_tex_ && w == rt_w_ && h == rt_h_) return true;
    safe_release(rt_srv_);
    safe_release(rt_rtv_);
    safe_release(rt_tex_);

    D3D11_TEXTURE2D_DESC td = {};
    td.Width            = w;
    td.Height           = h;
    td.MipLevels        = 1;
    td.ArraySize        = 1;
    td.Format           = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc.Count = 1;
    td.Usage            = D3D11_USAGE_DEFAULT;
    td.BindFlags        = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(device_->CreateTexture2D(&td, nullptr, &rt_tex_))) {
        std::fprintf(stderr, "D3D11: CreateTexture2D(RT %dx%d) упал.\n", w, h);
        tile_supported_ = false;
        return false;
    }
    if (FAILED(device_->CreateRenderTargetView(rt_tex_, nullptr, &rt_rtv_)) ||
        FAILED(device_->CreateShaderResourceView(rt_tex_, nullptr, &rt_srv_))) {
        std::fprintf(stderr, "D3D11: RTV/SRV(RT) упал.\n");
        tile_supported_ = false;
        return false;
    }
    rt_w_ = w;
    rt_h_ = h;
    return true;
}

bool D3D11Display::ensure_tile_tex(int idx) {
    while ((int)tile_texs_.size() <= idx) {
        D3D11_TEXTURE2D_DESC td = {};
        td.Width            = tile_tex_w_;
        td.Height           = tile_tex_h_;
        td.MipLevels        = 1;
        td.ArraySize        = 1;
        td.Format           = tile_fmt_;
        td.SampleDesc.Count = 1;
        td.Usage            = D3D11_USAGE_DYNAMIC;
        td.BindFlags        = D3D11_BIND_SHADER_RESOURCE;
        td.CPUAccessFlags   = D3D11_CPU_ACCESS_WRITE;
        ID3D11Texture2D* t = nullptr;
        if (FAILED(device_->CreateTexture2D(&td, nullptr, &t))) {
            std::fprintf(stderr, "D3D11: CreateTexture2D(tile) упал.\n");
            tile_supported_ = false;
            return false;
        }
        ID3D11ShaderResourceView* s = nullptr;
        if (FAILED(device_->CreateShaderResourceView(t, nullptr, &s))) {
            std::fprintf(stderr, "D3D11: SRV(tile) упал.\n");
            safe_release(t);
            tile_supported_ = false;
            return false;
        }
        tile_texs_.push_back(t);
        tile_srvs_.push_back(s);
    }
    return true;
}

bool D3D11Display::tile_frame_begin(const TileFrameDesc& d) {
    if (!tile_supported_ || closed_) return false;
    tile_desc_ = d;

    const DXGI_FORMAT fmt = (d.channels == 1)
        ? DXGI_FORMAT_R8_UNORM : DXGI_FORMAT_R8G8B8A8_UNORM;
    if (fmt != tile_fmt_ || d.tile_w != tile_tex_w_ || d.tile_h != tile_tex_h_) {
        for (auto* s : tile_srvs_) safe_release(s);
        for (auto* t : tile_texs_) safe_release(t);
        tile_srvs_.clear();
        tile_texs_.clear();
        tile_fmt_      = fmt;
        tile_tex_w_    = d.tile_w;
        tile_tex_h_    = d.tile_h;
        tile_channels_ = d.channels;
    }
    if (!ensure_rt(d.canvas_w, d.canvas_h)) return false;
    tile_add_idx_ = 0;

    // Очищаем offscreen RT и настраиваем стейт под наложение тайлов.
    const float black[4] = { 0.f, 0.f, 0.f, 1.f };
    ctx_->OMSetRenderTargets(1, &rt_rtv_, nullptr);
    ctx_->ClearRenderTargetView(rt_rtv_, black);
    set_viewport(0, 0, rt_w_, rt_h_);
    ctx_->RSSetState(raster_);

    ComposeCB cb = {};
    cb.canvas[0] = (float)rt_w_;  cb.canvas[1] = (float)rt_h_;
    cb.tilesz[0] = (float)d.tile_w; cb.tilesz[1] = (float)d.tile_h;
    cb.scale    = d.scale;
    cb.zp       = (float)d.zero_point;
    cb.channels = (float)d.channels;
    cb.dtype    = (float)d.dtype;
    cb.range    = (float)d.range;
    ctx_->UpdateSubresource(cmp_cb_, 0, nullptr, &cb, 0, 0);

    ctx_->IASetInputLayout(cmp_layout_);
    ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    ctx_->VSSetShader(cmp_vs_, nullptr, 0);
    ctx_->PSSetShader(cmp_ps_, nullptr, 0);
    ctx_->VSSetConstantBuffers(0, 1, &cmp_cb_);
    ctx_->PSSetConstantBuffers(0, 1, &cmp_cb_);
    ctx_->PSSetSamplers(0, 1, &cmp_sampler_);
    ctx_->OMSetBlendState(blend_, nullptr, 0xffffffff);  // SRC_ALPHA/INV
    return true;
}

void D3D11Display::tile_frame_add(const void* raw, int dst_x, int dst_y,
                                  int ramp_x, int ramp_y) {
    if (!tile_supported_ || !raw) return;
    if (!ensure_tile_tex(tile_add_idx_)) return;
    const int idx = tile_add_idx_++;
    ID3D11Texture2D* tex = tile_texs_[idx];

    // Заливаем тайл в DYNAMIC-текстуру построчно (RowPitch ≥ ширина).
    D3D11_MAPPED_SUBRESOURCE m = {};
    if (FAILED(ctx_->Map(tex, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) return;
    const auto* src = static_cast<const uint8_t*>(raw);
    auto* dst = static_cast<uint8_t*>(m.pData);
    if (tile_channels_ == 1) {
        for (int y = 0; y < tile_tex_h_; ++y) {
            std::memcpy(dst + (size_t)y * m.RowPitch,
                        src + (size_t)y * tile_tex_w_,
                        (size_t)tile_tex_w_);
        }
    } else {
        // RGB → RGBA (DXGI не знает 3-байтных форматов).
        for (int y = 0; y < tile_tex_h_; ++y) {
            const uint8_t* sp = src + (size_t)y * tile_tex_w_ * 3;
            uint8_t*       dp = dst + (size_t)y * m.RowPitch;
            for (int x = 0; x < tile_tex_w_; ++x) {
                dp[x*4+0] = sp[x*3+0];
                dp[x*4+1] = sp[x*3+1];
                dp[x*4+2] = sp[x*3+2];
                dp[x*4+3] = 255;
            }
        }
    }
    ctx_->Unmap(tex, 0);

    // Квад в canvas-координатах + uv по тайлу (uv.y=0 — верх тайла).
    const float x0 = (float)dst_x, y0 = (float)dst_y;
    const float x1 = (float)(dst_x + tile_tex_w_);
    const float y1 = (float)(dst_y + tile_tex_h_);
    const float verts[16] = {
        x0, y0,  0.f, 0.f,   // TL
        x1, y0,  1.f, 0.f,   // TR
        x0, y1,  0.f, 1.f,   // BL
        x1, y1,  1.f, 1.f,   // BR
    };
    D3D11_MAPPED_SUBRESOURCE mv = {};
    if (FAILED(ctx_->Map(cmp_vb_, 0, D3D11_MAP_WRITE_DISCARD, 0, &mv))) return;
    std::memcpy(mv.pData, verts, sizeof(verts));
    ctx_->Unmap(cmp_vb_, 0);

    // Обновляем только ramp в CB (остальное стабильно за кадр).
    ComposeCB cb = {};
    cb.canvas[0] = (float)rt_w_;       cb.canvas[1] = (float)rt_h_;
    cb.tilesz[0] = (float)tile_tex_w_; cb.tilesz[1] = (float)tile_tex_h_;
    cb.scale    = tile_desc_.scale;
    cb.zp       = (float)tile_desc_.zero_point;
    cb.channels = (float)tile_desc_.channels;
    cb.dtype    = (float)tile_desc_.dtype;
    cb.range    = (float)tile_desc_.range;
    cb.ramp[0]  = (float)ramp_x; cb.ramp[1] = (float)ramp_y;
    ctx_->UpdateSubresource(cmp_cb_, 0, nullptr, &cb, 0, 0);

    UINT stride = 4 * sizeof(float), offset = 0;
    ctx_->IASetVertexBuffers(0, 1, &cmp_vb_, &stride, &offset);
    ctx_->PSSetShaderResources(0, 1, &tile_srvs_[idx]);
    ctx_->Draw(4, 0);
}

bool D3D11Display::tile_frame_present() {
    if (closed_) return false;
    if (!tile_supported_ || !rt_srv_) return false;
    // Отвязываем RT-текстуру с выхода (она станет входом для кадрового draw).
    ID3D11ShaderResourceView* none = nullptr;
    ctx_->PSSetShaderResources(0, 1, &none);
    tex_w_ = rt_w_;
    tex_h_ = rt_h_;
    cur_frame_srv_ = rt_srv_;
    render();           // letterbox + боксы + оверлей + Present
    cur_frame_srv_ = nullptr;
    poll();
    return !closed_;
}

bool D3D11Display::tile_frame_readback(std::vector<uint8_t>& rgb,
                                       int& w, int& h) {
    if (!tile_supported_ || !rt_tex_) return false;
    w = rt_w_;
    h = rt_h_;
    if (!rb_tex_ || rb_w_ != w || rb_h_ != h) {
        safe_release(rb_tex_);
        D3D11_TEXTURE2D_DESC td = {};
        td.Width            = w;
        td.Height           = h;
        td.MipLevels        = 1;
        td.ArraySize        = 1;
        td.Format           = DXGI_FORMAT_R8G8B8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage            = D3D11_USAGE_STAGING;
        td.CPUAccessFlags   = D3D11_CPU_ACCESS_READ;
        if (FAILED(device_->CreateTexture2D(&td, nullptr, &rb_tex_))) {
            std::fprintf(stderr, "D3D11: CreateTexture2D(staging) упал.\n");
            return false;
        }
        rb_w_ = w;
        rb_h_ = h;
    }
    ctx_->CopyResource(rb_tex_, rt_tex_);
    D3D11_MAPPED_SUBRESOURCE m = {};
    if (FAILED(ctx_->Map(rb_tex_, 0, D3D11_MAP_READ, 0, &m))) return false;
    // RT row0 = верх кадра (D3D top-left), staging повторяет → переворот не
    // нужен. RGBA → RGB.
    rgb.resize((size_t)w * h * 3);
    const auto* src = static_cast<const uint8_t*>(m.pData);
    for (int y = 0; y < h; ++y) {
        const uint8_t* sp = src + (size_t)y * m.RowPitch;
        uint8_t*       dp = rgb.data() + (size_t)y * w * 3;
        for (int x = 0; x < w; ++x) {
            dp[x*3+0] = sp[x*4+0];
            dp[x*3+1] = sp[x*4+1];
            dp[x*3+2] = sp[x*4+2];
        }
    }
    ctx_->Unmap(rb_tex_, 0);
    return true;
}

bool D3D11Display::poll() {
    if (closed_) return false;
    MSG msg;
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
        if (closed_) return false;
    }
    return !closed_;
}

bool D3D11Display::wait() {
    if (closed_) return false;
    MSG msg;
    BOOL r = GetMessageW(&msg, nullptr, 0, 0);
    if (r <= 0) {  // WM_QUIT (0) или ошибка (-1)
        closed_ = true;
        return false;
    }
    TranslateMessage(&msg);
    DispatchMessageW(&msg);
    return !closed_;
}

void D3D11Display::cleanup() {
    if (ctx_) ctx_->ClearState();

    // ---- GPU-сборка тайлов ----
    for (auto* s : tile_srvs_) safe_release(s);
    for (auto* t : tile_texs_) safe_release(t);
    tile_srvs_.clear();
    tile_texs_.clear();
    safe_release(rb_tex_);
    safe_release(rt_srv_);
    safe_release(rt_rtv_);
    safe_release(rt_tex_);
    safe_release(cmp_sampler_);
    safe_release(cmp_cb_);
    safe_release(cmp_vb_);
    safe_release(cmp_layout_);
    safe_release(cmp_ps_);
    safe_release(cmp_vs_);

    safe_release(raster_);
    safe_release(blend_);
    safe_release(ov_vb_);
    safe_release(ov_cb_);
    safe_release(ov_layout_);
    safe_release(ov_ps_);
    safe_release(ov_vs_);

    safe_release(sampler_);
    safe_release(srv_);
    safe_release(texture_);
    safe_release(quad_vb_);
    safe_release(tex_layout_);
    safe_release(tex_ps_);
    safe_release(tex_vs_);

    safe_release(rtv_);
    safe_release(swap_);
    safe_release(ctx_);
    safe_release(device_);

    if (hwnd_) { DestroyWindow(hwnd_); hwnd_ = nullptr; }
    if (wnd_class_) {
        UnregisterClassW(kWndClassName, hinst_);
        wnd_class_ = 0;
    }
}

}  // namespace

std::unique_ptr<Display> make_display() {
    return std::make_unique<D3D11Display>();
}
