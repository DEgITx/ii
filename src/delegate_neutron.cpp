// Опциональный модуль внешнего NPU-делегата (TFLite external delegate).
//
// Пример аппаратного делегата: модуль лишь сообщает раннеру платформенный
// дефолтный путь к .so-плагину делегата на Linux, чтобы его не нужно было
// каждый раз указывать через --delegate. Ядро раннера про этот модуль
// ничего не знает — он подключается опцией CMake (USE_NEUTRON_DELEGATE) и
// виден остальному коду только через module_default_delegate_path()
// (см. delegate.cpp). На любой другой платформе путь пуст.
//
// Сам плагин делегата (libneutron_delegate.so) в репозиторий не входит и
// поставляется отдельно вместе с драйверами/BSP целевой платформы.

namespace inf {

const char* module_default_delegate_path() {
#if defined(__linux__)
    return "/lib/libneutron_delegate.so";
#else
    return "";
#endif
}

}  // namespace inf
