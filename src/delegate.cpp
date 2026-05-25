// Выбор внешнего делегата ускорения TFLite — платформенно-нейтральная
// часть. Сам раннер не привязан ни к одному конкретному ускорителю или
// делегату: путь к плагину задаётся пользователем через --delegate, и по
// умолчанию делегат не используется (CPU / выбранный бэкенд).
//
// Опциональные модули делегатов (по одному на конкретный ускоритель)
// могут подставлять платформенный дефолтный путь. Такой модуль
// подключается отдельным .cpp по опции CMake и виден остальному коду
// только через module_default_delegate_path() ниже. Если ни один модуль
// не собран, дефолт пуст — это и есть желаемое поведение «из коробки».

#include "inference.h"

namespace inf {

#if defined(INF_HAS_DELEGATE_MODULE)
// Определяется опциональным модулем делегата (см., например,
// delegate_neutron.cpp). Линкуется только когда соответствующая опция
// CMake включена.
const char* module_default_delegate_path();
#endif

const char* default_delegate_path() {
#if defined(INF_HAS_DELEGATE_MODULE)
    return module_default_delegate_path();
#else
    return "";
#endif
}

}  // namespace inf
