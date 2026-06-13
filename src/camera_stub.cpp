// Заглушка модуля Camera для платформ без реального backend'а захвата
// (например macOS) или при сборке с -DUSE_CAMERA=OFF. На Linux backend —
// camera_v4l2.cpp, на Windows — camera_mediafoundation.cpp. Возвращает
// nullptr — клиентский код (ii.cpp) увидит это и сообщит «поддержка
// камеры не собрана».

#include "camera.h"

std::unique_ptr<Camera> make_camera() { return nullptr; }
