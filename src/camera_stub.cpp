// Заглушка модуля Camera для платформ без V4L2 (Windows/macOS) или
// при сборке с -DUSE_CAMERA=OFF. Возвращает nullptr — клиентский код
// (ii.cpp) увидит это и сообщит «поддержка камеры не собрана».

#include "camera.h"

std::unique_ptr<Camera> make_camera() { return nullptr; }
