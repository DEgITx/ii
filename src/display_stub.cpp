// Заглушка Display для сборки без поддержки графического вывода
// (USE_DISPLAY=OFF). make_display() вернёт nullptr — клиентский код
// обязан это проверять и корректно работать без визуализации.

#include "display.h"

std::unique_ptr<Display> make_display() {
    return nullptr;
}
