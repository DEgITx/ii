// Заглушка Display для сборки без поддержки графического вывода
// (USE_DISPLAY=OFF). make_display() вернёт nullptr — клиентский код
// обязан это проверять и корректно работать без визуализации.

#include "display.h"

std::unique_ptr<Display> make_display() {
    return nullptr;
}

// Заглушки методов не нужны — без USE_DISPLAY ни один Display
// не создаётся, а значит set_overlay_text/etc. никогда не вызываются.
