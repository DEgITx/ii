// Мелкие хелперы ii::Tensor, вынесенные из заголовка (не инлайнятся в
// горячих циклах, поэтому держать их в .cpp дешевле для времени сборки).

#include "engine/tensor.h"

namespace ii {

std::string shape_to_string(const Shape& s) {
    std::string out = "[";
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (i) out += ",";
        out += std::to_string(s[i]);
    }
    out += "]";
    return out;
}

std::vector<std::int64_t> row_major_strides(const Shape& s) {
    std::vector<std::int64_t> st(s.size(), 1);
    for (int i = static_cast<int>(s.size()) - 2; i >= 0; --i) {
        st[i] = st[i + 1] * s[i + 1];
    }
    return st;
}

}  // namespace ii
