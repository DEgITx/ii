// IEEE 754 binary16 -> binary32 — единственная реализация на весь проект.
//
// Раньше эта функция жила инлайном в inference.h и дублировалась в загрузчике
// ONNX движка. Вынесена в отдельный заголовок без зависимостей, чтобы её
// могли разделять и backend-контракт (inference.h), и раннер (tensor_utils,
// image_proc), и встроенный движок `ii` (engine/onnx.cpp), не таща друг друга.
//
// Имя/пространство имён сохранены (inf::half_to_float) ради совместимости —
// это часть публичного набора числовых помощников inference.h.

#pragma once

#include <cstdint>
#include <cstring>

namespace inf {

// Без зависимости от __fp16 / F16C — работает на любом ARMv8 / x86-64 / MSVC.
// Header-inline: вызывается в горячих циклах деквантования.
inline float half_to_float(std::uint16_t h) {
    const std::uint32_t sign = (std::uint32_t)(h >> 15) & 0x1u;
    const std::uint32_t exp  = (std::uint32_t)(h >> 10) & 0x1Fu;
    std::uint32_t       mant = (std::uint32_t)h & 0x3FFu;
    std::uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign << 31;
        } else {
            int e = -1;
            do { ++e; mant <<= 1; } while ((mant & 0x400u) == 0);
            mant &= 0x3FFu;
            bits = (sign << 31) | ((127u - 15u - (std::uint32_t)e) << 23)
                 | (mant << 13);
        }
    } else if (exp == 31) {
        bits = (sign << 31) | (0xFFu << 23) | (mant << 13);
    } else {
        bits = (sign << 31) | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

}  // namespace inf
