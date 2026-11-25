// SPDX-License-Identifier: MIT
#include "core/compress.h"

#include <array>

namespace ii {

namespace {

constexpr int kMaxBits = 15;  // longest Huffman code
constexpr int kMaxLiteralCodes = 286;
constexpr int kMaxDistanceCodes = 30;
constexpr int kFixedLiteralCodes = 288;

struct Huffman {
    std::array<std::uint16_t, kMaxBits + 1> count{};         // codes per length
    std::array<std::uint16_t, kFixedLiteralCodes> symbol{};  // symbols ordered by code
};

class Inflater {
public:
    Inflater(std::span<const std::byte> in, std::size_t max_output) : in_(in), max_output_(max_output) {}

    Status run(std::vector<std::byte>& out) {
        out_ = &out;
        bool last = false;
        while (!last) {
            last = bits(1) != 0;
            const int type = bits(2);
            if (!status_.ok()) {
                return status_;
            }
            switch (type) {
            case 0:
                II_TRY(stored());
                break;
            case 1:
                II_TRY(fixed());
                break;
            case 2:
                II_TRY(dynamic());
                break;
            default:
                return errors::data_corrupted("deflate: invalid block type");
            }
        }
        return status_;
    }

private:
    int bits(int need) {
        std::uint32_t value = bit_buffer_;
        while (bit_count_ < need) {
            if (pos_ >= in_.size()) {
                if (status_.ok()) {
                    status_ = errors::data_corrupted("deflate: unexpected end of data");
                }
                return 0;
            }
            value |= static_cast<std::uint32_t>(in_[pos_++]) << bit_count_;
            bit_count_ += 8;
        }
        bit_buffer_ = value >> need;
        bit_count_ -= need;
        return static_cast<int>(value & ((1u << need) - 1));
    }

    Status emit(std::byte value) {
        if (out_->size() >= max_output_) {
            return errors::data_corrupted("deflate: output exceeds {} bytes", max_output_);
        }
        out_->push_back(value);
        return ok_status();
    }

    Status stored() {
        bit_buffer_ = 0;
        bit_count_ = 0;
        if (in_.size() - pos_ < 4) {
            return errors::data_corrupted("deflate: truncated stored block");
        }
        const unsigned length = static_cast<unsigned>(in_[pos_]) | (static_cast<unsigned>(in_[pos_ + 1]) << 8);
        const unsigned check = static_cast<unsigned>(in_[pos_ + 2]) | (static_cast<unsigned>(in_[pos_ + 3]) << 8);
        pos_ += 4;
        if (length != (~check & 0xFFFFu)) {
            return errors::data_corrupted("deflate: stored block length check failed");
        }
        if (in_.size() - pos_ < length) {
            return errors::data_corrupted("deflate: truncated stored block");
        }
        if (out_->size() + length > max_output_) {
            return errors::data_corrupted("deflate: output exceeds {} bytes", max_output_);
        }
        out_->insert(out_->end(), in_.begin() + static_cast<std::ptrdiff_t>(pos_),
                     in_.begin() + static_cast<std::ptrdiff_t>(pos_ + length));
        pos_ += length;
        return ok_status();
    }

    /// Decodes one symbol (canonical Huffman, bit by bit).
    int decode(const Huffman& h) {
        int code = 0;
        int first = 0;
        int index = 0;
        for (int len = 1; len <= kMaxBits; ++len) {
            code |= bits(1);
            if (!status_.ok()) {
                return -1;
            }
            const int count = h.count[static_cast<std::size_t>(len)];
            if (code - count < first) {
                return h.symbol[static_cast<std::size_t>(index + (code - first))];
            }
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        return -1;  // ran out of codes
    }

    /// Builds a decoding table from code lengths; fails on over-subscribed codes. Incomplete codes are allowed only
    /// when `allow_incomplete` (a distance code with a single code, as permitted by the format).
    static Status construct(Huffman& h, std::span<const std::uint8_t> lengths) {
        h.count.fill(0);
        for (const std::uint8_t len : lengths) {
            ++h.count[len];
        }
        if (h.count[0] == lengths.size()) {
            return ok_status();  // no codes: complete, but decoding will fail
        }
        int left = 1;
        for (int len = 1; len <= kMaxBits; ++len) {
            left <<= 1;
            left -= h.count[static_cast<std::size_t>(len)];
            if (left < 0) {
                return errors::data_corrupted("deflate: over-subscribed Huffman code");
            }
        }
        std::array<std::uint16_t, kMaxBits + 1> offsets{};
        for (int len = 1; len < kMaxBits; ++len) {
            offsets[static_cast<std::size_t>(len + 1)] = static_cast<std::uint16_t>(
                offsets[static_cast<std::size_t>(len)] + h.count[static_cast<std::size_t>(len)]);
        }
        for (std::size_t symbol = 0; symbol < lengths.size(); ++symbol) {
            if (lengths[symbol] != 0) {
                h.symbol[offsets[lengths[symbol]]++] = static_cast<std::uint16_t>(symbol);
            }
        }
        return ok_status();
    }

    Status codes(const Huffman& literal, const Huffman& distance) {
        static constexpr std::uint16_t kLengthBase[29] = {3,  4,  5,  6,   7,   8,   9,   10,  11, 13,
                                                          15, 17, 19, 23,  27,  31,  35,  43,  51, 59,
                                                          67, 83, 99, 115, 131, 163, 195, 227, 258};
        static constexpr std::uint8_t kLengthExtra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                                          2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
        static constexpr std::uint16_t kDistBase[30] = {1,    2,    3,    4,    5,    7,    9,    13,    17,    25,
                                                        33,   49,   65,   97,   129,  193,  257,  385,   513,   769,
                                                        1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
        static constexpr std::uint8_t kDistExtra[30] = {0, 0, 0, 0, 1, 1, 2, 2,  3,  3,  4,  4,  5,  5,  6,
                                                        6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
        while (true) {
            int symbol = decode(literal);
            if (symbol < 0 || !status_.ok()) {
                return status_.ok() ? errors::data_corrupted("deflate: invalid literal/length code") : status_;
            }
            if (symbol < 256) {
                II_TRY(emit(static_cast<std::byte>(symbol)));
                continue;
            }
            if (symbol == 256) {
                return ok_status();
            }
            symbol -= 257;
            if (symbol >= 29) {
                return errors::data_corrupted("deflate: invalid length symbol");
            }
            const int length = kLengthBase[symbol] + bits(kLengthExtra[symbol]);
            const int dist_symbol = decode(distance);
            if (dist_symbol < 0 || dist_symbol >= 30 || !status_.ok()) {
                return status_.ok() ? errors::data_corrupted("deflate: invalid distance code") : status_;
            }
            const auto dist = static_cast<std::size_t>(kDistBase[dist_symbol] + bits(kDistExtra[dist_symbol]));
            if (!status_.ok()) {
                return status_;
            }
            if (dist > out_->size()) {
                return errors::data_corrupted("deflate: distance too far back");
            }
            if (out_->size() + static_cast<std::size_t>(length) > max_output_) {
                return errors::data_corrupted("deflate: output exceeds {} bytes", max_output_);
            }
            for (int i = 0; i < length; ++i) {
                out_->push_back((*out_)[out_->size() - dist]);
            }
        }
    }

    Status fixed() {
        static const auto tables = [] {
            std::pair<Huffman, Huffman> t;
            std::array<std::uint8_t, kFixedLiteralCodes> lengths{};
            for (int s = 0; s < 144; ++s) {
                lengths[static_cast<std::size_t>(s)] = 8;
            }
            for (int s = 144; s < 256; ++s) {
                lengths[static_cast<std::size_t>(s)] = 9;
            }
            for (int s = 256; s < 280; ++s) {
                lengths[static_cast<std::size_t>(s)] = 7;
            }
            for (int s = 280; s < kFixedLiteralCodes; ++s) {
                lengths[static_cast<std::size_t>(s)] = 8;
            }
            (void)construct(t.first, lengths);
            std::array<std::uint8_t, kMaxDistanceCodes> dist{};
            dist.fill(5);
            (void)construct(t.second, dist);
            return t;
        }();
        return codes(tables.first, tables.second);
    }

    Status dynamic() {
        const int nlen = bits(5) + 257;
        const int ndist = bits(5) + 1;
        const int ncode = bits(4) + 4;
        if (!status_.ok()) {
            return status_;
        }
        if (nlen > kMaxLiteralCodes || ndist > kMaxDistanceCodes) {
            return errors::data_corrupted("deflate: bad code counts");
        }
        static constexpr std::uint8_t kOrder[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
        std::array<std::uint8_t, 19> code_lengths{};
        for (int i = 0; i < ncode; ++i) {
            code_lengths[kOrder[i]] = static_cast<std::uint8_t>(bits(3));
        }
        Huffman lencode;
        II_TRY(construct(lencode, code_lengths));
        std::array<std::uint8_t, kMaxLiteralCodes + kMaxDistanceCodes> lengths{};
        int index = 0;
        while (index < nlen + ndist) {
            int symbol = decode(lencode);
            if (symbol < 0 || !status_.ok()) {
                return status_.ok() ? errors::data_corrupted("deflate: invalid code length code") : status_;
            }
            if (symbol < 16) {
                lengths[static_cast<std::size_t>(index++)] = static_cast<std::uint8_t>(symbol);
                continue;
            }
            std::uint8_t len = 0;
            int repeat = 0;
            if (symbol == 16) {
                if (index == 0) {
                    return errors::data_corrupted("deflate: repeat without a previous length");
                }
                len = lengths[static_cast<std::size_t>(index - 1)];
                repeat = 3 + bits(2);
            } else if (symbol == 17) {
                repeat = 3 + bits(3);
            } else {
                repeat = 11 + bits(7);
            }
            if (index + repeat > nlen + ndist) {
                return errors::data_corrupted("deflate: too many code lengths");
            }
            while (repeat-- > 0) {
                lengths[static_cast<std::size_t>(index++)] = len;
            }
        }
        if (lengths[256] == 0) {
            return errors::data_corrupted("deflate: no end-of-block code");
        }
        Huffman literal;
        Huffman distance;
        II_TRY(construct(literal, std::span<const std::uint8_t>(lengths.data(), static_cast<std::size_t>(nlen))));
        II_TRY(
            construct(distance, std::span<const std::uint8_t>(lengths.data() + nlen, static_cast<std::size_t>(ndist))));
        return codes(literal, distance);
    }

    std::span<const std::byte> in_;
    std::size_t pos_ = 0;
    std::uint32_t bit_buffer_ = 0;
    int bit_count_ = 0;
    std::size_t max_output_;
    std::vector<std::byte>* out_ = nullptr;
    Status status_;
};

}  // namespace

Result<std::vector<std::byte>> inflate(std::span<const std::byte> data, std::size_t max_output) {
    std::vector<std::byte> out;
    Inflater inflater(data, max_output);
    II_TRY(inflater.run(out));
    return out;
}

std::uint32_t crc32(std::span<const std::byte> data, std::uint32_t crc) noexcept {
    static const auto table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t n = 0; n < 256; ++n) {
            std::uint32_t c = n;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            }
            t[n] = c;
        }
        return t;
    }();
    crc = ~crc;
    for (const std::byte b : data) {
        crc = table[(crc ^ static_cast<std::uint32_t>(b)) & 0xFF] ^ (crc >> 8);
    }
    return ~crc;
}

}  // namespace ii
