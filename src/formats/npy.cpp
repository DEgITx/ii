// SPDX-License-Identifier: MIT
// NumPy .npy / .npz reading and writing.
#include <ii/io.h>

#include "core/compress.h"
#include "core/platform/mapped_file.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <format>
#include <fstream>
#include <limits>
#include <optional>

namespace ii {

namespace {

static_assert(std::endian::native == std::endian::little, "the NumPy readers assume a little-endian host");

constexpr char kMagic[] = "\x93NUMPY";

std::string utf8(const std::filesystem::path& path) {
    const std::u8string text = path.u8string();
    return std::string(text.begin(), text.end());
}

/// NumPy type string ("<f4") of a dtype.
std::optional<std::string> descr_of(DType dtype) {
    switch (dtype) {
    case DType::kF32:
        return "<f4";
    case DType::kF64:
        return "<f8";
    case DType::kF16:
        return "<f2";
    case DType::kI8:
        return "|i1";
    case DType::kU8:
        return "|u1";
    case DType::kI16:
        return "<i2";
    case DType::kU16:
        return "<u2";
    case DType::kI32:
        return "<i4";
    case DType::kU32:
        return "<u4";
    case DType::kI64:
        return "<i8";
    case DType::kU64:
        return "<u8";
    case DType::kBool:
        return "|b1";
    default:
        return std::nullopt;
    }
}

DType dtype_of_descr(std::string_view descr) {
    if (descr.size() < 3) {
        return DType::kInvalid;
    }
    const char order = descr[0];
    const std::string_view type = descr.substr(1);
    const bool single_byte = type == "i1" || type == "u1" || type == "b1";
    if (order != '<' && order != '=' && order != '|' && !(order == '>' && single_byte)) {
        return DType::kInvalid;  // big-endian
    }
    for (const DType dtype : {DType::kF32, DType::kF64, DType::kF16, DType::kI8, DType::kU8, DType::kI16, DType::kU16,
                              DType::kI32, DType::kU32, DType::kI64, DType::kU64, DType::kBool}) {
        if (descr_of(dtype)->substr(1) == type) {
            return dtype;
        }
    }
    return DType::kInvalid;
}

/// The value of `key` in the header dictionary: the text after "'key':" up to the next top-level ',' or '}'.
std::optional<std::string_view> header_value(std::string_view header, std::string_view key) {
    const std::string quoted = std::format("'{}'", key);
    std::size_t pos = header.find(quoted);
    if (pos == std::string_view::npos) {
        return std::nullopt;
    }
    pos = header.find(':', pos + quoted.size());
    if (pos == std::string_view::npos) {
        return std::nullopt;
    }
    ++pos;
    while (pos < header.size() && header[pos] == ' ') {
        ++pos;
    }
    int depth = 0;
    std::size_t end = pos;
    for (; end < header.size(); ++end) {
        const char c = header[end];
        if (c == '(') {
            ++depth;
        } else if (c == ')') {
            --depth;
        } else if ((c == ',' || c == '}') && depth == 0) {
            break;
        }
    }
    return header.substr(pos, end - pos);
}

std::uint32_t read_u32(std::span<const std::byte> data, std::size_t offset) {
    std::uint32_t value = 0;
    std::memcpy(&value, data.data() + offset, 4);
    return value;
}

std::uint16_t read_u16(std::span<const std::byte> data, std::size_t offset) {
    std::uint16_t value = 0;
    std::memcpy(&value, data.data() + offset, 2);
    return value;
}

std::uint64_t read_u64(std::span<const std::byte> data, std::size_t offset) {
    std::uint64_t value = 0;
    std::memcpy(&value, data.data() + offset, 8);
    return value;
}

Status write_bytes(const std::filesystem::path& path, std::span<const std::byte> data) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    out.close();
    if (!out) {
        return errors::io_error("cannot write '{}'", utf8(path));
    }
    return ok_status();
}

void append(std::vector<std::byte>& out, std::uint64_t value, int bytes) {
    for (int i = 0; i < bytes; ++i) {
        out.push_back(static_cast<std::byte>((value >> (8 * i)) & 0xFF));
    }
}

}  // namespace

Result<Tensor> parse_npy(std::span<const std::byte> data) {
    if (data.size() < 10 || std::memcmp(data.data(), kMagic, 6) != 0) {
        return errors::data_corrupted("not an .npy file");
    }
    const auto major = static_cast<int>(data[6]);
    std::size_t header_start = 10;
    std::size_t header_size = read_u16(data, 8);
    if (major >= 2) {
        if (data.size() < 12) {
            return errors::data_corrupted(".npy: truncated header");
        }
        header_start = 12;
        header_size = read_u32(data, 8);
    }
    if (header_size > data.size() - header_start) {
        return errors::data_corrupted(".npy: truncated header");
    }
    const std::string_view header(reinterpret_cast<const char*>(data.data() + header_start), header_size);
    const std::optional<std::string_view> descr = header_value(header, "descr");
    const std::optional<std::string_view> fortran = header_value(header, "fortran_order");
    const std::optional<std::string_view> shape_text = header_value(header, "shape");
    if (!descr || !fortran || !shape_text || descr->size() < 2) {
        return errors::data_corrupted(".npy: malformed header");
    }
    const std::string_view type = descr->substr(1, descr->size() - 2);  // strip quotes
    const DType dtype = dtype_of_descr(type);
    if (dtype == DType::kInvalid) {
        return errors::unsupported(".npy: element type '{}' is not supported", type);
    }
    // Shape: "(2, 3)" / "(5,)" / "()".
    Shape shape;
    std::int64_t numel = 1;
    std::int64_t current = -1;
    for (const char c : *shape_text) {
        if (c >= '0' && c <= '9') {
            current = (current < 0 ? 0 : current) * 10 + (c - '0');
            if (current > (std::int64_t{1} << 48)) {
                return errors::data_corrupted(".npy: dimension too large");
            }
        } else if (c == ',' || c == ')') {
            if (current >= 0) {
                if (shape.rank() >= kMaxRank) {
                    return errors::unsupported(".npy: more than {} dimensions", kMaxRank);
                }
                shape.push_back(current);
                if (current != 0 && numel > (std::int64_t{1} << 48) / current) {
                    return errors::data_corrupted(".npy: too many elements");
                }
                numel *= current;
                current = -1;
            }
        } else if (c != '(' && c != ' ' && c != 'L') {
            return errors::data_corrupted(".npy: malformed shape '{}'", *shape_text);
        }
    }
    const std::size_t offset = header_start + header_size;
    const std::size_t nbytes = storage_size(dtype, numel);
    if (data.size() - offset < nbytes) {
        return errors::data_corrupted(".npy: {} bytes of data, expected {}", data.size() - offset, nbytes);
    }
    const bool fortran_order = fortran->starts_with("True");
    if (!fortran_order || shape.rank() < 2) {
        return Tensor::from_bytes(dtype, shape, data.data() + offset, nbytes);
    }
    // Fortran order: the data is the transpose in C order.
    Shape reversed;
    std::vector<int> perm;
    for (int d = shape.rank() - 1; d >= 0; --d) {
        reversed.push_back(shape[d]);
        perm.push_back(d);
    }
    II_ASSIGN_OR_RETURN(Tensor transposed, Tensor::from_bytes(dtype, reversed, data.data() + offset, nbytes));
    II_ASSIGN_OR_RETURN(Tensor view, transposed.permute(perm));
    return view.contiguous();
}

Result<std::vector<std::byte>> serialize_npy(const Tensor& tensor) {
    const std::optional<std::string> descr = descr_of(tensor.dtype());
    if (!descr) {
        return errors::unsupported(".npy: element type {} has no NumPy equivalent", to_string(tensor.dtype()));
    }
    std::string shape = "(";
    for (int d = 0; d < tensor.rank(); ++d) {
        shape += std::format("{}{}", tensor.shape()[d], tensor.rank() == 1 ? "," : d + 1 < tensor.rank() ? ", " : "");
    }
    shape += ")";
    std::string header = std::format("{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}", *descr, shape);
    // Pad with spaces so that the data starts at a multiple of 64 bytes; the header ends with a newline.
    const std::size_t total = 10 + header.size() + 1;
    header.append((64 - total % 64) % 64, ' ');
    header += '\n';
    if (header.size() > 65535) {
        return errors::unsupported(".npy: header too long");
    }
    II_ASSIGN_OR_RETURN(const Tensor contiguous, tensor.contiguous());
    std::string prefix(kMagic, 6);
    prefix += static_cast<char>(1);  // format version 1.0
    prefix += static_cast<char>(0);
    prefix += static_cast<char>(header.size() & 0xFF);
    prefix += static_cast<char>(header.size() >> 8);
    prefix += header;
    std::vector<std::byte> out(prefix.size() + contiguous.nbytes());
    std::memcpy(out.data(), prefix.data(), prefix.size());
    if (contiguous.nbytes() != 0) {
        std::memcpy(out.data() + prefix.size(), contiguous.data(), contiguous.nbytes());
    }
    return out;
}

Result<Tensor> load_npy(const std::filesystem::path& path) {
    II_ASSIGN_OR_RETURN(platform::MappedFile file, platform::MappedFile::open(path));
    Result<Tensor> tensor = parse_npy(file.bytes());
    if (!tensor.ok()) {
        Status status = tensor.status();
        return status.add_context(utf8(path));
    }
    return tensor;
}

Status save_npy(const std::filesystem::path& path, const Tensor& tensor) {
    II_ASSIGN_OR_RETURN(const std::vector<std::byte> data, serialize_npy(tensor));
    return write_bytes(path, data);
}

Result<std::vector<NamedTensor>> load_npz(const std::filesystem::path& path) {
    II_ASSIGN_OR_RETURN(platform::MappedFile file, platform::MappedFile::open(path));
    Result<std::vector<NamedTensor>> arrays = parse_npz(file.bytes());
    if (!arrays.ok()) {
        Status status = arrays.status();
        return status.add_context(utf8(path));
    }
    return arrays;
}

Result<std::vector<NamedTensor>> parse_npz(std::span<const std::byte> data) {
    const auto corrupted = [&](std::string_view what) {
        return errors::data_corrupted("not a valid .npz archive: {}", what);
    };
    // End of central directory: the last 22+ bytes, possibly followed by a comment.
    if (data.size() < 64) {
        return corrupted("too small");
    }
    std::size_t eocd = std::string::npos;
    const std::size_t lowest = data.size() > 22 + 65535 ? data.size() - 22 - 65535 : 0;
    for (std::size_t pos = data.size() - 22 + 1; pos-- > lowest;) {
        if (read_u32(data, pos) == 0x06054b50) {
            eocd = pos;
            break;
        }
    }
    if (eocd == std::string::npos) {
        return corrupted("no end of central directory");
    }
    std::uint64_t entries = read_u16(data, eocd + 10);
    std::uint64_t directory_size = read_u32(data, eocd + 12);
    std::uint64_t directory_offset = read_u32(data, eocd + 16);
    if (directory_offset == 0xFFFFFFFF || entries == 0xFFFF) {
        // ZIP64: the locator precedes the end of central directory.
        if (eocd < 20 || read_u32(data, eocd - 20) != 0x07064b50) {
            return corrupted("missing ZIP64 locator");
        }
        const std::uint64_t record = read_u64(data, eocd - 20 + 8);
        if (record > data.size() - 56 || read_u32(data, static_cast<std::size_t>(record)) != 0x06064b50) {
            return corrupted("bad ZIP64 record");
        }
        entries = read_u64(data, static_cast<std::size_t>(record) + 32);
        directory_size = read_u64(data, static_cast<std::size_t>(record) + 40);
        directory_offset = read_u64(data, static_cast<std::size_t>(record) + 48);
    }
    if (directory_offset > data.size() || directory_size > data.size() - directory_offset) {
        return corrupted("central directory outside of the file");
    }
    std::vector<NamedTensor> arrays;
    std::size_t pos = static_cast<std::size_t>(directory_offset);
    const std::size_t directory_end = pos + static_cast<std::size_t>(directory_size);
    for (std::uint64_t e = 0; e < entries; ++e) {
        if (directory_end - pos < 46 || read_u32(data, pos) != 0x02014b50) {
            return corrupted("bad central directory entry");
        }
        const std::uint16_t method = read_u16(data, pos + 10);
        const std::uint32_t crc = read_u32(data, pos + 16);
        std::uint64_t compressed = read_u32(data, pos + 20);
        std::uint64_t uncompressed = read_u32(data, pos + 24);
        const std::uint16_t name_size = read_u16(data, pos + 28);
        const std::uint16_t extra_size = read_u16(data, pos + 30);
        const std::uint16_t comment_size = read_u16(data, pos + 32);
        std::uint64_t local = read_u32(data, pos + 42);
        if (directory_end - pos - 46 < static_cast<std::size_t>(name_size) + extra_size + comment_size) {
            return corrupted("truncated central directory entry");
        }
        std::string name(reinterpret_cast<const char*>(data.data() + pos + 46), name_size);
        // ZIP64 extended information.
        std::size_t extra = pos + 46 + name_size;
        const std::size_t extra_end = extra + extra_size;
        while (extra + 4 <= extra_end) {
            const std::uint16_t id = read_u16(data, extra);
            const std::uint16_t size = read_u16(data, extra + 2);
            if (extra + 4 + size > extra_end) {
                break;
            }
            if (id == 0x0001) {
                std::size_t field = extra + 4;
                const auto next = [&](std::uint64_t& value) {
                    if (value == 0xFFFFFFFF && field + 8 <= extra + 4 + size) {
                        value = read_u64(data, field);
                        field += 8;
                    }
                };
                next(uncompressed);
                next(compressed);
                next(local);
            }
            extra += 4 + static_cast<std::size_t>(size);
        }
        pos += 46 + static_cast<std::size_t>(name_size) + extra_size + comment_size;

        if (local > data.size() - 30 || read_u32(data, static_cast<std::size_t>(local)) != 0x04034b50) {
            return corrupted(std::format("bad local header of '{}'", name));
        }
        const std::size_t start = static_cast<std::size_t>(local) + 30 +
                                  read_u16(data, static_cast<std::size_t>(local) + 26) +
                                  read_u16(data, static_cast<std::size_t>(local) + 28);
        if (start > data.size() || compressed > data.size() - start) {
            return corrupted(std::format("data of '{}' outside of the file", name));
        }
        const std::span<const std::byte> stored = data.subspan(start, static_cast<std::size_t>(compressed));
        std::vector<std::byte> inflated;
        std::span<const std::byte> bytes;
        if (method == 0) {
            bytes = stored;
        } else if (method == 8) {
            if (uncompressed > (std::uint64_t{1} << 40)) {
                return corrupted("entry too large");
            }
            Result<std::vector<std::byte>> result = inflate(stored, static_cast<std::size_t>(uncompressed));
            if (!result.ok()) {
                Status status = result.status();
                return status.add_context(std::format("'{}'", name));
            }
            inflated = std::move(*result);
            bytes = inflated;
        } else {
            return errors::unsupported("'{}' uses ZIP compression method {}", name, method);
        }
        if (bytes.size() != uncompressed || crc32(bytes) != crc) {
            return corrupted(std::format("'{}' fails its size or CRC check", name));
        }
        Result<Tensor> tensor = parse_npy(bytes);
        if (!tensor.ok()) {
            Status status = tensor.status();
            return status.add_context(std::format("'{}'", name));
        }
        if (name.ends_with(".npy")) {
            name.resize(name.size() - 4);
        }
        arrays.emplace_back(std::move(name), std::move(*tensor));
    }
    return arrays;
}

Status save_npz(const std::filesystem::path& path, std::span<const NamedTensor> arrays) {
    std::vector<std::byte> out;
    std::vector<std::byte> directory;
    constexpr std::uint16_t kDosDate = (0 << 9) | (1 << 5) | 1;  // 1980-01-01
    for (const auto& [name, tensor] : arrays) {
        II_ASSIGN_OR_RETURN(const std::vector<std::byte> npy, serialize_npy(tensor));
        const std::string entry = name + ".npy";
        if (npy.size() >= 0xFFFFFFFFu || out.size() >= 0xFFFFFFFFu || entry.size() > 0xFFFF) {
            return errors::unsupported("save_npz: archives larger than 4 GiB are not supported");
        }
        const std::uint32_t crc = crc32(npy);
        const std::size_t local = out.size();
        append(out, 0x04034b50, 4);
        append(out, 20, 2);  // version needed
        append(out, 0, 2);   // flags
        append(out, 0, 2);   // stored
        append(out, 0, 2);   // time
        append(out, kDosDate, 2);
        append(out, crc, 4);
        append(out, npy.size(), 4);
        append(out, npy.size(), 4);
        append(out, entry.size(), 2);
        append(out, 0, 2);
        for (const char c : entry) {
            out.push_back(static_cast<std::byte>(c));
        }
        out.insert(out.end(), npy.begin(), npy.end());

        append(directory, 0x02014b50, 4);
        append(directory, 20, 2);  // version made by
        append(directory, 20, 2);  // version needed
        append(directory, 0, 2);
        append(directory, 0, 2);
        append(directory, 0, 2);
        append(directory, kDosDate, 2);
        append(directory, crc, 4);
        append(directory, npy.size(), 4);
        append(directory, npy.size(), 4);
        append(directory, entry.size(), 2);
        append(directory, 0, 2);  // extra
        append(directory, 0, 2);  // comment
        append(directory, 0, 2);  // disk
        append(directory, 0, 2);  // internal attributes
        append(directory, 0, 4);  // external attributes
        append(directory, local, 4);
        for (const char c : entry) {
            directory.push_back(static_cast<std::byte>(c));
        }
    }
    const std::size_t directory_offset = out.size();
    out.insert(out.end(), directory.begin(), directory.end());
    if (arrays.size() > 0xFFFF || out.size() >= 0xFFFFFFFFu) {
        return errors::unsupported("save_npz: archives larger than 4 GiB are not supported");
    }
    append(out, 0x06054b50, 4);
    append(out, 0, 2);
    append(out, 0, 2);
    append(out, arrays.size(), 2);
    append(out, arrays.size(), 2);
    append(out, directory.size(), 4);
    append(out, directory_offset, 4);
    append(out, 0, 2);
    return write_bytes(path, out);
}

}  // namespace ii
