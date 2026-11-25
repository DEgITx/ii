// SPDX-License-Identifier: MIT
#include "core/platform/mapped_file.h"
#include "core/platform/platform.h"
#include "core/platform/windows/win32_util.h"

#include <windows.h>

namespace ii::platform {

namespace {

Status file_error(const char* operation, const std::filesystem::path& path) {
    const DWORD error = GetLastError();
    const StatusCode code =
        (error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND) ? StatusCode::kNotFound : StatusCode::kIoError;
    return Status(code, std::string(operation) + " '" + path_to_utf8(path) + "': " + win32::error_message(error));
}

struct HandleCloser {
    HANDLE handle;
    ~HandleCloser() {
        if (handle != nullptr && handle != INVALID_HANDLE_VALUE) {
            CloseHandle(handle);
        }
    }
};

}  // namespace

Result<MappedFile> MappedFile::open(const std::filesystem::path& path) {
    const HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                    FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        return file_error("cannot open", path);
    }
    const HandleCloser file_closer{file};

    LARGE_INTEGER size{};
    if (!GetFileSizeEx(file, &size)) {
        return file_error("cannot get the size of", path);
    }
    if (size.QuadPart == 0) {
        return MappedFile();
    }
    if (static_cast<unsigned long long>(size.QuadPart) > static_cast<unsigned long long>(SIZE_MAX)) {
        return errors::out_of_range("'{}' is too large to map", path_to_utf8(path));
    }

    const HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping == nullptr) {
        return file_error("cannot create a file mapping for", path);
    }
    const HandleCloser mapping_closer{mapping};

    // The view keeps the mapping and the file alive after their handles are closed.
    const void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (view == nullptr) {
        return file_error("cannot map", path);
    }
    return MappedFile(view, static_cast<std::size_t>(size.QuadPart));
}

void MappedFile::unmap() noexcept {
    if (data_ != nullptr) {
        UnmapViewOfFile(data_);
        data_ = nullptr;
        size_ = 0;
    }
}

}  // namespace ii::platform
