// SPDX-License-Identifier: MIT
#include "core/platform/mapped_file.h"
#include "core/platform/platform.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <string>

namespace ii::platform {

namespace {

Status errno_error(const char* operation, const std::filesystem::path& path, int error) {
    const StatusCode code = (error == ENOENT || error == ENOTDIR) ? StatusCode::kNotFound : StatusCode::kIoError;
    return Status(code, std::string(operation) + " '" + path_to_utf8(path) + "': " + std::strerror(error));
}

}  // namespace

Result<MappedFile> MappedFile::open(const std::filesystem::path& path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return errno_error("cannot open", path, errno);
    }
    struct stat st{};
    if (fstat(fd, &st) != 0) {
        const int error = errno;
        ::close(fd);
        return errno_error("cannot stat", path, error);
    }
    if (!S_ISREG(st.st_mode)) {
        ::close(fd);
        return errors::invalid_argument("'{}' is not a regular file", path_to_utf8(path));
    }
    if (st.st_size == 0) {
        ::close(fd);
        return MappedFile();
    }
    const auto size = static_cast<std::size_t>(st.st_size);
    void* data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    const int error = errno;
    ::close(fd);  // the mapping stays valid after the descriptor is closed
    if (data == MAP_FAILED) {
        return errno_error("cannot map", path, error);
    }
    return MappedFile(data, size);
}

void MappedFile::unmap() noexcept {
    if (data_ != nullptr) {
        munmap(const_cast<void*>(data_), size_);
        data_ = nullptr;
        size_ = 0;
    }
}

}  // namespace ii::platform
