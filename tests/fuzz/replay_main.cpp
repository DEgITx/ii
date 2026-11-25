// SPDX-License-Identifier: MIT
// Runs a fuzz target over corpus files without libFuzzer, so that every build (any compiler) replays the corpus as a
// regression test: replay <file-or-directory>...
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size);

namespace {

int replay(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    LLVMFuzzerTestOneInput(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    int count = 0;
    for (int i = 1; i < argc; ++i) {
        const std::filesystem::path path(argv[i]);
        if (std::filesystem::is_directory(path)) {
            for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
                if (entry.is_regular_file()) {
                    count += replay(entry.path());
                }
            }
        } else {
            count += replay(path);
        }
    }
    std::printf("replayed %d inputs\n", count);
    return count > 0 ? 0 : 1;
}
