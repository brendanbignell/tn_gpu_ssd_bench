#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>
#include <random>
#include <chrono>
#include <iostream>
#include <cstring>
#include <cstdlib>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << ": " << cudaGetErrorString(err) << std::endl;       \
            return 1;                                                         \
        }                                                                     \
    } while (0)

size_t parse_size_arg(const char* arg) {
    char unit = arg[strlen(arg) - 1];
    size_t multiplier = 1;
    if (unit == 'G' || unit == 'g') multiplier = 1024L * 1024 * 1024;
    else if (unit == 'M' || unit == 'm') multiplier = 1024L * 1024;
    else if (unit == 'K' || unit == 'k') multiplier = 1024L;
    else return std::stoul(arg);
    return std::stoul(std::string(arg, strlen(arg) - 1)) * multiplier;
}

int main(int argc, char** argv) {
    // Defaults
    size_t total_size = 5L * 1024 * 1024 * 1024; // 5 GB
    int device_id = 0;
    const size_t chunk_size = 16 *1024 * 1024;
    const char* filepath = "/mnt/kvcache/huge_test_4tb.bin";

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--total_size") == 0 && i + 1 < argc) {
            total_size = parse_size_arg(argv[++i]);
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device_id = std::atoi(argv[++i]);
        } else {
            std::cerr << "Usage: " << argv[0]
                      << " [--total_size <bytes|10G|500M>] [--device <cuda_id>]\n";
            return 1;
        }
    }

    CUDA_CHECK(cudaSetDevice(device_id));

    int fd = open(filepath, O_RDONLY);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        perror("fstat");
        return 1;
    }
    size_t file_size = st.st_size;

    std::mt19937_64 rng(getpid() + time(nullptr));
    size_t max_offset = file_size - total_size;
    size_t offset = (rng() % (max_offset / chunk_size)) * chunk_size;

    void* src = mmap(nullptr, total_size, PROT_READ, MAP_PRIVATE, fd, offset);
    if (src == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    madvise(src, total_size, MADV_WILLNEED);

    // Touch pages
    /*
    volatile char dummy = 0;
    for (size_t i = 0; i < total_size; i += 4096) {
        dummy += ((volatile char*)src)[i];
    }
    */
    void* dst;
    CUDA_CHECK(cudaMalloc(&dst, total_size));

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < total_size; i += chunk_size) {
        const void* chunk_src = static_cast<const char*>(src) + i;
        void* chunk_dst = static_cast<char*>(dst) + i;
        CUDA_CHECK(cudaMemcpyAsync(chunk_dst, chunk_src, chunk_size, cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    double gbps = total_size / elapsed / (1024.0 * 1024.0 * 1024.0);

    std::cout << "Elapsed time: " << elapsed << " s\n";
    std::cout << "Transfer bandwidth: " << gbps << " GB/s\n";

    CUDA_CHECK(cudaFree(dst));
    munmap(src, total_size);
    close(fd);
    return 0;
}
