#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>
#include <random>
#include <chrono>
#include <iostream>
#include <cstring>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << ": " << cudaGetErrorString(err) << std::endl;       \
            return 1;                                                         \
        }                                                                     \
    } while (0)

int main() {
    //const size_t size = 256 * 1024 * 1024; 
    const size_t size = 1L * 1024 * 1024 * 1024; // 1 GB

    const size_t block_size = 1024 * 1024;       // 1 MB
    const char* filepath = "/mnt/kvcache/huge_test_4tb.bin";

    // Open file
    int fd = open(filepath, O_RDONLY);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    // Get file size
    struct stat st;
    if (fstat(fd, &st) < 0) {
        perror("fstat");
        return 1;
    }
    size_t file_size = st.st_size;

    // Random 1MB-aligned offset
    std::mt19937_64 rng(getpid() + time(nullptr));
    size_t max_offset = file_size - size;
    size_t offset = (rng() % (max_offset / block_size)) * block_size;

    // mmap the region
    void* src = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, offset);
    if (src == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    // Advise the kernel to prefetch
    if (madvise(src, size, MADV_WILLNEED) != 0) {
        perror("madvise");
    }

    // Fault in all pages to avoid skewed timing
    volatile char dummy = 0;
    for (size_t i = 0; i < size; i += 4096) {
        dummy += ((volatile char*)src)[i];
    }

    // Allocate GPU memory
    void* dst;
    CUDA_CHECK(cudaMalloc(&dst, size));

    // Time GPU transfer
    auto start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count();
    double gbps = size / elapsed / (1024.0 * 1024.0 * 1024.0);

    std::cout << "Elapsed time: " << elapsed << " s" << std::endl;
    std::cout << "Transfer bandwidth: " << gbps << " GB/s" << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(dst));
    munmap(src, size);
    close(fd);

    return 0;
}
